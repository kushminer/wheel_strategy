"""Main trading loop - orchestrates monitoring, scanning, and execution."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING, List, Optional, Tuple

import pytz

from csp.data.models import OptionContract
from csp.data.options import GreeksCalculator
from csp.data.vix import VixDataFetcher
from csp.trading.models import ActivePosition, ExitReason, RiskCheckResult

if TYPE_CHECKING:
    from csp.config import StrategyConfig
    from csp.data.manager import DataManager
    from csp.signals.scanner import StrategyScanner
    from csp.trading.execution import ExecutionEngine
    from csp.trading.portfolio import PortfolioManager
    from csp.trading.risk import RiskManager

logger = logging.getLogger(__name__)


class TradingLoop:
    """
    Main trading loop that orchestrates the CSP strategy.

    Responsibilities:
    1. Check market hours
    2. Monitor VIX regime
    3. Check existing positions for exits
    4. Scan for new opportunities
    5. Execute trades
    """

    def __init__(
        self,
        config: "StrategyConfig",
        data_manager: "DataManager",
        scanner: "StrategyScanner",
        portfolio: "PortfolioManager",
        risk_manager: "RiskManager",
        execution: "ExecutionEngine",
        vix_fetcher: VixDataFetcher,
        greeks_calc: GreeksCalculator,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.scanner = scanner
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution = execution
        self.vix_fetcher = vix_fetcher
        self.greeks_calc = greeks_calc

        self.eastern = pytz.timezone("US/Eastern")
        self._running = False
        self._session_vix_open: Optional[Tuple[object, float]] = None

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        now = datetime.now(self.eastern)

        if now.weekday() > 4:
            return False

        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        return market_open <= now.time() <= market_close

    def get_session_vix_reference(self) -> float:
        """Get VIX reference for current session."""
        session_date = datetime.now(self.eastern).date()

        if self._session_vix_open is None:
            _, vix_open = self.vix_fetcher.get_session_reference_vix()
            self._session_vix_open = (session_date, vix_open)

        if self._session_vix_open[0] != session_date:
            _, vix_open = self.vix_fetcher.get_session_reference_vix()
            self._session_vix_open = (session_date, vix_open)

        return self._session_vix_open[1]

    def check_global_vix_stop(self, current_vix: float) -> bool:
        """Check if global VIX stop is triggered."""
        reference_vix = self.get_session_vix_reference()
        threshold = reference_vix * self.config.vix_spike_multiplier
        return current_vix >= threshold

    def monitor_positions(
        self, current_vix: float
    ) -> List[Tuple[ActivePosition, RiskCheckResult, float]]:
        """
        Check all positions for exit conditions.

        Returns:
            List of (position, risk_result, current_premium) that should be closed
        """
        exits_needed: List[Tuple[ActivePosition, RiskCheckResult, float]] = []
        reference_vix = self.get_session_vix_reference()

        if not self.data_manager.equity_fetcher or not self.data_manager.options_fetcher:
            return exits_needed

        for position in self.portfolio.get_active_positions():
            try:
                current_stock_price = self.data_manager.equity_fetcher.get_current_price(
                    position.symbol
                )

                snapshots = self.data_manager.options_fetcher.get_option_snapshots(
                    [position.option_symbol]
                )

                if position.option_symbol not in snapshots:
                    logger.warning("No data for %s", position.option_symbol)
                    continue

                snapshot = snapshots[position.option_symbol]
                current_premium = snapshot.get("ask", 0.0)
                current_delta = snapshot.get("delta")

                if current_delta is None and snapshot.get("bid") and snapshot.get("ask"):
                    mid = (snapshot["bid"] + snapshot["ask"]) / 2
                    greeks = self.greeks_calc.compute_greeks_from_price(
                        mid,
                        current_stock_price,
                        position.strike,
                        position.current_dte,
                        "put",
                    )
                    current_delta = greeks.get("delta", position.entry_delta)

                if current_delta is None:
                    current_delta = position.entry_delta

                risk_result = self.risk_manager.evaluate_position(
                    position=position,
                    current_delta=current_delta,
                    current_stock_price=current_stock_price,
                    current_vix=current_vix,
                    current_premium=current_premium,
                    reference_vix=reference_vix,
                )

                if risk_result.should_exit and risk_result.exit_reason:
                    exits_needed.append((position, risk_result, current_premium))

            except Exception as e:
                logger.exception("Error monitoring %s: %s", position.symbol, e)

        return exits_needed

    def execute_exit(
        self,
        position: ActivePosition,
        risk_result: RiskCheckResult,
        current_premium: float,
    ) -> bool:
        """Execute exit for a position. Returns True if exit order submitted."""
        if risk_result.exit_reason:
            logger.info(
                "Exiting %s: %s - %s",
                position.symbol,
                risk_result.exit_reason.value,
                risk_result.details,
            )

        result = self.execution.buy_to_close(
            option_symbol=position.option_symbol,
            quantity=abs(position.quantity),
            limit_price=None,
        )

        if result.success:
            exit_premium = current_premium
            early_exit = risk_result.current_values.get("early_exit")
            if isinstance(early_exit, dict) and "current_premium" in early_exit:
                exit_premium = early_exit["current_premium"]

            self.portfolio.close_position(
                position_id=position.position_id,
                exit_premium=exit_premium,
                exit_reason=risk_result.exit_reason or ExitReason.MANUAL,
                exit_details=risk_result.details,
            )

            pnl = position.calculate_pnl(exit_premium)
            logger.info("Exit order submitted. Est. P&L: $%.2f", pnl)
            return True

        logger.warning("Exit order failed: %s", result.message)
        return False

    def scan_and_enter(self, deployable_cash: float) -> int:
        """Scan for new opportunities and enter positions. Returns count entered."""
        available_cash = self.portfolio.get_available_cash(deployable_cash)

        if available_cash <= 0:
            return 0

        if self.portfolio.active_count >= self.config.num_tickers:
            return 0

        active_symbols = set(self.portfolio.active_symbols)

        candidates: List[OptionContract] = self.scanner.get_all_candidates(
            skip_equity_filter=False,
            max_total=20,
        )

        candidates = [c for c in candidates if c.underlying not in active_symbols]

        entered = 0
        current_vix = self.vix_fetcher.get_current_vix()

        for candidate in candidates:
            if not self.portfolio.can_add_position(
                candidate.collateral_required, deployable_cash
            ):
                continue

            if candidate.underlying in self.portfolio.active_symbols:
                continue

            logger.info(
                "Entering %s: %s | Strike: $%.2f, Premium: $%.2f, DTE: %d",
                candidate.underlying,
                candidate.symbol,
                candidate.strike,
                candidate.bid,
                candidate.dte,
            )

            result = self.execution.sell_to_open(
                option_symbol=candidate.symbol,
                quantity=1,
                limit_price=candidate.bid,
            )

            if result.success:
                position = ActivePosition(
                    position_id="",
                    symbol=candidate.underlying,
                    option_symbol=candidate.symbol,
                    entry_date=datetime.now(),
                    entry_stock_price=candidate.stock_price,
                    entry_delta=candidate.delta or -0.20,
                    entry_premium=candidate.bid,
                    entry_vix=current_vix,
                    entry_iv=candidate.implied_volatility or 0.30,
                    strike=candidate.strike,
                    expiration=candidate.expiration,
                    dte_at_entry=candidate.dte,
                    quantity=-1,
                    entry_order_id=result.order_id,
                )

                self.portfolio.add_position(position)
                logger.info("Position opened: %s", position.position_id)
                entered += 1
            else:
                logger.warning("Order failed: %s", result.message)

        return entered

    def run_cycle(self) -> dict:
        """Run a single trading cycle. Returns cycle summary dict."""
        cycle_start = datetime.now()
        summary: dict = {
            "timestamp": cycle_start.isoformat(),
            "market_open": self.is_market_open(),
            "exits": 0,
            "entries": 0,
            "errors": [],
        }

        try:
            current_vix = self.vix_fetcher.get_current_vix()
            summary["current_vix"] = current_vix

            deployable_cash = self.config.get_deployable_cash(current_vix)
            summary["deployable_cash"] = deployable_cash

            if self.check_global_vix_stop(current_vix):
                logger.warning(
                    "GLOBAL VIX STOP TRIGGERED - VIX: %.2f", current_vix
                )
                summary["global_vix_stop"] = True

                vix_result = RiskCheckResult(
                    should_exit=True,
                    exit_reason=ExitReason.VIX_SPIKE,
                    details=f"Global VIX stop: {current_vix:.2f}",
                    current_values={"current_vix": current_vix},
                )

                for position in self.portfolio.get_active_positions():
                    if self.data_manager.options_fetcher:
                        snapshots = self.data_manager.options_fetcher.get_option_snapshots(
                            [position.option_symbol]
                        )
                        premium = (
                            snapshots.get(position.option_symbol, {}).get(
                                "ask", 0.0
                            )
                        )
                    else:
                        premium = 0.0

                    if self.execute_exit(position, vix_result, premium):
                        summary["exits"] += 1

                summary["portfolio"] = self.portfolio.get_summary()
                return summary

            exits_needed = self.monitor_positions(current_vix)

            for position, risk_result, current_premium in exits_needed:
                if self.execute_exit(position, risk_result, current_premium):
                    summary["exits"] += 1

            if self.is_market_open() and deployable_cash > 0:
                entries = self.scan_and_enter(deployable_cash)
                summary["entries"] = entries

            summary["portfolio"] = self.portfolio.get_summary()

        except Exception as e:
            summary["errors"].append(str(e))
            logger.exception("Cycle error: %s", e)

        return summary

    def run(
        self,
        poll_interval: int = 60,
        max_cycles: Optional[int] = None,
    ) -> None:
        """Run the main trading loop."""
        self._running = True
        cycle_count = 0

        logger.info(
            "CSP TRADING LOOP STARTED | Poll: %ds | Paper: %s",
            poll_interval,
            self.execution.paper,
        )

        try:
            while self._running:
                cycle_count += 1

                logger.info(
                    "Cycle %d @ %s",
                    cycle_count,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )

                summary = self.run_cycle()

                logger.info(
                    "VIX: %.2f | Market: %s | Exits: %d | Entries: %d",
                    summary.get("current_vix", 0),
                    summary.get("market_open", False),
                    summary.get("exits", 0),
                    summary.get("entries", 0),
                )

                if "portfolio" in summary:
                    p = summary["portfolio"]
                    logger.info(
                        "Positions: %d | Collateral: $%.2f",
                        p.get("active_positions", 0),
                        p.get("total_collateral", 0),
                    )

                if max_cycles and cycle_count >= max_cycles:
                    logger.info("Max cycles (%d) reached. Stopping.", max_cycles)
                    break

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Loop stopped by user.")
        finally:
            self._running = False
            logger.info(
                "Trading loop ended. Cycles: %d | %s",
                cycle_count,
                self.portfolio.get_summary(),
            )

    def stop(self) -> None:
        """Stop the trading loop."""
        self._running = False
