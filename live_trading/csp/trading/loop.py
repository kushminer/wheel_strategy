"""Main trading loop that orchestrates the CSP strategy."""

import asyncio
import re
from datetime import date, datetime, time as dt_time
from itertools import groupby
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytz

from csp.clients import AlpacaClientManager
from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.data.manager import DataManager
from csp.data.vix import VixDataFetcher
from csp.signals.scanner import StrategyScanner
from csp.storage import build_storage_backend
from csp.trading.daily_log import DailyLog
from csp.trading.execution import ExecutionEngine
from csp.trading.metadata import StrategyMetadataStore
from csp.trading.models import ExitReason, OrderResult, RiskCheckResult
from csp.trading.risk import RiskManager


async def _arun(fn, *args, **kwargs):
    """Run a sync function in a thread pool (non-blocking)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


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
        config: 'StrategyConfig',
        data_manager: 'DataManager',
        scanner: 'StrategyScanner',
        metadata_store: 'StrategyMetadataStore',
        risk_manager: RiskManager,
        execution: ExecutionEngine,
        vix_fetcher: 'VixDataFetcher',
        greeks_calc: 'GreeksCalculator',
        alpaca_manager: 'AlpacaClientManager' = None
    ):
        self.config = config
        self.data_manager = data_manager
        self.scanner = scanner
        self.metadata = metadata_store
        self.risk_manager = risk_manager
        self.execution = execution
        self.vix_fetcher = vix_fetcher
        self.greeks_calc = greeks_calc
        self.alpaca_manager = alpaca_manager
        
        self.eastern = pytz.timezone('US/Eastern')
        self._running = False
        self._session_vix_open = None
        
        # Daily scan state
        self._equity_passing: Optional[List[str]] = None  # Symbols passing equity filter today
        self._equity_scan_date: Optional[date] = None     # Date of last equity scan
        self._monitor_only: bool = False                  # True = no new entries, only track exits
        self._last_scan_results: List = []                # Full scan results for diagnostics        
        self._cycle_count: int = 0
        self._last_step_log: List[dict] = []  # Step-by-step data from last order attempt
        self._api_semaphore = asyncio.Semaphore(config.max_concurrent_options_fetches)
        self._storage_backend = build_storage_backend(config)
        self.logger = DailyLog(log_dir="logs", backend=self._storage_backend)
        print(f"  Daily log: {self.logger.today_path}")

        # Summary-mode cycle counters
        self._last_equity_passed = 0
        self._last_options_evaluated = 0

    def _vprint(self, msg: str):
        """Print only in verbose mode."""
        if self.config.print_mode == "verbose":
            print(msg)

    # ── Alpaca helpers (source of truth for positions) ──────────

    async def _get_alpaca_positions(self) -> list:
        """Fetch current positions from Alpaca."""
        try:
            return await _arun(self.alpaca_manager.trading_client.get_all_positions)
        except Exception as e:
            self._vprint(f"  Warning: Could not fetch Alpaca positions: {e}")
            return []

    async def _get_option_positions(self) -> list:
        """Filter Alpaca positions to option positions only."""
        return [
            p for p in await self._get_alpaca_positions()
            if len(p.symbol) > 10 and any(c.isdigit() for c in p.symbol)
        ]

    async def _get_active_symbols(self) -> set:
        """Get underlying symbols from active Alpaca option positions."""
        symbols = set()
        for pos in await self._get_option_positions():
            match = re.match(r'^([A-Z]+)\d', pos.symbol)
            if match:
                symbols.add(match.group(1))
        return symbols

    async def _get_position_count(self) -> int:
        """Count active option positions on Alpaca."""
        return len(await self._get_option_positions())

    def _build_position_proxy(self, alpaca_pos, meta: dict):
        """Build lightweight object with position-like attributes
        so RiskManager works without changes."""
        strike = AlpacaClientManager.parse_strike_from_symbol(alpaca_pos.symbol)
        expiration = AlpacaClientManager.parse_expiration_from_symbol(alpaca_pos.symbol)
        entry_date_str = meta.get('entry_date', datetime.now().isoformat())

        proxy = SimpleNamespace(
            symbol=meta.get('underlying', ''),
            option_symbol=alpaca_pos.symbol,
            quantity=int(float(alpaca_pos.qty)),
            strike=strike,
            expiration=expiration or date.today(),
            position_id=alpaca_pos.symbol,
            entry_delta=meta.get('entry_delta', 0),
            entry_iv=meta.get('entry_iv', 0),
            entry_vix=meta.get('entry_vix', 0),
            entry_stock_price=meta.get('entry_stock_price', 0),
            entry_premium=meta.get('entry_premium', 0),
            entry_daily_return=meta.get('entry_daily_return', 0),
            dte_at_entry=meta.get('dte_at_entry', 0),
            entry_order_id=meta.get('entry_order_id', ''),
        )
        proxy.current_dte = (proxy.expiration - date.today()).days
        proxy.days_held = (date.today() - datetime.fromisoformat(entry_date_str).date()).days
        proxy.collateral_required = abs(proxy.quantity) * proxy.strike * 100
        def calculate_pnl(exit_premium, _p=proxy):
            return (_p.entry_premium - exit_premium) * abs(_p.quantity) * 100
        proxy.calculate_pnl = calculate_pnl
        return proxy

    async def _get_portfolio_summary(self) -> dict:
        """Build portfolio summary from Alpaca + metadata."""
        positions = await self._get_option_positions()
        collateral = await _arun(self.alpaca_manager.get_short_collateral)
        return {
            'active_positions': len(positions),
            'total_collateral': collateral,
            'symbols': list(await self._get_active_symbols()),
        }

    async def is_market_open(self) -> bool:
        """Check if US market is currently open using Alpaca calendar API.
        Caches the trading-day check per calendar date so only one API call per day.
        """
        now = datetime.now(self.eastern)
        today = now.date()

        # Weekday check (Mon=0, Fri=4) — fast reject weekends
        if now.weekday() > 4:
            return False

        # Check if today is a trading day (holiday check, cached per day)
        if not hasattr(self, '_trading_day_cache') or self._trading_day_cache.get('date') != today:
            try:
                from alpaca.trading.requests import GetCalendarRequest
                cal_req = GetCalendarRequest(start=today, end=today)
                cal = await _arun(self.alpaca_manager.trading_client.get_calendar, cal_req)
                # cal[0].date is a date object; cal[0].open/close are datetime objects
                is_trading_day = len(cal) > 0 and cal[0].date == today
                self._trading_day_cache = {
                    'date': today,
                    'is_trading_day': is_trading_day,
                    'open': cal[0].open.time() if is_trading_day else None,
                    'close': cal[0].close.time() if is_trading_day else None,
                }
                if not is_trading_day:
                    print(f"  Market closed today ({today} is not a trading day)")
            except Exception as e:
                self._vprint(f"  Warning: Alpaca calendar check failed ({e}), falling back to time-only check")
                self._trading_day_cache = {'date': today, 'is_trading_day': True, 'open': None, 'close': None}

        if not self._trading_day_cache['is_trading_day']:
            return False

        # Time check — use Alpaca hours if available, else default 9:30-16:00 ET
        market_open = self._trading_day_cache.get('open') or dt_time(9, 30)
        market_close = self._trading_day_cache.get('close') or dt_time(16, 0)

        return market_open <= now.time() <= market_close
    
    async def get_session_vix_reference(self) -> float:
        """
        Get VIX reference for current session.
        Uses session open VIX, cached for the day.
        """
        session_date = datetime.now(self.eastern).date()

        if self._session_vix_open is None:
            # Get last session's open
            _, vix_open = await _arun(self.vix_fetcher.get_session_reference_vix)
            self._session_vix_open = (session_date, vix_open)

        # Reset if new day
        if self._session_vix_open[0] != session_date:
            _, vix_open = await _arun(self.vix_fetcher.get_session_reference_vix)
            self._session_vix_open = (session_date, vix_open)

        return self._session_vix_open[1]

    async def check_global_vix_stop(self, current_vix: float) -> bool:
        """
        Check if global VIX stop is triggered.
        If VIX >= 1.15x session open, close ALL positions.

        Returns:
            True if global stop triggered
        """
        reference_vix = await self.get_session_vix_reference()
        threshold = reference_vix * self.config.vix_spike_multiplier

        return current_vix >= threshold
    
    async def monitor_positions(self, current_vix: float) -> List[Tuple]:
        """
        Check all positions for exit conditions (parallel).

        Returns:
            List of (position, risk_result, current_premium) tuples that should be closed
        """
        exits_needed = []

        # Check for assignments first (no market data needed)
        assignment_exits = await self._check_assignments()
        exits_needed.extend(assignment_exits)
        assigned_ids = {pos.option_symbol for pos, _, _ in assignment_exits}

        reference_vix = await self.get_session_vix_reference()

        # Build position proxies, filtering out assigned and unknown
        positions_to_check = []
        for alpaca_pos in await self._get_option_positions():
            option_sym = alpaca_pos.symbol
            if option_sym in assigned_ids:
                continue

            meta = self.metadata.get(option_sym)
            if meta is None:
                self._vprint(f"  Warning: No metadata for {option_sym}, skipping risk checks")
                continue
            positions_to_check.append(self._build_position_proxy(alpaca_pos, meta))

        # Check all positions in parallel
        if positions_to_check:
            results = await asyncio.gather(
                *[self._check_single_position(pos, current_vix, reference_vix)
                  for pos in positions_to_check]
            )
            exits_needed.extend(r for r in results if r is not None)

        return exits_needed

    async def _check_single_position(
        self, position, current_vix: float, reference_vix: float
    ) -> Optional[Tuple]:
        """Check a single position for exit conditions. Safe for asyncio.gather()."""
        async with self._api_semaphore:
            try:
                # Check expiration proximity
                days_to_expiry = (position.expiration - date.today()).days
                if days_to_expiry <= self.config.close_before_expiry_days:
                    # Fetch premium for P&L tracking
                    snapshots = await _arun(
                        self.data_manager.options_fetcher.get_option_snapshots,
                        [position.option_symbol]
                    )
                    snapshot = snapshots.get(position.option_symbol, {})
                    current_premium = snapshot.get('ask', 0)

                    print(f"  {position.symbol}: Expiring in {days_to_expiry} day(s), triggering exit")
                    expiry_result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.EXPIRY,
                        details=f"Position expiring in {days_to_expiry} day(s) (threshold: {self.config.close_before_expiry_days}d)",
                        current_values={
                            'days_to_expiry': days_to_expiry,
                            'expiration': position.expiration.isoformat(),
                            'current_premium': current_premium,
                        }
                    )
                    return (position, expiry_result, current_premium)

                # Get current data for the position
                current_stock_price = await _arun(
                    self.data_manager.equity_fetcher.get_current_price,
                    position.symbol
                )

                # Get current option data
                snapshots = await _arun(
                    self.data_manager.options_fetcher.get_option_snapshots,
                    [position.option_symbol]
                )

                if position.option_symbol not in snapshots:
                    self._vprint(f"  Warning: No data for {position.option_symbol}")
                    return None

                snapshot = snapshots[position.option_symbol]
                current_premium = snapshot.get('ask', 0)  # Use ask to buy back
                current_delta = snapshot.get('delta')

                # Calculate delta if not provided by snapshot
                if current_delta is None and snapshot.get('bid') and snapshot.get('ask'):
                    mid = (snapshot['bid'] + snapshot['ask']) / 2
                    greeks = self.greeks_calc.compute_greeks_from_price(
                        mid, current_stock_price, position.strike,
                        position.current_dte, 'put'
                    )
                    current_delta = greeks.get('delta')  # No fallback to entry_delta

                # Handle missing delta
                if current_delta is None:
                    if self.config.exit_on_missing_delta:
                        self._vprint(f"  WARNING: Delta unavailable for {position.symbol}, triggering exit (exit_on_missing_delta=True)")
                        data_result = RiskCheckResult(
                            should_exit=True,
                            exit_reason=ExitReason.DATA_UNAVAILABLE,
                            details=f"Delta could not be retrieved or computed for {position.option_symbol}",
                            current_values={
                                'entry_delta': position.entry_delta,
                                'current_premium': current_premium,
                            }
                        )
                        return (position, data_result, current_premium)
                    else:
                        self._vprint(f"  WARNING: Delta unavailable for {position.symbol}, using entry_delta as fallback")
                        current_delta = position.entry_delta

                # Run risk evaluation
                risk_result = self.risk_manager.evaluate_position(
                    position=position,
                    current_delta=current_delta,
                    current_stock_price=current_stock_price,
                    current_vix=current_vix,
                    current_premium=current_premium,
                    reference_vix=reference_vix
                )

                if risk_result.should_exit:
                    return (position, risk_result, current_premium)

                return None

            except Exception as e:
                print(f"  Error monitoring {position.symbol}: {e}")
                return None


    async def execute_exit(
        self,
        position,
        risk_result: RiskCheckResult,
        current_premium: float = 0.0,
    ) -> bool:
        """
        Execute exit for a position. Routes to appropriate order type based on exit reason.

        Args:
            position: The position to close
            risk_result: Risk check result with exit reason and details
            current_premium: Current option premium (ask price), passed directly
                             from monitor_positions for reliable P&L tracking

        Returns:
            True if exit completed successfully
        """
        print(f"  Exiting {position.symbol}: {risk_result.exit_reason.value}")
        print(f"    {risk_result.details}")

        exit_premium = current_premium

        # === Assignment: no order needed ===
        if risk_result.exit_reason == ExitReason.ASSIGNED:
            exit_premium = 0.0
            print(f"    Assignment detected -- no order needed")

            self.metadata.record_exit(
                option_symbol=position.option_symbol,
                exit_reason=risk_result.exit_reason.value,
                exit_details=risk_result.details,
                exit_order_id="assignment",
            )
            pnl = position.calculate_pnl(exit_premium)
            print(f"    Position closed (assigned). Option P&L: ${pnl:.2f}")
            return True

        # === Early exit & Expiry: use stepped limit ===
        if risk_result.exit_reason in (ExitReason.EARLY_EXIT, ExitReason.EXPIRY):
            result_tuple = await self._execute_stepped_exit(position)

            if result_tuple is not None:
                result, filled_price = result_tuple
                exit_premium = filled_price

                self.metadata.record_exit(
                    option_symbol=position.option_symbol,
                    exit_reason=risk_result.exit_reason.value,
                    exit_details=risk_result.details,
                    exit_order_id=result.order_id,
                )
                pnl = position.calculate_pnl(exit_premium)
                print(f"    Stepped exit filled @ ${exit_premium:.2f}. P&L: ${pnl:.2f}")
                return True
            else:
                # Stepped exit exhausted -- fall back to market order
                print(f"    Stepped exit exhausted, falling back to market order")
                result = await _arun(
                    self.execution.buy_to_close,
                    option_symbol=position.option_symbol,
                    quantity=abs(position.quantity),
                    limit_price=None,
                )
                if result.success:
                    self.metadata.record_exit(
                        option_symbol=position.option_symbol,
                        exit_reason=risk_result.exit_reason.value,
                        exit_details=risk_result.details,
                        exit_order_id=result.order_id,
                    )
                    pnl = position.calculate_pnl(exit_premium)
                    print(f"    Market fallback submitted. Est. P&L: ${pnl:.2f}")
                    return True
                else:
                    print(f"    Market fallback also failed: {result.message}")
                    return False

        # === Stop-loss exits: always market order (immediate) ===
        print(f"    Stop-loss: market order")
        result = await _arun(
            self.execution.buy_to_close,
            option_symbol=position.option_symbol,
            quantity=abs(position.quantity),
            limit_price=None,
        )

        if result and result.success:
            self.metadata.record_exit(
                option_symbol=position.option_symbol,
                exit_reason=risk_result.exit_reason.value,
                exit_details=risk_result.details,
                exit_order_id=result.order_id,
            )
            pnl = position.calculate_pnl(exit_premium)
            print(f"    Exit order submitted. Est. P&L: ${pnl:.2f}")
            return True
        else:
            msg = result.message if result else "No order result"
            print(f"    Exit order failed: {msg}")
            return False


    async def _refresh_equity_scan(self) -> List[str]:
        """
        Run equity scan once per day. Cache passing symbols and full results.
        Returns list of equity-passing symbols.
        """
        today = datetime.now(self.eastern).date()

        if self._equity_scan_date == today and self._equity_passing is not None:
            return self._equity_passing

        scan_start = datetime.now()
        print(f"  Initiating equity scan at {datetime.now(self.eastern).strftime('%H:%M:%S %Z')}...")
        scan_results = await _arun(self.scanner.scan_universe, skip_equity_filter=False)
        scan_elapsed = (datetime.now() - scan_start).total_seconds()
        
        passing_equity = [r for r in scan_results if r.equity_result.passes]
        self._equity_passing = [r.symbol for r in passing_equity]
        self._equity_scan_date = today
        self._last_scan_results = scan_results  # Cache full results for diagnostics
        self._last_equity_passed = len(passing_equity)
        
        print(f"  Symbols scanned:                         {len(scan_results)}")
        print(f"  Passed equity filter:                     {len(passing_equity)}")
        print(f"  Scan completed in {scan_elapsed:.1f}s")
        
        # Print equity-passing table
        if passing_equity:
            bb_label = f"BB{self.config.bb_period}"
            self._vprint(f"\n  \u2713 Equity-passing symbols ({len(passing_equity)}):")
            self._vprint(f"  {'Symbol':<8} {'Price':>9} {'SMA8':>9} {'SMA20':>9} {'SMA50':>9} {bb_label:>9} {'RSI':>6} {'Collateral':>12}")
            self._vprint("  " + "-" * 72)
            for result in passing_equity:
                r = result.equity_result
                collateral = r.current_price * 100
                self._vprint(
                    f"  {r.symbol:<8} "
                    f"${r.current_price:>8.2f} "
                    f"{r.sma_8:>9.2f} "
                    f"{r.sma_20:>9.2f} "
                    f"{r.sma_50:>9.2f} "
                    f"{r.bb_upper:>9.2f} "
                    f"{r.rsi:>6.1f} "
                    f"${collateral:>10,.0f}"
                )
        else:
            print("\n  \u26a0 No symbols passed the equity filter.")
        
        # Log equity scan
        self.logger.log_equity_scan(
            [r.equity_result for r in scan_results],
            self._equity_passing
        )
        self.logger.flush()

        return self._equity_passing

    def _get_sort_key(self, contract):
        """Get sort key based on configured per-ticker rank mode."""
        if self.config.contract_rank_mode == "daily_return_per_delta":
            return contract.daily_return_per_delta
        elif self.config.contract_rank_mode == "days_since_strike":
            return contract.days_since_strike or 0
        elif self.config.contract_rank_mode == "lowest_strike_price":
            return -contract.strike
        elif self.config.contract_rank_mode == "lowest_delta":
            return -(abs(contract.delta) if contract.delta else 1.0)
        else:  # "daily_return_on_collateral"
            return contract.daily_return_on_collateral

    def _get_universe_sort_key(self, contract):
        """Get sort key for cross-ticker ranking (universe_rank_mode)."""
        if self.config.universe_rank_mode == "daily_return_per_delta":
            return contract.daily_return_per_delta
        elif self.config.universe_rank_mode == "days_since_strike":
            return contract.days_since_strike or 0
        elif self.config.universe_rank_mode == "lowest_strike_price":
            return -contract.strike
        elif self.config.universe_rank_mode == "lowest_delta":
            return -(abs(contract.delta) if contract.delta else 1.0)
        else:  # "daily_return_on_collateral"
            return contract.daily_return_on_collateral

    def compute_target_quantity(self, collateral_per_contract: float, available_cash: float) -> int:
        """Compute number of CSP contracts for a ticker.
        If cash >= max_position_pct of portfolio: qty = floor(max_position_pct * portfolio / collateral)
        Else: qty = floor(available_cash / collateral)
        Capped by max_contracts_per_ticker.
        """
        if collateral_per_contract <= 0:
            return 1
        target_allocation = self.config.starting_cash * self.config.max_position_pct
        if available_cash >= target_allocation:
            n = int(target_allocation // collateral_per_contract)
        else:
            n = int(available_cash // collateral_per_contract)
        n = min(n, self.config.max_contracts_per_ticker)
        return max(1, n)

    async def _execute_market_entry(
        self,
        candidate: 'OptionContract',
        qty: int,
    ) -> Optional[Tuple['OrderResult', float]]:
        """Execute a market order entry for a CSP position.

        Used when entry_order_type='market'. Reliable with delayed (indicative) data
        since no limit price depends on stale quotes.
        """
        symbol = candidate.symbol
        self._vprint(f"    Market order: selling {qty}x {symbol}")

        self._last_step_log = []

        result = await _arun(
            self.execution.sell_to_open,
            option_symbol=symbol,
            quantity=qty,
            limit_price=None,
        )

        if not result.success:
            self._vprint(f"    Market order failed: {result.message}")
            self._last_step_log.append({
                "step": 0, "type": "market", "status": "failed",
                "message": result.message,
            })
            return None

        # Brief wait for fill to propagate
        await asyncio.sleep(2)

        status = await _arun(self.execution.get_order_status, result.order_id)
        filled_price = (
            float(status['filled_avg_price'])
            if status and status.get('filled_avg_price')
            else candidate.bid
        )
        order_status = status['status'] if status else 'unknown'

        self._last_step_log.append({
            "step": 0, "type": "market", "status": order_status,
            "filled_price": filled_price,
        })

        if order_status in ('filled', 'partially_filled'):
            self._vprint(f"    FILLED @ ${filled_price:.2f}")
            return (result, filled_price)

        self._vprint(f"    Market order status: {order_status} (expected fill)")
        return (result, filled_price)

    async def _execute_stepped_entry(
        self,
        candidate: 'OptionContract',
        qty: int,
        current_vix: float,
    ) -> Optional[Tuple['OrderResult', float]]:
        """Execute a stepped limit order entry for a CSP position.

        Starts at mid (or bid) and steps down toward bid over multiple
        attempts, optionally re-fetching the snapshot and re-validating
        the contract between steps.

        Returns:
            Tuple of (OrderResult, filled_price) if filled, or None if exhausted.
        """
        cfg = self.config
        symbol = candidate.symbol

        bid = candidate.bid
        ask = candidate.ask
        mid = candidate.mid
        spread = ask - bid

        # Initial limit price
        if cfg.entry_start_price == "mid":
            limit_price = mid
        else:
            limit_price = bid

        # Floor: never go below bid; also respect the max-step computed floor
        floor_from_steps = mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread)
        price_floor = max(bid, floor_from_steps)
        limit_price = round(max(limit_price, price_floor), 2)

        self._vprint(f"    Stepped entry: start=${limit_price:.2f}, "
              f"bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, "
              f"spread=${spread:.2f}, floor=${price_floor:.2f}")

        self._last_step_log = []

        for step in range(cfg.entry_max_steps + 1):
            self._vprint(f"    Step {step}/{cfg.entry_max_steps}: limit @ ${limit_price:.2f}")

            result = await _arun(
                self.execution.sell_to_open,
                option_symbol=symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                self._vprint(f"    Step {step}: order submission failed — {result.message}")
                return None

            order_id = result.order_id

            self._vprint(f"    Step {step}: waiting {cfg.entry_step_interval}s for fill...")
            await asyncio.sleep(cfg.entry_step_interval)

            status = await _arun(self.execution.get_order_status, order_id)
            self._last_step_log.append({
                "step": step, "limit_price": limit_price,
                "status": status['status'] if status else 'unknown',
                "duration_s": cfg.entry_step_interval,
                "bid": bid, "ask": ask, "mid": round(mid, 2), "spread": round(spread, 2),
            })

            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                tag = "FILLED" if status['status'] == 'filled' else f"PARTIAL ({status['filled_qty']}/{qty})"
                self._vprint(f"    Step {step}: {tag} @ ${filled_price:.2f}")
                return (result, filled_price)

            # Not filled — cancel
            self._vprint(f"    Step {step}: not filled (status={status['status'] if status else 'unknown'}), cancelling...")
            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)  # brief pause for cancel to propagate

            # Re-check in case fill happened during cancel
            status = await _arun(self.execution.get_order_status, order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                self._vprint(f"    Step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            # Last step — give up
            if step >= cfg.entry_max_steps:
                self._vprint(f"    All {cfg.entry_max_steps} steps exhausted. Giving up on {candidate.underlying}.")
                return None

            # Optionally re-fetch snapshot
            if cfg.entry_refetch_snapshot:
                snapshots = await _arun(self.data_manager.options_fetcher.get_option_snapshots, [symbol])
                if symbol not in snapshots:
                    self._vprint(f"    Snapshot unavailable after re-fetch. Aborting.")
                    return None

                snap = snapshots[symbol]
                new_bid = float(snap.get('bid', 0) or 0)
                new_ask = float(snap.get('ask', 0) or 0)

                if new_bid <= 0:
                    self._vprint(f"    Bid is zero after re-fetch. Aborting.")
                    return None

                new_mid = (new_bid + new_ask) / 2
                new_spread = new_ask - new_bid

                # Update candidate for re-validation
                candidate.bid = new_bid
                candidate.ask = new_ask
                candidate.mid = new_mid
                if snap.get('delta') is not None:
                    candidate.delta = snap['delta']
                if snap.get('implied_volatility') is not None:
                    candidate.implied_volatility = snap['implied_volatility']
                if snap.get('volume') is not None:
                    candidate.volume = snap['volume']
                if snap.get('open_interest') is not None:
                    candidate.open_interest = snap['open_interest']

                # Re-validate against filters
                filter_result = self.scanner.options_filter.evaluate(candidate)
                if not filter_result.passes:
                    self._vprint(f"    Contract no longer passes filters: {filter_result.failure_reasons}")
                    return None

                bid, ask, mid, spread = new_bid, new_ask, new_mid, new_spread
                price_floor = max(bid, mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread))

                self._vprint(f"    Refreshed: bid=${bid:.2f}, ask=${ask:.2f}, "
                      f"mid=${mid:.2f}, spread=${spread:.2f}, floor=${price_floor:.2f}")

            # Compute next step price
            next_step = step + 1
            if cfg.entry_start_price == "mid":
                limit_price = mid - (next_step * cfg.entry_step_pct * spread)
            else:
                limit_price = bid - (next_step * cfg.entry_step_pct * spread)

            limit_price = round(max(limit_price, price_floor), 2)

        return None

    async def _execute_single_entry(
        self,
        candidate,
        qty: int,
        current_vix: float,
        index: int,
        total_count: int,
    ) -> bool:
        """Execute a single entry order. Safe for asyncio.gather().

        Args:
            candidate: The OptionContract to trade
            qty: Number of contracts (pre-sized by sizing pass)
            current_vix: Current VIX level
            index: 1-based position in the entry queue (for logging)
            total_count: Total number of entries being attempted

        Returns:
            True if entry was filled successfully.
        """
        total_collateral = candidate.collateral_required * qty
        target_val = self.config.starting_cash * self.config.max_position_pct
        spread = candidate.ask - candidate.bid if candidate.ask and candidate.bid else 0
        delta_str = f"{abs(candidate.delta):.3f}" if candidate.delta else "N/A"

        print(f"\n  [{index}/{total_count}] ENTERING {candidate.underlying}: {candidate.symbol}")
        print(f"    Stock: ${candidate.stock_price:.2f} | Strike: ${candidate.strike:.2f} | DTE: {candidate.dte}")
        self._vprint(f"    Bid: ${candidate.bid:.2f} | Ask: ${candidate.ask:.2f} | Mid: ${candidate.mid:.2f} | Spread: ${spread:.2f}")
        self._vprint(f"    Delta: {delta_str} | IV: {candidate.implied_volatility:.1%} | Daily: {candidate.daily_return_on_collateral:.4%}")
        print(f"    Qty: {qty} | Collateral: ${total_collateral:,.0f} (target: ${target_val:,.0f})")

        # Execute entry
        if self.config.entry_order_type == "market":
            entry_result = await self._execute_market_entry(
                candidate=candidate, qty=qty,
            )
        else:
            entry_result = await self._execute_stepped_entry(
                candidate=candidate, qty=qty, current_vix=current_vix,
            )

        # Capture step log immediately (no await between entry return and copy,
        # so this is safe even when multiple entries run via asyncio.gather)
        step_log = list(self._last_step_log)

        # Log order attempt
        self.logger.log_order_attempt(
            action="entry",
            symbol=candidate.underlying,
            contract=candidate.symbol,
            steps=step_log,
            outcome="filled" if entry_result is not None else "exhausted",
            filled_price=entry_result[1] if entry_result else None,
            start_price=candidate.mid,
            floor_price=candidate.bid,
            qty=qty,
        )

        if entry_result is not None:
            result, filled_price = entry_result
            improvement = filled_price - candidate.bid

            self.metadata.record_entry(
                option_symbol=candidate.symbol,
                underlying=candidate.underlying,
                strike=candidate.strike,
                expiration=candidate.expiration.isoformat(),
                entry_delta=candidate.delta,
                entry_iv=candidate.implied_volatility,
                entry_vix=current_vix,
                entry_stock_price=candidate.stock_price,
                entry_premium=filled_price,
                entry_daily_return=candidate.daily_return_on_collateral,
                dte_at_entry=candidate.dte,
                quantity=-qty,
                entry_order_id=result.order_id,
            )
            print(f"    FILLED: {candidate.symbol}")
            print(f"      {qty} contracts @ ${filled_price:.2f} (bid was ${candidate.bid:.2f}, improvement: ${improvement:+.2f})")
            print(f"      Total premium: ${filled_price * qty * 100:,.2f} | Collateral: ${total_collateral:,.0f}")
            return True
        else:
            if self.config.entry_order_type == "market":
                print(f"    FAILED: Market order entry failed for {candidate.underlying}")
            else:
                print(f"    FAILED: Entry exhausted for {candidate.underlying} after {self.config.entry_max_steps} steps")
            return False

    async def _execute_stepped_exit(
        self,
        position,
    ) -> Optional[Tuple['OrderResult', float]]:
        """Execute a stepped limit order exit (buy-to-close) for a CSP position.

        Starts at mid (or ask) and steps UP toward ask over multiple
        attempts, optionally re-fetching the snapshot between steps.

        Returns:
            Tuple of (OrderResult, filled_price) if filled, or None if exhausted.
        """
        cfg = self.config
        option_symbol = position.option_symbol
        qty = abs(position.quantity)

        # Fetch current snapshot
        snapshots = await _arun(self.data_manager.options_fetcher.get_option_snapshots, [option_symbol])
        if option_symbol not in snapshots:
            self._vprint(f"    Stepped exit: no snapshot for {option_symbol}")
            return None

        snap = snapshots[option_symbol]
        bid = float(snap.get('bid', 0) or 0)
        ask = float(snap.get('ask', 0) or 0)

        if ask <= 0:
            self._vprint(f"    Stepped exit: ask is zero, aborting")
            return None

        mid = (bid + ask) / 2
        spread = ask - bid

        # Initial limit price
        if cfg.exit_start_price == "ask":
            limit_price = ask
        else:
            limit_price = mid

        # Ceiling: never go above ask
        ceiling_from_steps = mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread)
        price_ceiling = min(ask, ceiling_from_steps)
        limit_price = round(min(limit_price, price_ceiling), 2)

        self._vprint(f"    Stepped exit: start=${limit_price:.2f}, "
              f"bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, "
              f"spread=${spread:.2f}, ceiling=${price_ceiling:.2f}")

        for step in range(cfg.exit_max_steps + 1):
            self._vprint(f"    Step {step}/{cfg.exit_max_steps}: limit @ ${limit_price:.2f}")

            result = await _arun(
                self.execution.buy_to_close,
                option_symbol=option_symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                self._vprint(f"    Step {step}: order submission failed -- {result.message}")
                return None

            order_id = result.order_id

            self._vprint(f"    Step {step}: waiting {cfg.exit_step_interval}s for fill...")
            await asyncio.sleep(cfg.exit_step_interval)

            status = await _arun(self.execution.get_order_status, order_id)

            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                tag = "FILLED" if status['status'] == 'filled' else f"PARTIAL ({status['filled_qty']}/{qty})"
                self._vprint(f"    Step {step}: {tag} @ ${filled_price:.2f}")
                return (result, filled_price)

            # Not filled -- cancel
            self._vprint(f"    Step {step}: not filled (status={status['status'] if status else 'unknown'}), cancelling...")
            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)

            # Re-check in case fill happened during cancel
            status = await _arun(self.execution.get_order_status, order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                self._vprint(f"    Step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            # Last step -- give up
            if step >= cfg.exit_max_steps:
                self._vprint(f"    All {cfg.exit_max_steps} steps exhausted for exit of {position.symbol}.")
                return None

            # Optionally re-fetch snapshot
            if cfg.exit_refetch_snapshot:
                snapshots = await _arun(self.data_manager.options_fetcher.get_option_snapshots, [option_symbol])
                if option_symbol not in snapshots:
                    self._vprint(f"    Snapshot unavailable after re-fetch. Aborting.")
                    return None

                snap = snapshots[option_symbol]
                new_bid = float(snap.get('bid', 0) or 0)
                new_ask = float(snap.get('ask', 0) or 0)

                if new_ask <= 0:
                    self._vprint(f"    Ask is zero after re-fetch. Aborting.")
                    return None

                new_mid = (new_bid + new_ask) / 2
                new_spread = new_ask - new_bid

                bid, ask, mid, spread = new_bid, new_ask, new_mid, new_spread
                price_ceiling = min(ask, mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread))

                self._vprint(f"    Refreshed: bid=${bid:.2f}, ask=${ask:.2f}, "
                      f"mid=${mid:.2f}, spread=${spread:.2f}, ceiling=${price_ceiling:.2f}")

            # Compute next step price (step UP toward ask)
            next_step = step + 1
            if cfg.exit_start_price == "ask":
                limit_price = ask  # Already at ask, stay there
            else:
                limit_price = mid + (next_step * cfg.exit_step_pct * spread)

            limit_price = round(min(limit_price, price_ceiling), 2)

        return None

    async def _check_assignments(self) -> List[Tuple]:
        """Detect assignments by comparing internal portfolio against Alpaca holdings.

        Assignment signal: our portfolio has an active short put for symbol X,
        but Alpaca no longer has the option position AND Alpaca now holds
        shares of X with side='long'.

        Returns:
            List of (position, risk_result, current_premium=0) for assigned positions.
        """
        assigned = []

        if not self.alpaca_manager:
            return assigned

        try:
            alpaca_positions = await _arun(self.alpaca_manager.trading_client.get_all_positions)
        except Exception as e:
            self._vprint(f"  Warning: Could not fetch Alpaca positions for assignment check: {e}")
            return assigned

        # Build lookup maps
        alpaca_option_symbols = set()
        alpaca_stock_holdings = {}  # symbol -> qty (float)

        for pos in alpaca_positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)

            # Options use OCC format (long symbol with digits), stocks are short tickers
            if any(c.isdigit() for c in symbol) and len(symbol) > 10:
                alpaca_option_symbols.add(symbol)
            else:
                if side == 'long' or qty > 0:
                    alpaca_stock_holdings[symbol] = abs(qty)

        # Check each position we have metadata for
        for option_sym, meta in self.metadata.get_active().items():
            if option_sym not in alpaca_option_symbols:
                underlying = meta.get('underlying', '')
                shares_held = alpaca_stock_holdings.get(underlying, 0)
                expected_shares = abs(meta.get('quantity', 1)) * 100

                if shares_held >= expected_shares:
                    print(f"  ASSIGNMENT DETECTED: {underlying}")
                    print(f"    Option {option_sym} no longer in Alpaca")
                    print(f"    Now holding {shares_held:.0f} shares of {underlying}")

                    proxy = SimpleNamespace(
                        symbol=underlying, option_symbol=option_sym,
                        position_id=option_sym,
                        quantity=meta.get('quantity', -1),
                        strike=meta.get('strike', 0),
                        expiration=date.fromisoformat(meta['expiration']) if meta.get('expiration') else date.today(),
                        entry_delta=meta.get('entry_delta', 0),
                        entry_iv=meta.get('entry_iv', 0),
                        entry_premium=meta.get('entry_premium', 0),
                        current_dte=0,
                    )
                    proxy.calculate_pnl = lambda ep, m=meta: (m.get('entry_premium', 0) - ep) * abs(m.get('quantity', 1)) * 100

                    result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.ASSIGNED,
                        details=f"Assigned: {option_sym} removed, now hold {shares_held:.0f} shares of {underlying}",
                        current_values={'shares_held': shares_held, 'option_symbol': option_sym}
                    )
                    assigned.append((proxy, result, 0.0))

        return assigned


    async def _fetch_symbol_options(self, symbol: str) -> Optional[dict]:
        """Fetch options chain data for a single symbol. Safe for asyncio.gather()."""
        async with self._api_semaphore:
            try:
                stock_price = await _arun(self.data_manager.equity_fetcher.get_current_price, symbol)

                sma_ceiling = None
                if self.config.max_strike_mode == "sma" and hasattr(self, '_last_scan_results'):
                    for sr in self._last_scan_results:
                        if sr.symbol == symbol:
                            sma_ceiling = getattr(sr.equity_result, f"sma_{self.config.max_strike_sma_period}", None)
                            break

                puts = await _arun(
                    self.data_manager.options_fetcher.get_puts_chain,
                    symbol, stock_price, self.config, sma_ceiling=sma_ceiling
                )

                price_history = await _arun(
                    self.data_manager.equity_fetcher.get_close_history,
                    [symbol], days=self.config.history_days
                )
                if symbol in price_history:
                    prices = price_history[symbol]
                    for put in puts:
                        at_or_below = prices[prices <= put.strike]
                        if at_or_below.empty:
                            put.days_since_strike = 999
                        else:
                            last_date = at_or_below.index[-1]
                            put.days_since_strike = (prices.index[-1] - last_date).days

                ranked, filter_results = await _arun(self.scanner.options_filter.filter_and_rank, puts)

                return {
                    "symbol": symbol,
                    "stock_price": stock_price,
                    "puts_count": len(puts),
                    "ranked": ranked,
                    "filter_results": filter_results,
                }
            except Exception as e:
                self._vprint(f"  Error fetching options for {symbol}: {e}")
                return None

    async def scan_and_enter(self, deployable_cash: float) -> int:
        """
        Scan for new opportunities and enter positions.
        Uses cached daily equity scan; only fetches options for passing symbols.
        Produces verbose output matching the Universe Scan cell diagnostic format.

        Returns:
            Number of new positions entered, or -1 if no candidates for the day
        """
        if self._monitor_only:
            return 0

        available_cash = min(await _arun(self.alpaca_manager.compute_available_capital), deployable_cash)

        if available_cash <= 0:
            return 0

        if await self._get_position_count() >= self.config.num_tickers:
            return 0
        
        # Run equity scan (cached per day)
        equity_passing = await self._refresh_equity_scan()
        
        if not equity_passing:
            print("  No symbols pass equity filter today.")
            return -1  # -1 means "nothing to do all day"

        # Only fetch options for equity-passing symbols (not the full universe)
        active_symbols = await self._get_active_symbols()
        skipped_active = [s for s in equity_passing if s in active_symbols]
        symbols_to_check = [s for s in equity_passing if s not in active_symbols]
        
        if skipped_active:
            self._vprint(f"\n  Already in portfolio (skipped): {skipped_active}")
        self._vprint(f"  Checking options for {len(symbols_to_check)} symbol(s): {symbols_to_check}")
        
        all_candidates = []
        all_filter_results_by_symbol = {}  # symbol -> (stock_price, puts_count, filter_results, ranked)
        all_failure_counts = {}

        # Fetch options chains in parallel (capped by semaphore)
        fetch_results = await asyncio.gather(
            *[self._fetch_symbol_options(sym) for sym in symbols_to_check]
        )

        for result in fetch_results:
            if result is None:
                continue
            symbol = result["symbol"]
            ranked = result["ranked"]
            filter_results = result["filter_results"]

            all_candidates.extend(ranked[:self.config.max_candidates_per_symbol])
            all_filter_results_by_symbol[symbol] = (
                result["stock_price"], result["puts_count"], filter_results, ranked
            )

            self.logger.log_options_scan(self._cycle_count, symbol, filter_results)

            # Tally failure reasons
            for r in filter_results:
                for reason in r.failure_reasons:
                    if "Daily return" in reason:
                        key = "Premium too low"
                    elif "Strike" in reason:
                        key = "Strike too high"
                    elif "Delta" in reason:
                        key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                    elif "DTE" in reason:
                        key = "DTE out of range"
                    else:
                        key = reason
                    all_failure_counts[key] = all_failure_counts.get(key, 0) + 1
        
        # Track total contracts evaluated for summary mode
        self._last_options_evaluated = sum(info[1] for info in all_filter_results_by_symbol.values())

        # Flush all options scan logs in one batch
        self.logger.flush()

        # Print options scan summary per symbol
        passing_both_count = sum(1 for s, (_, _, _, ranked) in all_filter_results_by_symbol.items() if ranked)
        print(f"  Passed equity + options filter:            {passing_both_count}")
        
        # Pick best 1 contract per ticker using configured rank mode
        all_candidates.sort(key=lambda c: c.underlying)
        best_per_ticker = []
        for ticker, group in groupby(all_candidates, key=lambda c: c.underlying):
            group_list = list(group)
            group_list.sort(key=lambda c: self._get_sort_key(c), reverse=True)
            best_per_ticker.append(group_list[0])

        # Re-rank across tickers
        best_per_ticker.sort(key=lambda c: self._get_universe_sort_key(c), reverse=True)
        
        # Check earnings & dividends only for final candidates
        candidate_symbols = list(set(c.underlying for c in best_per_ticker)) if best_per_ticker else []
        event_rejections = (await _arun(self.scanner.equity_filter.check_events, candidate_symbols)) if candidate_symbols else {}
        if event_rejections:
            print(f"\n  Event-based rejections (DTE window = {self.config.max_dte}d):")
            for sym in sorted(event_rejections):
                for reason in event_rejections[sym]:
                    print(f"    {sym:<8} {reason}")
            best_per_ticker = [c for c in best_per_ticker if c.underlying not in event_rejections]
        
        candidates = best_per_ticker[:self.config.max_candidates_total]
        
        if not candidates:
            print("\n  No options candidates passed all filters.")
            if all_failure_counts:
                reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(all_failure_counts.items(), key=lambda x: -x[1]))
                print(f"  Aggregate fail reasons: {reasons_str}")
            
            # Detailed per-symbol diagnostics (like Cell 34)
            failed_symbols = [(s, info) for s, info in all_filter_results_by_symbol.items() if not info[3]]
            if failed_symbols:
                self._vprint(f"\n  Diagnostic \u2014 {len(failed_symbols)} equity-passing symbol(s) failed options filter:")
                self._vprint("  " + "-" * 95)
                for symbol, (stock_price, puts_count, filter_results, _) in sorted(failed_symbols):
                    if puts_count == 0:
                        if self.config.max_strike_mode == "sma":
                            max_strike = stock_price
                        else:
                            max_strike = stock_price * self.config.max_strike_pct
                        min_strike = stock_price * self.config.min_strike_pct
                        self._vprint(f"\n    {symbol} @ ${stock_price:.2f}: 0 puts returned from API "
                              f"(strike range ${min_strike:.0f}-${max_strike:.0f}, DTE {self.config.min_dte}-{self.config.max_dte})")
                        continue

                    # Tally per-symbol failure reasons
                    sym_failure_counts = {}
                    for r in filter_results:
                        for reason in r.failure_reasons:
                            if "Daily return" in reason:
                                key = "Premium too low"
                            elif "Strike" in reason:
                                key = "Strike too high"
                            elif "Delta" in reason:
                                key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                            elif "DTE" in reason:
                                key = "DTE out of range"
                            else:
                                key = reason
                            sym_failure_counts[key] = sym_failure_counts.get(key, 0) + 1

                    reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(sym_failure_counts.items(), key=lambda x: -x[1]))
                    self._vprint(f"\n    {symbol} @ ${stock_price:.2f}: {puts_count} puts, 0 passed \u2014 {reasons_str}")

                    # Show nearest misses (top 5 by daily return)
                    near_misses = sorted(filter_results, key=lambda r: r.daily_return, reverse=True)[:5]
                    self._vprint(f"      {'Contract':<26} {'Strike':>8} {'DTE':>5} {'Bid':>8} {'Delta':>8} {'Daily%':>10}  Fail Reasons")
                    self._vprint(f"      {'-'*91}")
                    for r in near_misses:
                        c = r.contract
                        delta_str = f"{r.delta_abs:.3f}" if r.delta_abs else "N/A"
                        reasons = "; ".join(r.failure_reasons) if r.failure_reasons else "\u2713"
                        self._vprint(
                            f"      {c.symbol:<26} "
                            f"${c.strike:>7.2f} "
                            f"{c.dte:>5} "
                            f"${c.bid:>7.2f} "
                            f"{delta_str:>8} "
                            f"{r.daily_return:>9.2%}  "
                            f"{reasons}"
                        )
            
            return 0  # 0 means "none right now, keep trying"
        
        # === Print full candidate table ===
        self._vprint(f"\n  {len(all_candidates)} total option candidates, {len(candidates)} selected for entry")

        # Sort for display: by symbol ascending, then daily return descending
        display_candidates = sorted(all_candidates, key=lambda c: (c.underlying, -c.daily_return_on_collateral))

        self._vprint(f"\n  {'Symbol':<26} {'Price':>9} {'Strike':>8} {'Drop%':>7} {'Days':>5} {'DTE':>5} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Sprd%':>7} {'Delta':>7} {'Daily%':>9} {'Vol':>6} {'OI':>6}")
        self._vprint("  " + "-" * 135)
        for c in display_candidates:
            delta_str = f"{abs(c.delta):.3f}" if c.delta else "N/A"
            spread = c.ask - c.bid if c.ask and c.bid else 0
            spread_pct = spread / c.mid if c.mid > 0 else 0
            vol_str = f"{c.volume:>6}" if c.volume is not None else "     0"
            oi_str = f"{c.open_interest:>6}" if c.open_interest is not None else "   N/A"
            drop_pct = (c.stock_price - c.strike) / c.stock_price if c.stock_price > 0 else 0
            days_str = str(c.days_since_strike) if c.days_since_strike is not None and c.days_since_strike < 999 else ">60"
            self._vprint(
                f"  {c.symbol:<26} "
                f"${c.stock_price:>8.2f} "
                f"${c.strike:>7.2f} "
                f"{drop_pct:>6.1%} "
                f"{days_str:>5} "
                f"{c.dte:>5} "
                f"${c.bid:>7.2f} "
                f"${c.ask:>7.2f} "
                f"${spread:>7.2f} "
                f"{spread_pct:>6.0%} "
                f"{delta_str:>7} "
                f"{c.daily_return_on_collateral:>8.4%} "
                f"{vol_str} "
                f"{oi_str} "
            )

        # === Best Pick Per Ticker by Ranking Mode ===
        if len(all_candidates) > 1:

            def _days_since(c):
                return c.days_since_strike if c.days_since_strike is not None else 0

            rank_modes = {
                "daily_ret/delta": lambda c: c.daily_return_per_delta,
                "days_since_strike": lambda c: _days_since(c),
                "daily_return_on_collateral": lambda c: c.daily_return_on_collateral,
                "lowest_strike": lambda c: -c.strike,
            }

            sorted_by_ticker = sorted(all_candidates, key=lambda c: c.underlying)
            tickers = []
            for ticker, grp in groupby(sorted_by_ticker, key=lambda c: c.underlying):
                tickers.append((ticker, list(grp)))

            if tickers:
                self._vprint(f"\n  {'='*120}")
                self._vprint(f"  Best Pick Per Ticker by Ranking Mode   (per-ticker: {self.config.contract_rank_mode}, universe: {self.config.universe_rank_mode})")
                self._vprint(f"  {'='*120}")
                self._vprint(f"    {'Ticker':<8} | {'daily_ret/delta':<30} | {'days_since_strike':<30} | {'daily_ret':<30} | {'lowest_strike':<30}")
                self._vprint(f"    {'-'*8}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")

                for ticker, contracts in tickers:
                    picks = {}
                    for mode_name, key_fn in rank_modes.items():
                        best = max(contracts, key=key_fn)
                        val = key_fn(best)
                        if mode_name == "daily_ret/delta":
                            val_str = f"{best.symbol[-15:]}  ({val:.4f})"
                        elif mode_name == "days_since_strike":
                            days_val = int(val) if val < 999 else ">60"
                            val_str = f"{best.symbol[-15:]}  ({days_val}d)"
                        elif mode_name == "lowest_strike":
                            val_str = f"{best.symbol[-15:]}  (${best.strike:.0f})"
                        else:
                            val_str = f"{best.symbol[-15:]}  (${val:.3f}/d)"
                        picks[mode_name] = val_str

                    self._vprint(
                        f"    {ticker:<8} | {picks['daily_ret/delta']:<30} | {picks['days_since_strike']:<30} | {picks['daily_return_on_collateral']:<30} | {picks['lowest_strike']:<30}"
                    )
        
        # Show which symbols had no passing options (diagnostic for completeness)
        symbols_no_opts = [s for s in symbols_to_check if s in all_filter_results_by_symbol and not all_filter_results_by_symbol[s][3]]
        if symbols_no_opts and all_failure_counts:
            self._vprint(f"\n  {len(symbols_no_opts)} symbol(s) had no passing options: {symbols_no_opts}")
            reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(all_failure_counts.items(), key=lambda x: -x[1]))
            self._vprint(f"  Aggregate fail reasons: {reasons_str}")
        
        # === Order Entry ===
        self._vprint(f"\n  {'='*80}")
        self._vprint(f"  ORDER ENTRY \u2014 {len(candidates)} candidate(s)")
        self._vprint(f"  {'='*80}")

        current_vix = await _arun(self.vix_fetcher.get_current_vix)

        # Phase 1: Sequential sizing pass (pure arithmetic, no API calls)
        available_cash = min(await _arun(self.alpaca_manager.compute_available_capital), deployable_cash)
        remaining_cash = available_cash
        sized_orders = []

        self._vprint(f"\n  Sizing pass: ${remaining_cash:,.0f} available")

        for i, candidate in enumerate(candidates, 1):
            # Guard: skip contracts with missing Greeks
            if candidate.delta is None or candidate.implied_volatility is None:
                self._vprint(f"  [{i}/{len(candidates)}] Skipping {candidate.underlying}: missing Greeks (delta={candidate.delta}, iv={candidate.implied_volatility})")
                continue

            collateral_per_contract = candidate.collateral_required
            if remaining_cash < collateral_per_contract:
                self._vprint(f"  [{i}/{len(candidates)}] Skipping {candidate.underlying}: insufficient cash (need ${collateral_per_contract:,.0f}, have ${remaining_cash:,.0f})")
                continue

            qty = self.compute_target_quantity(collateral_per_contract, remaining_cash)
            total_collateral = collateral_per_contract * qty

            sized_orders.append((candidate, qty))
            remaining_cash -= total_collateral
            self._vprint(f"  [{i}/{len(candidates)}] {candidate.underlying}: {qty} contracts, ${total_collateral:,.0f} collateral, ${remaining_cash:,.0f} remaining")

        if not sized_orders:
            print(f"\n  No orders sized (insufficient cash or missing Greeks)")
            return 0

        print(f"\n  Sized {len(sized_orders)} orders, ${available_cash - remaining_cash:,.0f} allocated, ${remaining_cash:,.0f} reserved")

        # Phase 2: Parallel execution
        print(f"\n  Executing {len(sized_orders)} entries in parallel...")
        results = await asyncio.gather(
            *[self._execute_single_entry(candidate, qty, current_vix, i, len(sized_orders))
              for i, (candidate, qty) in enumerate(sized_orders, 1)]
        )

        entered = sum(1 for r in results if r)
        print(f"\n  Entry complete: {entered}/{len(sized_orders)} positions opened")
        return entered

    async def run_cycle(self) -> dict:
        """
        Run a single trading cycle.

        Returns:
            Cycle summary dict
        """
        cycle_start = datetime.now()
        summary = {
            'timestamp': cycle_start.isoformat(),
            'market_open': await self.is_market_open(),
            'exits': 0,
            'entries': 0,
            'errors': [],
        }

        try:
            # Print capital banner
            if self.alpaca_manager:
                account_info = await _arun(self.alpaca_manager.get_account_info)
                short_collateral = await _arun(self.alpaca_manager.get_short_collateral)
                avail_capital = account_info['cash'] - short_collateral
                target_pos = avail_capital * self.config.max_position_pct
                
                self._vprint(f"\n  {'='*60}")
                self._vprint(f"  Alpaca cash:                    ${account_info['cash']:>12,.2f}")
                self._vprint(f"  Short position collateral:      ${short_collateral:>12,.2f}")
                self._vprint(f"  Available capital:               ${avail_capital:>12,.2f}")
                self._vprint(f"  Max position size ({self.config.max_position_pct*100:.1f}%):     ${target_pos:>12,.2f}")
                self._vprint(f"  Active positions:                {await self._get_position_count():>12}")
                self._vprint(f"  {'='*60}")

            # Check liquidate_all toggle
            if self.config.liquidate_all and self.alpaca_manager:
                print("LIQUIDATE ALL: config.liquidate_all is True")
                liq_result = await _arun(self.alpaca_manager.liquidate_all_holdings)
                summary['liquidation'] = liq_result

                # Mark all metadata entries as exited
                stale_count = 0
                for sym in list(self.metadata.get_active().keys()):
                    self.metadata.record_exit(
                        option_symbol=sym,
                        exit_reason=ExitReason.MANUAL.value,
                        exit_details="Closed by liquidate_all",
                    )
                    stale_count += 1
                if stale_count:
                    print(f"  Marked {stale_count} metadata entries as exited")

                # Refresh starting_cash after liquidation
                self.config.starting_cash = await _arun(self.alpaca_manager.compute_available_capital)
                print(f"  Starting cash refreshed: ${self.config.starting_cash:,.2f}")
                self.config.liquidate_all = False  # Reset toggle after execution
                print("  liquidate_all reset to False")
                return summary

            # Get current VIX
            current_vix = await _arun(self.vix_fetcher.get_current_vix)
            summary['current_vix'] = current_vix

            # Refresh starting_cash from live account data
            if self.alpaca_manager:
                self.config.starting_cash = await _arun(self.alpaca_manager.compute_available_capital)

            # Calculate deployable cash
            deployable_cash = self.config.get_deployable_cash(current_vix)
            summary['deployable_cash'] = deployable_cash

            # Check global VIX stop
            if await self.check_global_vix_stop(current_vix):
                print(f"🚨 GLOBAL VIX STOP TRIGGERED - VIX: {current_vix:.2f}")
                summary['global_vix_stop'] = True

                # Close all positions (parallel)
                exit_tasks = []
                for alpaca_pos in await self._get_option_positions():
                    meta = self.metadata.get(alpaca_pos.symbol)
                    if meta is None:
                        continue
                    position = self._build_position_proxy(alpaca_pos, meta)
                    result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.VIX_SPIKE,
                        details=f"Global VIX stop: {current_vix:.2f}",
                        current_values={'current_vix': current_vix}
                    )
                    exit_tasks.append(self.execute_exit(position, result, current_premium=0.0))

                if exit_tasks:
                    exit_results = await asyncio.gather(*exit_tasks)
                    summary['exits'] = sum(1 for r in exit_results if r)

                return summary

            # Monitor existing positions
            exits_needed = await self.monitor_positions(current_vix)

            if exits_needed:
                exit_results = await asyncio.gather(
                    *[self.execute_exit(pos, rr, current_premium=ep)
                      for pos, rr, ep in exits_needed]
                )
                summary['exits'] = sum(1 for r in exit_results if r)

            # Scan for new entries (only if market is open and not monitor-only)
            if await self.is_market_open() and deployable_cash > 0 and not self._monitor_only:
                entries = await self.scan_and_enter(deployable_cash)
                summary['entries'] = entries

                # No candidates available today
                if entries == -1:
                    summary['entries'] = 0
                    has_positions = await self._get_position_count() > 0
                    if has_positions:
                        print("  → Switching to monitor-only mode (tracking exits only)")
                        self._monitor_only = True
                    else:
                        print("  → No positions and no candidates. Shutting down for the day.")
                        self._running = False
                        summary['shutdown_reason'] = 'no_candidates_no_positions'
                        return summary

            # Update summary with portfolio state
            portfolio_summary = await self._get_portfolio_summary()
            summary['portfolio'] = portfolio_summary
            
        except Exception as e:
            summary['errors'].append(str(e))
            print(f"Cycle error: {e}")
        
        return summary
    
    async def run(
        self,
        poll_interval: int = 60,
        max_cycles: Optional[int] = None
    ):
        """
        Run the main trading loop.

        Args:
            poll_interval: Seconds between cycles
            max_cycles: Maximum cycles to run (None for infinite)
        """
        self._running = True
        cycle_count = 0
        self._cycle_count = 0

        # Log config snapshot for the day
        self.logger.log_config(self.config)
        self.logger.flush()

        print("\n" + "=" * 60)
        print("CSP TRADING LOOP STARTED")
        print(f"Poll Interval: {poll_interval}s")
        print(f"Paper Trading: {self.execution.paper}")
        print("=" * 60 + "\n")

        if self.config.print_mode == "summary":
            print(f"  {'Date':<12}| {'Time':<10}| {'Cycle':>5} | {'Deployable':>12} | {'VIX':>5} | {'Pos':>3} | {'Eq':>3} | {'Opts':>5} | {'Exits':>5} | {'Entries':>7}")

        try:
            while self._running:
                cycle_count += 1
                self._cycle_count = cycle_count

                self._vprint(f"\n--- Cycle {cycle_count} @ {datetime.now(self.eastern).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

                # Reset daily state if new trading day
                today = datetime.now(self.eastern).date()
                if self._equity_scan_date is not None and self._equity_scan_date != today:
                    print("  New trading day — resetting equity scan and monitor-only flag")
                    self._equity_passing = None
                    self._equity_scan_date = None
                    self._monitor_only = False

                summary = await self.run_cycle()

                # If market is closed today, exit cleanly
                if not summary.get('market_open', False) and not await self.is_market_open():
                    print("  Market closed — exiting loop.")
                    break

                # Print cycle summary
                if self.config.print_mode == "summary":
                    now = datetime.now(self.eastern)
                    p = summary.get('portfolio', {})
                    vix_val = summary.get('current_vix', 0)
                    print(
                        f"  {now.strftime('%Y-%m-%d'):<12}| {now.strftime('%H:%M:%S'):<10}| {cycle_count:>5} | "
                        f"${summary.get('deployable_cash', 0):>11,.0f} | {vix_val:>5.1f} | "
                        f"{p.get('active_positions', 0):>3} | {self._last_equity_passed:>3} | "
                        f"{self._last_options_evaluated:>5} | {summary.get('exits', 0):>5} | "
                        f"{summary.get('entries', 0):>7}"
                    )
                else:
                    mode = "MONITOR-ONLY" if self._monitor_only else "ACTIVE"
                    print(f"  Mode: {mode}")
                    vix_val = summary.get('current_vix')
                    vix_str = f"{vix_val:.2f}" if isinstance(vix_val, (int, float)) else "N/A"
                    print(f"  VIX: {vix_str} | Deployable: ${summary.get('deployable_cash', 0):,.0f}")
                    print(f"  Market Open: {summary.get('market_open', False)}")
                    print(f"  Exits: {summary.get('exits', 0)}, Entries: {summary.get('entries', 0)}")

                    if 'portfolio' in summary:
                        p = summary['portfolio']
                        print(f"  Positions: {p['active_positions']}, Collateral: ${p['total_collateral']:,.2f}")

                # Log cycle
                summary['monitor_only'] = self._monitor_only
                self.logger.log_cycle(cycle_count, summary,
                    options_checked=self._equity_passing or [],
                    failure_tally=summary.get('failure_tally', {}),
                )
                self.logger.flush()

                if 'portfolio' in summary:
                    # In monitor-only mode, stop once all positions are closed
                    if self._monitor_only and p['active_positions'] == 0:
                        print("\n  All positions closed in monitor-only mode. Done for the day.")
                        break

                # Check max cycles
                if max_cycles and cycle_count >= max_cycles:
                    print(f"\nMax cycles ({max_cycles}) reached. Stopping.")
                    break

                # Wait for next cycle
                await asyncio.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nLoop stopped by user.")
        finally:
            self._running = False
            portfolio_summary = await self._get_portfolio_summary()

            # Log shutdown
            self.logger.log_shutdown(
                reason="keyboard_interrupt" if cycle_count > 0 else "error",
                total_cycles=cycle_count,
                portfolio_summary=portfolio_summary,
            )

            print("\nTrading loop ended.")
            print(f"Total cycles: {cycle_count}")
            print(f"Final portfolio: {portfolio_summary}")
                
    def stop(self):
        """Stop the trading loop."""
        self._running = False
