"""Covered Call Manager — orchestrates the CC lifecycle for stock positions."""

import asyncio
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pytz

from covered_call.config import CoveredCallConfig
from covered_call.store import WheelPositionStore
from csp.config import StrategyConfig
from csp.data.manager import DataManager
from csp.data.options import OptionContract
from csp.trading.execution import ExecutionEngine
from csp.trading.metadata import StrategyMetadataStore
from csp.trading.models import ExitReason, OrderResult, RiskCheckResult
from csp.trading.risk import RiskManager


async def _arun(fn, *args, **kwargs):
    """Run a sync function in a thread pool (non-blocking)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


class CoveredCallManager:
    """Manages the covered call lifecycle for stock positions.

    Detects stock positions from Alpaca (from CSP assignment or standalone),
    sells covered calls, monitors them, and terminates when recovery conditions
    are met.
    """

    def __init__(
        self,
        cc_config: CoveredCallConfig,
        strategy_config: StrategyConfig,
        store: WheelPositionStore,
        data_manager: DataManager,
        execution: ExecutionEngine,
        risk_manager: RiskManager,
        metadata_store: StrategyMetadataStore,
        alpaca_manager,
    ):
        self.cc_config = cc_config
        self.config = strategy_config
        self.store = store
        self.data_manager = data_manager
        self.execution = execution
        self.risk_manager = risk_manager
        self.metadata = metadata_store
        self.alpaca_manager = alpaca_manager
        self.eastern = pytz.timezone("US/Eastern")

    def _vprint(self, msg: str):
        if self.config.print_mode == "verbose":
            print(msg)

    # ── Main cycle ────────────────────────────────────────────────

    async def run_cycle(self, current_vix: float) -> dict:
        """Run one CC management cycle.

        1. Detect new stock positions needing CC management
        2. For each wheel position: check termination, enter CC, or monitor
        3. Persist state
        """
        summary = {"cc_detected": 0, "cc_entered": 0, "cc_exited": 0, "cc_terminated": 0}

        # Step 1: Detect stock positions
        new_count = await self._detect_stock_positions()
        summary["cc_detected"] = new_count

        # Step 2: Process each active wheel position
        active = self.store.get_active()
        if not active:
            return summary

        print(f"\n  CC Manager: {len(active)} active wheel position(s)")

        for underlying, pos in active.items():
            status = pos.get("status")

            if status == "awaiting_cc_entry":
                # Check termination first
                if await self._check_termination(underlying, pos, current_vix):
                    summary["cc_terminated"] += 1
                    continue

                # Enter a new covered call
                if await self._enter_covered_call(underlying, pos, current_vix):
                    summary["cc_entered"] += 1

            elif status == "cc_active":
                result = await self._monitor_covered_call(underlying, pos, current_vix)
                if result == "exited":
                    summary["cc_exited"] += 1
                elif result == "terminated":
                    summary["cc_terminated"] += 1

        return summary

    # ── Detection ─────────────────────────────────────────────────

    async def _detect_stock_positions(self) -> int:
        """Detect stock positions from Alpaca that need CC management.

        Scans all long stock positions, cross-references with existing
        wheel positions to avoid duplicates, and creates new entries.
        """
        try:
            alpaca_positions = await _arun(
                self.alpaca_manager.trading_client.get_all_positions
            )
        except Exception as e:
            self._vprint(f"  CC: Could not fetch Alpaca positions: {e}")
            return 0

        new_count = 0
        existing_underlyings = set(self.store.get_active().keys())

        for pos in alpaca_positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)

            # Skip options (OCC format: long symbols with digits)
            if any(c.isdigit() for c in symbol) and len(symbol) > 10:
                continue

            # Only long stock positions
            if side != "long" and qty <= 0:
                continue

            # Skip if already tracked
            if symbol in existing_underlyings:
                continue

            shares = int(abs(qty))
            if shares < 100:
                self._vprint(f"  CC: Skipping {symbol} — only {shares} shares (need >= 100)")
                continue

            # Determine cost basis and source
            cost_basis, source, csp_premium = self._resolve_cost_basis(symbol, pos)

            self.store.add_position(
                underlying=symbol,
                shares=shares,
                cost_basis=cost_basis,
                source=source,
                csp_entry_premium=csp_premium,
            )
            print(f"  CC: Detected {symbol} — {shares} shares @ ${cost_basis:.2f} (source: {source})")
            new_count += 1

        return new_count

    def _resolve_cost_basis(self, symbol: str, alpaca_pos) -> Tuple[float, str, Optional[float]]:
        """Determine cost basis for a stock position.

        Checks CSP metadata for assignment info first (put strike as cost_basis),
        falls back to Alpaca avg_entry_price for standalone positions.

        Returns: (cost_basis, source, csp_entry_premium)
        """
        # Check CSP metadata for a matching assignment
        for option_sym, meta in self.metadata.entries.items():
            if (
                meta.get("underlying") == symbol
                and meta.get("exit_reason") == ExitReason.ASSIGNED.value
                and meta.get("exit_date") is not None
            ):
                strike = meta.get("strike", 0)
                premium = meta.get("entry_premium", 0)
                if strike > 0:
                    return (strike, "csp_assignment", premium)

        # Fallback: use Alpaca's avg_entry_price
        avg_price = float(alpaca_pos.avg_entry_price) if alpaca_pos.avg_entry_price else 0.0
        return (avg_price, "alpaca_position", None)

    # ── Termination check ─────────────────────────────────────────

    async def _check_termination(
        self, underlying: str, pos: dict, current_vix: float
    ) -> bool:
        """Check if the wheel position should be terminated.

        strike_recovery: Best available CC strike >= cost_basis
        premium_recovery: Total CC premiums >= unrealized loss at assignment
        """
        mode = self.cc_config.cc_exit_mode
        cost_basis = pos.get("cost_basis", 0)

        if mode == "premium_recovery":
            total_premiums = pos.get("total_cc_premiums", 0)
            try:
                stock_price = await _arun(
                    self.data_manager.equity_fetcher.get_current_price, underlying
                )
            except Exception:
                return False

            assignment_loss = max(cost_basis - stock_price, 0)
            csp_premium = pos.get("csp_entry_premium") or 0
            net_loss = assignment_loss - csp_premium

            if total_premiums >= net_loss and net_loss > 0:
                print(f"  CC TERMINATION ({underlying}): Premium recovery achieved")
                print(f"    CC premiums: ${total_premiums:.2f} >= net loss: ${net_loss:.2f}")
                await self._sell_shares(underlying, pos)
                return True

        elif mode == "strike_recovery":
            try:
                stock_price = await _arun(
                    self.data_manager.equity_fetcher.get_current_price, underlying
                )
                calls = await _arun(
                    self.data_manager.options_fetcher.get_calls_chain,
                    underlying, stock_price,
                    self.cc_config.cc_min_dte, self.cc_config.cc_max_dte,
                )
            except Exception as e:
                self._vprint(f"  CC: Could not check termination for {underlying}: {e}")
                return False

            if not calls:
                return False

            # Check if any available CC strike >= cost_basis
            best_strike = max(c.strike for c in calls)
            if best_strike >= cost_basis:
                print(f"  CC TERMINATION ({underlying}): Strike recovery achieved")
                print(f"    Best CC strike: ${best_strike:.2f} >= cost basis: ${cost_basis:.2f}")
                await self._sell_shares(underlying, pos)
                return True

        return False

    # ── CC entry ──────────────────────────────────────────────────

    async def _enter_covered_call(
        self, underlying: str, pos: dict, current_vix: float
    ) -> bool:
        """Select and sell a covered call for a wheel position."""
        try:
            stock_price = await _arun(
                self.data_manager.equity_fetcher.get_current_price, underlying
            )
        except Exception as e:
            print(f"  CC: Could not get price for {underlying}: {e}")
            return False

        # Fetch call chain
        try:
            calls = await _arun(
                self.data_manager.options_fetcher.get_calls_chain,
                underlying, stock_price,
                self.cc_config.cc_min_dte, self.cc_config.cc_max_dte,
            )
        except Exception as e:
            print(f"  CC: Could not fetch calls for {underlying}: {e}")
            return False

        if not calls:
            self._vprint(f"  CC: No call contracts available for {underlying}")
            return False

        # Select contract by mode
        selected = self._select_contract(calls, stock_price)
        if selected is None:
            self._vprint(f"  CC: No contract matched selection criteria for {underlying}")
            return False

        # Calculate quantity (1 contract per 100 shares)
        shares = pos.get("shares", 0)
        qty = shares // 100
        if qty <= 0:
            return False

        delta_str = f"{abs(selected.delta):.3f}" if selected.delta else "N/A"
        print(f"\n  CC ENTRY: {underlying} — {selected.symbol}")
        print(f"    Strike: ${selected.strike:.2f} | DTE: {selected.dte} | Delta: {delta_str}")
        print(f"    Bid: ${selected.bid:.2f} | Ask: ${selected.ask:.2f} | Qty: {qty}")

        # Execute sell_to_open (stepped or market)
        entry_result = await self._execute_cc_entry(selected, qty)

        if entry_result is not None:
            result, filled_price = entry_result
            self.store.record_cc_entry(
                underlying=underlying,
                option_symbol=selected.symbol,
                strike=selected.strike,
                expiration=selected.expiration.isoformat(),
                entry_premium=filled_price,
                quantity=qty,
                order_id=result.order_id,
            )
            print(f"    FILLED @ ${filled_price:.2f} — premium: ${filled_price * qty * 100:,.2f}")
            return True
        else:
            print(f"    FAILED: CC entry exhausted for {underlying}")
            return False

    def _select_contract(
        self, calls: List[OptionContract], stock_price: float
    ) -> Optional[OptionContract]:
        """Select a call contract based on the configured strike mode."""
        mode = self.cc_config.cc_strike_mode

        if mode == "delta":
            target = self.cc_config.cc_strike_delta
            valid = [c for c in calls if c.delta is not None]
            if not valid:
                return None
            return min(valid, key=lambda c: abs(abs(c.delta) - target))

        elif mode == "min_delta":
            target = self.cc_config.cc_strike_delta
            valid = [c for c in calls if c.delta is not None and abs(c.delta) >= target]
            if not valid:
                return None
            return min(valid, key=lambda c: abs(abs(c.delta) - target))

        elif mode == "pct_change":
            target_strike = stock_price * (1 + self.cc_config.cc_strike_pct)
            return min(calls, key=lambda c: abs(c.strike - target_strike))

        elif mode == "min_pct_change":
            target_strike = stock_price * (1 + self.cc_config.cc_strike_pct)
            valid = [c for c in calls if c.strike >= target_strike]
            if not valid:
                return None
            return min(valid, key=lambda c: abs(c.strike - target_strike))

        elif mode == "min_daily_return":
            target = self.cc_config.cc_min_daily_return_pct
            valid = [c for c in calls if c.daily_return_on_collateral >= target]
            if not valid:
                return None
            return min(valid, key=lambda c: abs(c.daily_return_on_collateral - target))

        else:
            print(f"  CC: Unknown strike mode: {mode}")
            return None

    async def _execute_cc_entry(
        self, candidate: OptionContract, qty: int
    ) -> Optional[Tuple[OrderResult, float]]:
        """Execute a stepped limit sell-to-open for a covered call.

        Uses the same stepped order logic as CSP entries: start at mid,
        step down toward bid.
        """
        cfg = self.config
        symbol = candidate.symbol

        bid = candidate.bid
        ask = candidate.ask
        mid = candidate.mid
        spread = ask - bid

        if cfg.entry_start_price == "mid":
            limit_price = mid
        else:
            limit_price = bid

        floor_from_steps = mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread)
        price_floor = max(bid, floor_from_steps)
        limit_price = round(max(limit_price, price_floor), 2)

        self._vprint(f"    CC stepped entry: start=${limit_price:.2f}, "
                     f"bid=${bid:.2f}, ask=${ask:.2f}, spread=${spread:.2f}")

        for step in range(cfg.entry_max_steps + 1):
            self._vprint(f"    Step {step}/{cfg.entry_max_steps}: limit @ ${limit_price:.2f}")

            result = await _arun(
                self.execution.sell_to_open,
                option_symbol=symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                self._vprint(f"    Step {step}: order failed — {result.message}")
                return None

            order_id = result.order_id
            await asyncio.sleep(cfg.entry_step_interval)

            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                filled_price = (
                    float(status["filled_avg_price"])
                    if status.get("filled_avg_price")
                    else limit_price
                )
                return (result, filled_price)

            # Not filled — cancel
            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)

            # Re-check fill during cancel
            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                filled_price = (
                    float(status["filled_avg_price"])
                    if status.get("filled_avg_price")
                    else limit_price
                )
                return (result, filled_price)

            if step >= cfg.entry_max_steps:
                return None

            # Optionally re-fetch snapshot
            if cfg.entry_refetch_snapshot:
                snapshots = await _arun(
                    self.data_manager.options_fetcher.get_option_snapshots, [symbol]
                )
                if symbol not in snapshots:
                    return None
                snap = snapshots[symbol]
                new_bid = float(snap.get("bid", 0) or 0)
                new_ask = float(snap.get("ask", 0) or 0)
                if new_bid <= 0:
                    return None
                bid, ask = new_bid, new_ask
                mid = (bid + ask) / 2
                spread = ask - bid
                price_floor = max(bid, mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread))

            next_step = step + 1
            if cfg.entry_start_price == "mid":
                limit_price = mid - (next_step * cfg.entry_step_pct * spread)
            else:
                limit_price = bid - (next_step * cfg.entry_step_pct * spread)
            limit_price = round(max(limit_price, price_floor), 2)

        return None

    # ── CC monitoring ─────────────────────────────────────────────

    async def _monitor_covered_call(
        self, underlying: str, pos: dict, current_vix: float
    ) -> Optional[str]:
        """Monitor an active covered call position.

        Returns:
            "exited" if CC was closed early (risk trigger)
            "terminated" if CC was assigned (shares called away)
            None if CC is still active and healthy
        """
        cc = pos.get("current_cc")
        if not cc:
            self.store.update(underlying, status="awaiting_cc_entry")
            return None

        option_symbol = cc["option_symbol"]

        # Check if CC still exists in Alpaca
        try:
            alpaca_positions = await _arun(
                self.alpaca_manager.trading_client.get_all_positions
            )
        except Exception as e:
            self._vprint(f"  CC: Could not fetch positions for {underlying}: {e}")
            return None

        # Build lookup
        has_option = False
        has_shares = False
        shares_held = 0

        for p in alpaca_positions:
            if p.symbol == option_symbol:
                has_option = True
            elif p.symbol == underlying:
                side = p.side.value if hasattr(p.side, "value") else str(p.side)
                if side == "long" or float(p.qty) > 0:
                    has_shares = True
                    shares_held = int(abs(float(p.qty)))

        if not has_option:
            if not has_shares:
                # CC assigned — shares called away
                print(f"  CC ASSIGNED: {underlying} — shares called away")
                self.store.record_cc_exit(
                    underlying=underlying,
                    exit_reason="cc_assigned",
                    exit_premium=0.0,
                )
                self.store.terminate(underlying, reason="shares_called_away")
                return "terminated"
            else:
                # CC expired/closed worthless — ready for next round
                print(f"  CC EXPIRED: {underlying} — still hold {shares_held} shares")
                self.store.record_cc_exit(
                    underlying=underlying,
                    exit_reason="expired",
                    exit_premium=0.0,
                )
                return "exited"

        # CC still active — run risk checks
        try:
            snapshots = await _arun(
                self.data_manager.options_fetcher.get_option_snapshots, [option_symbol]
            )
        except Exception:
            self._vprint(f"  CC: Could not fetch snapshot for {option_symbol}")
            return None

        snap = snapshots.get(option_symbol, {})
        current_delta = snap.get("delta")
        current_bid = float(snap.get("bid", 0) or 0)

        if current_delta is None and self.config.exit_on_missing_delta:
            print(f"  CC: Delta unavailable for {option_symbol}, triggering exit")
            await self._exit_covered_call(underlying, pos, option_symbol, "data_unavailable")
            return "exited"

        if current_delta is None:
            self._vprint(f"  CC: Delta unavailable for {option_symbol}, skipping risk check")
            return None

        # Build a position proxy for the RiskManager
        from types import SimpleNamespace

        entry_premium = cc.get("entry_premium", 0)
        proxy = SimpleNamespace(
            entry_delta=current_delta * 0.5,  # approximate entry delta
            entry_stock_price=pos.get("cost_basis", 0),
            entry_premium=entry_premium,
            entry_vix=current_vix,
            entry_daily_return=0,
            strike=cc.get("strike", 0),
            days_held=max((datetime.now() - datetime.fromisoformat(cc["entry_date"])).days, 0)
            if cc.get("entry_date")
            else 0,
        )

        try:
            stock_price = await _arun(
                self.data_manager.equity_fetcher.get_current_price, underlying
            )
        except Exception:
            stock_price = pos.get("cost_basis", 0)

        # Check stop-loss conditions
        stop_result = self.risk_manager.check_all_stops(
            proxy, current_delta, stock_price, current_vix
        )

        if stop_result.should_exit:
            print(f"  CC RISK EXIT ({underlying}): {stop_result.details}")
            await self._exit_covered_call(
                underlying, pos, option_symbol, stop_result.exit_reason.value
            )
            return "exited"

        # Check early exit
        if self.config.enable_early_exit:
            early_result = self.risk_manager.check_early_exit(proxy, current_bid)
            if early_result.should_exit:
                print(f"  CC EARLY EXIT ({underlying}): {early_result.details}")
                await self._exit_covered_call(
                    underlying, pos, option_symbol, "early_exit"
                )
                return "exited"

        self._vprint(
            f"  CC: {underlying} — {option_symbol} healthy "
            f"(delta={abs(current_delta):.3f}, bid=${current_bid:.2f})"
        )
        return None

    async def _exit_covered_call(
        self,
        underlying: str,
        pos: dict,
        option_symbol: str,
        exit_reason: str,
    ):
        """Buy-to-close a covered call using stepped limit order."""
        cc = pos.get("current_cc", {})
        qty = cc.get("quantity", 1)

        # Try stepped exit (buy-to-close)
        result = await self._execute_cc_exit(option_symbol, qty)

        if result is not None:
            _, filled_price = result
            print(f"    CC exit filled @ ${filled_price:.2f}")
            self.store.record_cc_exit(
                underlying=underlying,
                exit_reason=exit_reason,
                exit_premium=filled_price,
            )
        else:
            # Fallback to market order
            print(f"    CC stepped exit failed, using market order")
            market_result = await _arun(
                self.execution.buy_to_close,
                option_symbol=option_symbol,
                quantity=qty,
            )
            if market_result.success:
                self.store.record_cc_exit(
                    underlying=underlying,
                    exit_reason=exit_reason,
                    exit_premium=0.0,
                )
            else:
                print(f"    CC market exit also failed: {market_result.message}")

    async def _execute_cc_exit(
        self, option_symbol: str, qty: int
    ) -> Optional[Tuple[OrderResult, float]]:
        """Execute a stepped limit buy-to-close for a covered call.

        Steps UP from mid toward ask (same logic as CSP exits).
        """
        cfg = self.config

        snapshots = await _arun(
            self.data_manager.options_fetcher.get_option_snapshots, [option_symbol]
        )
        if option_symbol not in snapshots:
            return None

        snap = snapshots[option_symbol]
        bid = float(snap.get("bid", 0) or 0)
        ask = float(snap.get("ask", 0) or 0)
        if ask <= 0:
            return None

        mid = (bid + ask) / 2
        spread = ask - bid

        if cfg.exit_start_price == "ask":
            limit_price = ask
        else:
            limit_price = mid

        ceiling = min(ask, mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread))
        limit_price = round(min(limit_price, ceiling), 2)

        for step in range(cfg.exit_max_steps + 1):
            result = await _arun(
                self.execution.buy_to_close,
                option_symbol=option_symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                return None

            order_id = result.order_id
            await asyncio.sleep(cfg.exit_step_interval)

            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                filled_price = (
                    float(status["filled_avg_price"])
                    if status.get("filled_avg_price")
                    else limit_price
                )
                return (result, filled_price)

            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)

            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                filled_price = (
                    float(status["filled_avg_price"])
                    if status.get("filled_avg_price")
                    else limit_price
                )
                return (result, filled_price)

            if step >= cfg.exit_max_steps:
                return None

            if cfg.exit_refetch_snapshot:
                snapshots = await _arun(
                    self.data_manager.options_fetcher.get_option_snapshots,
                    [option_symbol],
                )
                if option_symbol not in snapshots:
                    return None
                snap = snapshots[option_symbol]
                bid = float(snap.get("bid", 0) or 0)
                ask = float(snap.get("ask", 0) or 0)
                if ask <= 0:
                    return None
                mid = (bid + ask) / 2
                spread = ask - bid
                ceiling = min(ask, mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread))

            next_step = step + 1
            if cfg.exit_start_price == "ask":
                limit_price = ask
            else:
                limit_price = mid + (next_step * cfg.exit_step_pct * spread)
            limit_price = round(min(limit_price, ceiling), 2)

        return None

    # ── Share liquidation ─────────────────────────────────────────

    async def _sell_shares(self, underlying: str, pos: dict):
        """Sell all shares via stepped limit order (termination)."""
        shares = pos.get("shares", 0)
        if shares <= 0:
            self.store.terminate(underlying, reason="no_shares")
            return

        # Get current stock quote for stepped pricing
        try:
            stock_price = await _arun(
                self.data_manager.equity_fetcher.get_current_price, underlying
            )
        except Exception as e:
            print(f"  CC: Could not get price for share sale of {underlying}: {e}")
            # Fallback to market order
            result = await _arun(
                self.execution.sell_stock, underlying, shares
            )
            if result.success:
                print(f"    Shares sold (market): {underlying} x {shares}")
                self.store.terminate(underlying, reason="shares_sold_market")
            else:
                print(f"    Share sale failed: {result.message}")
            return

        # Use stepped limit: start at a reasonable price, step down
        cfg = self.config
        limit_price = round(stock_price * 0.999, 2)  # Start just below current price
        price_floor = round(stock_price * 0.995, 2)   # Floor at 0.5% below

        print(f"  Selling {shares} shares of {underlying} @ ~${limit_price:.2f}")

        for step in range(cfg.exit_max_steps + 1):
            result = await _arun(
                self.execution.sell_stock,
                underlying, shares,
                limit_price=limit_price,
            )

            if not result.success:
                break

            order_id = result.order_id
            await asyncio.sleep(cfg.exit_step_interval)

            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                filled = status.get("filled_avg_price", limit_price)
                print(f"    Shares sold @ ${filled}")
                self.store.terminate(underlying, reason="shares_sold")
                return

            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)

            status = await _arun(self.execution.get_order_status, order_id)
            if status and status["status"] in ("filled", "partially_filled"):
                self.store.terminate(underlying, reason="shares_sold")
                return

            if step >= cfg.exit_max_steps:
                break

            # Step down
            limit_price = round(limit_price - (stock_price * 0.001), 2)
            limit_price = max(limit_price, price_floor)

        # Fallback to market
        print(f"    Stepped share sale exhausted, using market order")
        result = await _arun(self.execution.sell_stock, underlying, shares)
        if result.success:
            self.store.terminate(underlying, reason="shares_sold_market")
        else:
            print(f"    Market share sale also failed: {result.message}")
