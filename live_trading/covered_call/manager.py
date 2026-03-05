"""Covered Call Manager — orchestrates the CC lifecycle for stock positions."""

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
from csp.trading.models import ExitReason, OrderResult, PositionProxy, RiskCheckResult
from csp.trading.risk import RiskManager
from csp.trading.utils import _arun, is_option_symbol, update_candidate_from_snapshot, build_execution_components


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

        snapshot_fetcher = lambda symbols: _arun(
            self.data_manager.options_fetcher.get_option_snapshots, symbols
        )
        self.stepped_executor, self.exit_router = build_execution_components(
            execution=self.execution,
            config=strategy_config,
            snapshot_fetcher=snapshot_fetcher,
            vprint=self._vprint,
        )

    def _vprint(self, msg: str):
        if self.config.print_mode == "verbose":
            print(msg)

    # ── Main cycle ────────────────────────────────────────────────

    async def run_cycle(self, current_vix: float) -> dict:
        """Run one CC management cycle.

        1. Detect new stock positions needing CC management
        2. For each wheel position: enter CC or monitor active CC
        3. CC terminates only when: expired, assigned, or early-exit met

        No pre-entry termination check — the CC runs its natural course.
        Assignment is the natural exit when the stock recovers above the
        CC strike.
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
                # Always enter a CC — the exit comes from assignment,
                # expiration, or early-exit on the active contract.
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
            if is_option_symbol(symbol):
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
                self.cc_config.cc_min_strike_pct, self.cc_config.cc_max_strike_pct,
            )
        except Exception as e:
            print(f"  CC: Could not fetch calls for {underlying}: {e}")
            return False

        if not calls:
            print(f"  CC: No call contracts available for {underlying}")
            if self.cc_config.cc_verbose:
                await self._diagnose_empty_chain(underlying, stock_price)
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

        cost_basis = pos.get("cost_basis", 0)
        delta_str = f"{abs(selected.delta):.3f}" if selected.delta else "N/A"
        now_et = datetime.now(self.eastern).strftime("%H:%M:%S ET")
        print(f"\n  CC ENTRY: {underlying} — {selected.symbol}  [{now_et}]")
        print(f"    Stock: ${stock_price:.2f} | Cost basis: ${cost_basis:.2f}")
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

    def _validate_cc_candidate(self, candidate: OptionContract) -> bool:
        """Re-validate a CC candidate after snapshot refetch.

        Checks that the candidate still meets CC selection criteria
        based on the configured strike mode.
        """
        if candidate.bid <= 0:
            return False

        mode = self.cc_config.cc_strike_mode

        if mode in ("delta", "min_delta"):
            if candidate.delta is None:
                return False
            if mode == "min_delta" and abs(candidate.delta) < self.cc_config.cc_strike_delta:
                return False

        elif mode == "min_daily_return":
            if candidate.daily_return_on_collateral < self.cc_config.cc_min_daily_return_pct:
                return False

        return True

    async def _execute_cc_entry(
        self, candidate: OptionContract, qty: int
    ) -> Optional[Tuple[OrderResult, float]]:
        """Execute a stepped limit sell-to-open for a covered call.

        Delegates to SteppedOrderExecutor with CC-specific callbacks
        for greek updates and re-validation between steps.
        """
        def update_fn(snap):
            update_candidate_from_snapshot(candidate, snap)

        def validate_fn(snap):
            return self._validate_cc_candidate(candidate)

        return await self.stepped_executor.execute_entry(
            candidate.symbol, qty, candidate.bid, candidate.ask, candidate.mid,
            validate_fn=validate_fn, update_fn=update_fn,
        )

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
                    exit_reason=ExitReason.ASSIGNED.value,
                    exit_premium=0.0,
                )
                self.store.terminate(underlying, reason="shares_called_away")
                return "terminated"
            else:
                # CC expired/closed worthless — ready for next round
                print(f"  CC EXPIRED: {underlying} — still hold {shares_held} shares")
                self.store.record_cc_exit(
                    underlying=underlying,
                    exit_reason=ExitReason.EXPIRY.value,
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
            await self._exit_covered_call(underlying, pos, option_symbol, ExitReason.DATA_UNAVAILABLE)
            return "exited"

        if current_delta is None:
            self._vprint(f"  CC: Delta unavailable for {option_symbol}, skipping risk check")
            return None

        # Build a position proxy for the RiskManager
        proxy = PositionProxy.from_cc_store(underlying, cc, pos)
        # Override entry_delta with approximate from current (CC doesn't track entry delta)
        proxy.entry_delta = current_delta * 0.5
        proxy.entry_vix = current_vix

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
                underlying, pos, option_symbol, stop_result.exit_reason
            )
            return "exited"

        # Check early exit
        if self.config.enable_early_exit:
            early_result = self.risk_manager.check_early_exit(proxy, current_bid)
            if early_result.should_exit:
                print(f"  CC EARLY EXIT ({underlying}): {early_result.details}")
                await self._exit_covered_call(
                    underlying, pos, option_symbol, ExitReason.EARLY_EXIT
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
        exit_reason: ExitReason,
    ):
        """Buy-to-close a covered call. Delegates to ExitRouter."""
        cc = pos.get("current_cc", {})
        qty = cc.get("quantity", 1)

        def record_fn(order_id, filled_price):
            self.store.record_cc_exit(
                underlying=underlying,
                exit_reason=exit_reason.value,
                exit_premium=filled_price,
            )

        success = await self.exit_router.execute_exit(
            option_symbol=option_symbol,
            quantity=qty,
            exit_reason=exit_reason,
            record_fn=record_fn,
        )

        if not success:
            print(f"    CC exit failed for {option_symbol}")

    # ── Diagnostics ─────────────────────────────────────────────

    async def _diagnose_empty_chain(self, underlying: str, stock_price: float):
        """Print detailed diagnostics explaining why no call contracts were returned.

        Checks each stage of the pipeline to pinpoint the cause:
        1. Are there ANY call contracts in the configured DTE range (ignoring strike)?
        2. How many fall within the configured strike range?
        3. How many have snapshots / liquidity (bid > 0)?
        4. If nothing in the current DTE range, when is the nearest expiry?
        """
        cc = self.cc_config
        min_strike = stock_price * cc.cc_min_strike_pct
        max_strike = stock_price * cc.cc_max_strike_pct

        print(f"\n    ── CC Diagnostic: {underlying} @ ${stock_price:.2f} ──")
        print(f"    Config: DTE {cc.cc_min_dte}-{cc.cc_max_dte} | "
              f"Strike ${min_strike:.2f}-${max_strike:.2f} "
              f"({cc.cc_min_strike_pct:.0%}-{cc.cc_max_strike_pct:.0%} of price)")

        # Step 1: Fetch all calls in DTE range WITHOUT strike filter
        try:
            all_contracts = await _arun(
                self.data_manager.options_fetcher.get_option_contracts,
                underlying=underlying,
                contract_type="call",
                min_dte=cc.cc_min_dte,
                max_dte=cc.cc_max_dte,
            )
        except Exception as e:
            print(f"    API error: {e}")
            return

        if not all_contracts:
            print(f"    Result: API returned 0 call contracts in DTE {cc.cc_min_dte}-{cc.cc_max_dte}")

            # Check wider DTE range for nearest expiry
            try:
                wider = await _arun(
                    self.data_manager.options_fetcher.get_option_contracts,
                    underlying=underlying,
                    contract_type="call",
                    min_dte=1,
                    max_dte=45,
                )
            except Exception:
                wider = []

            if wider:
                expirations = sorted(set(c["expiration"] for c in wider))
                print(f"    Nearest call expirations (DTE 1-45): "
                      f"{', '.join(str(e) for e in expirations[:5])}")
            else:
                print(f"    No call contracts found even in DTE 1-45 range")
            return

        # Step 2: How many are in the configured strike range?
        in_range = [c for c in all_contracts
                    if min_strike <= c["strike"] <= max_strike]

        print(f"    API contracts in DTE range: {len(all_contracts)}")
        print(f"    In strike range: {len(in_range)}")

        if not in_range:
            strikes = sorted(c["strike"] for c in all_contracts)
            print(f"    All strikes outside range. Available: "
                  f"${strikes[0]:.2f} - ${strikes[-1]:.2f}")
            return

        # Step 3: Check snapshots / liquidity
        symbols = [c["symbol"] for c in in_range]
        try:
            snapshots = await _arun(
                self.data_manager.options_fetcher.get_option_snapshots, symbols
            )
        except Exception as e:
            print(f"    Snapshot fetch error: {e}")
            return

        has_snapshot = sum(1 for s in symbols if s in snapshots)
        has_bid = sum(1 for s in symbols
                      if float(snapshots.get(s, {}).get("bid", 0) or 0) > 0)

        print(f"    With snapshot data: {has_snapshot}/{len(symbols)}")
        print(f"    With bid > 0 (liquid): {has_bid}/{len(symbols)}")

        if has_bid == 0:
            print(f"    Result: All contracts have $0 bid — no liquidity")
        elif has_snapshot < len(symbols):
            print(f"    Result: {len(symbols) - has_snapshot} contracts missing snapshot data")

