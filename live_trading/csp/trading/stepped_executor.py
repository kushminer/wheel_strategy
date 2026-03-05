"""Stepped limit order executor — shared by CSP and CC strategies.

Implements the step-down (entry) and step-up (exit) limit order patterns
that both strategies use for price improvement.

Entry: sell-to-open, starting at mid/bid and stepping DOWN toward bid.
Exit:  buy-to-close, starting at mid/ask and stepping UP toward ask.
"""

import asyncio
from typing import Callable, List, Optional, Tuple

from csp.trading.execution import ExecutionEngine
from csp.trading.execution_config import ExecutionConfig
from csp.trading.models import OrderResult


async def _arun(fn, *args, **kwargs):
    """Run a sync function in a thread pool (non-blocking)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


class SteppedOrderExecutor:
    """Executes stepped limit orders for options entry and exit.

    Both CSP and CC strategies delegate to this class instead of
    maintaining their own copies of the stepping logic.
    """

    def __init__(
        self,
        execution: ExecutionEngine,
        config: ExecutionConfig,
        snapshot_fetcher: Callable = None,
        vprint: Callable = None,
    ):
        self.execution = execution
        self.config = config
        self._fetch_snapshots = snapshot_fetcher
        self._vprint = vprint or (lambda msg: None)
        self.last_step_log: List[dict] = []

    async def execute_entry(
        self,
        symbol: str,
        qty: int,
        bid: float,
        ask: float,
        mid: float,
        *,
        validate_fn: Callable = None,
        update_fn: Callable = None,
    ) -> Optional[Tuple[OrderResult, float]]:
        """Execute a stepped limit sell-to-open.

        Args:
            symbol: OCC option symbol
            qty: number of contracts
            bid, ask, mid: current prices
            validate_fn: Optional callback(snapshot) -> bool.
                Called after refetch + update. Return False to abort.
            update_fn: Optional callback(snapshot) -> None.
                Called after refetch to update candidate greeks.

        Returns:
            Tuple of (OrderResult, filled_price) or None if exhausted.
        """
        cfg = self.config
        spread = ask - bid

        # Initial limit price
        if cfg.entry_start_price == "mid":
            limit_price = mid
        else:
            limit_price = bid

        # Floor: never go below bid
        floor_from_steps = mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread)
        price_floor = max(bid, floor_from_steps)
        limit_price = round(max(limit_price, price_floor), 2)

        self._vprint(f"    Stepped entry: start=${limit_price:.2f}, "
                     f"bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, "
                     f"spread=${spread:.2f}, floor=${price_floor:.2f}")

        self.last_step_log = []

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
            self.last_step_log.append({
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
                self._vprint(f"    All {cfg.entry_max_steps} steps exhausted.")
                return None

            # Optionally re-fetch snapshot
            if cfg.entry_refetch_snapshot and self._fetch_snapshots:
                snapshots = await self._fetch_snapshots([symbol])
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

                # Update candidate greeks via callback
                if update_fn:
                    update_fn(snap)

                # Re-validate via callback
                if validate_fn and not validate_fn(snap):
                    self._vprint(f"    Contract no longer passes validation after re-fetch.")
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

    async def execute_exit(
        self,
        option_symbol: str,
        qty: int,
        bid: float = None,
        ask: float = None,
        mid: float = None,
    ) -> Optional[Tuple[OrderResult, float]]:
        """Execute a stepped limit buy-to-close.

        If bid/ask/mid are not provided, fetches a snapshot first.
        Steps UP from mid toward ask.

        Returns:
            Tuple of (OrderResult, filled_price) or None if exhausted.
        """
        cfg = self.config

        # If prices not provided, fetch snapshot
        if bid is None or ask is None:
            if not self._fetch_snapshots:
                self._vprint(f"    Stepped exit: no prices and no snapshot fetcher")
                return None
            snapshots = await self._fetch_snapshots([option_symbol])
            if option_symbol not in snapshots:
                self._vprint(f"    Stepped exit: no snapshot for {option_symbol}")
                return None
            snap = snapshots[option_symbol]
            bid = float(snap.get('bid', 0) or 0)
            ask = float(snap.get('ask', 0) or 0)

        if ask <= 0:
            self._vprint(f"    Stepped exit: ask is zero, aborting")
            return None

        if mid is None:
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
                self._vprint(f"    Step {step}: order submission failed — {result.message}")
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

            # Not filled — cancel
            self._vprint(f"    Step {step}: not filled (status={status['status'] if status else 'unknown'}), cancelling...")
            await _arun(self.execution.cancel_order, order_id)
            await asyncio.sleep(1)

            # Re-check in case fill happened during cancel
            status = await _arun(self.execution.get_order_status, order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                self._vprint(f"    Step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            # Last step — give up
            if step >= cfg.exit_max_steps:
                self._vprint(f"    All {cfg.exit_max_steps} steps exhausted for exit of {option_symbol}.")
                return None

            # Optionally re-fetch snapshot
            if cfg.exit_refetch_snapshot and self._fetch_snapshots:
                snapshots = await self._fetch_snapshots([option_symbol])
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
