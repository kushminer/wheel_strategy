"""Shared exit order routing — used by both CSP and CC strategies.

Routes exit orders by ExitReason:
  EARLY_EXIT / EXPIRY → stepped limit order, market fallback
  Stop-losses (DELTA_STOP, VIX_SPIKE, etc.) → immediate market order
"""

from typing import Callable, Optional, Tuple

from csp.trading.models import ExitReason, OrderResult
from csp.trading.utils import _arun


# Exit reasons that use stepped limit orders with market fallback
_STEPPED_EXIT_REASONS = frozenset({ExitReason.EARLY_EXIT, ExitReason.EXPIRY})


class ExitRouter:
    """Routes exit orders to the appropriate execution method.

    Both CSP and CC strategies delegate to this class instead of
    maintaining their own exit routing logic. Each strategy provides
    a `record_fn` callback to handle its own metadata recording.
    """

    def __init__(
        self,
        execution: ExecutionEngine,
        stepped_executor: SteppedOrderExecutor,
        vprint: Callable = None,
    ):
        self.execution = execution
        self.stepped_executor = stepped_executor
        self._vprint = vprint or (lambda msg: None)

    async def execute_exit(
        self,
        option_symbol: str,
        quantity: int,
        exit_reason: ExitReason,
        *,
        record_fn: Callable[[str, float], None] = None,
    ) -> bool:
        """Execute an exit order, routing by exit reason.

        Args:
            option_symbol: OCC option symbol to close
            quantity: number of contracts (positive)
            exit_reason: determines routing (stepped vs market)
            record_fn: callback(order_id, filled_price) called on success

        Returns:
            True if exit completed successfully.
        """
        if exit_reason in _STEPPED_EXIT_REASONS:
            return await self._exit_stepped_with_fallback(
                option_symbol, quantity, record_fn,
            )
        else:
            return await self._exit_market(
                option_symbol, quantity, record_fn,
            )

    async def _exit_stepped_with_fallback(
        self,
        option_symbol: str,
        quantity: int,
        record_fn: Callable = None,
    ) -> bool:
        """Try stepped limit exit, fall back to market order."""
        result_tuple = await self.stepped_executor.execute_exit(
            option_symbol, quantity,
        )

        if result_tuple is not None:
            result, filled_price = result_tuple
            self._vprint(f"    Stepped exit filled @ ${filled_price:.2f}")
            if record_fn:
                record_fn(result.order_id, filled_price)
            return True

        # Stepped exhausted — fall back to market order
        self._vprint(f"    Stepped exit exhausted, falling back to market order")
        return await self._exit_market(option_symbol, quantity, record_fn)

    async def _exit_market(
        self,
        option_symbol: str,
        quantity: int,
        record_fn: Callable = None,
    ) -> bool:
        """Execute immediate market order buy-to-close."""
        self._vprint(f"    Market order: buy-to-close {quantity}x {option_symbol}")

        result = await _arun(
            self.execution.buy_to_close,
            option_symbol=option_symbol,
            quantity=quantity,
            limit_price=None,
        )

        if result and result.success:
            if record_fn:
                record_fn(result.order_id, 0.0)
            return True
        else:
            msg = result.message if result else "No order result"
            self._vprint(f"    Market exit failed: {msg}")
            return False
