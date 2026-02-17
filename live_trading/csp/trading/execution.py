"""Order execution via Alpaca trading API."""

from typing import List, Optional

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from csp.trading.models import OrderResult


class ExecutionEngine:
    """Handles order execution via Alpaca for CSP strategy."""

    def __init__(self, alpaca_manager, config):
        self.trading_client = alpaca_manager.trading_client
        self.config = config
        self.paper = alpaca_manager.paper

    def sell_to_open(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force=TimeInForce.DAY,
    ) -> OrderResult:
        """Sell to open a put option (enter CSP position)."""
        try:
            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=option_symbol, qty=quantity,
                    side=OrderSide.SELL, limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol, qty=quantity,
                    side=OrderSide.SELL, time_in_force=time_in_force,
                )

            order = self.trading_client.submit_order(order_request)
            return OrderResult(
                success=True,
                order_id=str(order.id),
                message=f"Order submitted: {order.status.value}",
                order_details={
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': str(order.qty),
                    'type': order.type.value,
                    'status': order.status.value,
                    'limit_price': str(order.limit_price) if order.limit_price else None,
                },
            )
        except Exception as e:
            return OrderResult(success=False, order_id=None, message=f"Order failed: {e}")

    def buy_to_close(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force=TimeInForce.DAY,
    ) -> OrderResult:
        """Buy to close a put option (exit CSP position)."""
        try:
            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=option_symbol, qty=quantity,
                    side=OrderSide.BUY, limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol, qty=quantity,
                    side=OrderSide.BUY, time_in_force=time_in_force,
                )

            order = self.trading_client.submit_order(order_request)
            return OrderResult(
                success=True,
                order_id=str(order.id),
                message=f"Order submitted: {order.status.value}",
                order_details={
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': str(order.qty),
                    'type': order.type.value,
                    'status': order.status.value,
                    'limit_price': str(order.limit_price) if order.limit_price else None,
                },
            )
        except Exception as e:
            return OrderResult(success=False, order_id=None, message=f"Order failed: {e}")

    def get_order_status(self, order_id: str) -> Optional[dict]:
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': str(order.qty),
                'filled_qty': str(order.filled_qty),
                'type': order.type.value,
                'status': order.status.value,
                'filled_avg_price': str(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception:
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_positions(self) -> List[dict]:
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': str(pos.qty),
                    'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    'avg_entry_price': str(pos.avg_entry_price),
                    'current_price': str(pos.current_price),
                    'market_value': str(pos.market_value),
                    'unrealized_pl': str(pos.unrealized_pl),
                }
                for pos in positions
            ]
        except Exception:
            return []
