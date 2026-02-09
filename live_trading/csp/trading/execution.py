"""Execution engine for Alpaca order placement."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from alpaca.trading.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

from csp.trading.models import OrderResult

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Handles order execution via Alpaca.

    For CSP strategy:
    - Entry: Sell to Open (STO) put options
    - Exit: Buy to Close (BTC) put options
    """

    def __init__(
        self,
        alpaca_manager: "AlpacaClientManager",
        config: "StrategyConfig",
    ) -> None:
        self.trading_client = alpaca_manager.trading_client
        self.config = config
        self.paper = alpaca_manager.paper

    def sell_to_open(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> OrderResult:
        """
        Sell to open a put option (enter CSP position).
        """
        try:
            if limit_price is not None:
                order_request = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=time_in_force,
                )

            order = self.trading_client.submit_order(order_request)

            return OrderResult(
                success=True,
                order_id=order.id,
                message=f"Order submitted: {order.status.value}",
                order_details={
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": str(order.qty),
                    "type": order.type.value,
                    "status": order.status.value,
                    "limit_price": (
                        str(order.limit_price) if order.limit_price else None
                    ),
                },
            )

        except Exception as e:
            logger.warning("Sell to open failed: %s", e)
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Order failed: {e}",
            )

    def buy_to_close(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> OrderResult:
        """
        Buy to close a put option (exit CSP position).
        """
        try:
            if limit_price is not None:
                order_request = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=time_in_force,
                )

            order = self.trading_client.submit_order(order_request)

            return OrderResult(
                success=True,
                order_id=order.id,
                message=f"Order submitted: {order.status.value}",
                order_details={
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": str(order.qty),
                    "type": order.type.value,
                    "status": order.status.value,
                    "limit_price": (
                        str(order.limit_price) if order.limit_price else None
                    ),
                },
            )

        except Exception as e:
            logger.warning("Buy to close failed: %s", e)
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Order failed: {e}",
            )

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an order.

        Returns:
            Order details dict or None if not found
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": str(order.qty),
                "filled_qty": str(order.filled_qty),
                "type": order.type.value,
                "status": order.status.value,
                "filled_avg_price": (
                    str(order.filled_avg_price)
                    if order.filled_avg_price
                    else None
                ),
            }
        except Exception as e:
            logger.warning("Failed to get order status: %s", e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Returns:
            True if cancelled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order: %s", e)
            return False

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions from Alpaca.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": str(pos.qty),
                    "side": (
                        pos.side.value
                        if hasattr(pos.side, "value")
                        else str(pos.side)
                    ),
                    "avg_entry_price": str(pos.avg_entry_price),
                    "current_price": str(pos.current_price),
                    "market_value": str(pos.market_value),
                    "unrealized_pl": str(pos.unrealized_pl),
                }
                for pos in positions
            ]
        except Exception as e:
            logger.warning("Error fetching positions: %s", e)
            return []
