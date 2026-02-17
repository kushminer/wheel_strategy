"""Emergency liquidation — cancel all orders and close all positions."""

import re
import time
from datetime import date

from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


def _parse_strike(symbol: str) -> float:
    match = re.search(r'[PC](\d+)$', symbol)
    if match:
        return int(match.group(1)) / 1000.0
    return 0.0


def _parse_expiration(symbol: str):
    match = re.search(r'(\d{6})[PC]', symbol)
    if match:
        d = match.group(1)
        return date(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]))
    return None


def liquidate_all_holdings(alpaca) -> None:
    """Cancel all open orders and close all positions.

    Args:
        alpaca: AlpacaClientManager instance.
    """
    print("=" * 80)
    print("LIQUIDATING ALL HOLDINGS")
    print("=" * 80)
    print()

    # Categorize positions
    expired_positions = []
    active_positions = []
    today = date.today()

    try:
        positions = alpaca.trading_client.get_all_positions()
        if positions:
            print(f"Open Positions ({len(positions)}):")
            total_collateral = 0
            for pos in positions:
                qty = float(pos.qty)
                side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                strike = _parse_strike(pos.symbol)
                expiration = _parse_expiration(pos.symbol)
                current_price = float(pos.current_price)
                is_expired = expiration and expiration < today
                is_worthless = current_price == 0.0
                if side == 'short' or qty < 0:
                    collateral = abs(qty) * strike * 100
                    total_collateral += collateral
                status = " [EXPIRED]" if is_expired else (" [WORTHLESS]" if is_worthless else "")
                print(
                    f"  {pos.symbol:<20} qty={qty:>6.0f} side={side:<6} "
                    f"strike=${strike:>7.2f} exp={expiration} "
                    f"price=${current_price:>6.2f}{status}"
                )
                if is_expired or is_worthless:
                    expired_positions.append(pos)
                else:
                    active_positions.append(pos)
            print(f"  Total collateral: ${total_collateral:,.2f}")
        else:
            print("No positions to close.")
    except Exception as e:
        print(f"Error fetching positions: {e}")

    # Step 1: Cancel all open orders
    print("\nCancelling all open orders...")
    try:
        open_orders = alpaca.trading_client.get_orders(GetOrdersRequest(status='open', limit=50))
        for order in (open_orders or []):
            try:
                alpaca.trading_client.cancel_order_by_id(order.id)
                print(f"  Cancelled: {order.symbol}")
            except Exception as e:
                print(f"  Failed to cancel {order.symbol}: {e}")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(2)

    # Step 2: Close active positions at market
    if active_positions:
        print("\nClosing active positions...")
        for pos in active_positions:
            try:
                qty = float(pos.qty)
                close_side = OrderSide.BUY if qty < 0 else OrderSide.SELL
                close_qty = int(abs(qty))
                order = alpaca.trading_client.submit_order(MarketOrderRequest(
                    symbol=pos.symbol, qty=close_qty,
                    side=close_side, time_in_force=TimeInForce.DAY,
                ))
                print(f"  Closing {pos.symbol} qty={close_qty} order_id={order.id}")
            except Exception as e:
                print(f"  Failed: {pos.symbol}: {e}")

    # Step 3: Close expired/worthless at $0.01
    if expired_positions:
        print("\nClosing expired/worthless positions...")
        for pos in expired_positions:
            try:
                qty = float(pos.qty)
                close_side = OrderSide.BUY if qty < 0 else OrderSide.SELL
                close_qty = int(abs(qty))
                order = alpaca.trading_client.submit_order(LimitOrderRequest(
                    symbol=pos.symbol, qty=close_qty,
                    side=close_side, limit_price=0.01,
                    time_in_force=TimeInForce.DAY,
                ))
                print(f"  Closing {pos.symbol} at $0.01 order_id={order.id}")
            except Exception as e:
                print(f"  Failed: {pos.symbol}: {e}")

    # Verify
    if active_positions or expired_positions:
        time.sleep(5)
        remaining = alpaca.trading_client.get_all_positions()
        if remaining:
            print(f"\nWarning: {len(remaining)} positions still open")
        else:
            print("\nAll positions closed.")

    # Final status
    info = alpaca.get_account_info()
    print(f"\nFinal: Cash=${info['cash']:,.2f} | Buying power=${info['buying_power']:,.2f}")
