"""Portfolio status display — account, positions, and order history."""

import re
from datetime import datetime, timedelta


def _parse_strike(symbol: str) -> float:
    match = re.search(r'[PC](\d+)$', symbol)
    if match:
        return int(match.group(1)) / 1000.0
    return 0.0


def print_account_info(alpaca) -> None:
    """Print Alpaca account summary."""
    info = alpaca.get_account_info()
    print("Alpaca Account Information")
    print("=" * 80)
    print(f"Account status:              {info['status']}")
    print(f"Cash available:              ${info['cash']:,.2f}")
    print(f"Buying power (with margin):  ${info['buying_power']:,.2f}")
    print(f"Portfolio value:             ${info['portfolio_value']:,.2f}")
    print(f"Options trading level:       {info['options_trading_level']}")
    print(f"Trading blocked:             {info['trading_blocked']}")
    print()


def print_positions(alpaca) -> None:
    """Print current positions with collateral breakdown."""
    try:
        positions = alpaca.trading_client.get_all_positions()
        info = alpaca.get_account_info()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return

    if not positions:
        print("Current Positions: None")
        print()
        return

    print(f"Current Positions ({len(positions)}):")
    header = (
        f"  {'Symbol':<20} {'Qty':>8} {'Side':<6} {'Strike':>10} "
        f"{'Entry':>12} {'Current':>14} {'Mkt Value':>14} "
        f"{'Unreal P/L':>14} {'Collateral':>12}"
    )
    print(header)
    print("  " + "-" * 120)

    total_collateral = 0
    for pos in positions:
        qty = float(pos.qty)
        market_val = float(pos.market_value)
        side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
        strike = _parse_strike(pos.symbol)
        collateral = abs(qty) * strike * 100 if (side == 'short' or qty < 0) else 0
        total_collateral += collateral
        print(
            f"  {pos.symbol:<20} "
            f"{qty:>8.0f} "
            f"{side:<6} "
            f"${strike:>9.2f} "
            f"${float(pos.avg_entry_price):>11.2f} "
            f"${float(pos.current_price):>13.2f} "
            f"${market_val:>13,.2f} "
            f"${float(pos.unrealized_pl):>13,.2f} "
            f"${collateral:>11,.2f}"
        )

    print(f"\n  Total collateral tied up: ${total_collateral:,.2f}")
    avail = info['cash'] - total_collateral
    print(f"  Available cash (after collateral): ${avail:,.2f}")
    print()


def print_open_orders(alpaca) -> None:
    """Print open orders."""
    from alpaca.trading.requests import GetOrdersRequest

    try:
        orders = alpaca.trading_client.get_orders(GetOrdersRequest(status='open', limit=50))
    except Exception as e:
        print(f"Error fetching open orders: {e}")
        return

    if not orders:
        print("Open Orders: None")
        print()
        return

    print(f"Open Orders ({len(orders)}):")
    header = (
        f"  {'Symbol':<20} {'Side':<6} {'Qty':>8} "
        f"{'Type':<8} {'Status':<12} {'Limit':>12} {'Filled':>8}"
    )
    print(header)
    print("  " + "-" * 100)

    for order in orders:
        lp = order.limit_price
        limit_str = f"${float(lp):.2f}" if lp else "Market"
        filled_qty = float(order.filled_qty) if order.filled_qty else 0
        print(
            f"  {order.symbol:<20} "
            f"{order.side.value:<6} "
            f"{float(order.qty):>8.0f} "
            f"{order.type.value:<8} "
            f"{order.status.value:<12} "
            f"{limit_str:>12} "
            f"{filled_qty:>8.0f}"
        )
    print()


def print_order_history(alpaca, days: int = 7) -> None:
    """Print recent filled/closed/canceled orders."""
    from alpaca.trading.requests import GetOrdersRequest

    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        all_orders = alpaca.trading_client.get_orders(GetOrdersRequest(
            status='all', limit=100, until=end_time, after=start_time,
        ))
        filled = [o for o in all_orders if o.status.value in ('filled', 'closed', 'canceled')]
    except Exception as e:
        print(f"Error fetching order history: {e}")
        return

    if not filled:
        print("Recent Order History: None")
        print()
        return

    print(f"Recent Order History (last {days} days, {len(filled)} orders):")
    header = (
        f"  {'Time':<20} {'Symbol':<20} {'Side':<6} "
        f"{'Qty':>8} {'Type':<8} {'Status':<12} {'Avg Price':>12}"
    )
    print(header)
    print("  " + "-" * 110)

    for order in sorted(filled, key=lambda x: x.created_at, reverse=True)[:20]:
        fp = order.filled_avg_price
        price_str = f"${float(fp):.2f}" if fp else "N/A"
        ca = order.created_at
        time_str = ca.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ca, 'strftime') else str(ca)[:19]
        print(
            f"  {time_str:<20} "
            f"{order.symbol:<20} "
            f"{order.side.value:<6} "
            f"{float(order.qty):>8.0f} "
            f"{order.type.value:<8} "
            f"{order.status.value:<12} "
            f"{price_str:>12}"
        )
    if len(filled) > 20:
        print(f"  ... and {len(filled) - 20} more orders")
    print()


def print_portfolio_status(alpaca) -> None:
    """Print full portfolio status: account, positions, orders, history."""
    print_account_info(alpaca)
    print_positions(alpaca)
    print_open_orders(alpaca)
    print_order_history(alpaca)
