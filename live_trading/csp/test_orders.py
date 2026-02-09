"""Order verification utilities - replenish buying power and test order types."""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig
    from csp.data.equity import EquityDataFetcher
    from csp.data.options import OptionsDataFetcher
    from csp.trading.execution import ExecutionEngine

logger = logging.getLogger(__name__)


def replenish_buying_power(alpaca_manager: "AlpacaClientManager") -> None:
    """
    Cancel all open orders and close all positions to free up buying power.
    """
    logger.info("Replenishing buying power: cancelling orders and closing positions")

    try:
        alpaca_manager.trading_client.cancel_orders()
        logger.info("Orders cancelled")
    except Exception as e:
        logger.warning("Cancel orders error: %s", e)

    time.sleep(2)

    try:
        positions = alpaca_manager.trading_client.get_all_positions()
        if not positions:
            logger.info("No open positions")
        else:
            for pos in positions:
                try:
                    alpaca_manager.trading_client.close_position(pos.symbol)
                    logger.info("Closed: %s (qty=%s)", pos.symbol, pos.qty)
                except Exception as e:
                    logger.warning("Failed to close %s: %s", pos.symbol, e)
            time.sleep(3)
    except Exception as e:
        logger.warning("Get positions error: %s", e)

    try:
        info = alpaca_manager.get_account_info()
        logger.info(
            "Cash: $%.2f | Buying power: $%.2f",
            info["cash"],
            info["buying_power"],
        )
    except Exception as e:
        logger.warning("Account info error: %s", e)


def test_all_order_types(
    execution_engine: "ExecutionEngine",
    config: "StrategyConfig",
    equity_fetcher: "EquityDataFetcher",
    options_fetcher: "OptionsDataFetcher",
    alpaca_manager: "AlpacaClientManager",
    num_tickers: int = 1,
    do_replenish: bool = True,
) -> List[Dict[str, Any]]:
    """
    Test all order type combinations via ExecutionEngine.

    During market hours: all 4 combos (limit/limit, limit/market, market/limit, market/market).
    Outside market hours: limit/limit only (submit, verify accepted, cancel).

    Args:
        execution_engine: The ExecutionEngine instance
        config: StrategyConfig (must be paper_trading=True)
        equity_fetcher: EquityDataFetcher for prices
        options_fetcher: OptionsDataFetcher for put chain
        alpaca_manager: AlpacaClientManager for clock/replenish
        num_tickers: Number of random tickers per combo
        do_replenish: Cancel orders & close positions before/between/after tests

    Returns:
        List of result dicts per test
    """
    if not execution_engine.paper:
        raise ValueError("Safety check: must be on paper trading!")

    try:
        clock = alpaca_manager.trading_client.get_clock()
        market_open = clock.is_open
    except Exception:
        market_open = False

    if market_open:
        order_combos = [
            ("limit", "limit"),
            ("limit", "market"),
            ("market", "limit"),
            ("market", "market"),
        ]
    else:
        order_combos = [("limit", "limit")]
        logger.info(
            "Market closed. Limit sell only (submit, verify, cancel). "
            "Re-run during market hours for full round-trip."
        )

    all_tickers = list(config.ticker_universe)
    random.shuffle(all_tickers)
    test_tickers = all_tickers[:num_tickers]

    logger.info(
        "ORDER TYPE VERIFICATION | %d combo(s) x %d ticker(s): %s | Market: %s",
        len(order_combos),
        len(test_tickers),
        test_tickers,
        market_open,
    )

    if do_replenish:
        replenish_buying_power(alpaca_manager)

    results: List[Dict[str, Any]] = []

    for entry_type, exit_type in order_combos:
        logger.info("COMBO: entry=%s / exit=%s", entry_type.upper(), exit_type.upper())

        if do_replenish and results:
            replenish_buying_power(alpaca_manager)

        for sym in test_tickers:
            try:
                prices = equity_fetcher.get_current_prices([sym])
                price = prices.get(sym)
                if not price:
                    logger.warning("Skip %s: no price data", sym)
                    results.append({
                        "combo": f"{entry_type}/{exit_type}",
                        "symbol": sym,
                        "sell": "SKIP",
                        "buy": "SKIP",
                    })
                    continue

                puts = options_fetcher.get_puts_chain(sym, price, config)
                if not puts:
                    logger.warning("Skip %s: no put contracts", sym)
                    results.append({
                        "combo": f"{entry_type}/{exit_type}",
                        "symbol": sym,
                        "sell": "SKIP",
                        "buy": "SKIP",
                    })
                    continue

                pick = random.choice(puts)
                delta_str = f"{abs(pick.delta):.3f}" if pick.delta else "N/A"

                sell_limit = round(pick.bid, 2) if entry_type == "limit" else None
                sell_result = execution_engine.sell_to_open(
                    option_symbol=pick.symbol,
                    quantity=1,
                    limit_price=sell_limit,
                )

                logger.info(
                    "SELL (%s): success=%s id=%s msg=%s",
                    entry_type,
                    sell_result.success,
                    sell_result.order_id,
                    sell_result.message,
                )

                if not sell_result.success:
                    results.append({
                        "combo": f"{entry_type}/{exit_type}",
                        "symbol": sym,
                        "sell": "REJECTED",
                        "buy": sell_result.message[:50],
                    })
                    continue

                sell_status = "accepted"
                time.sleep(3)

                order_info = execution_engine.get_order_status(
                    sell_result.order_id or ""
                )
                fill_status = (
                    order_info.get("status", "unknown") if order_info else "unknown"
                )
                logger.info("SELL fill check: %s", fill_status)

                if fill_status == "filled":
                    close_limit = (
                        round(pick.ask, 2) if exit_type == "limit" else None
                    )
                    close_result = execution_engine.buy_to_close(
                        option_symbol=pick.symbol,
                        quantity=1,
                        limit_price=close_limit,
                    )
                    buy_status = "accepted" if close_result.success else "REJECTED"
                    logger.info(
                        "BUY (%s): success=%s id=%s msg=%s",
                        exit_type,
                        close_result.success,
                        close_result.order_id,
                        close_result.message,
                    )
                    time.sleep(2)
                else:
                    cancelled = execution_engine.cancel_order(
                        sell_result.order_id or ""
                    )
                    logger.info("Sell not filled. Cancel: %s", "OK" if cancelled else "FAILED")
                    sell_status = "accepted (verified, cancelled)"
                    buy_status = "SKIPPED (sell not filled)"

                results.append({
                    "combo": f"{entry_type}/{exit_type}",
                    "symbol": sym,
                    "contract": pick.symbol,
                    "sell": sell_status,
                    "buy": buy_status,
                })

            except Exception as e:
                logger.exception("Error testing %s: %s", sym, e)
                results.append({
                    "combo": f"{entry_type}/{exit_type}",
                    "symbol": sym,
                    "sell": "ERROR",
                    "buy": str(e)[:60],
                })

    if do_replenish:
        replenish_buying_power(alpaca_manager)

    passed = sum(
        1
        for r in results
        if r["sell"] not in ("FAIL", "ERROR", "SKIP", "REJECTED")
    )
    logger.info("SUMMARY: %d/%d tests passed", passed, len(results))

    return results
