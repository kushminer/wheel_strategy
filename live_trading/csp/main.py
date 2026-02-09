#!/usr/bin/env python3
"""CSP strategy CLI - test orders or run full trading loop."""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env before other imports that may use env vars
load_dotenv()


def _setup_logging(log_level: str = "INFO") -> None:
    """Configure Python logging for the application."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _run_test_orders(config, alpaca, data_manager, execution) -> int:
    """Run order type verification. Returns exit code."""
    from csp.test_orders import test_all_order_types

    if not data_manager.equity_fetcher or not data_manager.options_fetcher:
        logging.getLogger(__name__).error(
            "Data manager requires Alpaca (equity/options fetchers)"
        )
        return 1

    results = test_all_order_types(
        execution_engine=execution,
        config=config,
        equity_fetcher=data_manager.equity_fetcher,
        options_fetcher=data_manager.options_fetcher,
        alpaca_manager=alpaca,
        num_tickers=1,
        do_replenish=True,
    )

    passed = sum(
        1
        for r in results
        if r.get("sell") not in ("FAIL", "ERROR", "SKIP", "REJECTED")
    )
    return 0 if passed == len(results) else 1


def _run_strategy(
    config, alpaca, data_manager, execution, max_cycles=None
) -> None:
    """Run the full trading loop."""
    from csp.data.options import GreeksCalculator
    from csp.data.vix import VixDataFetcher
    from csp.signals.scanner import StrategyScanner
    from csp.trading.loop import TradingLoop
    from csp.trading.portfolio import PortfolioManager
    from csp.trading.risk import RiskManager

    greeks_calc = GreeksCalculator()
    vix_fetcher = VixDataFetcher()

    if not data_manager.equity_fetcher or not data_manager.options_fetcher:
        raise RuntimeError(
            "Data manager requires Alpaca (equity/options fetchers)"
        )

    scanner = StrategyScanner(
        config=config,
        equity_fetcher=data_manager.equity_fetcher,
        options_fetcher=data_manager.options_fetcher,
        greeks_calc=greeks_calc,
    )

    portfolio = PortfolioManager(
        config=config,
        persistence_path=os.path.join(
            os.getcwd(), "portfolio_state.json"
        ),
    )

    risk_manager = RiskManager(config=config)

    trading_loop = TradingLoop(
        config=config,
        data_manager=data_manager,
        scanner=scanner,
        portfolio=portfolio,
        risk_manager=risk_manager,
        execution=execution,
        vix_fetcher=vix_fetcher,
        greeks_calc=greeks_calc,
    )

    trading_loop.run(
        poll_interval=config.poll_interval_seconds,
        max_cycles=max_cycles,
    )


def main() -> int:
    """CLI entry point."""
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig
    from csp.data.manager import DataManager
    from csp.trading.execution import ExecutionEngine

    parser = argparse.ArgumentParser(
        description="CSP (Cash-Secured Put) Strategy"
    )
    parser.add_argument(
        "--test-orders",
        action="store_true",
        help="Run order type verification before trading",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full trading loop",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Max trading cycles (for testing); default is unlimited",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level (default from config/env)",
    )
    args = parser.parse_args()

    if not args.test_orders and not args.run:
        parser.print_help()
        return 0

    config = StrategyConfig()
    log_level = args.log_level or config.log_level
    _setup_logging(log_level)

    logger = logging.getLogger(__name__)

    try:
        alpaca = AlpacaClientManager(paper=config.paper_trading)
    except ValueError as e:
        logger.error("Alpaca credentials: %s", e)
        return 1

    data_manager = DataManager(alpaca_manager=alpaca, config=config)
    execution = ExecutionEngine(alpaca_manager=alpaca, config=config)

    if args.test_orders:
        return _run_test_orders(config, alpaca, data_manager, execution)

    if args.run:
        _run_strategy(
            config, alpaca, data_manager, execution,
            max_cycles=args.max_cycles,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
