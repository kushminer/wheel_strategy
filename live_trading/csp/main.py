"""CLI entry point for Cloud Run Job deployment.

Usage:
    python -m csp.main              # uses env vars for config
    python -m csp.main --dry-run    # prints config and exits

Environment variables:
    ALPACA_API_KEY          Alpaca API key (required)
    ALPACA_SECRET_KEY       Alpaca secret key (required)
    PAPER_TRADING           "true" or "false" (default: true)
    STARTING_CASH           Starting cash amount (default: 1000000)
    STORAGE_BACKEND         "local" or "gcs" (default: local)
    GCS_BUCKET_NAME         GCS bucket name (required if storage_backend=gcs)
    GCS_PREFIX              GCS path prefix (default: "paper")
    POLL_INTERVAL           Seconds between cycles (default: 60)
    MAX_CYCLES              Max cycles before exit, 0=unlimited (default: 0)
    PRINT_MODE              "summary" or "verbose" (default: summary)
"""

import asyncio
import os
import sys

from csp.config import StrategyConfig
from csp.clients import AlpacaClientManager
from csp.data.vix import VixDataFetcher
from csp.data.greeks import GreeksCalculator
from csp.data.manager import DataManager
from csp.signals.scanner import StrategyScanner
from csp.storage import build_storage_backend
from csp.trading.execution import ExecutionEngine
from csp.trading.risk import RiskManager
from csp.trading.metadata import StrategyMetadataStore
from csp.trading.loop import TradingLoop


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def build_config() -> StrategyConfig:
    """Build StrategyConfig from environment variables.

    Only overrides StrategyConfig defaults when the env var is explicitly set.
    All defaults live in config.py as the single source of truth.
    """
    overrides = {}

    if os.getenv("PAPER_TRADING"):
        overrides["paper_trading"] = _env_bool("PAPER_TRADING", True)
    if os.getenv("STARTING_CASH"):
        overrides["starting_cash"] = _env_float("STARTING_CASH", 1_000_000)
    if os.getenv("STORAGE_BACKEND"):
        overrides["storage_backend"] = os.getenv("STORAGE_BACKEND")
    if os.getenv("GCS_BUCKET_NAME"):
        overrides["gcs_bucket_name"] = os.getenv("GCS_BUCKET_NAME")
    if os.getenv("GCS_PREFIX"):
        overrides["gcs_prefix"] = os.getenv("GCS_PREFIX")
    if os.getenv("POLL_INTERVAL"):
        overrides["poll_interval_seconds"] = _env_int("POLL_INTERVAL", 60)
    if os.getenv("ENTRY_ORDER_TYPE"):
        overrides["entry_order_type"] = os.getenv("ENTRY_ORDER_TYPE")
    if os.getenv("PRINT_MODE"):
        overrides["print_mode"] = os.getenv("PRINT_MODE")

    # Covered call config from env vars
    if _env_bool("CC_ENABLED", False):
        from covered_call.config import CoveredCallConfig

        cc_overrides = {}
        cc_overrides["enabled"] = True
        if os.getenv("CC_STRIKE_MODE"):
            cc_overrides["cc_strike_mode"] = os.getenv("CC_STRIKE_MODE")
        if os.getenv("CC_STRIKE_DELTA"):
            cc_overrides["cc_strike_delta"] = _env_float("CC_STRIKE_DELTA", 0.30)
        if os.getenv("CC_STRIKE_PCT"):
            cc_overrides["cc_strike_pct"] = _env_float("CC_STRIKE_PCT", 0.02)
        if os.getenv("CC_MIN_DAILY_RETURN_PCT"):
            cc_overrides["cc_min_daily_return_pct"] = _env_float("CC_MIN_DAILY_RETURN_PCT", 0.0015)
        if os.getenv("CC_MIN_DTE"):
            cc_overrides["cc_min_dte"] = _env_int("CC_MIN_DTE", 1)
        if os.getenv("CC_MAX_DTE"):
            cc_overrides["cc_max_dte"] = _env_int("CC_MAX_DTE", 6)
        if os.getenv("CC_EXIT_MODE"):
            cc_overrides["cc_exit_mode"] = os.getenv("CC_EXIT_MODE")

        overrides["covered_call_config"] = CoveredCallConfig(**cc_overrides)

    return StrategyConfig(**overrides)


def build_components(config: StrategyConfig):
    """Wire up all strategy components."""
    alpaca = AlpacaClientManager(paper=config.paper_trading)
    vix_fetcher = VixDataFetcher()
    greeks_calc = GreeksCalculator()
    data_manager = DataManager(alpaca, config)

    scanner = StrategyScanner(
        config=config,
        equity_fetcher=data_manager.equity_fetcher,
        options_fetcher=data_manager.options_fetcher,
        greeks_calc=greeks_calc,
    )

    execution = ExecutionEngine(alpaca, config)
    risk_manager = RiskManager(config)

    storage = build_storage_backend(config)
    metadata = StrategyMetadataStore(
        path="strategy_metadata.json",
        backend=storage,
    )

    # Covered call manager (optional)
    cc_manager = None
    cc_config = config.covered_call_config
    if cc_config is not None and cc_config.enabled:
        from covered_call.manager import CoveredCallManager
        from covered_call.store import WheelPositionStore

        wheel_store = WheelPositionStore(
            path="wheel_positions.json",
            backend=storage,
        )
        cc_manager = CoveredCallManager(
            cc_config=cc_config,
            strategy_config=config,
            store=wheel_store,
            data_manager=data_manager,
            execution=execution,
            risk_manager=risk_manager,
            metadata_store=metadata,
            alpaca_manager=alpaca,
        )
        print(f"  Covered calls:    ENABLED (mode={cc_config.cc_strike_mode}, "
              f"exit={cc_config.cc_exit_mode})")

    loop = TradingLoop(
        config=config,
        data_manager=data_manager,
        scanner=scanner,
        metadata_store=metadata,
        risk_manager=risk_manager,
        execution=execution,
        vix_fetcher=vix_fetcher,
        greeks_calc=greeks_calc,
        alpaca_manager=alpaca,
        cc_manager=cc_manager,
    )

    return loop


async def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("CSP Strategy — Cloud Run Job  [v3]")
    print("=" * 60)

    config = build_config()

    print(f"  Paper trading:    {config.paper_trading}")
    print(f"  Starting cash:    ${config.starting_cash:,.0f}")
    print(f"  Storage backend:  {config.storage_backend}")
    if config.storage_backend == "gcs":
        print(f"  GCS bucket:       {config.gcs_bucket_name}")
        print(f"  GCS prefix:       {config.gcs_prefix}")
    print(f"  Poll interval:    {config.poll_interval_seconds}s")
    print(f"  Print mode:       {config.print_mode}")
    print(f"  Universe size:    {len(config.ticker_universe)} symbols")

    if dry_run:
        print("\n--dry-run: config looks good, exiting.")
        return

    loop = build_components(config)

    poll_interval = config.poll_interval_seconds
    max_cycles = _env_int("MAX_CYCLES", 0) or None  # 0 means unlimited

    print(f"\nStarting loop (poll={poll_interval}s, max_cycles={max_cycles})")
    print("=" * 60)

    await loop.run(poll_interval=poll_interval, max_cycles=max_cycles)

    print("\nLoop exited cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
