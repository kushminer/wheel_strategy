"""Shared trading utilities — used across all strategy modules.

Consolidates infrastructure helpers that were duplicated across
loop.py, manager.py, exit_router.py, and market_session.py.
"""

import asyncio


async def _arun(fn, *args, **kwargs):
    """Run a sync function in a thread pool (non-blocking)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def is_option_symbol(symbol: str) -> bool:
    """Check if an Alpaca symbol is an option (OCC format, not a stock ticker)."""
    return len(symbol) > 10 and any(c.isdigit() for c in symbol)


def update_candidate_from_snapshot(candidate, snap: dict) -> None:
    """Update an OptionContract candidate with fresh snapshot data.

    Used as the update_fn callback in stepped order entry execution.
    """
    candidate.bid = float(snap.get('bid', 0) or 0)
    candidate.ask = float(snap.get('ask', 0) or 0)
    candidate.mid = (candidate.bid + candidate.ask) / 2
    for attr in ('delta', 'implied_volatility', 'volume', 'open_interest'):
        if snap.get(attr) is not None:
            setattr(candidate, attr, snap[attr])


def build_execution_components(execution, config, snapshot_fetcher, vprint):
    """Factory: build SteppedOrderExecutor + ExitRouter pair.

    Both CSP and CC strategies need the same pair with the same wiring.
    Returns (stepped_executor, exit_router).
    """
    from csp.trading.execution_config import ExecutionConfig
    from csp.trading.stepped_executor import SteppedOrderExecutor
    from csp.trading.exit_router import ExitRouter

    exec_config = ExecutionConfig.from_strategy_config(config)
    stepped = SteppedOrderExecutor(
        execution=execution,
        config=exec_config,
        snapshot_fetcher=snapshot_fetcher,
        vprint=vprint,
    )
    router = ExitRouter(
        execution=execution,
        stepped_executor=stepped,
        vprint=vprint,
    )
    return stepped, router
