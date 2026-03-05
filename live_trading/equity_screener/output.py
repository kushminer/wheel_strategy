"""Serialize screener results to the standard JSON output format."""

import json
from datetime import datetime
from typing import Dict, List

import pytz

from equity_screener.config import EquityScreenerConfig
from equity_screener.filter import EquityFilterResult


def build_output(
    config: EquityScreenerConfig,
    results: List[EquityFilterResult],
    events: Dict[str, List[str]],
    universe_size: int,
) -> dict:
    """Build the output dict matching the standard schema.

    Event-rejected symbols (passed technical but failed event checks) are
    excluded from ``pass`` and listed separately under ``event_rejected``.
    """
    technically_passing = [r for r in results if r.passes]
    failing = [r for r in results if not r.passes]

    # Split technically passing into truly passing vs event-rejected
    event_symbols = set(events.keys())
    passing = [r for r in technically_passing if r.symbol not in event_symbols]
    event_rejected = [r for r in technically_passing if r.symbol in event_symbols]

    now_et = datetime.now(pytz.timezone("US/Eastern")).isoformat()

    def _entry(r, **extra):
        d = {
            "symbol": r.symbol,
            "price": round(r.current_price, 2),
            "sma_8": round(r.sma_8, 2),
            "sma_20": round(r.sma_20, 2),
            "sma_50": round(r.sma_50, 2),
            "rsi": round(r.rsi, 1),
            "bb_upper": round(r.bb_upper, 2),
            "sma_50_trending": r.sma_50_trending,
            "checks": {k: bool(v) if v is not None else None for k, v in r.checks.items()},
        }
        d.update(extra)
        return d

    return {
        "screener": config.name,
        "screened_at": now_et,
        "universe_source": config.universe_source,
        "universe_size": universe_size,
        "pass_count": len(passing),
        "pass": [_entry(r) for r in passing],
        "fail": [_entry(r, reasons=r.failure_reasons) for r in failing],
        "event_rejected": [
            _entry(r, events=events[r.symbol]) for r in event_rejected
        ],
        "events": events,
    }


def save_output(output: dict, path: str):
    """Write output dict to JSON file."""
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def default_output_path(config_name: str) -> str:
    """Generate default output filename: equity_screened_{name}.json"""
    return f"equity_screened_{config_name}.json"
