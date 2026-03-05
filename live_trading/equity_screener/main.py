"""CLI entry point for the equity screener.

Usage:
    python -m equity_screener --config csp_bullish
    python -m equity_screener --config-file /path/to/custom.json
    python -m equity_screener --config csp_bullish -o custom_output.json
"""

import argparse
import json

from dotenv import load_dotenv

from csp.clients import AlpacaClientManager
from csp.data.equity import EquityDataFetcher
from equity_screener.config import EquityScreenerConfig
from equity_screener.filter import EquityFilter
from equity_screener.calendars import check_events
from equity_screener.output import build_output, save_output, default_output_path


def load_universe(source_path: str) -> list:
    """Load ticker universe from screened_universe.json or similar.

    Accepts two formats:
    - Dict with "pass" key: {"pass": ["AAPL", "MSFT", ...]}
    - Plain list: ["AAPL", "MSFT", ...]
    """
    with open(source_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "pass" in data:
        return data["pass"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Cannot parse universe from {source_path}")


def run_screener(config: EquityScreenerConfig, output_path: str = None,
                 *, tickers: list = None, verbose: bool = False) -> dict:
    """Run the full equity screening pipeline.

    Args:
        config: EquityScreenerConfig with filter/event settings.
        output_path: where to write JSON results (default: equity_screened_{name}.json).
        tickers: optional ticker list to screen (overrides config.universe_source).
        verbose: if True, print progress and summary.

    Returns the output dict (also saved to disk).
    """
    if verbose:
        print(f"Equity Screener: {config.name}")

    # Load universe
    if tickers is not None:
        universe = tickers
    else:
        universe = load_universe(config.universe_source)

    if verbose:
        print(f"  Universe: {len(universe)} tickers")

    # Connect to Alpaca
    alpaca = AlpacaClientManager(paper=True)
    fetcher = EquityDataFetcher(alpaca)

    # Fetch price history
    if verbose:
        print(f"  Fetching {config.history_days}d price history...")
    price_history = fetcher.get_close_history(universe, days=config.history_days)
    if verbose:
        print(f"  Got prices for {len(price_history)} symbols")

    # Run technical filter
    eq_filter = EquityFilter(config)
    passing, results = eq_filter.filter_universe(price_history)
    if verbose:
        print(f"  Passed technical filter: {len(passing)}/{len(results)}")

    # Run event checks on passing symbols
    events = check_events(passing, config)
    if verbose:
        if events:
            print(f"  Event rejections: {len(events)} symbols")
        else:
            print(f"  No event rejections")

    # Build and save output
    output = build_output(config, results, events, universe_size=len(universe))

    if output_path is None:
        output_path = default_output_path(config.name)

    save_output(output, output_path)
    if verbose:
        print(f"  Saved to {output_path}")
        n_ev = len(output.get('event_rejected', []))
        print(f"  Pass: {output['pass_count']} | Fail: {len(output['fail'])} | Event-rejected: {n_ev}")

        # Per-check results table
        _CK_KEYS = ["price_cap", "sma8", "sma20", "sma50", "bb_upper", "band", "trend", "rsi"]
        _CK_COLS = ["$Cap", "SMA8", "SMA20", "SMA50", "BB\u2191", "Band", "Trend", "RSI"]
        _EV_KEYS = ["earnings", "dividends", "fomc"]
        _EV_COLS = ["Earn", "Div", "FOMC"]

        def _ck(val):
            if val is None: return "-"
            return "\u2713" if val else "\u2717"

        def _event_status(symbol, tech_passes):
            if not tech_passes:
                return {"earnings": None, "dividends": None, "fomc": None}
            reasons = events.get(symbol, [])
            return {
                "earnings": None if config.trade_during_earnings else not any("Earnings" in r for r in reasons),
                "dividends": None if config.trade_during_dividends else not any("Ex-div" in r for r in reasons),
                "fomc": None if config.trade_during_fomc else not any("FOMC" in r for r in reasons),
            }

        rows = []
        for r in results:
            ev = _event_status(r.symbol, r.passes)
            overall = r.passes and r.symbol not in events
            rows.append((r, overall, ev))
        rows.sort(key=lambda x: x[0].symbol)

        if rows:
            hdr = f"    {'Symbol':<6} {'':>3} {'Price':>9}"
            sep = f"    {'\u2500'*6} {'\u2500'*3} {'\u2500'*9}"
            for col in _CK_COLS:
                hdr += f" {col:>5}"
                sep += f" {'\u2500'*5}"
            for col in _EV_COLS:
                hdr += f" {col:>4}"
                sep += f" {'\u2500'*4}"
            print(f"\n{hdr}")
            print(sep)

            for r, overall, ev in rows:
                status = "\u2713" if overall else "\u2717"
                line = f"    {r.symbol:<6} {status:>3} ${r.current_price:>8.2f}"
                for key in _CK_KEYS:
                    line += f" {_ck(r.checks.get(key)):>5}"
                for key in _EV_KEYS:
                    line += f" {_ck(ev[key]):>4}"
                print(line)

        if output.get('event_rejected'):
            print(f"\n  Event rejections ({len(output['event_rejected'])}):")
            for entry in output['event_rejected']:
                print(f"    {entry['symbol']:<6} \u2014 {', '.join(entry['events'])}")

    return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Equity screener — technical filter + event calendar pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str,
        help="Named config profile (e.g., 'csp_bullish')"
    )
    group.add_argument(
        "--config-file", type=str,
        help="Path to a custom JSON config file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path (default: equity_screened_{name}.json)"
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    if args.config:
        config = EquityScreenerConfig.from_name(args.config)
    else:
        config = EquityScreenerConfig.from_json(args.config_file)

    run_screener(config, output_path=args.output, verbose=True)
