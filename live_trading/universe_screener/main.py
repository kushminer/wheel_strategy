"""Universe screener — screens options universe for liquidity on upcoming Fridays.

Usage:
    python -m universe_screener                     # screen full universe, save to screened_universe.json
    python -m universe_screener --sample 50         # screen 50 random tickers (fast test)
    python -m universe_screener -o my_output.json   # custom output path
"""

import argparse
import json
import random
from datetime import date, datetime, timedelta

import pytz
from dotenv import load_dotenv

from csp.clients import AlpacaClientManager
from csp.data.options import OptionsDataFetcher
from universe_screener.universe import get_combined_universe


def get_next_fridays() -> tuple:
    """Return (coming_friday, subsequent_friday).

    If today is Friday, the coming Friday is next week (skip 0-DTE).
    """
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    coming = today + timedelta(days=days_ahead)
    return coming, coming + timedelta(days=7)


def screen_tickers(tickers, options_fetcher, sample_size=None, verbose=False):
    """Screen tickers for call options expiring on both the next two Fridays.

    Args:
        tickers: full list of ticker symbols
        options_fetcher: OptionsDataFetcher instance
        sample_size: if set, randomly sample this many tickers (for fast testing)
        verbose: if True, print progress and summary

    Returns:
        dict with pass_list, fail_list, and metadata
    """
    coming_fri, next_fri = get_next_fridays()

    candidates = tickers
    if sample_size and sample_size < len(tickers):
        candidates = random.sample(tickers, sample_size)

    if verbose:
        print(f"Target expirations: {coming_fri} and {next_fri}")
        print(f"Screening {len(candidates)} tickers...\n")

    pass_list = []
    fail_list = []

    for i, ticker in enumerate(candidates):
        try:
            contracts = options_fetcher.get_option_contracts(
                underlying=ticker,
                contract_type="call",
                min_dte=1,
                max_dte=14,
            )
            expirations = set(c["expiration"] for c in contracts)

            if coming_fri in expirations and next_fri in expirations:
                pass_list.append(ticker)
            else:
                missing = []
                if coming_fri not in expirations:
                    missing.append(str(coming_fri))
                if next_fri not in expirations:
                    missing.append(str(next_fri))
                fail_list.append({"ticker": ticker, "reason": missing})
        except Exception as e:
            fail_list.append({"ticker": ticker, "reason": [f"error: {e}"]})

        if verbose and (i + 1) % 25 == 0:
            print(f"  Screened {i+1}/{len(candidates)}: "
                  f"{len(pass_list)} pass, {len(fail_list)} fail")

    now_et = datetime.now(pytz.timezone("US/Eastern")).isoformat()
    results = {
        "screened_at": now_et,
        "target_fridays": [str(coming_fri), str(next_fri)],
        "universe_size": len(candidates),
        "pass_count": len(pass_list),
        "fail_count": len(fail_list),
        "pass": sorted(pass_list),
        "fail": fail_list,
    }

    if verbose:
        print(f"\nDone: {len(pass_list)} pass / {len(fail_list)} fail "
              f"(of {len(candidates)} screened)")
        print(f"\nNext Fridays: {coming_fri} and {next_fri}")
        print(f"Passing tickers ({results['pass_count']}): {results['pass']}")
        print(f"Failed tickers ({results['fail_count']}): "
              f"{[f['ticker'] for f in results['fail']]}")

    return results


def save_results(results, output_path):
    """Write screening results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Screen options universe for CC/CSP availability")
    parser.add_argument("--sample", type=int, default=None,
                        help="Screen a random sample instead of full universe")
    parser.add_argument("-o", "--output", default="screened_universe.json",
                        help="Output JSON path (default: screened_universe.json)")
    args = parser.parse_args()

    load_dotenv(override=True)

    print("Building ticker universe...")
    tickers = get_combined_universe(verbose=True)

    print("\nConnecting to Alpaca...")
    alpaca = AlpacaClientManager(paper=True)
    options_fetcher = OptionsDataFetcher(alpaca)
    print("Connected.\n")

    results = screen_tickers(tickers, options_fetcher, sample_size=args.sample, verbose=True)
    save_results(results, args.output)
