"""Scan diagnostics — equity filter, options filter, and candidate display."""

from itertools import groupby


def print_scan_diagnostics(scanner, config, alpaca):
    """Run full universe scan and print equity filter, candidates, and diagnostics.

    Args:
        scanner: StrategyScanner instance (already initialized).
        config: StrategyConfig instance.
        alpaca: AlpacaClientManager instance.
    """
    print("Universe Scan")
    print("=" * 80)

    # Refresh starting_cash from live account
    account_info = alpaca.get_account_info()
    short_collateral = alpaca.get_short_collateral()
    config.starting_cash = account_info['cash'] - short_collateral
    target_position_dollars = config.starting_cash * config.max_position_pct

    print(f"Alpaca cash:                               ${account_info['cash']:,.2f}")
    print(f"Short position collateral:                 ${short_collateral:,.2f}")
    print(f"Available capital (cash - collateral):      ${config.starting_cash:,.2f}")
    print(f"Max position size ({config.max_position_pct*100:.1f}%):                ${target_position_dollars:,.2f}")
    print()

    # Fetch price history (shared by equity filter display + days_since_strike)
    price_history = scanner.equity_fetcher.get_close_history(
        config.ticker_universe, days=config.history_days,
    )

    # Run scan
    scan_results = []
    for symbol in config.ticker_universe:
        if symbol not in price_history:
            continue
        result = scanner.scan_symbol(symbol, price_history[symbol], skip_equity_filter=False)
        scan_results.append(result)

    passing_equity = [r for r in scan_results if r.equity_result.passes]
    passing_both = [r for r in passing_equity if r.has_candidates]

    print(f"Symbols scanned:                         {len(scan_results)}")
    print(f"Passed equity filter:                     {len(passing_equity)}")
    print(f"Passed equity + options filter:            {len(passing_both)}")

    # Event-based rejections (earnings, dividends)
    candidate_symbols = list(set(r.symbol for r in passing_both))
    event_rejections = scanner.equity_filter.check_events(candidate_symbols)

    if event_rejections:
        print(f"\nEvent-based rejections (DTE window = {config.max_dte}d):")
        for sym in sorted(event_rejections):
            for reason in event_rejections[sym]:
                print(f"  {sym:<8} {reason}")
        passing_both = [r for r in passing_both if r.symbol not in event_rejections]
        print(f"Passed after event filter:                 {len(passing_both)}")

    # ── Equity filter results ────────────────────────────────────
    _print_equity_results(passing_equity, config)

    # ── Candidates ───────────────────────────────────────────────
    # Collect all candidates from scan results (already filtered)
    all_candidates = []
    for result in scan_results:
        all_candidates.extend(result.options_candidates)

    # Remove event-rejected
    if event_rejections:
        all_candidates = [c for c in all_candidates if c.underlying not in event_rejections]

    if all_candidates:
        all_candidates.sort(key=lambda c: (c.underlying, -c.daily_return_on_collateral))
        _print_candidates_table(all_candidates, price_history)
        _print_best_picks(all_candidates, price_history, config)
    else:
        print("\nNo candidates found with equity filter enabled.")
        _print_failure_diagnostics(passing_equity, scanner, config, price_history)


# ── Private display helpers ──────────────────────────────────────


def _print_equity_results(passing_equity, config):
    """Print equity filter pass/fail table."""
    if not passing_equity:
        print("\nNo symbols passed the equity filter.")
        return

    bb_label = f"BB{config.bb_period}"
    print(f"\nPassed equity filter ({len(passing_equity)}):")
    print(
        f"  {'Symbol':<8} {'Price':>9} {'SMA8':>9} {'SMA20':>9} "
        f"{'SMA50':>9} {bb_label:>9} {'RSI':>6} {'Collateral':>12} {'Opts':>5}"
    )
    print("  " + "-" * 88)
    for result in passing_equity:
        r = result.equity_result
        collateral = r.current_price * 100
        print(
            f"  {r.symbol:<8} "
            f"${r.current_price:>8.2f} "
            f"{r.sma_8:>9.2f} "
            f"{r.sma_20:>9.2f} "
            f"{r.sma_50:>9.2f} "
            f"{r.bb_upper:>9.2f} "
            f"{r.rsi:>6.1f} "
            f"${collateral:>10,.0f} "
            f"{len(result.options_candidates):>5}"
        )


def _days_since_strike_str(contract, price_history):
    """Return string for days since price was at/below strike."""
    if contract.underlying not in price_history:
        return "N/A"
    prices = price_history[contract.underlying]
    at_or_below = prices[prices <= contract.strike]
    if at_or_below.empty:
        return ">60"
    last_date = at_or_below.index[-1]
    return str((prices.index[-1] - last_date).days)


def _days_since_strike_int(contract, price_history):
    """Return int for days since price was at/below strike (for ranking)."""
    if contract.underlying not in price_history:
        return 0
    prices = price_history[contract.underlying]
    at_or_below = prices[prices <= contract.strike]
    if at_or_below.empty:
        return 999
    last_date = at_or_below.index[-1]
    return (prices.index[-1] - last_date).days


def _print_candidates_table(candidates, price_history):
    """Print full candidate table."""
    print(
        f"\n{'Symbol':<26} {'Price':>9} {'Strike':>8} {'Drop%':>7} {'Days':>5} "
        f"{'DTE':>5} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Sprd%':>7} "
        f"{'Delta':>7} {'Daily%':>9} {'Vol':>6} {'OI':>6}"
    )
    print("-" * 135)
    for c in candidates:
        delta_str = f"{abs(c.delta):.3f}" if c.delta else "N/A"
        spread = c.ask - c.bid if c.ask and c.bid else 0
        spread_pct = spread / c.mid if c.mid > 0 else 0
        vol_str = f"{c.volume:>6}" if c.volume is not None else "     0"
        oi_str = f"{c.open_interest:>6}" if c.open_interest is not None else "   N/A"
        drop_pct = (c.stock_price - c.strike) / c.stock_price
        days_str = _days_since_strike_str(c, price_history)
        print(
            f"{c.symbol:<26} "
            f"${c.stock_price:>8.2f} "
            f"${c.strike:>7.2f} "
            f"{drop_pct:>6.1%} "
            f"{days_str:>5} "
            f"{c.dte:>5} "
            f"${c.bid:>7.2f} "
            f"${c.ask:>7.2f} "
            f"${spread:>7.2f} "
            f"{spread_pct:>6.0%} "
            f"{delta_str:>7} "
            f"{c.daily_return_on_collateral:>8.4%} "
            f"{vol_str} "
            f"{oi_str} "
        )


def _print_best_picks(candidates, price_history, config):
    """Print best pick per ticker by each ranking mode."""
    rank_modes = {
        "daily_ret/delta": lambda c: c.daily_return_per_delta,
        "days_since_strike": lambda c: (
            c.days_since_strike if c.days_since_strike is not None
            else _days_since_strike_int(c, price_history)
        ),
        "daily_return_on_collateral": lambda c: c.daily_return_on_collateral,
        "lowest_strike": lambda c: -c.strike,
    }

    sorted_by_ticker = sorted(candidates, key=lambda c: c.underlying)
    tickers = []
    for ticker, grp in groupby(sorted_by_ticker, key=lambda c: c.underlying):
        tickers.append((ticker, list(grp)))

    print(f"\n{'='*120}")
    print(f"Best Pick Per Ticker by Ranking Mode   (active mode: {config.contract_rank_mode})")
    print(f"{'='*120}")
    print(
        f"  {'Ticker':<8} | {'daily_ret/delta':<30} | {'days_since_strike':<30} "
        f"| {'daily_ret':<30} | {'lowest_strike':<30}"
    )
    print(f"  {'-'*8}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")

    for ticker, contracts in tickers:
        picks = {}
        for mode_name, key_fn in rank_modes.items():
            best = max(contracts, key=key_fn)
            val = key_fn(best)
            if mode_name == "daily_ret/delta":
                val_str = f"{best.symbol[-15:]}  ({val:.4f})"
            elif mode_name == "days_since_strike":
                days_val = int(val) if val < 999 else ">60"
                val_str = f"{best.symbol[-15:]}  ({days_val}d)"
            elif mode_name == "lowest_strike":
                val_str = f"{best.symbol[-15:]}  (${best.strike:.0f})"
            else:
                val_str = f"{best.symbol[-15:]}  (${val:.3f}/d)"
            picks[mode_name] = val_str

        print(
            f"  {ticker:<8} | {picks['daily_ret/delta']:<30} "
            f"| {picks['days_since_strike']:<30} "
            f"| {picks['daily_return_on_collateral']:<30} "
            f"| {picks['lowest_strike']:<30}"
        )


def _print_failure_diagnostics(passing_equity, scanner, config, price_history):
    """Print diagnostics for equity-passing symbols that had zero options candidates."""
    no_options = [r for r in passing_equity if not r.options_candidates]
    if not no_options:
        print("  (No symbols passed the equity filter, so no options were evaluated.)")
        return

    print(f"\nDiagnostic — {len(no_options)} equity-passing symbol(s) failed options filter:")
    print("-" * 95)

    for result in no_options:
        sma_ceiling = None
        if config.max_strike_mode == "sma":
            sma_ceiling = getattr(result.equity_result, f"sma_{config.max_strike_sma_period}", None)
        puts = scanner.options_fetcher.get_puts_chain(
            result.symbol, result.stock_price, config, sma_ceiling=sma_ceiling,
        )

        if not puts:
            if config.max_strike_mode == "sma" and sma_ceiling:
                max_strike = sma_ceiling
            else:
                max_strike = result.stock_price * config.max_strike_pct
            min_strike = result.stock_price * config.min_strike_pct
            print(
                f"\n  {result.symbol} @ ${result.stock_price:.2f}: "
                f"0 puts returned from API "
                f"(strike range ${min_strike:.0f}-${max_strike:.0f}, "
                f"DTE {config.min_dte}-{config.max_dte})"
            )
            continue

        _, all_filter_results = scanner.options_filter.filter_and_rank(puts)

        # Tally failure reasons
        failure_counts = {}
        for r in all_filter_results:
            for reason in r.failure_reasons:
                if "Daily return" in reason:
                    key = "Premium too low"
                elif "Strike" in reason:
                    key = "Strike too high"
                elif "Delta" in reason:
                    key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                elif "DTE" in reason:
                    key = "DTE out of range"
                else:
                    key = reason
                failure_counts[key] = failure_counts.get(key, 0) + 1

        reasons_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(failure_counts.items(), key=lambda x: -x[1])
        )
        print(f"\n  {result.symbol} @ ${result.stock_price:.2f}: {len(puts)} puts, 0 passed — {reasons_str}")

        # Closest misses (top 5 by daily return)
        near_misses = sorted(all_filter_results, key=lambda r: r.daily_return, reverse=True)[:5]
        print(
            f"    {'Contract':<26} {'Strike':>8} {'DTE':>5} "
            f"{'Bid':>8} {'Delta':>8} {'Daily%':>10}  Fail Reasons"
        )
        print(f"    {'-'*91}")
        for r in near_misses:
            c = r.contract
            delta_str = f"{r.delta_abs:.3f}" if r.delta_abs else "N/A"
            reasons = "; ".join(r.failure_reasons) if r.failure_reasons else "PASS"
            print(
                f"    {c.symbol:<26} "
                f"${c.strike:>7.2f} "
                f"{c.dte:>5} "
                f"${c.bid:>7.2f} "
                f"{delta_str:>8} "
                f"{r.daily_return:>9.2%}  "
                f"{reasons}"
            )
