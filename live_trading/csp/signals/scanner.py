"""Strategy scanner: equity filter + options filter pipeline."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.data.options import OptionContract
from csp.signals.equity_filter import EquityFilter, EquityFilterResult
from csp.signals.options_filter import OptionsFilter
from equity_screener.config import EquityScreenerConfig


@dataclass
class ScanResult:
    """Complete scan result for a symbol."""
    symbol: str
    stock_price: float
    equity_result: EquityFilterResult
    options_candidates: List[OptionContract]

    @property
    def has_candidates(self) -> bool:
        return len(self.options_candidates) > 0


class StrategyScanner:
    """Combined scanner that runs equity filter then options filter."""

    def __init__(
        self,
        config: StrategyConfig,
        equity_fetcher,
        options_fetcher,
        greeks_calc: GreeksCalculator,
        equity_screener_config: EquityScreenerConfig = None,
    ):
        self.config = config
        self.equity_fetcher = equity_fetcher
        self.options_fetcher = options_fetcher
        if equity_screener_config is None:
            equity_screener_config = EquityScreenerConfig()
        self.equity_filter = EquityFilter(equity_screener_config)
        self.options_filter = OptionsFilter(config, greeks_calc)

    def scan_symbol(
        self,
        symbol: str,
        prices: pd.Series,
        skip_equity_filter: bool = False,
    ) -> ScanResult:
        stock_price = prices.iloc[-1]
        equity_result = self.equity_filter.evaluate(symbol, prices)

        if not equity_result.passes and not skip_equity_filter:
            return ScanResult(
                symbol=symbol, stock_price=stock_price,
                equity_result=equity_result, options_candidates=[],
            )

        sma_ceiling = None
        if self.config.max_strike_mode == "sma":
            sma_ceiling = getattr(equity_result, f"sma_{self.config.max_strike_sma_period}", None)

        puts = self.options_fetcher.get_puts_chain(
            symbol, stock_price, self.config, sma_ceiling=sma_ceiling
        )

        for put in puts:
            at_or_below = prices[prices <= put.strike]
            if at_or_below.empty:
                put.days_since_strike = 999
            else:
                last_date = at_or_below.index[-1]
                put.days_since_strike = (prices.index[-1] - last_date).days

        candidates = self.options_filter.get_best_candidates(
            puts, max_candidates=self.config.max_candidates_per_symbol
        )

        return ScanResult(
            symbol=symbol, stock_price=stock_price,
            equity_result=equity_result, options_candidates=candidates,
        )

    def scan_universe(self, skip_equity_filter: bool = False) -> List[ScanResult]:
        price_history = self.equity_fetcher.get_close_history(
            self.config.ticker_universe, days=self.config.history_days,
        )
        results = []
        for symbol in self.config.ticker_universe:
            if symbol not in price_history:
                continue
            result = self.scan_symbol(
                symbol, price_history[symbol], skip_equity_filter=skip_equity_filter,
            )
            results.append(result)
        return results

    def get_all_candidates(
        self,
        skip_equity_filter: bool = False,
        max_total: Optional[int] = None,
    ) -> List[OptionContract]:
        if max_total is None:
            max_total = self.config.max_candidates_total
        scan_results = self.scan_universe(skip_equity_filter=skip_equity_filter)

        all_candidates = []
        for result in scan_results:
            all_candidates.extend(result.options_candidates)

        def _sort_key(c):
            if self.config.contract_rank_mode == "daily_return_per_delta":
                return c.daily_return_per_delta
            elif self.config.contract_rank_mode == "days_since_strike":
                return c.days_since_strike or 0
            elif self.config.contract_rank_mode == "lowest_strike_price":
                return -c.strike
            else:
                return c.daily_return_on_collateral

        all_candidates.sort(key=_sort_key, reverse=True)
        return all_candidates[:max_total]

    def scan_tickers(
        self,
        tickers: List[str],
        skip_equity_filter: bool = False,
        verbose: bool = False,
    ) -> List[ScanResult]:
        """Scan an explicit list of tickers for options candidates.

        Args:
            tickers: symbols to scan.
            skip_equity_filter: if True, skip equity technical filter.
            verbose: if True, print progress and results table.

        Returns:
            list of ScanResult (one per ticker).
        """
        if verbose:
            self._print_config_header(len(tickers))

        price_history = self.equity_fetcher.get_close_history(
            tickers, days=self.config.history_days,
        )

        if verbose:
            print(f"  Got prices for {len(price_history)}/{len(tickers)} symbols")

        results = []
        for symbol in tickers:
            if symbol not in price_history:
                if verbose:
                    print(f"  {symbol}: no price data, skipping")
                continue
            result = self.scan_symbol(
                symbol, price_history[symbol],
                skip_equity_filter=skip_equity_filter,
            )
            results.append(result)

        if verbose:
            self._print_scan_results(results, price_history)

        return results

    def _print_config_header(self, n_tickers: int):
        """Print full options filter configuration."""
        c = self.config
        print(f"\nOptions Scan: {n_tickers} tickers")
        print(f"  Contract selection:  rank={c.contract_rank_mode}, "
              f"universe_rank={c.universe_rank_mode}, "
              f"max_per_ticker={c.max_contracts_per_ticker}")
        print(f"  Filter params:       DTE=[{c.min_dte}, {c.max_dte}], "
              f"delta=[{c.delta_min}, {c.delta_max}], "
              f"strike=[{c.min_strike_pct:.0%}, {c.max_strike_pct:.0%}], "
              f"strike_mode={c.max_strike_mode}")
        print(f"  Premium/liquidity:   min_daily_return={c.min_daily_return:.4%}, "
              f"min_vol={c.min_volume}, min_oi={c.min_open_interest}, "
              f"max_spread={c.max_spread_pct:.0%}")
        print(f"  Limits:              max_per_symbol={c.max_candidates_per_symbol}, "
              f"max_total={c.max_candidates_total}")

    def _print_contract_row(self, c, days_str: str):
        """Print a single contract row."""
        delta_str = f"{abs(c.delta):.3f}" if c.delta else "N/A"
        spread = c.ask - c.bid if c.ask and c.bid else 0
        spread_pct = spread / c.mid if c.mid > 0 else 0
        vol_str = f"{c.volume:>6}" if c.volume is not None else "     0"
        oi_str = f"{c.open_interest:>6}" if c.open_interest is not None else "   N/A"
        drop_pct = (c.stock_price - c.strike) / c.stock_price
        return (
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
            f"{oi_str}"
        )

    def _days_since_strike_str(self, contract, price_history):
        """Return string for days since price was at/below strike."""
        if contract.underlying not in price_history:
            return "N/A"
        prices = price_history[contract.underlying]
        at_or_below = prices[prices <= contract.strike]
        if at_or_below.empty:
            return ">60"
        last_date = at_or_below.index[-1]
        return str((prices.index[-1] - last_date).days)

    def _print_scan_results(self, results: List[ScanResult], price_history: dict):
        """Print accepted and rejected contract tables."""
        _TABLE_HEADER = (
            f"{'Symbol':<26} {'Price':>9} {'Strike':>8} {'Drop%':>7} {'Days':>5} "
            f"{'DTE':>5} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Sprd%':>7} "
            f"{'Delta':>7} {'Daily%':>9} {'Vol':>6} {'OI':>6}"
        )
        _TABLE_SEP = "-" * 135

        # ── Accepted candidates ──
        all_candidates = []
        for r in results:
            all_candidates.extend(r.options_candidates)

        if all_candidates:
            all_candidates.sort(key=lambda c: (c.underlying, c.strike))
            print(f"\n  Accepted Candidates ({len(all_candidates)}):")
            print(f"  {_TABLE_HEADER}")
            print(f"  {_TABLE_SEP}")
            for c in all_candidates:
                days_str = self._days_since_strike_str(c, price_history)
                print(f"  {self._print_contract_row(c, days_str)}")

        # ── Rejected contracts ──
        rejected_rows = []
        for r in results:
            sma_ceiling = None
            if self.config.max_strike_mode == "sma":
                sma_ceiling = getattr(
                    r.equity_result, f"sma_{self.config.max_strike_sma_period}", None
                )
            puts = self.options_fetcher.get_puts_chain(
                r.symbol, r.stock_price, self.config, sma_ceiling=sma_ceiling,
            ) if r.stock_price > 0 else []
            if not puts:
                continue
            _, filter_results = self.options_filter.filter_and_rank(puts)
            for fr in filter_results:
                if not fr.passes:
                    days_str = self._days_since_strike_str(fr.contract, price_history)
                    rejected_rows.append((fr, days_str))

        if rejected_rows:
            rejected_rows.sort(key=lambda x: (x[0].contract.underlying, x[0].contract.strike))
            print(f"\n  {'='*155}")
            print(f"  Rejected Contracts ({len(rejected_rows)})")
            print(f"  {'='*155}")
            print(f"  {_TABLE_HEADER}  Fail Reasons")
            print(f"  {'-'*175}")
            for fr, days_str in rejected_rows:
                reasons = "; ".join(fr.failure_reasons)
                print(f"  {self._print_contract_row(fr.contract, days_str)}  {reasons}")

        # ── Summary ──
        total_pass = sum(1 for r in results if r.has_candidates)
        print(f"\n  Summary: {total_pass}/{len(results)} tickers have candidates, "
              f"{len(all_candidates)} accepted, {len(rejected_rows)} rejected")
