"""Strategy scanner: equity filter + options filter pipeline."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.data.options import OptionContract
from csp.signals.equity_filter import EquityFilter, EquityFilterResult
from csp.signals.options_filter import OptionsFilter


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
    ):
        self.config = config
        self.equity_fetcher = equity_fetcher
        self.options_fetcher = options_fetcher
        self.equity_filter = EquityFilter(config)
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
