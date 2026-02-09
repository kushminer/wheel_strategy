"""Combined strategy scanner - equity filter + options filter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import pandas as pd

from csp.data.models import OptionContract
from csp.data.options import GreeksCalculator, OptionsDataFetcher
from csp.signals.equity_filter import EquityFilter, EquityFilterResult
from csp.signals.options_filter import OptionsFilter

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig
    from csp.data.equity import EquityDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """
    Complete scan result for a symbol.
    """

    symbol: str
    stock_price: float
    equity_result: EquityFilterResult
    options_candidates: List[OptionContract]

    @property
    def has_candidates(self) -> bool:
        return len(self.options_candidates) > 0


class StrategyScanner:
    """
    Combined scanner that runs equity filter then options filter.
    """

    def __init__(
        self,
        config: "StrategyConfig",
        equity_fetcher: "EquityDataFetcher",
        options_fetcher: OptionsDataFetcher,
        greeks_calc: GreeksCalculator,
    ) -> None:
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
        """
        Scan a single symbol through both filters.

        Args:
            symbol: Ticker symbol
            prices: Price history
            skip_equity_filter: If True, skip equity filter (for testing)

        Returns:
            ScanResult with equity filter result and option candidates
        """
        stock_price = float(prices.iloc[-1])

        # Run equity filter
        equity_result = self.equity_filter.evaluate(symbol, prices)

        # If equity fails and we're not skipping, return empty options
        if not equity_result.passes and not skip_equity_filter:
            return ScanResult(
                symbol=symbol,
                stock_price=stock_price,
                equity_result=equity_result,
                options_candidates=[],
            )

        # Get options chain
        puts = self.options_fetcher.get_puts_chain(symbol, stock_price, self.config)

        # Filter and rank options
        candidates = self.options_filter.get_best_candidates(puts, max_candidates=5)

        return ScanResult(
            symbol=symbol,
            stock_price=stock_price,
            equity_result=equity_result,
            options_candidates=candidates,
        )

    def scan_universe(
        self,
        skip_equity_filter: bool = False,
    ) -> List[ScanResult]:
        """
        Scan entire universe.

        Args:
            skip_equity_filter: If True, scan options for all symbols

        Returns:
            List of ScanResults for each symbol
        """
        price_history = self.equity_fetcher.get_close_history(
            self.config.ticker_universe,
            days=self.config.history_days,
        )

        results: List[ScanResult] = []
        for symbol in self.config.ticker_universe:
            if symbol not in price_history:
                continue

            result = self.scan_symbol(
                symbol,
                price_history[symbol],
                skip_equity_filter=skip_equity_filter,
            )
            results.append(result)

        return results

    def get_all_candidates(
        self,
        skip_equity_filter: bool = False,
        max_total: int = 20,
    ) -> List[OptionContract]:
        """
        Get all option candidates across universe, ranked by premium/day.

        Args:
            skip_equity_filter: If True, include all symbols
            max_total: Maximum total candidates to return

        Returns:
            List of top option candidates across all symbols
        """
        scan_results = self.scan_universe(skip_equity_filter=skip_equity_filter)

        all_candidates: List[OptionContract] = []
        for result in scan_results:
            all_candidates.extend(result.options_candidates)

        all_candidates.sort(key=lambda c: c.premium_per_day, reverse=True)

        return all_candidates[:max_total]
