"""Equity technical filter for the CSP strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import pandas as pd

from csp.signals.indicators import TechnicalIndicators

if TYPE_CHECKING:
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class EquityFilterResult:
    """
    Result of equity technical filter.
    """

    symbol: str
    passes: bool
    current_price: float
    sma_8: float
    sma_20: float
    sma_50: float
    rsi: float
    bb_upper: float
    sma_50_trending: bool
    failure_reasons: List[str]

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        reasons = ", ".join(self.failure_reasons) if self.failure_reasons else "All criteria met"
        return f"{self.symbol}: {status} | {reasons}"


class EquityFilter:
    """
    Filters equities based on technical criteria.

    Filter Rules:
    1. SMA(8) < SMA(20) < SMA(50) < current_price < BB_upper(50, 1std)
    2. SMA(50) has been rising for last 3 days
    3. RSI is between 30 and 70
    """

    def __init__(self, config: "StrategyConfig") -> None:
        self.config = config
        self.indicators = TechnicalIndicators()

    def evaluate(self, symbol: str, prices: pd.Series) -> EquityFilterResult:
        """
        Evaluate a single equity against filter criteria.

        Args:
            symbol: Ticker symbol
            prices: Series of closing prices (at least 50 days)

        Returns:
            EquityFilterResult with pass/fail and details
        """
        failure_reasons: List[str] = []

        # Check minimum data
        if len(prices) < 50:
            return EquityFilterResult(
                symbol=symbol,
                passes=False,
                current_price=float(prices.iloc[-1]) if len(prices) > 0 else 0.0,
                sma_8=0.0,
                sma_20=0.0,
                sma_50=0.0,
                rsi=0.0,
                bb_upper=0.0,
                sma_50_trending=False,
                failure_reasons=["Insufficient price history"],
            )

        # Calculate indicators
        current_price = float(prices.iloc[-1])

        sma_8 = float(self.indicators.sma(prices, 8).iloc[-1])
        sma_20 = float(self.indicators.sma(prices, 20).iloc[-1])
        sma_50 = float(self.indicators.sma(prices, 50).iloc[-1])

        rsi_val = self.indicators.rsi(prices, self.config.rsi_period).iloc[-1]
        rsi = float(rsi_val) if pd.notna(rsi_val) else 0.0

        _, _, bb_upper = self.indicators.bollinger_bands(
            prices,
            self.config.bb_period,
            self.config.bb_std,
        )
        bb_upper_val = float(bb_upper.iloc[-1])

        sma_50_trending = self.indicators.sma_trend(
            prices,
            50,
            self.config.sma_trend_lookback,
        )

        # Check criteria

        # 1. SMA alignment: SMA(8) < SMA(20) < SMA(50) < price
        if not (sma_8 < sma_20):
            failure_reasons.append(f"SMA(8) {sma_8:.2f} >= SMA(20) {sma_20:.2f}")
        if not (sma_20 < sma_50):
            failure_reasons.append(f"SMA(20) {sma_20:.2f} >= SMA(50) {sma_50:.2f}")
        if not (sma_50 < current_price):
            failure_reasons.append(f"SMA(50) {sma_50:.2f} >= Price {current_price:.2f}")

        # 2. Price below BB upper
        if not (current_price < bb_upper_val):
            failure_reasons.append(f"Price {current_price:.2f} >= BB_upper {bb_upper_val:.2f}")

        # 3. SMA(50) trending up
        if not sma_50_trending:
            failure_reasons.append("SMA(50) not trending up")

        # 4. RSI in range
        if not (self.config.rsi_lower < rsi < self.config.rsi_upper):
            failure_reasons.append(
                f"RSI {rsi:.1f} outside [{self.config.rsi_lower}, {self.config.rsi_upper}]"
            )

        passes = len(failure_reasons) == 0

        return EquityFilterResult(
            symbol=symbol,
            passes=passes,
            current_price=current_price,
            sma_8=sma_8,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi=rsi,
            bb_upper=bb_upper_val,
            sma_50_trending=sma_50_trending,
            failure_reasons=failure_reasons,
        )

    def filter_universe(
        self,
        price_history: Dict[str, pd.Series],
    ) -> Tuple[List[str], List[EquityFilterResult]]:
        """
        Filter entire universe and return passing symbols.

        Args:
            price_history: Dict mapping symbol -> price Series

        Returns:
            Tuple of (passing_symbols, all_results)
        """
        results: List[EquityFilterResult] = []
        passing: List[str] = []

        for symbol, prices in price_history.items():
            result = self.evaluate(symbol, prices)
            results.append(result)
            if result.passes:
                passing.append(symbol)

        return passing, results
