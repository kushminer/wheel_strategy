"""Equity technical filter — evaluates stocks against configurable criteria."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from equity_screener.config import EquityScreenerConfig
from csp.signals.indicators import TechnicalIndicators


@dataclass
class EquityFilterResult:
    """Result of equity technical filter for a single symbol.

    checks: per-check results. True=pass, False=fail, None=disabled.
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
    checks: Dict[str, Optional[bool]] = field(default_factory=dict)


class EquityFilter:
    """Filters equities based on technical criteria from EquityScreenerConfig."""

    def __init__(self, config: EquityScreenerConfig):
        self.config = config
        self.indicators = TechnicalIndicators()

    def evaluate(self, symbol: str, prices: pd.Series) -> EquityFilterResult:
        """Evaluate a single symbol against technical filters."""
        failure_reasons = []
        checks: Dict[str, Optional[bool]] = {}

        if len(prices) < 50:
            return EquityFilterResult(
                symbol=symbol, passes=False,
                current_price=prices.iloc[-1] if len(prices) > 0 else 0,
                sma_8=0, sma_20=0, sma_50=0, rsi=0, bb_upper=0,
                sma_50_trending=False,
                failure_reasons=["Insufficient price history"],
            )

        current_price = prices.iloc[-1]

        # Share price cap
        if self.config.share_price_max is not None:
            ok = bool(current_price <= self.config.share_price_max)
            checks["price_cap"] = ok
            if not ok:
                failure_reasons.append(
                    f"Price ${current_price:.2f} > max ${self.config.share_price_max:.2f}"
                )
        else:
            checks["price_cap"] = None

        sma_8 = self.indicators.sma(prices, 8).iloc[-1]
        sma_20 = self.indicators.sma(prices, 20).iloc[-1]
        sma_50 = self.indicators.sma(prices, 50).iloc[-1]
        rsi = self.indicators.rsi(prices, self.config.rsi_period).iloc[-1]
        _, _, bb_upper = self.indicators.bollinger_bands(
            prices, self.config.bb_period, self.config.bb_std
        )
        bb_upper_val = bb_upper.iloc[-1]
        sma_50_trending = self.indicators.sma_trend(
            prices, 50, self.config.sma_trend_lookback
        )

        if self.config.enable_sma8_check:
            ok = bool(current_price > sma_8)
            checks["sma8"] = ok
            if not ok:
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(8) {sma_8:.2f}")
        else:
            checks["sma8"] = None

        if self.config.enable_sma20_check:
            ok = bool(current_price > sma_20)
            checks["sma20"] = ok
            if not ok:
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(20) {sma_20:.2f}")
        else:
            checks["sma20"] = None

        if self.config.enable_sma50_check:
            ok = bool(current_price > sma_50)
            checks["sma50"] = ok
            if not ok:
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(50) {sma_50:.2f}")
        else:
            checks["sma50"] = None

        if self.config.enable_bb_upper_check:
            ok = bool(current_price > bb_upper_val)
            checks["bb_upper"] = ok
            if not ok:
                failure_reasons.append(
                    f"Price {current_price:.2f} <= BB_upper({self.config.bb_period}) {bb_upper_val:.2f}"
                )
        else:
            checks["bb_upper"] = None

        if self.config.enable_band_check:
            band_period = self.config.sma_bb_period
            sma_band = self.indicators.sma(prices, band_period).iloc[-1]
            _, _, bb_band_upper = self.indicators.bollinger_bands(
                prices, band_period, self.config.bb_std
            )
            bb_band_upper_val = bb_band_upper.iloc[-1]
            ok = bool(sma_band <= current_price <= bb_band_upper_val)
            checks["band"] = ok
            if not ok:
                if current_price < sma_band:
                    failure_reasons.append(
                        f"Price {current_price:.2f} < SMA({band_period}) {sma_band:.2f}"
                    )
                else:
                    failure_reasons.append(
                        f"Price {current_price:.2f} > BB_upper({band_period}) {bb_band_upper_val:.2f}"
                    )
        else:
            checks["band"] = None

        if self.config.enable_sma50_trend_check:
            ok = bool(sma_50_trending)
            checks["trend"] = ok
            if not ok:
                failure_reasons.append("SMA(50) not trending up")
        else:
            checks["trend"] = None

        if self.config.enable_rsi_check:
            ok = bool(self.config.rsi_lower < rsi < self.config.rsi_upper)
            checks["rsi"] = ok
            if not ok:
                failure_reasons.append(
                    f"RSI {rsi:.1f} outside [{self.config.rsi_lower}, {self.config.rsi_upper}]"
                )
        else:
            checks["rsi"] = None

        passes = len(failure_reasons) == 0

        return EquityFilterResult(
            symbol=symbol, passes=passes, current_price=current_price,
            sma_8=sma_8, sma_20=sma_20, sma_50=sma_50, rsi=rsi,
            bb_upper=bb_upper_val, sma_50_trending=sma_50_trending,
            failure_reasons=failure_reasons, checks=checks,
        )

    def filter_universe(
        self,
        price_history: Dict[str, pd.Series],
    ) -> Tuple[List[str], List[EquityFilterResult]]:
        """Batch evaluate all symbols. Returns (passing_symbols, all_results)."""
        results = []
        passing = []
        for symbol, prices in price_history.items():
            result = self.evaluate(symbol, prices)
            results.append(result)
            if result.passes:
                passing.append(symbol)
        return passing, results

    def check_events(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Check events for backward compatibility with loop.py.

        Delegates to the standalone check_events() in calendars module.
        """
        from equity_screener.calendars import check_events as _check_events
        return _check_events(symbols, self.config)
