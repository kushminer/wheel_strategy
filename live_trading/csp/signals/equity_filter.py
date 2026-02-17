"""Equity technical filter and event calendars."""

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from csp.config import StrategyConfig
from csp.signals.indicators import TechnicalIndicators


class EarningsCalendar:
    """Fetches upcoming earnings dates from Alpha Vantage per-symbol."""

    def __init__(self, max_dte: int = 10):
        self._cache: Dict[str, List[date]] = {}
        self._cache_date: Optional[date] = None
        self._max_dte = max_dte
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv("ALPHAVANTAGE_API_KEY") or ""
        return self._api_key if self._api_key else None

    def _reset_if_new_day(self):
        today = date.today()
        if self._cache_date != today:
            self._cache = {}
            self._cache_date = today

    def _select_horizon(self) -> str:
        horizon_date = date.today() + timedelta(days=self._max_dte)
        if horizon_date <= date.today() + timedelta(days=90):
            return "3month"
        if horizon_date <= date.today() + timedelta(days=180):
            return "6month"
        return "12month"

    def _fetch_symbol(self, symbol: str) -> List[date]:
        api_key = self._get_api_key()
        if not api_key:
            return []

        horizon = self._select_horizon()
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=EARNINGS_CALENDAR"
            f"&symbol={symbol}"
            f"&horizon={horizon}"
            f"&apikey={api_key}"
        )

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            reader = csv.DictReader(resp.text.strip().splitlines())
            dates = []
            for row in reader:
                report_date_str = row.get("reportDate", "").strip()
                if report_date_str:
                    try:
                        dates.append(datetime.strptime(report_date_str, "%Y-%m-%d").date())
                    except ValueError:
                        continue
            return dates
        except Exception:
            return []

    def prefetch(self, symbols: List[str]):
        self._reset_if_new_day()
        api_key = self._get_api_key()
        if not api_key:
            return
        to_fetch = [s for s in symbols if s not in self._cache]
        for symbol in to_fetch:
            self._cache[symbol] = self._fetch_symbol(symbol)

    def has_earnings_in_window(self, symbol: str, max_dte: int) -> bool:
        self._reset_if_new_day()
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        dates = self._cache.get(symbol, [])
        if not dates:
            return False
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return any(today <= d <= window_end for d in dates)

    def next_earnings_date(self, symbol: str) -> Optional[date]:
        self._reset_if_new_day()
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        dates = self._cache.get(symbol, [])
        today = date.today()
        future = [d for d in dates if d >= today]
        return min(future) if future else None


class DividendCalendar:
    """Fetches upcoming ex-dividend dates from Alpha Vantage."""

    def __init__(self, max_dte: int = 10):
        self._cache: Dict[str, Optional[date]] = {}
        self._cache_date: Optional[date] = None
        self._max_dte = max_dte
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv("ALPHAVANTAGE_API_KEY") or ""
        return self._api_key if self._api_key else None

    def _reset_if_new_day(self):
        today = date.today()
        if self._cache_date != today:
            self._cache = {}
            self._cache_date = today

    def _fetch_symbol(self, symbol: str) -> Optional[date]:
        api_key = self._get_api_key()
        if not api_key:
            return None
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=DIVIDENDS"
            f"&symbol={symbol}"
            f"&datatype=json"
            f"&apikey={api_key}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for record in data.get("data", []):
                ex_date_str = record.get("ex_dividend_date", "").strip()
                if ex_date_str and ex_date_str != "None":
                    try:
                        return datetime.strptime(ex_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
            return None
        except Exception:
            return None

    def prefetch(self, symbols: List[str]):
        self._reset_if_new_day()
        api_key = self._get_api_key()
        if not api_key:
            return
        to_fetch = [s for s in symbols if s not in self._cache]
        for symbol in to_fetch:
            self._cache[symbol] = self._fetch_symbol(symbol)

    def has_exdiv_in_window(self, symbol: str, max_dte: int) -> bool:
        self._reset_if_new_day()
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        ex_date = self._cache.get(symbol)
        if ex_date is None:
            return False
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return today <= ex_date <= window_end

    def next_exdiv_date(self, symbol: str) -> Optional[date]:
        self._reset_if_new_day()
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        return self._cache.get(symbol)


class FomcCalendar:
    """FOMC meeting schedule. Hardcoded dates refreshed manually."""

    _MEETING_DATES = [
        (date(2025, 1, 28), date(2025, 1, 29)),
        (date(2025, 3, 18), date(2025, 3, 19)),
        (date(2025, 5, 6),  date(2025, 5, 7)),
        (date(2025, 6, 17), date(2025, 6, 18)),
        (date(2025, 7, 29), date(2025, 7, 30)),
        (date(2025, 9, 16), date(2025, 9, 17)),
        (date(2025, 10, 28), date(2025, 10, 29)),
        (date(2025, 12, 9), date(2025, 12, 10)),
        (date(2026, 1, 27), date(2026, 1, 28)),
        (date(2026, 3, 17), date(2026, 3, 18)),
        (date(2026, 4, 28), date(2026, 4, 29)),
        (date(2026, 6, 16), date(2026, 6, 17)),
        (date(2026, 7, 28), date(2026, 7, 29)),
        (date(2026, 9, 15), date(2026, 9, 16)),
        (date(2026, 10, 27), date(2026, 10, 28)),
        (date(2026, 12, 8), date(2026, 12, 9)),
    ]

    @classmethod
    def _all_meeting_days(cls) -> List[date]:
        days = []
        for start, end in cls._MEETING_DATES:
            d = start
            while d <= end:
                days.append(d)
                d += timedelta(days=1)
        return days

    @classmethod
    def has_fomc_in_window(cls, max_dte: int) -> bool:
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return any(today <= d <= window_end for d in cls._all_meeting_days())

    @classmethod
    def next_fomc_date(cls, max_dte: int) -> Optional[date]:
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        upcoming = [d for d in cls._all_meeting_days() if today <= d <= window_end]
        return min(upcoming) if upcoming else None


@dataclass
class EquityFilterResult:
    """Result of equity technical filter."""
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


class EquityFilter:
    """Filters equities based on technical criteria."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.earnings_calendar = EarningsCalendar(max_dte=config.max_dte)
        self.dividend_calendar = DividendCalendar(max_dte=config.max_dte)

    def evaluate(self, symbol: str, prices: pd.Series) -> EquityFilterResult:
        failure_reasons = []

        if len(prices) < 50:
            return EquityFilterResult(
                symbol=symbol, passes=False,
                current_price=prices.iloc[-1] if len(prices) > 0 else 0,
                sma_8=0, sma_20=0, sma_50=0, rsi=0, bb_upper=0,
                sma_50_trending=False,
                failure_reasons=["Insufficient price history"],
            )

        current_price = prices.iloc[-1]
        sma_8 = self.indicators.sma(prices, 8).iloc[-1]
        sma_20 = self.indicators.sma(prices, 20).iloc[-1]
        sma_50 = self.indicators.sma(prices, 50).iloc[-1]
        rsi = self.indicators.rsi(prices, self.config.rsi_period).iloc[-1]
        _, _, bb_upper = self.indicators.bollinger_bands(prices, self.config.bb_period, self.config.bb_std)
        bb_upper_val = bb_upper.iloc[-1]
        sma_50_trending = self.indicators.sma_trend(prices, 50, self.config.sma_trend_lookback)

        if self.config.enable_sma8_check:
            if not (current_price > sma_8):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(8) {sma_8:.2f}")
        if self.config.enable_sma20_check:
            if not (current_price > sma_20):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(20) {sma_20:.2f}")
        if self.config.enable_sma50_check:
            if not (current_price > sma_50):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(50) {sma_50:.2f}")

        if self.config.enable_bb_upper_check:
            if not (current_price > bb_upper_val):
                failure_reasons.append(f"Price {current_price:.2f} <= BB_upper({self.config.bb_period}) {bb_upper_val:.2f}")

        if self.config.enable_band_check:
            band_period = self.config.sma_bb_period
            sma_band = self.indicators.sma(prices, band_period).iloc[-1]
            _, _, bb_band_upper = self.indicators.bollinger_bands(prices, band_period, self.config.bb_std)
            bb_band_upper_val = bb_band_upper.iloc[-1]
            if not (sma_band <= current_price <= bb_band_upper_val):
                if current_price < sma_band:
                    failure_reasons.append(f"Price {current_price:.2f} < SMA({band_period}) {sma_band:.2f}")
                else:
                    failure_reasons.append(f"Price {current_price:.2f} > BB_upper({band_period}) {bb_band_upper_val:.2f}")

        if self.config.enable_sma50_trend_check:
            if not sma_50_trending:
                failure_reasons.append("SMA(50) not trending up")

        if self.config.enable_rsi_check:
            if not (self.config.rsi_lower < rsi < self.config.rsi_upper):
                failure_reasons.append(f"RSI {rsi:.1f} outside [{self.config.rsi_lower}, {self.config.rsi_upper}]")

        if self.config.enable_position_size_check:
            max_position_value = self.config.starting_cash * self.config.max_position_pct
            collateral_required = current_price * 100
            if collateral_required > max_position_value:
                failure_reasons.append(
                    f"Collateral ${collateral_required:,.0f} > {self.config.max_position_pct:.0%} of portfolio (${max_position_value:,.0f})"
                )

        passes = len(failure_reasons) == 0

        return EquityFilterResult(
            symbol=symbol, passes=passes, current_price=current_price,
            sma_8=sma_8, sma_20=sma_20, sma_50=sma_50, rsi=rsi,
            bb_upper=bb_upper_val, sma_50_trending=sma_50_trending,
            failure_reasons=failure_reasons,
        )

    def filter_universe(
        self,
        price_history: Dict[str, pd.Series],
    ) -> Tuple[List[str], List[EquityFilterResult]]:
        results = []
        passing = []
        for symbol, prices in price_history.items():
            result = self.evaluate(symbol, prices)
            results.append(result)
            if result.passes:
                passing.append(symbol)
        return passing, results

    def check_events(self, symbols: List[str]) -> Dict[str, List[str]]:
        rejections: Dict[str, List[str]] = {}

        if not self.config.trade_during_fomc:
            if FomcCalendar.has_fomc_in_window(self.config.max_dte):
                fomc_date = FomcCalendar.next_fomc_date(self.config.max_dte)
                reason = f"FOMC meeting on {fomc_date.isoformat()} within {self.config.max_dte}d window"
                for symbol in symbols:
                    rejections.setdefault(symbol, []).append(reason)
                return rejections

        if not self.config.trade_during_earnings:
            self.earnings_calendar.prefetch(symbols)
            for symbol in symbols:
                if self.earnings_calendar.has_earnings_in_window(symbol, self.config.max_dte):
                    next_date = self.earnings_calendar.next_earnings_date(symbol)
                    rejections.setdefault(symbol, []).append(
                        f"Earnings on {next_date.isoformat()} within {self.config.max_dte}d window"
                    )

        if not self.config.trade_during_dividends:
            self.dividend_calendar.prefetch(symbols)
            for symbol in symbols:
                if self.dividend_calendar.has_exdiv_in_window(symbol, self.config.max_dte):
                    next_date = self.dividend_calendar.next_exdiv_date(symbol)
                    rejections.setdefault(symbol, []).append(
                        f"Ex-div on {next_date.isoformat()} within {self.config.max_dte}d window"
                    )

        return rejections
