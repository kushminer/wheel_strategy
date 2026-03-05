"""Event calendars — earnings, dividends, and FOMC meeting schedules."""

import csv
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests


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


def check_events(symbols: List[str], config) -> Dict[str, List[str]]:
    """Check earnings, dividends, and FOMC events for symbols.

    Args:
        symbols: list of ticker symbols to check
        config: object with trade_during_earnings, trade_during_dividends,
                trade_during_fomc, and max_dte attributes

    Returns:
        Dict of symbol -> list of rejection reasons.
    """
    rejections: Dict[str, List[str]] = {}

    if not config.trade_during_fomc:
        if FomcCalendar.has_fomc_in_window(config.max_dte):
            fomc_date = FomcCalendar.next_fomc_date(config.max_dte)
            reason = f"FOMC meeting on {fomc_date.isoformat()} within {config.max_dte}d window"
            for symbol in symbols:
                rejections.setdefault(symbol, []).append(reason)
            return rejections

    earnings_cal = EarningsCalendar(max_dte=config.max_dte)
    dividend_cal = DividendCalendar(max_dte=config.max_dte)

    if not config.trade_during_earnings:
        earnings_cal.prefetch(symbols)
        for symbol in symbols:
            if earnings_cal.has_earnings_in_window(symbol, config.max_dte):
                next_date = earnings_cal.next_earnings_date(symbol)
                rejections.setdefault(symbol, []).append(
                    f"Earnings on {next_date.isoformat()} within {config.max_dte}d window"
                )

    if not config.trade_during_dividends:
        dividend_cal.prefetch(symbols)
        for symbol in symbols:
            if dividend_cal.has_exdiv_in_window(symbol, config.max_dte):
                next_date = dividend_cal.next_exdiv_date(symbol)
                rejections.setdefault(symbol, []).append(
                    f"Ex-div on {next_date.isoformat()} within {config.max_dte}d window"
                )

    return rejections
