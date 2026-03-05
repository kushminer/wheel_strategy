"""Market hours and VIX session state — strategy-agnostic infrastructure.

Extracted from TradingLoop so any strategy can reuse market open checks,
session VIX reference caching, and global VIX stop detection.
"""

from datetime import date, datetime, time as dt_time
from typing import Callable, Optional, Tuple

import pytz

from csp.trading.utils import _arun


class MarketSession:
    """Strategy-agnostic market hours and VIX session management.

    Provides:
    - Market open/closed detection (Alpaca calendar API with daily cache)
    - Session-open VIX reference (cached per trading day)
    - Global VIX stop check (VIX spike from session open)
    """

    def __init__(
        self,
        alpaca_manager,
        vix_fetcher,
        vix_spike_multiplier: float = 1.15,
        vprint: Callable = None,
    ):
        self.alpaca_manager = alpaca_manager
        self.vix_fetcher = vix_fetcher
        self.vix_spike_multiplier = vix_spike_multiplier
        self._vprint = vprint or (lambda msg: None)

        self.eastern = pytz.timezone('US/Eastern')
        self._trading_day_cache: dict = {}
        self._session_vix_open: Optional[Tuple[date, float]] = None

    async def is_market_open(self) -> bool:
        """Check if US market is currently open using Alpaca calendar API.

        Caches the trading-day check per calendar date so only one API call per day.
        """
        now = datetime.now(self.eastern)
        today = now.date()

        # Weekday check (Mon=0, Fri=4) — fast reject weekends
        if now.weekday() > 4:
            return False

        # Check if today is a trading day (holiday check, cached per day)
        if self._trading_day_cache.get('date') != today:
            try:
                from alpaca.trading.requests import GetCalendarRequest
                cal_req = GetCalendarRequest(start=today, end=today)
                cal = await _arun(self.alpaca_manager.trading_client.get_calendar, cal_req)
                is_trading_day = len(cal) > 0 and cal[0].date == today
                self._trading_day_cache = {
                    'date': today,
                    'is_trading_day': is_trading_day,
                    'open': cal[0].open.time() if is_trading_day else None,
                    'close': cal[0].close.time() if is_trading_day else None,
                }
                if not is_trading_day:
                    print(f"  Market closed today ({today} is not a trading day)")
            except Exception as e:
                self._vprint(f"  Warning: Alpaca calendar check failed ({e}), falling back to time-only check")
                self._trading_day_cache = {'date': today, 'is_trading_day': True, 'open': None, 'close': None}

        if not self._trading_day_cache['is_trading_day']:
            return False

        # Time check — use Alpaca hours if available, else default 9:30-16:00 ET
        market_open = self._trading_day_cache.get('open') or dt_time(9, 30)
        market_close = self._trading_day_cache.get('close') or dt_time(16, 0)

        return market_open <= now.time() <= market_close

    async def get_session_vix_reference(self) -> float:
        """Get VIX reference for current session.

        Uses session open VIX, cached for the day.
        """
        session_date = datetime.now(self.eastern).date()

        if self._session_vix_open is None:
            _, vix_open = await _arun(self.vix_fetcher.get_session_reference_vix)
            self._session_vix_open = (session_date, vix_open)

        # Reset if new day
        if self._session_vix_open[0] != session_date:
            _, vix_open = await _arun(self.vix_fetcher.get_session_reference_vix)
            self._session_vix_open = (session_date, vix_open)

        return self._session_vix_open[1]

    async def check_global_vix_stop(self, current_vix: float) -> bool:
        """Check if global VIX stop is triggered.

        If VIX >= multiplier * session open VIX, signals to close ALL positions.

        Returns:
            True if global stop triggered.
        """
        reference_vix = await self.get_session_vix_reference()
        threshold = reference_vix * self.vix_spike_multiplier

        return current_vix >= threshold
