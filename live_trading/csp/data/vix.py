"""VIX data fetcher via Yahoo Finance."""

from datetime import datetime, timedelta, date
from typing import Tuple

import pandas as pd
import yfinance as yf


class VixDataFetcher:
    """Fetches VIX data from Yahoo Finance."""

    SYMBOL = "^VIX"

    def __init__(self):
        self._ticker = yf.Ticker(self.SYMBOL)
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = timedelta(minutes=1)

    def get_current_vix(self) -> float:
        if (self._cache_time and
                datetime.now() - self._cache_time < self._cache_ttl and
                'current' in self._cache):
            return self._cache['current']

        try:
            daily = self._ticker.history(period='5d')
            if daily.empty:
                raise RuntimeError("No VIX data available")
            vix = float(daily['Close'].iloc[-1])
            self._cache['current'] = vix
            self._cache_time = datetime.now()
            return vix
        except Exception as e:
            raise RuntimeError(f"Failed to fetch VIX data: {e}")

    def get_vix_history(self, days: int = 30) -> pd.DataFrame:
        history = self._ticker.history(period=f'{days}d')
        return history[['Open', 'High', 'Low', 'Close']]

    def get_last_session(self) -> dict:
        history = self._ticker.history(period='5d')
        if history.empty:
            raise RuntimeError("No VIX history available")
        last_row = history.iloc[-1]
        session_date = history.index[-1]
        return {
            'session_date': session_date.date() if hasattr(session_date, 'date') else session_date,
            'open': float(last_row['Open']),
            'high': float(last_row['High']),
            'low': float(last_row['Low']),
            'close': float(last_row['Close']),
        }

    def get_session_reference_vix(self) -> Tuple[date, float]:
        session = self.get_last_session()
        return session['session_date'], session['open']

    def check_vix_stop_loss(
        self,
        reference_vix: float,
        multiplier: float = 1.15,
    ) -> dict:
        current_vix = self.get_current_vix()
        threshold = reference_vix * multiplier
        triggered = current_vix >= threshold
        return {
            'triggered': triggered,
            'current_vix': current_vix,
            'reference_vix': reference_vix,
            'threshold': threshold,
            'pct_change': (current_vix / reference_vix - 1) * 100,
            'reason': f"VIX {current_vix:.2f} >= {threshold:.2f}" if triggered else "",
        }
