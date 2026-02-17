"""Technical indicator calculations for equity filtering."""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Technical indicator calculations. All methods are static and work with pandas Series."""

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple:
        """Bollinger Bands. Returns (lower, middle, upper) Series."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return lower, middle, upper

    @staticmethod
    def sma_trend(
        prices: pd.Series,
        sma_period: int,
        lookback_days: int = 3,
    ) -> bool:
        """Check if SMA is trending upward over lookback period."""
        sma = prices.rolling(window=sma_period).mean()
        if len(sma.dropna()) < lookback_days + 1:
            return False
        recent = sma.dropna().iloc[-(lookback_days + 1):]
        return bool(recent.iloc[-1] > recent.iloc[0])
