"""Technical indicator calculations for equity filtering."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicator calculations for equity filtering.
    All methods are static and work with pandas Series.
    """

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            prices: Series of prices
            period: Lookback period

        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            prices: Series of prices
            period: Lookback period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            prices: Series of prices
            period: Lookback period for SMA (default 20)
            num_std: Number of standard deviations (default 2.0)

        Returns:
            Tuple of (lower_band, middle_band, upper_band) as Series
        """
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
        """
        Check if SMA is trending upward over lookback period.

        Args:
            prices: Series of prices
            sma_period: Period for SMA calculation
            lookback_days: Number of days to check trend

        Returns:
            True if SMA has been rising for all lookback_days
        """
        sma = TechnicalIndicators.sma(prices, sma_period)

        if len(sma) < lookback_days + 1:
            return False

        # Check if each day's SMA > previous day's SMA
        for i in range(1, lookback_days + 1):
            if sma.iloc[-i] <= sma.iloc[-(i + 1)]:
                return False

        return True
