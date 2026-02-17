"""Unit tests for TechnicalIndicators."""

import math
import numpy as np
import pandas as pd
import pytest

from csp.signals.indicators import TechnicalIndicators


@pytest.fixture
def rising_prices():
    """60-day consistently rising price series."""
    return pd.Series(
        [100 + i * 0.5 for i in range(60)],
        index=pd.bdate_range(end="2026-02-16", periods=60),
    )


@pytest.fixture
def falling_prices():
    """60-day consistently falling price series."""
    return pd.Series(
        [130 - i * 0.5 for i in range(60)],
        index=pd.bdate_range(end="2026-02-16", periods=60),
    )


@pytest.fixture
def flat_prices():
    """60-day flat price series."""
    return pd.Series(
        [100.0] * 60,
        index=pd.bdate_range(end="2026-02-16", periods=60),
    )


class TestSMA:
    def test_basic_calculation(self):
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        sma = TechnicalIndicators.sma(prices, 5)
        assert sma.iloc[-1] == 3.0

    def test_insufficient_data_returns_nan(self):
        prices = pd.Series([1, 2, 3], dtype=float)
        sma = TechnicalIndicators.sma(prices, 5)
        assert all(math.isnan(v) for v in sma)


class TestRSI:
    def test_overbought_with_mostly_rising_prices(self):
        """Mostly rising with periodic tiny dips so RSI denominator is nonzero."""
        prices = [100.0]
        for i in range(1, 60):
            if i % 5 == 0:
                prices.append(prices[-1] - 0.2)  # tiny dip every 5 bars
            else:
                prices.append(prices[-1] + 0.6)  # clear rise
        rsi = TechnicalIndicators.rsi(pd.Series(prices, dtype=float))
        valid = rsi.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 70

    def test_oversold_with_falling_prices(self, falling_prices):
        rsi = TechnicalIndicators.rsi(falling_prices)
        assert rsi.dropna().iloc[-1] < 20

    def test_neutral_with_alternating_prices(self):
        # Alternating up/down should produce RSI near 50
        prices = pd.Series([100 + (i % 2) for i in range(60)], dtype=float)
        rsi = TechnicalIndicators.rsi(prices)
        last_rsi = rsi.dropna().iloc[-1]
        assert 40 < last_rsi < 60


class TestBollingerBands:
    def test_middle_equals_sma(self, rising_prices):
        lower, middle, upper = TechnicalIndicators.bollinger_bands(rising_prices, period=20)
        sma20 = TechnicalIndicators.sma(rising_prices, 20)
        pd.testing.assert_series_equal(middle, sma20)

    def test_upper_above_middle(self, rising_prices):
        lower, middle, upper = TechnicalIndicators.bollinger_bands(rising_prices, period=20)
        valid = middle.dropna()
        valid_upper = upper.loc[valid.index]
        assert (valid_upper >= valid).all()

    def test_lower_below_middle(self, rising_prices):
        lower, middle, upper = TechnicalIndicators.bollinger_bands(rising_prices, period=20)
        valid = middle.dropna()
        valid_lower = lower.loc[valid.index]
        assert (valid_lower <= valid).all()

    def test_width_proportional_to_std(self):
        prices = pd.Series(np.random.default_rng(42).normal(100, 5, 60), dtype=float)
        lower, middle, upper = TechnicalIndicators.bollinger_bands(prices, period=20, num_std=2.0)
        std = prices.rolling(20).std()
        expected_width = 2 * 2.0 * std  # 2 * num_std * std
        actual_width = upper - lower
        # Compare at a point where we have enough data
        idx = 50
        assert abs(actual_width.iloc[idx] - expected_width.iloc[idx]) < 0.01


class TestSMATrend:
    def test_upward_trend_returns_true(self, rising_prices):
        assert TechnicalIndicators.sma_trend(rising_prices, sma_period=20) is True

    def test_flat_prices_returns_false(self, flat_prices):
        assert TechnicalIndicators.sma_trend(flat_prices, sma_period=20) is False

    def test_insufficient_data_returns_false(self):
        short = pd.Series([100, 101, 102], dtype=float)
        assert TechnicalIndicators.sma_trend(short, sma_period=20) is False
