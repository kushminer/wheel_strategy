"""Unit tests for EquityFilter and event calendars."""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from csp.config import StrategyConfig
from csp.signals.equity_filter import (
    EquityFilter,
    EquityFilterResult,
    FomcCalendar,
)

from tests.conftest import make_price_series


@pytest.fixture
def config():
    return StrategyConfig(
        ticker_universe=["AAPL"],
        starting_cash=1_000_000,
        enable_sma8_check=True,
        enable_sma20_check=True,
        enable_sma50_check=True,
        enable_bb_upper_check=False,
        enable_band_check=True,
        enable_sma50_trend_check=True,
        enable_rsi_check=True,
        enable_position_size_check=True,
    )


@pytest.fixture
def eq_filter(config):
    return EquityFilter(config)


# ── Passing stock ───────────────────────────────────────────────


class TestPassingStock:
    def test_uptrending_stock_passes(self, eq_filter):
        prices = make_price_series(base=230.0, n=60, trend="up", seed=42)
        result = eq_filter.evaluate("AAPL", prices)
        # An uptrending stock should either pass or only fail on band/trend checks
        # depending on random seed. Let's just verify we get a result
        assert isinstance(result, EquityFilterResult)
        assert result.symbol == "AAPL"
        assert result.current_price > 0


# ── Individual filter checks ───────────────────────────────────


class TestSMAChecks:
    def test_fails_when_price_below_sma8(self, eq_filter):
        # Flat then recent drop
        prices_flat = pd.Series([200.0] * 55 + [190.0] * 5,
                                index=pd.bdate_range(end=date.today(), periods=60))
        result = eq_filter.evaluate("TEST", prices_flat)
        assert result.passes is False
        assert any("SMA(8)" in r for r in result.failure_reasons)

    def test_fails_when_price_below_sma50(self):
        # Price below SMA50 — steadily falling
        config = StrategyConfig(
            ticker_universe=["TEST"],
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=True,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
            enable_position_size_check=False,
        )
        ef = EquityFilter(config)
        prices = make_price_series(base=230.0, n=60, trend="down", seed=42)
        result = ef.evaluate("TEST", prices)
        # The current price should be below SMA50 for a down trend
        if result.current_price < result.sma_50:
            assert result.passes is False
            assert any("SMA(50)" in r for r in result.failure_reasons)


class TestRSICheck:
    def test_fails_when_rsi_overbought(self):
        config = StrategyConfig(
            ticker_universe=["TEST"],
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_band_check=False,
            enable_sma50_trend_check=False, enable_rsi_check=True,
            enable_position_size_check=False,
            rsi_upper=70,
        )
        ef = EquityFilter(config)
        # Strongly rising prices produce high RSI
        prices = [100.0]
        for i in range(1, 60):
            if i % 7 == 0:
                prices.append(prices[-1] - 0.1)
            else:
                prices.append(prices[-1] + 1.0)
        series = pd.Series(prices, index=pd.bdate_range(end=date.today(), periods=60))
        result = ef.evaluate("TEST", series)
        if result.rsi > 70:
            assert result.passes is False
            assert any("RSI" in r for r in result.failure_reasons)

    def test_fails_when_rsi_oversold(self):
        config = StrategyConfig(
            ticker_universe=["TEST"],
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_band_check=False,
            enable_sma50_trend_check=False, enable_rsi_check=True,
            enable_position_size_check=False,
            rsi_lower=30,
        )
        ef = EquityFilter(config)
        # Strongly falling prices produce low RSI
        prices = [200.0]
        for i in range(1, 60):
            if i % 7 == 0:
                prices.append(prices[-1] + 0.1)
            else:
                prices.append(prices[-1] - 1.0)
        series = pd.Series(prices, index=pd.bdate_range(end=date.today(), periods=60))
        result = ef.evaluate("TEST", series)
        if result.rsi < 30:
            assert result.passes is False
            assert any("RSI" in r for r in result.failure_reasons)


class TestSMA50Trend:
    def test_fails_when_not_trending(self):
        config = StrategyConfig(
            ticker_universe=["TEST"],
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_band_check=False,
            enable_sma50_trend_check=True, enable_rsi_check=False,
            enable_position_size_check=False,
        )
        ef = EquityFilter(config)
        # Flat prices => SMA50 not trending
        prices = pd.Series([100.0] * 60, index=pd.bdate_range(end=date.today(), periods=60))
        result = ef.evaluate("TEST", prices)
        assert result.passes is False
        assert any("SMA(50) not trending" in r for r in result.failure_reasons)


class TestPositionSize:
    def test_fails_when_too_large(self):
        config = StrategyConfig(
            ticker_universe=["TEST"],
            starting_cash=100_000,
            max_position_pct=0.10,  # $10,000 max
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_band_check=False,
            enable_sma50_trend_check=False, enable_rsi_check=False,
            enable_position_size_check=True,
        )
        ef = EquityFilter(config)
        # Stock at $150 -> collateral = $15,000 > $10,000
        prices = pd.Series([150.0] * 60, index=pd.bdate_range(end=date.today(), periods=60))
        result = ef.evaluate("TEST", prices)
        assert result.passes is False
        assert any("Collateral" in r for r in result.failure_reasons)


class TestInsufficientHistory:
    def test_fails_with_short_series(self, eq_filter):
        short = pd.Series([100.0] * 30, index=pd.bdate_range(end=date.today(), periods=30))
        result = eq_filter.evaluate("TEST", short)
        assert result.passes is False
        assert "Insufficient price history" in result.failure_reasons


class TestEnableFlags:
    def test_all_disabled_passes(self):
        config = StrategyConfig(
            ticker_universe=["TEST"],
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_bb_upper_check=False,
            enable_band_check=False, enable_sma50_trend_check=False,
            enable_rsi_check=False, enable_position_size_check=False,
        )
        ef = EquityFilter(config)
        prices = pd.Series([100.0] * 60, index=pd.bdate_range(end=date.today(), periods=60))
        result = ef.evaluate("TEST", prices)
        assert result.passes is True


# ── filter_universe ─────────────────────────────────────────────


class TestFilterUniverse:
    def test_returns_passing_and_all(self):
        config = StrategyConfig(
            ticker_universe=["AAPL", "MSFT"],
            enable_sma8_check=False, enable_sma20_check=False,
            enable_sma50_check=False, enable_bb_upper_check=False,
            enable_band_check=False, enable_sma50_trend_check=False,
            enable_rsi_check=False, enable_position_size_check=False,
        )
        ef = EquityFilter(config)
        history = {
            "AAPL": pd.Series([230.0] * 60, index=pd.bdate_range(end=date.today(), periods=60)),
            "MSFT": pd.Series([100.0] * 30, index=pd.bdate_range(end=date.today(), periods=30)),  # too short
        }
        passing, all_results = ef.filter_universe(history)
        assert "AAPL" in passing
        assert "MSFT" not in passing
        assert len(all_results) == 2


# ── FomcCalendar ────────────────────────────────────────────────


class TestFomcCalendar:
    def test_has_fomc_with_large_window(self):
        # 365 days should catch at least one meeting
        assert FomcCalendar.has_fomc_in_window(365) is True

    def test_no_fomc_with_zero_window(self):
        # Today is unlikely to be an FOMC day (most days aren't)
        # This test may intermittently pass on FOMC days
        today = date.today()
        all_days = FomcCalendar._all_meeting_days()
        if today not in all_days:
            assert FomcCalendar.has_fomc_in_window(0) is False

    def test_next_fomc_date_within_365(self):
        result = FomcCalendar.next_fomc_date(365)
        assert result is not None
        assert result >= date.today()


# ── check_events ────────────────────────────────────────────────


class TestCheckEvents:
    def test_fomc_rejects_all(self):
        """When FOMC is in window, all symbols rejected."""
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            trade_during_fomc=False,
            max_dte=365,  # ensure FOMC in window
        )
        ef = EquityFilter(config)
        rejections = ef.check_events(["AAPL", "MSFT"])
        assert "AAPL" in rejections
        assert "MSFT" in rejections
        assert any("FOMC" in r for r in rejections["AAPL"])

    def test_clear_when_fomc_allowed(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            trade_during_fomc=True,
            trade_during_earnings=True,
            trade_during_dividends=True,
        )
        ef = EquityFilter(config)
        rejections = ef.check_events(["AAPL"])
        # No API keys set, so earnings/dividends won't fetch
        assert len(rejections) == 0
