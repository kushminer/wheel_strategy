"""Tests for MarketSession — market hours and VIX session management."""

from datetime import date, datetime, time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytz

from csp.trading.market_session import MarketSession


EASTERN = pytz.timezone("US/Eastern")


def _make_session(vix_spike_multiplier=1.15):
    """Build a MarketSession with mocked dependencies."""
    alpaca_manager = MagicMock()
    vix_fetcher = MagicMock()
    messages = []
    session = MarketSession(
        alpaca_manager=alpaca_manager,
        vix_fetcher=vix_fetcher,
        vix_spike_multiplier=vix_spike_multiplier,
        vprint=messages.append,
    )
    return session, alpaca_manager, vix_fetcher, messages


class TestIsMarketOpen:
    """Test market open/closed detection."""

    async def test_weekend_returns_false(self):
        session, _, _, _ = _make_session()

        # Saturday
        with patch("csp.trading.market_session.datetime") as mock_dt:
            sat = datetime(2026, 2, 21, 12, 0, 0, tzinfo=EASTERN)  # Saturday
            mock_dt.now.return_value = sat
            result = await session.is_market_open()

        assert not result

    async def test_sunday_returns_false(self):
        session, _, _, _ = _make_session()

        with patch("csp.trading.market_session.datetime") as mock_dt:
            sun = datetime(2026, 2, 22, 12, 0, 0, tzinfo=EASTERN)  # Sunday
            mock_dt.now.return_value = sun
            result = await session.is_market_open()

        assert not result

    async def test_weekday_during_hours_returns_true(self):
        session, alpaca_manager, _, _ = _make_session()

        # Mock Alpaca calendar response
        cal_entry = SimpleNamespace(
            date=date(2026, 2, 23),
            open=SimpleNamespace(time=lambda: dt_time(9, 30)),
            close=SimpleNamespace(time=lambda: dt_time(16, 0)),
        )
        alpaca_manager.trading_client.get_calendar.return_value = [cal_entry]

        with patch("csp.trading.market_session.datetime") as mock_dt:
            # Monday at noon
            mock_dt.now.return_value = datetime(2026, 2, 23, 12, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = await session.is_market_open()

        assert result

    async def test_weekday_before_open_returns_false(self):
        session, alpaca_manager, _, _ = _make_session()

        cal_entry = SimpleNamespace(
            date=date(2026, 2, 23),
            open=SimpleNamespace(time=lambda: dt_time(9, 30)),
            close=SimpleNamespace(time=lambda: dt_time(16, 0)),
        )
        alpaca_manager.trading_client.get_calendar.return_value = [cal_entry]

        with patch("csp.trading.market_session.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 23, 8, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = await session.is_market_open()

        assert not result

    async def test_weekday_after_close_returns_false(self):
        session, alpaca_manager, _, _ = _make_session()

        cal_entry = SimpleNamespace(
            date=date(2026, 2, 23),
            open=SimpleNamespace(time=lambda: dt_time(9, 30)),
            close=SimpleNamespace(time=lambda: dt_time(16, 0)),
        )
        alpaca_manager.trading_client.get_calendar.return_value = [cal_entry]

        with patch("csp.trading.market_session.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 23, 17, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = await session.is_market_open()

        assert not result

    async def test_holiday_returns_false(self):
        """If Alpaca returns no calendar entries, today is not a trading day."""
        session, alpaca_manager, _, _ = _make_session()

        alpaca_manager.trading_client.get_calendar.return_value = []

        with patch("csp.trading.market_session.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 23, 12, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = await session.is_market_open()

        assert not result


class TestTradingDayCache:
    """Test that the calendar API is only called once per day."""

    async def test_caches_after_first_call(self):
        session, alpaca_manager, _, _ = _make_session()

        cal_entry = SimpleNamespace(
            date=date(2026, 2, 23),
            open=SimpleNamespace(time=lambda: dt_time(9, 30)),
            close=SimpleNamespace(time=lambda: dt_time(16, 0)),
        )
        alpaca_manager.trading_client.get_calendar.return_value = [cal_entry]

        with patch("csp.trading.market_session.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 23, 12, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat

            await session.is_market_open()
            await session.is_market_open()
            await session.is_market_open()

        # Calendar should only be fetched once
        assert alpaca_manager.trading_client.get_calendar.call_count == 1

    async def test_api_failure_falls_back_to_time_check(self):
        """If Alpaca calendar raises, assume trading day and use default hours."""
        session, alpaca_manager, _, messages = _make_session()

        alpaca_manager.trading_client.get_calendar.side_effect = Exception("API down")

        with patch("csp.trading.market_session.datetime") as mock_dt:
            # Monday at noon — within default 9:30-16:00
            mock_dt.now.return_value = datetime(2026, 2, 23, 12, 0, 0, tzinfo=EASTERN)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = await session.is_market_open()

        assert result  # Falls back to time-only check
        assert any("falling back" in m.lower() for m in messages)


class TestSessionVixReference:
    """Test VIX reference caching."""

    async def test_fetches_vix_on_first_call(self):
        session, _, vix_fetcher, _ = _make_session()
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 18.5)

        result = await session.get_session_vix_reference()

        assert result == 18.5
        vix_fetcher.get_session_reference_vix.assert_called_once()

    async def test_caches_vix_for_same_day(self):
        session, _, vix_fetcher, _ = _make_session()
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 20.0)

        v1 = await session.get_session_vix_reference()
        v2 = await session.get_session_vix_reference()
        v3 = await session.get_session_vix_reference()

        assert v1 == v2 == v3 == 20.0
        vix_fetcher.get_session_reference_vix.assert_called_once()

    async def test_resets_on_new_day(self):
        session, _, vix_fetcher, _ = _make_session()

        # First call: today
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 18.0)
        await session.get_session_vix_reference()

        # Simulate new day by changing cached date
        session._session_vix_open = (date(2026, 1, 1), 18.0)
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 22.0)

        result = await session.get_session_vix_reference()

        assert result == 22.0
        assert vix_fetcher.get_session_reference_vix.call_count == 2


class TestGlobalVixStop:
    """Test global VIX stop detection."""

    async def test_no_stop_when_below_threshold(self):
        session, _, vix_fetcher, _ = _make_session(vix_spike_multiplier=1.15)
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 20.0)

        # 20.0 * 1.15 = 23.0 threshold; 22.0 < 23.0
        result = await session.check_global_vix_stop(22.0)
        assert not result

    async def test_stop_triggered_at_threshold(self):
        session, _, vix_fetcher, _ = _make_session(vix_spike_multiplier=1.15)
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 20.0)

        # 20.0 * 1.15 = 23.0; 23.0 >= 23.0
        result = await session.check_global_vix_stop(23.0)
        assert result

    async def test_stop_triggered_above_threshold(self):
        session, _, vix_fetcher, _ = _make_session(vix_spike_multiplier=1.15)
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 20.0)

        # 20.0 * 1.15 = 23.0; 30.0 >= 23.0
        result = await session.check_global_vix_stop(30.0)
        assert result

    async def test_custom_multiplier(self):
        session, _, vix_fetcher, _ = _make_session(vix_spike_multiplier=1.25)
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 20.0)

        # 20.0 * 1.25 = 25.0; 24.0 < 25.0
        assert not await session.check_global_vix_stop(24.0)

        # 25.0 >= 25.0
        assert await session.check_global_vix_stop(25.0)

    async def test_uses_cached_reference_vix(self):
        """Global VIX stop should reuse cached session reference."""
        session, _, vix_fetcher, _ = _make_session()
        vix_fetcher.get_session_reference_vix.return_value = (date.today(), 18.0)

        await session.check_global_vix_stop(19.0)
        await session.check_global_vix_stop(20.0)
        await session.check_global_vix_stop(21.0)

        # Only one fetch despite three checks
        vix_fetcher.get_session_reference_vix.assert_called_once()
