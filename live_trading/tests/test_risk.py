"""Unit tests for RiskManager — highest value unit tests."""

import pytest
from types import SimpleNamespace

from csp.config import StrategyConfig
from csp.trading.risk import RiskManager
from csp.trading.models import ExitReason


@pytest.fixture
def config():
    return StrategyConfig(
        ticker_universe=["AAPL"],
        # Enable all stops for testing
        enable_delta_stop=True,
        enable_delta_absolute_stop=True,
        enable_stock_drop_stop=True,
        enable_vix_spike_stop=True,
        enable_early_exit=True,
    )


@pytest.fixture
def risk(config):
    return RiskManager(config)


@pytest.fixture
def position():
    """Standard position for risk checks."""
    return SimpleNamespace(
        entry_delta=-0.25,
        entry_stock_price=230.0,
        entry_vix=18.0,
        entry_premium=1.50,
        entry_daily_return=0.0015,
        strike=220.0,
        days_held=3,
    )


# ── Delta Stop (relative: 2x entry) ────────────────────────────


class TestDeltaStop:
    def test_triggers_at_2x(self, risk, position):
        result = risk.check_delta_stop(position, current_delta=-0.50)
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.DELTA_STOP

    def test_not_below_2x(self, risk, position):
        result = risk.check_delta_stop(position, current_delta=-0.49)
        assert result.should_exit is False
        assert result.exit_reason is None

    def test_exact_boundary(self, risk, position):
        # 2x of 0.25 = 0.50 — should trigger (>=)
        result = risk.check_delta_stop(position, current_delta=-0.50)
        assert result.should_exit is True


# ── Delta Absolute Stop (hard cap) ─────────────────────────────


class TestDeltaAbsoluteStop:
    def test_triggers_at_ceiling(self, risk, position):
        result = risk.check_delta_absolute_stop(position, current_delta=-0.40)
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.DELTA_ABSOLUTE

    def test_not_below_ceiling(self, risk, position):
        result = risk.check_delta_absolute_stop(position, current_delta=-0.39)
        assert result.should_exit is False

    def test_above_ceiling(self, risk, position):
        result = risk.check_delta_absolute_stop(position, current_delta=-0.55)
        assert result.should_exit is True


# ── Stock Drop Stop ─────────────────────────────────────────────


class TestStockDropStop:
    def test_triggers_at_5pct(self, risk, position):
        # 230 * 0.95 = 218.5
        result = risk.check_stock_drop_stop(position, current_stock_price=218.5)
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.STOCK_DROP

    def test_not_at_4_99pct(self, risk, position):
        # Just above the threshold
        result = risk.check_stock_drop_stop(position, current_stock_price=218.6)
        assert result.should_exit is False

    def test_exact_boundary(self, risk, position):
        # Exactly at threshold (<=)
        threshold = 230.0 * 0.95  # 218.5
        result = risk.check_stock_drop_stop(position, current_stock_price=threshold)
        assert result.should_exit is True


# ── VIX Spike Stop ──────────────────────────────────────────────


class TestVixSpikeStop:
    def test_triggers_at_15pct(self, risk, position):
        # 18 * 1.15 = 20.7
        result = risk.check_vix_spike_stop(position, current_vix=20.7)
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.VIX_SPIKE

    def test_not_below_15pct(self, risk, position):
        result = risk.check_vix_spike_stop(position, current_vix=20.6)
        assert result.should_exit is False

    def test_reference_vix_override(self, risk, position):
        # Use reference_vix=20 instead of entry_vix=18
        # 20 * 1.15 = 23.0
        result = risk.check_vix_spike_stop(position, current_vix=23.0, reference_vix=20.0)
        assert result.should_exit is True

    def test_reference_vix_override_no_trigger(self, risk, position):
        result = risk.check_vix_spike_stop(position, current_vix=22.9, reference_vix=20.0)
        assert result.should_exit is False


# ── Early Exit ──────────────────────────────────────────────────


class TestEarlyExit:
    def test_triggers_when_premium_captured(self, risk, position):
        # days_held=3, daily_return=0.0015, strike=220, buffer=100%
        # expected = 3 * 0.0015 * 220 = 0.99
        # target = 0.99 + 0.99 = 1.98
        # need premium_captured >= 1.98 → current_premium <= 1.50 - 1.98 = -0.48
        # That's negative, so let's use a position with more days held
        pos = SimpleNamespace(
            entry_delta=-0.25, entry_stock_price=230.0, entry_vix=18.0,
            entry_premium=3.00, entry_daily_return=0.0015,
            strike=220.0, days_held=5,
        )
        # expected = 5 * 0.0015 * 220 = 1.65, target = 1.65 + 1.65 = 3.30
        # premium_captured = 3.00 - (-0.30) = 3.30 >= 3.30 → triggers
        result = risk.check_early_exit(pos, current_premium=-0.30)
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.EARLY_EXIT

    def test_day_zero_returns_false(self, risk):
        pos = SimpleNamespace(
            entry_delta=-0.25, entry_stock_price=230.0, entry_vix=18.0,
            entry_premium=1.50, entry_daily_return=0.0015,
            strike=220.0, days_held=0,
        )
        result = risk.check_early_exit(pos, current_premium=0.50)
        assert result.should_exit is False

    def test_return_source_config(self, risk, position):
        """When early_exit_return_source='config', uses config.min_daily_return."""
        risk.config.early_exit_return_source = "config"
        risk.config.min_daily_return = 0.002
        # days_held=3, daily_return=0.002, strike=220
        # expected = 3 * 0.002 * 220 = 1.32, target = 1.32 + 1.32 = 2.64
        # Use -1.20 to clearly exceed target (captured = 2.70 > 2.64)
        result = risk.check_early_exit(position, current_premium=-1.20)
        assert result.should_exit is True

    def test_buffer_effect(self):
        """Lower buffer makes early exit easier to trigger."""
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            early_exit_buffer_pct=0.0,  # no buffer
        )
        risk = RiskManager(config)
        pos = SimpleNamespace(
            entry_delta=-0.25, entry_stock_price=230.0, entry_vix=18.0,
            entry_premium=1.50, entry_daily_return=0.0015,
            strike=220.0, days_held=3,
        )
        # expected = 3 * 0.0015 * 220 = 0.99, target = 0.99 (no buffer)
        # Use 0.50 to clearly exceed target (captured = 1.00 > 0.99)
        result = risk.check_early_exit(pos, current_premium=0.50)
        assert result.should_exit is True


# ── check_all_stops ─────────────────────────────────────────────


class TestCheckAllStops:
    def test_returns_first_triggered(self, risk, position):
        # Both delta absolute and stock drop triggered — delta absolute checked first
        result = risk.check_all_stops(
            position,
            current_delta=-0.50,
            current_stock_price=200.0,
            current_vix=25.0,
        )
        assert result.should_exit is True
        # Delta stop is checked before delta absolute, and 0.50 >= 2*0.25=0.50
        assert result.exit_reason == ExitReason.DELTA_STOP

    def test_returns_no_exit(self, risk, position):
        result = risk.check_all_stops(
            position,
            current_delta=-0.30,
            current_stock_price=228.0,
            current_vix=19.0,
        )
        assert result.should_exit is False
        assert result.exit_reason is None

    def test_respects_enable_flags(self, position):
        """When all stops are disabled, nothing triggers."""
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            enable_delta_stop=False,
            enable_delta_absolute_stop=False,
            enable_stock_drop_stop=False,
            enable_vix_spike_stop=False,
        )
        risk = RiskManager(config)
        result = risk.check_all_stops(
            position,
            current_delta=-0.90,
            current_stock_price=100.0,
            current_vix=50.0,
        )
        assert result.should_exit is False


# ── evaluate_position ───────────────────────────────────────────


class TestEvaluatePosition:
    def test_stop_priority_over_early_exit(self, risk, position):
        """Stop-loss triggers even if early exit would too."""
        result = risk.evaluate_position(
            position,
            current_delta=-0.50,
            current_stock_price=228.0,
            current_vix=19.0,
            current_premium=0.01,  # huge capture
        )
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.DELTA_STOP

    def test_hold_when_all_clear(self, risk, position):
        result = risk.evaluate_position(
            position,
            current_delta=-0.30,
            current_stock_price=228.0,
            current_vix=19.0,
            current_premium=1.20,  # not enough capture
        )
        assert result.should_exit is False
        assert result.exit_reason is None

    def test_early_exit_when_no_stop(self, risk):
        pos = SimpleNamespace(
            entry_delta=-0.25, entry_stock_price=230.0, entry_vix=18.0,
            entry_premium=3.00, entry_daily_return=0.0015,
            strike=220.0, days_held=5,
        )
        result = risk.evaluate_position(
            pos,
            current_delta=-0.20,
            current_stock_price=232.0,
            current_vix=18.5,
            current_premium=-0.30,  # captured 3.30
        )
        assert result.should_exit is True
        assert result.exit_reason == ExitReason.EARLY_EXIT
