"""Unit tests for StrategyConfig."""

import pytest
from csp.config import StrategyConfig


@pytest.fixture
def config():
    """Config with small universe to avoid Wikipedia fetch."""
    return StrategyConfig(ticker_universe=["AAPL", "MSFT"])


class TestVixDeploymentMultiplier:
    def test_vix_below_12_returns_zero(self, config):
        assert config.get_deployment_multiplier(11) == 0.0

    def test_vix_12_to_15_returns_0_2(self, config):
        assert config.get_deployment_multiplier(13) == 0.2

    def test_vix_15_to_18_returns_0_8(self, config):
        assert config.get_deployment_multiplier(16) == 0.8

    def test_vix_18_to_21_returns_0_9(self, config):
        assert config.get_deployment_multiplier(19) == 0.9

    def test_vix_above_21_returns_1_0(self, config):
        assert config.get_deployment_multiplier(25) == 1.0

    def test_exact_boundary_12_goes_to_upper_band(self, config):
        assert config.get_deployment_multiplier(12.0) == 0.2

    def test_exact_boundary_15_goes_to_upper_band(self, config):
        assert config.get_deployment_multiplier(15.0) == 0.8

    def test_negative_vix_returns_zero(self, config):
        assert config.get_deployment_multiplier(-1) == 0.0


class TestDeployableCash:
    def test_scales_with_multiplier(self, config):
        # default starting_cash=1_000_000, VIX=16 -> multiplier=0.8
        assert config.get_deployable_cash(16) == 800_000

    def test_zero_when_vix_low(self, config):
        assert config.get_deployable_cash(10) == 0.0

    def test_full_deployment_high_vix(self, config):
        assert config.get_deployable_cash(25) == 1_000_000

    def test_custom_vix_rules(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            vix_deployment_rules={(0, 50): 0.5, (50, float('inf')): 1.0},
        )
        assert config.get_deployment_multiplier(30) == 0.5
        assert config.get_deployment_multiplier(60) == 1.0


class TestDefaults:
    def test_default_filter_toggles(self, config):
        assert config.enable_sma8_check is True
        assert config.enable_bb_upper_check is False
        assert config.enable_delta_stop is False
        assert config.enable_delta_absolute_stop is True

    def test_default_risk_params(self, config):
        assert config.delta_stop_multiplier == 2.0
        assert config.delta_absolute_stop == 0.40
        assert config.stock_drop_stop_pct == 0.05
        assert config.vix_spike_multiplier == 1.15
