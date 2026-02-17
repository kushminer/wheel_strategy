"""Unit tests for GreeksCalculator."""

import pytest
from csp.data.greeks import GreeksCalculator


@pytest.fixture
def calc():
    return GreeksCalculator(risk_free_rate=0.04)


# ── compute_iv ──────────────────────────────────────────────────


class TestComputeIV:
    def test_standard_put(self, calc):
        iv = calc.compute_iv(
            option_price=1.60,
            stock_price=230.0,
            strike=220.0,
            dte=7,
            option_type='put',
        )
        assert iv is not None
        assert 0.0 < iv < 2.0  # reasonable IV range

    def test_standard_call(self, calc):
        iv = calc.compute_iv(
            option_price=1.50,
            stock_price=230.0,
            strike=235.0,
            dte=14,
            option_type='call',
        )
        assert iv is not None
        assert 0.0 < iv < 2.0

    def test_zero_price_returns_none(self, calc):
        iv = calc.compute_iv(0.0, 230.0, 220.0, 7)
        assert iv is None

    def test_zero_dte_returns_none(self, calc):
        iv = calc.compute_iv(1.60, 230.0, 220.0, 0)
        assert iv is None

    def test_negative_price_returns_none(self, calc):
        iv = calc.compute_iv(-1.0, 230.0, 220.0, 7)
        assert iv is None

    def test_negative_strike_returns_none(self, calc):
        iv = calc.compute_iv(1.60, 230.0, -220.0, 7)
        assert iv is None


# ── compute_delta ───────────────────────────────────────────────


class TestComputeDelta:
    def test_standard_put(self, calc):
        delta = calc.compute_delta(230.0, 220.0, 7, iv=0.30, option_type='put')
        assert delta is not None
        assert -1.0 < delta < 0.0  # put delta is negative

    def test_atm_put_near_minus_half(self, calc):
        """ATM put should have delta near -0.5."""
        delta = calc.compute_delta(230.0, 230.0, 30, iv=0.25, option_type='put')
        assert delta is not None
        assert -0.6 < delta < -0.4

    def test_deep_otm_put_near_zero(self, calc):
        delta = calc.compute_delta(230.0, 180.0, 7, iv=0.25, option_type='put')
        assert delta is not None
        assert -0.05 < delta < 0.0

    def test_none_iv_returns_none(self, calc):
        delta = calc.compute_delta(230.0, 220.0, 7, iv=None)
        assert delta is None

    def test_zero_dte_returns_none(self, calc):
        delta = calc.compute_delta(230.0, 220.0, 0, iv=0.30)
        assert delta is None

    def test_negative_iv_returns_none(self, calc):
        delta = calc.compute_delta(230.0, 220.0, 7, iv=-0.30)
        assert delta is None


# ── compute_all_greeks ──────────────────────────────────────────


class TestComputeAllGreeks:
    def test_all_keys_present(self, calc):
        result = calc.compute_all_greeks(230.0, 220.0, 7, iv=0.30)
        assert 'delta' in result
        assert 'gamma' in result
        assert 'theta' in result
        assert 'vega' in result

    def test_gamma_positive(self, calc):
        result = calc.compute_all_greeks(230.0, 220.0, 7, iv=0.30)
        assert result['gamma'] is not None
        assert result['gamma'] > 0

    def test_theta_negative_for_put(self, calc):
        result = calc.compute_all_greeks(230.0, 220.0, 7, iv=0.30, option_type='put')
        assert result['theta'] is not None
        assert result['theta'] < 0

    def test_invalid_iv_returns_nones(self, calc):
        result = calc.compute_all_greeks(230.0, 220.0, 7, iv=None)
        assert all(v is None for v in result.values())


# ── compute_greeks_from_price ───────────────────────────────────


class TestComputeGreeksFromPrice:
    def test_end_to_end(self, calc):
        result = calc.compute_greeks_from_price(1.60, 230.0, 220.0, 7, 'put')
        assert result['iv'] is not None
        assert result['delta'] is not None
        assert result['gamma'] is not None
        assert result['theta'] is not None
        assert result['vega'] is not None

    def test_propagates_none_on_failure(self, calc):
        result = calc.compute_greeks_from_price(0.0, 230.0, 220.0, 7)
        assert result['iv'] is None
        assert result['delta'] is None
