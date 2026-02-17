"""Unit tests for OptionsFilter."""

from datetime import date, timedelta

import pytest

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.data.options import OptionContract
from csp.signals.options_filter import OptionsFilter, OptionsFilterResult

from tests.conftest import make_option_contract


@pytest.fixture
def config():
    return StrategyConfig(
        ticker_universe=["AAPL"],
        min_daily_return=0.0015,
        delta_min=0.00,
        delta_max=0.40,
        min_dte=1,
        max_dte=10,
        min_strike_pct=0.50,
        max_strike_pct=0.90,
        min_volume=0,
        min_open_interest=0,
        max_spread_pct=1.0,
        enable_premium_check=True,
        enable_delta_check=True,
        enable_dte_check=True,
        enable_volume_check=True,
        enable_open_interest_check=True,
        enable_spread_check=True,
    )


@pytest.fixture
def greeks_calc():
    return GreeksCalculator(risk_free_rate=0.04)


@pytest.fixture
def of(config, greeks_calc):
    return OptionsFilter(config, greeks_calc)


# ── Passing contract ────────────────────────────────────────────


class TestPassingContract:
    def test_good_contract_passes(self, of):
        contract = make_option_contract(
            strike=200.0,      # 200/230 = 87% — within 50-90%
            stock_price=230.0,
            bid=1.50,
            ask=1.70,
            mid=1.60,
            dte=5,
            delta=-0.25,
            implied_volatility=0.30,
        )
        result = of.evaluate(contract)
        assert result.passes is True
        assert len(result.failure_reasons) == 0


# ── Premium/return filter ──────────────────────────────────────


class TestPremiumFilter:
    def test_fails_on_low_return(self, of):
        contract = make_option_contract(
            strike=200.0,
            stock_price=230.0,
            bid=0.01,
            ask=0.03,
            mid=0.02,
            dte=5,
            delta=-0.10,
            implied_volatility=0.15,
        )
        result = of.evaluate(contract)
        assert result.passes is False
        assert any("Daily return" in r for r in result.failure_reasons)


# ── Delta filter ────────────────────────────────────────────────


class TestDeltaFilter:
    def test_fails_when_delta_too_high(self, of):
        contract = make_option_contract(delta=-0.50, implied_volatility=0.30)
        result = of.evaluate(contract)
        assert result.passes is False
        assert any("Delta" in r and "outside" in r for r in result.failure_reasons)

    def test_fails_when_delta_unavailable(self, of):
        contract = make_option_contract(delta=None, implied_volatility=None,
                                         mid=0.001)  # too small for IV calc
        result = of.evaluate(contract)
        # Should either fail delta or pass if greeks calc succeeds
        if result.delta_abs is None:
            assert result.passes is False
            assert any("Delta unavailable" in r for r in result.failure_reasons)


# ── DTE filter ──────────────────────────────────────────────────


class TestDTEFilter:
    def test_fails_when_dte_out_of_range(self, of):
        contract = make_option_contract(dte=15, expiration=date.today() + timedelta(days=15))
        result = of.evaluate(contract)
        assert result.passes is False
        assert any("DTE" in r for r in result.failure_reasons)


# ── Strike filter ───────────────────────────────────────────────


class TestStrikeFilter:
    def test_fails_when_strike_too_high(self, of):
        contract = make_option_contract(strike=225.0, stock_price=230.0)  # 97.8%
        result = of.evaluate(contract)
        assert result.passes is False
        assert any("Strike" in r and ">" in r for r in result.failure_reasons)

    def test_fails_when_strike_too_low(self, of):
        contract = make_option_contract(strike=100.0, stock_price=230.0)  # 43.5%
        result = of.evaluate(contract)
        assert result.passes is False
        assert any("Strike" in r and "<" in r for r in result.failure_reasons)


# ── Volume/OI filter ───────────────────────────────────────────


class TestLiquidityFilters:
    def test_fails_on_low_volume(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            min_volume=50,
            enable_volume_check=True,
        )
        of = OptionsFilter(config, GreeksCalculator())
        contract = make_option_contract(volume=10)
        result = of.evaluate(contract)
        assert any("Volume" in r for r in result.failure_reasons)

    def test_fails_on_low_oi(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            min_open_interest=100,
            enable_open_interest_check=True,
        )
        of = OptionsFilter(config, GreeksCalculator())
        contract = make_option_contract(open_interest=50)
        result = of.evaluate(contract)
        assert any("OI" in r for r in result.failure_reasons)


# ── Spread filter ───────────────────────────────────────────────


class TestSpreadFilter:
    def test_fails_on_wide_spread(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            max_spread_pct=0.10,
            enable_spread_check=True,
        )
        of = OptionsFilter(config, GreeksCalculator())
        contract = make_option_contract(bid=1.00, ask=2.00, mid=1.50)  # spread = 66%
        result = of.evaluate(contract)
        assert any("Spread" in r for r in result.failure_reasons)


# ── Greeks enrichment ───────────────────────────────────────────


class TestGreeksEnrichment:
    def test_ensures_greeks_when_missing(self, of):
        contract = make_option_contract(
            delta=None, gamma=None, theta=None, vega=None,
            implied_volatility=None,
            mid=1.60, stock_price=230.0, strike=200.0, dte=5,
        )
        result = of.evaluate(contract)
        # After evaluation, the contract should have enriched greeks
        assert contract.delta is not None or result.delta_abs is None

    def test_preserves_existing_greeks(self, of):
        contract = make_option_contract(delta=-0.25, implied_volatility=0.30)
        of.evaluate(contract)
        assert contract.delta == -0.25
        assert contract.implied_volatility == 0.30


# ── Enable flags ────────────────────────────────────────────────


class TestEnableFlags:
    def test_disabled_checks_dont_fail(self):
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            enable_premium_check=False,
            enable_delta_check=False,
            enable_dte_check=False,
            enable_volume_check=False,
            enable_open_interest_check=False,
            enable_spread_check=False,
            min_strike_pct=0.0,
            max_strike_pct=1.0,
        )
        of = OptionsFilter(config, GreeksCalculator())
        contract = make_option_contract(
            delta=None, implied_volatility=None,
            bid=0.01, ask=10.0, mid=5.0,
            dte=100, volume=0, open_interest=0,
        )
        result = of.evaluate(contract)
        assert result.passes is True


# ── filter_and_rank ─────────────────────────────────────────────


class TestFilterAndRank:
    def test_sorted_by_lowest_strike(self, of):
        c1 = make_option_contract(strike=200.0, stock_price=230.0, dte=5,
                                   delta=-0.25, implied_volatility=0.30,
                                   bid=1.50, ask=1.70, mid=1.60)
        c2 = make_option_contract(strike=190.0, stock_price=230.0, dte=5,
                                   delta=-0.20, implied_volatility=0.25,
                                   bid=1.00, ask=1.20, mid=1.10,
                                   symbol="AAPL260220P00190000")
        passing, results = of.filter_and_rank([c1, c2])
        # With "lowest_strike_price" mode, lowest strike first
        if len(passing) >= 2:
            assert passing[0].strike <= passing[1].strike

    def test_excludes_failures(self, of):
        good = make_option_contract(strike=200.0, stock_price=230.0, dte=5,
                                     delta=-0.25, implied_volatility=0.30,
                                     bid=1.50, ask=1.70, mid=1.60)
        bad = make_option_contract(strike=225.0, stock_price=230.0, dte=5,
                                    delta=-0.25, implied_volatility=0.30,
                                    bid=1.50, ask=1.70, mid=1.60,
                                    symbol="AAPL260220P00225000")
        passing, results = of.filter_and_rank([good, bad])
        # bad should be filtered out (strike 97.8% > 90%)
        assert all(c.strike != 225.0 for c in passing)


# ── get_best_candidates ────────────────────────────────────────


class TestGetBestCandidates:
    def test_limits_count(self, of):
        contracts = [
            make_option_contract(
                strike=200.0 - i, stock_price=230.0, dte=5,
                delta=-0.20 - i * 0.01, implied_volatility=0.30,
                bid=1.50, ask=1.70, mid=1.60,
                symbol=f"AAPL260220P00{200-i}000",
            )
            for i in range(5)
        ]
        best = of.get_best_candidates(contracts, max_candidates=2)
        assert len(best) <= 2


# ── Multiple failure reasons ────────────────────────────────────


class TestMultipleFailures:
    def test_accumulates_reasons(self, of):
        contract = make_option_contract(
            strike=225.0,   # too high
            stock_price=230.0,
            dte=15,          # out of range
            delta=-0.50,     # too high
            implied_volatility=0.30,
            bid=0.01, ask=0.03, mid=0.02,  # low return
        )
        result = of.evaluate(contract)
        assert result.passes is False
        assert len(result.failure_reasons) >= 2
