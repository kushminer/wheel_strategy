"""Integration tests for StrategyScanner."""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.signals.scanner import StrategyScanner, ScanResult

from tests.conftest import make_option_contract, make_price_series


@pytest.fixture
def config():
    return StrategyConfig(
        ticker_universe=["AAPL", "MSFT"],
        # Disable most equity checks for predictable tests
        enable_sma8_check=False,
        enable_sma20_check=False,
        enable_sma50_check=False,
        enable_bb_upper_check=False,
        enable_band_check=False,
        enable_sma50_trend_check=False,
        enable_rsi_check=False,
        enable_position_size_check=False,
        # Options filter
        min_daily_return=0.0010,
        delta_min=0.0,
        delta_max=0.40,
        min_dte=1,
        max_dte=10,
        min_strike_pct=0.50,
        max_strike_pct=0.90,
        max_candidates_per_symbol=5,
        max_candidates_total=10,
    )


@pytest.fixture
def greeks_calc():
    return GreeksCalculator()


@pytest.fixture
def mock_equity_fetcher():
    fetcher = MagicMock()
    fetcher.get_close_history.return_value = {
        "AAPL": make_price_series(base=230.0, n=60, trend="up", seed=42),
        "MSFT": make_price_series(base=400.0, n=60, trend="up", seed=43),
    }
    return fetcher


@pytest.fixture
def good_puts():
    """List of option contracts that should pass filter."""
    return [
        make_option_contract(
            symbol="AAPL260220P00200000",
            underlying="AAPL",
            strike=200.0,
            stock_price=230.0,
            bid=1.50, ask=1.70, mid=1.60,
            dte=5,
            delta=-0.25,
            implied_volatility=0.30,
        ),
    ]


@pytest.fixture
def mock_options_fetcher(good_puts):
    fetcher = MagicMock()
    fetcher.get_puts_chain.return_value = good_puts
    return fetcher


@pytest.fixture
def scanner(config, mock_equity_fetcher, mock_options_fetcher, greeks_calc):
    return StrategyScanner(config, mock_equity_fetcher, mock_options_fetcher, greeks_calc)


# ── Basic scan ──────────────────────────────────────────────────


class TestScanSymbol:
    def test_equity_passes_and_options_found(self, scanner):
        prices = make_price_series(base=230.0, n=60, trend="up", seed=42)
        result = scanner.scan_symbol("AAPL", prices)
        assert isinstance(result, ScanResult)
        assert result.symbol == "AAPL"
        assert result.equity_result.passes is True
        assert result.has_candidates is True

    def test_equity_fails_returns_empty_candidates(self, config, mock_options_fetcher, greeks_calc):
        # Enable SMA50 trend check so flat prices fail
        config.enable_sma50_trend_check = True
        eq_fetcher = MagicMock()
        scanner = StrategyScanner(config, eq_fetcher, mock_options_fetcher, greeks_calc)

        flat_prices = pd.Series([100.0] * 60, index=pd.bdate_range(end=date.today(), periods=60))
        result = scanner.scan_symbol("TEST", flat_prices)
        assert result.equity_result.passes is False
        assert result.has_candidates is False
        # Options fetcher should NOT be called when equity fails
        mock_options_fetcher.get_puts_chain.assert_not_called()

    def test_skip_equity_filter_fetches_anyway(self, config, mock_options_fetcher, greeks_calc):
        config.enable_sma50_trend_check = True
        eq_fetcher = MagicMock()
        scanner = StrategyScanner(config, eq_fetcher, mock_options_fetcher, greeks_calc)

        flat_prices = pd.Series([100.0] * 60, index=pd.bdate_range(end=date.today(), periods=60))
        result = scanner.scan_symbol("TEST", flat_prices, skip_equity_filter=True)
        assert result.equity_result.passes is False
        # But options should still be fetched
        mock_options_fetcher.get_puts_chain.assert_called_once()


# ── Days since strike enrichment ────────────────────────────────


class TestDaysSinceStrike:
    def test_enriches_days_since_strike(self, scanner):
        prices = make_price_series(base=230.0, n=60, trend="up", seed=42)
        result = scanner.scan_symbol("AAPL", prices)
        if result.has_candidates:
            for c in result.options_candidates:
                assert c.days_since_strike is not None


# ── Universe scan ───────────────────────────────────────────────


class TestScanUniverse:
    def test_processes_all_symbols(self, scanner):
        results = scanner.scan_universe()
        assert len(results) == 2  # AAPL and MSFT


class TestGetAllCandidates:
    def test_respects_max_total(self, scanner):
        candidates = scanner.get_all_candidates(max_total=1)
        assert len(candidates) <= 1

    def test_collects_from_all_symbols(self, scanner, mock_options_fetcher):
        # Both symbols should have puts fetched
        candidates = scanner.get_all_candidates()
        assert mock_options_fetcher.get_puts_chain.call_count == 2
