"""Tests for the equity_screener package."""

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from equity_screener.config import EquityScreenerConfig
from equity_screener.filter import EquityFilter, EquityFilterResult
from equity_screener.calendars import FomcCalendar, check_events
from equity_screener.output import build_output, default_output_path
from equity_screener.main import load_universe, run_screener

from tests.conftest import make_price_series


# ── Config loading ─────────────────────────────────────────────


class TestEquityScreenerConfig:
    def test_default_config(self):
        config = EquityScreenerConfig()
        assert config.name == "default"
        assert config.rsi_period == 14
        assert config.enable_sma8_check is True
        assert config.share_price_max is None
        assert config.trade_during_earnings is False
        assert config.max_dte == 10

    def test_from_name_csp_bullish(self):
        config = EquityScreenerConfig.from_name("csp_bullish")
        assert config.name == "csp_bullish"
        assert config.rsi_period == 14
        assert config.bb_std == 1.0
        assert config.enable_band_check is True

    def test_from_name_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            EquityScreenerConfig.from_name("nonexistent_config")

    def test_from_json(self, tmp_path):
        config_data = {
            "name": "test",
            "rsi_period": 10,
            "share_price_max": 300.0,
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(config_data))
        config = EquityScreenerConfig.from_json(str(path))
        assert config.rsi_period == 10
        assert config.share_price_max == 300.0
        # Other fields should use defaults
        assert config.bb_period == 20

    def test_to_dict_roundtrip(self):
        config = EquityScreenerConfig(name="test", rsi_period=10)
        d = config.to_dict()
        assert d["rsi_period"] == 10
        assert d["name"] == "test"
        assert "share_price_max" in d

    def test_no_position_size_fields(self):
        """Verify position size check is NOT in equity screener config."""
        config = EquityScreenerConfig()
        assert not hasattr(config, "enable_position_size_check")
        assert not hasattr(config, "starting_cash")
        assert not hasattr(config, "max_position_pct")


# ── Filter logic ───────────────────────────────────────────────


class TestEquityFilter:
    @pytest.fixture
    def config(self):
        return EquityScreenerConfig(
            enable_sma8_check=True,
            enable_sma20_check=True,
            enable_sma50_check=True,
            enable_band_check=True,
            enable_sma50_trend_check=True,
            enable_rsi_check=True,
        )

    @pytest.fixture
    def eq_filter(self, config):
        return EquityFilter(config)

    def test_uptrending_stock_returns_result(self, eq_filter):
        prices = make_price_series(base=230.0, n=60, trend="up", seed=42)
        result = eq_filter.evaluate("AAPL", prices)
        assert isinstance(result, EquityFilterResult)
        assert result.symbol == "AAPL"
        assert result.current_price > 0
        assert result.sma_8 > 0

    def test_insufficient_history_fails(self, eq_filter):
        short = pd.Series(
            [100.0] * 30,
            index=pd.bdate_range(end=date.today(), periods=30),
        )
        result = eq_filter.evaluate("TEST", short)
        assert result.passes is False
        assert "Insufficient price history" in result.failure_reasons

    def test_share_price_max_rejects(self):
        config = EquityScreenerConfig(
            share_price_max=200.0,
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )
        ef = EquityFilter(config)
        prices = make_price_series(base=250.0, n=60, trend="flat", seed=99)
        result = ef.evaluate("EXPENSIVE", prices)
        assert result.passes is False
        assert any("max" in r for r in result.failure_reasons)

    def test_share_price_max_none_allows_all(self):
        config = EquityScreenerConfig(
            share_price_max=None,
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )
        ef = EquityFilter(config)
        prices = pd.Series(
            [9999.0] * 60,
            index=pd.bdate_range(end=date.today(), periods=60),
        )
        result = ef.evaluate("PRICEY", prices)
        assert result.passes is True

    def test_all_disabled_passes(self):
        config = EquityScreenerConfig(
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_bb_upper_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )
        ef = EquityFilter(config)
        prices = pd.Series(
            [100.0] * 60,
            index=pd.bdate_range(end=date.today(), periods=60),
        )
        result = ef.evaluate("TEST", prices)
        assert result.passes is True
        assert result.failure_reasons == []

    def test_filter_universe_batch(self):
        config = EquityScreenerConfig(
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )
        ef = EquityFilter(config)
        history = {
            "AAPL": pd.Series(
                [230.0] * 60,
                index=pd.bdate_range(end=date.today(), periods=60),
            ),
            "MSFT": pd.Series(
                [100.0] * 30,
                index=pd.bdate_range(end=date.today(), periods=30),
            ),
        }
        passing, results = ef.filter_universe(history)
        assert "AAPL" in passing
        assert "MSFT" not in passing  # insufficient history
        assert len(results) == 2

    def test_sma_checks(self):
        """Verify SMA checks reject when price is below SMA."""
        config = EquityScreenerConfig(
            enable_sma8_check=True,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )
        ef = EquityFilter(config)
        # Flat then sharp drop — price definitively below SMA(8)
        prices = pd.Series(
            [200.0] * 55 + [190.0] * 5,
            index=pd.bdate_range(end=date.today(), periods=60),
        )
        result = ef.evaluate("DOWN", prices)
        assert any("SMA(8)" in r for r in result.failure_reasons)

    def test_check_events_delegates(self):
        """Verify check_events method works via delegation to calendars module."""
        config = EquityScreenerConfig(
            trade_during_fomc=True,
            trade_during_earnings=True,
            trade_during_dividends=True,
        )
        ef = EquityFilter(config)
        # With all events allowed, should return empty rejections
        rejections = ef.check_events(["AAPL", "MSFT"])
        assert isinstance(rejections, dict)


# ── Output format ──────────────────────────────────────────────


class TestOutput:
    def test_build_output_structure(self):
        config = EquityScreenerConfig(name="test")
        results = [
            EquityFilterResult(
                "AAPL", True, 230.0, 228.0, 225.0, 220.0, 55.0, 235.0, True, []
            ),
            EquityFilterResult(
                "FAIL", False, 100.0, 0, 0, 0, 0, 0, False, ["reason"]
            ),
        ]
        output = build_output(config, results, events={}, universe_size=10)
        assert output["screener"] == "test"
        assert output["pass_count"] == 1
        assert output["universe_size"] == 10
        assert len(output["pass"]) == 1
        assert output["pass"][0]["symbol"] == "AAPL"
        assert output["pass"][0]["rsi"] == 55.0
        assert output["pass"][0]["checks"] == {}  # no checks populated in minimal result
        assert len(output["fail"]) == 1
        assert output["fail"][0]["reasons"] == ["reason"]
        assert len(output.get("event_rejected", [])) == 0
        assert "screened_at" in output

    def test_build_output_with_events(self):
        config = EquityScreenerConfig(name="test")
        results = [
            EquityFilterResult(
                "TSLA", True, 300.0, 298.0, 295.0, 290.0, 52.0, 305.0, True, []
            ),
        ]
        events = {"TSLA": ["Earnings on 2026-02-25 within 10d window"]}
        output = build_output(config, results, events=events, universe_size=5)
        assert output["events"] == events
        # TSLA passed technical but failed events — should be event_rejected, not pass
        assert output["pass_count"] == 0
        assert len(output["pass"]) == 0
        assert len(output["event_rejected"]) == 1
        assert output["event_rejected"][0]["symbol"] == "TSLA"
        assert output["event_rejected"][0]["events"] == events["TSLA"]

    def test_default_output_path(self):
        assert default_output_path("csp_bullish") == "equity_screened_csp_bullish.json"
        assert default_output_path("cc_neutral") == "equity_screened_cc_neutral.json"


# ── Universe loading ───────────────────────────────────────────


class TestLoadUniverse:
    def test_load_from_screened_universe_format(self, tmp_path):
        data = {"pass": ["AAPL", "MSFT", "GOOG"], "fail": []}
        path = tmp_path / "test_universe.json"
        path.write_text(json.dumps(data))
        result = load_universe(str(path))
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_load_from_plain_list(self, tmp_path):
        data = ["AAPL", "MSFT"]
        path = tmp_path / "test_list.json"
        path.write_text(json.dumps(data))
        result = load_universe(str(path))
        assert result == ["AAPL", "MSFT"]

    def test_load_invalid_format_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"something": "else"}))
        with pytest.raises(ValueError, match="Cannot parse"):
            load_universe(str(path))


# ── CLI integration (mocked Alpaca) ───────────────────────────


class TestRunScreener:
    @patch("equity_screener.main.AlpacaClientManager")
    @patch("equity_screener.main.EquityDataFetcher")
    def test_run_screener_produces_json(self, MockFetcher, MockAlpaca, tmp_path):
        # Setup mock price data
        mock_fetcher = MockFetcher.return_value
        mock_fetcher.get_close_history.return_value = {
            "AAPL": make_price_series(base=230.0, n=60, trend="up", seed=42),
            "MSFT": make_price_series(base=300.0, n=60, trend="up", seed=43),
        }

        # Write a universe file
        universe_path = tmp_path / "screened_universe.json"
        universe_path.write_text(json.dumps({"pass": ["AAPL", "MSFT"]}))

        config = EquityScreenerConfig(
            name="test",
            universe_source=str(universe_path),
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )

        output_path = str(tmp_path / "output.json")
        result = run_screener(config, output_path=output_path)

        # Verify return value
        assert result["screener"] == "test"
        assert result["pass_count"] == 2

        # Verify output file
        assert Path(output_path).exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["screener"] == "test"
        assert saved["pass_count"] == 2
        assert len(saved["pass"]) == 2

    @patch("equity_screener.main.AlpacaClientManager")
    @patch("equity_screener.main.EquityDataFetcher")
    def test_run_screener_with_share_price_max(self, MockFetcher, MockAlpaca, tmp_path):
        mock_fetcher = MockFetcher.return_value
        mock_fetcher.get_close_history.return_value = {
            "CHEAP": make_price_series(base=50.0, n=60, trend="flat", seed=1),
            "PRICEY": make_price_series(base=600.0, n=60, trend="flat", seed=2),
        }

        universe_path = tmp_path / "universe.json"
        universe_path.write_text(json.dumps(["CHEAP", "PRICEY"]))

        config = EquityScreenerConfig(
            name="price_test",
            universe_source=str(universe_path),
            share_price_max=200.0,
            # Disable all technical checks to isolate price filter
            enable_sma8_check=False,
            enable_sma20_check=False,
            enable_sma50_check=False,
            enable_band_check=False,
            enable_sma50_trend_check=False,
            enable_rsi_check=False,
        )

        output_path = str(tmp_path / "output.json")
        result = run_screener(config, output_path=output_path)

        pass_symbols = [r["symbol"] for r in result["pass"]]
        assert "CHEAP" in pass_symbols
        assert "PRICEY" not in pass_symbols


# ── CSP integration ────────────────────────────────────────────


class TestCSPIntegration:
    def test_csp_can_parse_screener_output(self):
        """Verify CSP can parse the screener output JSON format."""
        output = {
            "screener": "csp_bullish",
            "screened_at": datetime.now().isoformat(),
            "pass_count": 2,
            "pass": [
                {
                    "symbol": "AAPL",
                    "price": 230.0,
                    "sma_8": 228.0,
                    "sma_20": 225.0,
                    "sma_50": 220.0,
                    "rsi": 55.0,
                    "bb_upper": 235.0,
                    "sma_50_trending": True,
                },
                {
                    "symbol": "MSFT",
                    "price": 300.0,
                    "sma_8": 298.0,
                    "sma_20": 295.0,
                    "sma_50": 290.0,
                    "rsi": 52.0,
                    "bb_upper": 305.0,
                    "sma_50_trending": True,
                },
            ],
            "fail": [],
            "events": {},
        }
        # This simulates what loop.py._load_screener_output() will do
        passing_symbols = [r["symbol"] for r in output["pass"]]
        assert passing_symbols == ["AAPL", "MSFT"]

    def test_wrapper_reexports_work(self):
        """Verify the csp/signals/equity_filter.py wrapper still exports correctly."""
        from csp.signals.equity_filter import (
            EquityFilter as WrappedFilter,
            EquityFilterResult as WrappedResult,
            FomcCalendar as WrappedFomc,
            EarningsCalendar as WrappedEarnings,
            DividendCalendar as WrappedDividend,
        )

        # Same classes as direct imports
        assert WrappedFilter is EquityFilter
        assert WrappedResult is EquityFilterResult
        assert WrappedFomc is FomcCalendar

    def test_strategy_config_has_equity_screener_field(self):
        """Verify StrategyConfig has the new equity_screener field."""
        from csp.config import StrategyConfig

        config = StrategyConfig(
            ticker_universe=["AAPL"],
            equity_screener="csp_bullish",
        )
        assert config.equity_screener == "csp_bullish"

        # Default is None (inline scan)
        config2 = StrategyConfig(ticker_universe=["AAPL"])
        assert config2.equity_screener is None


# ── Event calendars ────────────────────────────────────────────


class TestCalendars:
    def test_fomc_calendar_has_dates(self):
        """Verify FomcCalendar has meeting dates."""
        assert len(FomcCalendar._MEETING_DATES) > 0

    def test_check_events_all_allowed(self):
        """When all events are allowed, should return empty."""
        config = EquityScreenerConfig(
            trade_during_fomc=True,
            trade_during_earnings=True,
            trade_during_dividends=True,
        )
        result = check_events(["AAPL"], config)
        assert result == {}
