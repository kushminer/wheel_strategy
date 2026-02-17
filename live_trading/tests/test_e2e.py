"""End-to-end smoke tests — full pipeline through mocked APIs.

These tests verify the complete flow from config → scan → filter → risk → exit
using mocked external services (Alpaca, yfinance). They validate that
all extracted modules compose correctly.
"""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.signals.scanner import StrategyScanner
from csp.signals.options_filter import OptionsFilter
from csp.trading.risk import RiskManager
from csp.trading.metadata import StrategyMetadataStore
from csp.trading.daily_log import DailyLog
from csp.trading.execution import ExecutionEngine
from csp.trading.models import ExitReason

from tests.conftest import (
    make_option_contract,
    make_price_series,
    make_position_proxy,
    make_alpaca_order,
)


@pytest.fixture
def config():
    return StrategyConfig(
        ticker_universe=["AAPL", "MSFT"],
        starting_cash=100_000,
        # Disable equity checks for simplicity
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
        # Risk
        enable_delta_stop=True,
        enable_delta_absolute_stop=True,
        enable_stock_drop_stop=False,
        enable_vix_spike_stop=True,
        enable_early_exit=True,
    )


@pytest.fixture
def greeks_calc():
    return GreeksCalculator()


@pytest.fixture
def risk(config):
    return RiskManager(config)


@pytest.fixture
def metadata(tmp_path):
    return StrategyMetadataStore(path=str(tmp_path / "metadata.json"))


@pytest.fixture
def daily_log(tmp_path):
    return DailyLog(log_dir=str(tmp_path / "logs"))


# ── E2E: Entry then Exit ───────────────────────────────────────


class TestFullCycleEntryThenExit:
    """Simulate: VIX check → scan → find candidate → enter → delta stop → exit → verify metadata."""

    def test_entry_then_delta_stop_exit(self, config, greeks_calc, risk, metadata, daily_log):
        # 1. VIX check: VIX=18 → multiplier=0.9 → deployable=$90,000
        vix = 18.0
        deployable = config.get_deployable_cash(vix)
        assert deployable == 90_000

        # 2. Scan: mock fetchers return candidate
        good_put = make_option_contract(
            symbol="AAPL260220P00200000",
            underlying="AAPL",
            strike=200.0,
            stock_price=230.0,
            bid=1.50, ask=1.70, mid=1.60,
            dte=5,
            delta=-0.25,
            implied_volatility=0.30,
        )

        mock_eq_fetcher = MagicMock()
        mock_eq_fetcher.get_close_history.return_value = {
            "AAPL": make_price_series(base=230.0, n=60, trend="up"),
        }
        mock_opt_fetcher = MagicMock()
        mock_opt_fetcher.get_puts_chain.return_value = [good_put]

        scanner = StrategyScanner(config, mock_eq_fetcher, mock_opt_fetcher, greeks_calc)
        result = scanner.scan_symbol("AAPL", mock_eq_fetcher.get_close_history()["AAPL"])
        assert result.has_candidates is True
        candidate = result.options_candidates[0]

        # 3. Enter: mock execution
        mock_alpaca = SimpleNamespace(trading_client=MagicMock(), paper=True)
        mock_order = make_alpaca_order(order_id="entry-001", status="filled", filled_avg_price=1.50)
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        engine = ExecutionEngine(mock_alpaca, config)
        entry_result = engine.sell_to_open(candidate.symbol, quantity=1, limit_price=1.60)
        assert entry_result.success is True

        # 4. Record metadata
        metadata.record_entry(
            candidate.symbol,
            underlying="AAPL",
            strike=candidate.strike,
            expiration=candidate.expiration.isoformat(),
            entry_delta=-0.25,
            entry_iv=0.30,
            entry_vix=vix,
            entry_stock_price=230.0,
            entry_premium=1.50,
            entry_daily_return=candidate.daily_return_on_collateral,
            dte_at_entry=5,
            quantity=1,
            entry_order_id="entry-001",
        )
        assert metadata.has_symbol("AAPL") is True

        # 5. Risk check: delta absolute stop triggers (delta now 0.45)
        position = make_position_proxy(
            entry_delta=-0.25,
            entry_stock_price=230.0,
            entry_vix=vix,
            entry_premium=1.50,
            entry_daily_return=candidate.daily_return_on_collateral,
            strike=200.0,
            days_held=2,
        )
        risk_result = risk.evaluate_position(
            position,
            current_delta=-0.45,
            current_stock_price=225.0,
            current_vix=19.0,
            current_premium=2.50,
        )
        assert risk_result.should_exit is True
        assert risk_result.exit_reason == ExitReason.DELTA_ABSOLUTE

        # 6. Exit: buy to close
        exit_order = make_alpaca_order(order_id="exit-001", side="buy", status="filled", filled_avg_price=2.50)
        mock_alpaca.trading_client.submit_order.return_value = exit_order

        exit_result = engine.buy_to_close(candidate.symbol, quantity=1, limit_price=2.50)
        assert exit_result.success is True

        # 7. Record exit metadata
        metadata.record_exit(
            candidate.symbol,
            exit_reason=ExitReason.DELTA_ABSOLUTE.value,
            exit_details="Delta 0.450 >= 0.400 (absolute cap)",
            exit_order_id="exit-001",
        )
        assert metadata.has_symbol("AAPL") is False
        meta = metadata.get(candidate.symbol)
        assert meta["exit_reason"] == "delta_exceeded_absolute"


# ── E2E: No Candidates ─────────────────────────────────────────


class TestNoCandidatesMonitorOnly:
    """Scan returns no candidates → monitor mode only."""

    def test_no_candidates_returns_empty(self, config, greeks_calc):
        mock_eq = MagicMock()
        mock_eq.get_close_history.return_value = {
            "AAPL": make_price_series(base=230.0, n=60, trend="up"),
        }
        mock_opt = MagicMock()
        mock_opt.get_puts_chain.return_value = []  # no puts available

        scanner = StrategyScanner(config, mock_eq, mock_opt, greeks_calc)
        results = scanner.scan_universe()
        all_candidates = []
        for r in results:
            all_candidates.extend(r.options_candidates)
        assert len(all_candidates) == 0


# ── E2E: API Failure ────────────────────────────────────────────


class TestAPIFailureMidCycle:
    """API failure during order → no crash, failure recorded."""

    def test_order_failure_no_crash(self, config):
        mock_alpaca = SimpleNamespace(trading_client=MagicMock(), paper=True)
        mock_alpaca.trading_client.submit_order.side_effect = Exception("Connection timeout")

        engine = ExecutionEngine(mock_alpaca, config)
        result = engine.sell_to_open("AAPL260220P00220000", quantity=1)
        assert result.success is False
        assert "Connection timeout" in result.message


# ── E2E: VIX Regime Change ─────────────────────────────────────


class TestVixRegimeChange:
    """Low VIX → no deployment → no new entries."""

    def test_low_vix_zero_deployment(self, config):
        vix = 10.0  # Below 12 → multiplier = 0
        deployable = config.get_deployable_cash(vix)
        assert deployable == 0.0

    def test_high_vix_full_deployment(self, config):
        vix = 25.0  # Above 21 → multiplier = 1.0
        deployable = config.get_deployable_cash(vix)
        assert deployable == 100_000


# ── E2E: Daily Log Integration ─────────────────────────────────


class TestDailyLogIntegration:
    """Log captures config, scan, and shutdown in one file."""

    def test_full_log_cycle(self, config, daily_log):
        daily_log.log_config(config)
        daily_log.log_cycle(1, {
            "current_vix": 18.0,
            "deployable_cash": 90000,
            "entries": 1,
            "exits": 0,
            "portfolio": {"active_positions": 1, "total_collateral": 22000},
        })
        daily_log.log_shutdown("market_closed", 1, {
            "active_positions": 1,
            "total_collateral": 22000,
        })

        import json
        data = json.loads(open(daily_log.today_path).read())
        assert data["config_snapshot"]["starting_cash"] == 100_000
        assert len(data["cycles"]) == 1
        assert data["shutdown"]["reason"] == "market_closed"
