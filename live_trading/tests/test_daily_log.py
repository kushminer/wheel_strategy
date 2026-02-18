"""Unit tests for DailyLog."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from csp.config import StrategyConfig
from csp.trading.daily_log import DailyLog


@pytest.fixture
def log(tmp_path):
    """Fresh DailyLog in temp directory."""
    return DailyLog(log_dir=str(tmp_path))


@pytest.fixture
def config():
    return StrategyConfig(ticker_universe=["AAPL"])


class TestLogConfig:
    def test_creates_snapshot(self, log, config):
        log.log_config(config)
        log.flush()
        data = json.loads(open(log.today_path).read())
        assert data["config_snapshot"]["starting_cash"] == config.starting_cash
        assert data["config_snapshot"]["paper_trading"] == config.paper_trading
        assert data["config_snapshot"]["delta_stop_multiplier"] == config.delta_stop_multiplier
        assert data["config_snapshot"]["contract_rank_mode"] == config.contract_rank_mode
        assert data["config_snapshot"]["universe_rank_mode"] == config.universe_rank_mode


class TestLogEquityScan:
    def test_stores_results(self, log):
        scan_result = SimpleNamespace(
            symbol="AAPL", passes=True,
            current_price=230.0, sma_8=228.0, sma_20=225.0, sma_50=220.0, rsi=55.0,
        )
        log.log_equity_scan([scan_result], ["AAPL"])
        log.flush()
        data = json.loads(open(log.today_path).read())
        assert data["equity_scan"]["scanned"] == 1
        assert "AAPL" in data["equity_scan"]["passed"]
        assert data["equity_scan"]["results"]["AAPL"]["price"] == 230.0


class TestLogOptionsScan:
    def test_appends_multiple(self, log):
        contract = SimpleNamespace(
            symbol="AAPL260220P00220000", strike=220.0, dte=5,
            bid=1.50, ask=1.70, implied_volatility=0.30,
        )
        filter_result = SimpleNamespace(
            contract=contract, delta_abs=0.25, daily_return=0.0015,
            passes=True, failure_reasons=[],
        )
        log.log_options_scan(1, "AAPL", [filter_result])
        log.log_options_scan(2, "MSFT", [filter_result])
        log.flush()
        data = json.loads(open(log.today_path).read())
        assert len(data["options_scans"]) == 2
        assert data["options_scans"][0]["symbol"] == "AAPL"
        assert data["options_scans"][1]["symbol"] == "MSFT"


class TestLogCycle:
    def test_appends_multiple(self, log):
        summary = {"current_vix": 18.5, "deployable_cash": 80000, "entries": 1, "exits": 0}
        log.log_cycle(1, summary)
        log.log_cycle(2, summary)
        log.flush()
        data = json.loads(open(log.today_path).read())
        assert len(data["cycles"]) == 2
        assert data["cycles"][0]["cycle"] == 1
        assert data["cycles"][1]["cycle"] == 2


class TestLogOrderAttempt:
    def test_stores_steps(self, log):
        steps = [
            {"step": 1, "limit_price": 1.60, "status": "cancelled"},
            {"step": 2, "limit_price": 1.50, "status": "filled"},
        ]
        log.log_order_attempt(
            action="entry", symbol="AAPL", contract="AAPL260220P00220000",
            steps=steps, outcome="filled", filled_price=1.50,
        )
        log.flush()
        data = json.loads(open(log.today_path).read())
        assert len(data["order_attempts"]) == 1
        attempt = data["order_attempts"][0]
        assert attempt["action"] == "entry"
        assert attempt["total_steps"] == 2
        assert attempt["outcome"] == "filled"
        assert attempt["filled_price"] == 1.50


class TestLogShutdown:
    def test_records_summary(self, log):
        log.log_shutdown("market_closed", 5, {"active_positions": 2, "total_collateral": 44000})
        data = json.loads(open(log.today_path).read())
        assert data["shutdown"]["reason"] == "market_closed"
        assert data["shutdown"]["total_cycles"] == 5
        assert data["shutdown"]["final_positions"] == 2


class TestTodayPath:
    def test_format_matches(self, log):
        expected = f"{date.today().isoformat()}.json"
        assert log.today_path.endswith(expected)


class TestExistingLogLoaded:
    def test_loads_existing(self, tmp_path):
        path = tmp_path / f"{date.today().isoformat()}.json"
        existing = {"date": date.today().isoformat(), "config_snapshot": {"test": True},
                     "equity_scan": {}, "options_scans": [], "cycles": [{"cycle": 0}],
                     "order_attempts": [], "shutdown": {}}
        with open(path, "w") as f:
            json.dump(existing, f)

        log = DailyLog(log_dir=str(tmp_path))
        log.log_cycle(1, {"current_vix": 18})
        log.flush()
        data = json.loads(open(log.today_path).read())
        # Should have original cycle 0 + new cycle 1
        assert len(data["cycles"]) == 2
        assert data["config_snapshot"]["test"] is True


class TestIdempotency:
    def test_multiple_calls_safe(self, log, config):
        log.log_config(config)
        log.log_config(config)
        log.flush()
        data = json.loads(open(log.today_path).read())
        # Config snapshot is overwritten, not duplicated
        assert isinstance(data["config_snapshot"], dict)
