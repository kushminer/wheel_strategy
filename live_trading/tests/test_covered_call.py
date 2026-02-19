"""Unit tests for the covered call module (config, store, manager)."""

import json
import os
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covered_call.config import CoveredCallConfig
from covered_call.store import WheelPositionStore
from covered_call.manager import CoveredCallManager
from csp.config import StrategyConfig
from csp.data.options import OptionContract
from csp.trading.models import ExitReason, OrderResult

from tests.conftest import make_option_contract, make_alpaca_order


# ── Config Tests ─────────────────────────────────────────────────


class TestCoveredCallConfig:
    def test_defaults(self):
        cfg = CoveredCallConfig()
        assert cfg.enabled is False
        assert cfg.cc_min_dte == 1
        assert cfg.cc_max_dte == 6
        assert cfg.cc_strike_mode == "delta"
        assert cfg.cc_strike_delta == 0.30
        assert cfg.cc_strike_pct == 0.02
        assert cfg.cc_min_daily_return_pct == 0.0015
        assert cfg.cc_exit_mode == "strike_recovery"

    def test_custom_values(self):
        cfg = CoveredCallConfig(
            enabled=True,
            cc_strike_mode="min_daily_return",
            cc_min_daily_return_pct=0.002,
            cc_exit_mode="premium_recovery",
        )
        assert cfg.enabled is True
        assert cfg.cc_strike_mode == "min_daily_return"
        assert cfg.cc_min_daily_return_pct == 0.002
        assert cfg.cc_exit_mode == "premium_recovery"


# ── Store Tests ──────────────────────────────────────────────────


@pytest.fixture
def wheel_store(tmp_path):
    return WheelPositionStore(path=str(tmp_path / "wheel.json"))


class TestWheelPositionStore:
    def test_add_and_get(self, wheel_store):
        wheel_store.add_position(
            underlying="AAPL",
            shares=100,
            cost_basis=220.0,
            source="csp_assignment",
            csp_entry_premium=1.50,
        )
        pos = wheel_store.get("AAPL")
        assert pos is not None
        assert pos["shares"] == 100
        assert pos["cost_basis"] == 220.0
        assert pos["source"] == "csp_assignment"
        assert pos["status"] == "awaiting_cc_entry"
        assert pos["total_cc_premiums"] == 0.0
        assert pos["cc_rounds"] == 0

    def test_get_active_excludes_terminated(self, wheel_store):
        wheel_store.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")
        wheel_store.add_position("MSFT", shares=200, cost_basis=400.0, source="alpaca_position")
        wheel_store.terminate("AAPL", reason="shares_sold")

        active = wheel_store.get_active()
        assert "AAPL" not in active
        assert "MSFT" in active

    def test_record_cc_entry(self, wheel_store):
        wheel_store.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")
        wheel_store.record_cc_entry(
            underlying="AAPL",
            option_symbol="AAPL260227C00225000",
            strike=225.0,
            expiration="2026-02-27",
            entry_premium=0.75,
            quantity=1,
            order_id="cc-order-1",
        )
        pos = wheel_store.get("AAPL")
        assert pos["status"] == "cc_active"
        assert pos["current_cc"]["option_symbol"] == "AAPL260227C00225000"
        assert pos["current_cc"]["strike"] == 225.0

    def test_record_cc_exit(self, wheel_store):
        wheel_store.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")
        wheel_store.record_cc_entry(
            underlying="AAPL",
            option_symbol="AAPL260227C00225000",
            strike=225.0,
            expiration="2026-02-27",
            entry_premium=0.75,
            quantity=1,
            order_id="cc-order-1",
        )
        wheel_store.record_cc_exit(
            underlying="AAPL",
            exit_reason="expired",
            exit_premium=0.0,
        )
        pos = wheel_store.get("AAPL")
        assert pos["status"] == "awaiting_cc_entry"
        assert pos["current_cc"] is None
        assert pos["cc_rounds"] == 1
        assert pos["total_cc_premiums"] == 0.75
        assert len(pos["cc_history"]) == 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "wheel.json")
        store1 = WheelPositionStore(path=path)
        store1.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")

        store2 = WheelPositionStore(path=path)
        assert store2.get("AAPL") is not None
        assert store2.get("AAPL")["cost_basis"] == 220.0

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = str(tmp_path / "wheel.json")
        with open(path, "w") as f:
            f.write("{bad json!!!")
        store = WheelPositionStore(path=path)
        assert len(store.positions) == 0

    def test_get_by_status(self, wheel_store):
        wheel_store.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")
        wheel_store.add_position("MSFT", shares=200, cost_basis=400.0, source="alpaca_position")
        wheel_store.record_cc_entry(
            underlying="AAPL",
            option_symbol="AAPL260227C00225000",
            strike=225.0,
            expiration="2026-02-27",
            entry_premium=0.75,
            quantity=1,
            order_id="cc-order-1",
        )

        awaiting = wheel_store.get_by_status("awaiting_cc_entry")
        assert "MSFT" in awaiting
        assert "AAPL" not in awaiting

        active = wheel_store.get_by_status("cc_active")
        assert "AAPL" in active

    def test_multiple_cc_rounds_accumulate(self, wheel_store):
        wheel_store.add_position("AAPL", shares=100, cost_basis=220.0, source="csp_assignment")

        # Round 1
        wheel_store.record_cc_entry(
            underlying="AAPL", option_symbol="CC1", strike=225.0,
            expiration="2026-02-27", entry_premium=0.75, quantity=1, order_id="o1",
        )
        wheel_store.record_cc_exit(underlying="AAPL", exit_reason="expired", exit_premium=0.0)

        # Round 2
        wheel_store.record_cc_entry(
            underlying="AAPL", option_symbol="CC2", strike=223.0,
            expiration="2026-03-06", entry_premium=0.50, quantity=1, order_id="o2",
        )
        wheel_store.record_cc_exit(underlying="AAPL", exit_reason="expired", exit_premium=0.10)

        pos = wheel_store.get("AAPL")
        assert pos["cc_rounds"] == 2
        assert pos["total_cc_premiums"] == pytest.approx(0.75 + 0.40)  # 0.75 + (0.50 - 0.10)
        assert len(pos["cc_history"]) == 2


# ── Strike Selection Tests ───────────────────────────────────────


def _make_call_contract(strike, delta, bid=0.50, ask=0.70, dte=3):
    """Helper to build a call OptionContract for selection tests."""
    return OptionContract(
        symbol=f"TEST{int(strike*100):08d}C",
        underlying="TEST",
        contract_type="call",
        strike=strike,
        expiration=date.today() + timedelta(days=dte),
        dte=dte,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        stock_price=100.0,
        delta=delta,
    )


@pytest.fixture
def cc_manager_for_selection():
    """Minimal CoveredCallManager for testing _select_contract."""
    cc_config = CoveredCallConfig(enabled=True)
    strategy_config = StrategyConfig(ticker_universe=["TEST"])
    mgr = CoveredCallManager(
        cc_config=cc_config,
        strategy_config=strategy_config,
        store=MagicMock(),
        data_manager=MagicMock(),
        execution=MagicMock(),
        risk_manager=MagicMock(),
        metadata_store=MagicMock(),
        alpaca_manager=MagicMock(),
    )
    return mgr


class TestStrikeSelection:
    def test_delta_mode(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "delta"
        mgr.cc_config.cc_strike_delta = 0.30

        calls = [
            _make_call_contract(102, 0.45),
            _make_call_contract(103, 0.32),
            _make_call_contract(105, 0.20),
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected.strike == 103  # delta 0.32 is closest to 0.30

    def test_min_delta_mode(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "min_delta"
        mgr.cc_config.cc_strike_delta = 0.30

        calls = [
            _make_call_contract(102, 0.45),
            _make_call_contract(103, 0.32),
            _make_call_contract(105, 0.20),  # below floor, excluded
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected.strike == 103  # delta 0.32 is closest to 0.30 and >= 0.30

    def test_min_delta_mode_excludes_below(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "min_delta"
        mgr.cc_config.cc_strike_delta = 0.40

        calls = [
            _make_call_contract(103, 0.32),
            _make_call_contract(105, 0.20),
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected is None  # all below 0.40

    def test_pct_change_mode(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "pct_change"
        mgr.cc_config.cc_strike_pct = 0.03  # target = 103.0

        calls = [
            _make_call_contract(101, 0.45),
            _make_call_contract(103, 0.30),
            _make_call_contract(105, 0.20),
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected.strike == 103  # closest to 103.0

    def test_min_pct_change_mode(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "min_pct_change"
        mgr.cc_config.cc_strike_pct = 0.03  # target = 103.0

        calls = [
            _make_call_contract(101, 0.45),  # below 103, excluded
            _make_call_contract(103, 0.30),
            _make_call_contract(105, 0.20),
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected.strike == 103  # closest to 103.0 and >= 103.0

    def test_min_daily_return_mode(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "min_daily_return"
        mgr.cc_config.cc_min_daily_return_pct = 0.0015

        # daily_return_on_collateral = premium_per_day / strike
        # For a contract: bid / dte / strike
        # strike=100, bid=0.50, dte=3 → 0.50 / 3 / 100 = 0.00167 (above threshold)
        # strike=105, bid=0.30, dte=3 → 0.30 / 3 / 105 = 0.000952 (below threshold)
        calls = [
            _make_call_contract(100, 0.40, bid=0.50, dte=3),   # 0.00167
            _make_call_contract(102, 0.35, bid=0.48, dte=3),   # 0.48/3/102 = 0.00157
            _make_call_contract(105, 0.20, bid=0.30, dte=3),   # 0.000952 (below)
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected.strike == 102  # closest to 0.0015 while >= 0.0015

    def test_no_valid_contracts_returns_none(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        mgr.cc_config.cc_strike_mode = "delta"
        mgr.cc_config.cc_strike_delta = 0.30

        # All contracts have None delta
        calls = [
            _make_call_contract(102, None),
            _make_call_contract(105, None),
        ]
        selected = mgr._select_contract(calls, 100.0)
        assert selected is None

    def test_empty_calls_returns_none(self, cc_manager_for_selection):
        mgr = cc_manager_for_selection
        selected = mgr._select_contract([], 100.0)
        assert selected is None


# ── Cost Basis Resolution Tests ──────────────────────────────────


class TestCostBasisResolution:
    def test_csp_assignment_uses_strike(self):
        metadata_store = MagicMock()
        metadata_store.entries = {
            "AAPL260220P00220000": {
                "underlying": "AAPL",
                "strike": 220.0,
                "entry_premium": 1.50,
                "exit_reason": "assigned",
                "exit_date": "2026-02-20T10:00:00",
            },
        }
        mgr = CoveredCallManager(
            cc_config=CoveredCallConfig(enabled=True),
            strategy_config=StrategyConfig(ticker_universe=["AAPL"]),
            store=MagicMock(),
            data_manager=MagicMock(),
            execution=MagicMock(),
            risk_manager=MagicMock(),
            metadata_store=metadata_store,
            alpaca_manager=MagicMock(),
        )

        alpaca_pos = SimpleNamespace(avg_entry_price="218.50")
        cost_basis, source, premium = mgr._resolve_cost_basis("AAPL", alpaca_pos)

        assert cost_basis == 220.0  # put strike, not avg_entry_price
        assert source == "csp_assignment"
        assert premium == 1.50

    def test_standalone_uses_avg_entry_price(self):
        metadata_store = MagicMock()
        metadata_store.entries = {}
        mgr = CoveredCallManager(
            cc_config=CoveredCallConfig(enabled=True),
            strategy_config=StrategyConfig(ticker_universe=["AAPL"]),
            store=MagicMock(),
            data_manager=MagicMock(),
            execution=MagicMock(),
            risk_manager=MagicMock(),
            metadata_store=metadata_store,
            alpaca_manager=MagicMock(),
        )

        alpaca_pos = SimpleNamespace(avg_entry_price="218.50")
        cost_basis, source, premium = mgr._resolve_cost_basis("AAPL", alpaca_pos)

        assert cost_basis == 218.50
        assert source == "alpaca_position"
        assert premium is None


# ── Config Integration Tests ─────────────────────────────────────


class TestStrategyConfigCC:
    def test_default_covered_call_config_is_none(self):
        cfg = StrategyConfig(ticker_universe=["AAPL"])
        assert cfg.covered_call_config is None

    def test_covered_call_config_set(self):
        cc = CoveredCallConfig(enabled=True, cc_strike_mode="pct_change")
        cfg = StrategyConfig(
            ticker_universe=["AAPL"],
            covered_call_config=cc,
        )
        assert cfg.covered_call_config is not None
        assert cfg.covered_call_config.enabled is True
        assert cfg.covered_call_config.cc_strike_mode == "pct_change"


# ── Options get_calls_chain Tests ────────────────────────────────


class TestGetCallsChain:
    def test_calls_chain_uses_call_type(self):
        """Verify get_calls_chain passes contract_type='call' to get_option_contracts."""
        from csp.data.options import OptionsDataFetcher

        mock_alpaca = SimpleNamespace(
            trading_client=MagicMock(),
            api_key="test",
            secret_key="test",
        )
        fetcher = OptionsDataFetcher(mock_alpaca)

        # Mock the chain to return empty
        fetcher.get_option_contracts = MagicMock(return_value=[])

        result = fetcher.get_calls_chain("AAPL", 230.0)
        assert result == []

        # Verify it was called with contract_type="call"
        call_args = fetcher.get_option_contracts.call_args
        assert call_args.kwargs.get("contract_type") == "call" or call_args[1].get("contract_type") == "call"


# ── Execution sell_stock Tests ───────────────────────────────────


class TestSellStock:
    def test_limit_sell(self):
        mock_alpaca = SimpleNamespace(
            trading_client=MagicMock(),
            paper=True,
        )
        mock_order = make_alpaca_order(order_id="stock-sell-1", status="accepted")
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        from csp.trading.execution import ExecutionEngine
        engine = ExecutionEngine(mock_alpaca, StrategyConfig(ticker_universe=["AAPL"]))

        result = engine.sell_stock("AAPL", 100, limit_price=220.0)
        assert result.success is True
        assert result.order_id == "stock-sell-1"

    def test_market_sell(self):
        mock_alpaca = SimpleNamespace(
            trading_client=MagicMock(),
            paper=True,
        )
        mock_order = make_alpaca_order(order_id="stock-sell-2", status="accepted")
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        from csp.trading.execution import ExecutionEngine
        engine = ExecutionEngine(mock_alpaca, StrategyConfig(ticker_universe=["AAPL"]))

        result = engine.sell_stock("AAPL", 100)
        assert result.success is True

    def test_sell_stock_failure(self):
        mock_alpaca = SimpleNamespace(
            trading_client=MagicMock(),
            paper=True,
        )
        mock_alpaca.trading_client.submit_order.side_effect = Exception("Insufficient shares")

        from csp.trading.execution import ExecutionEngine
        engine = ExecutionEngine(mock_alpaca, StrategyConfig(ticker_universe=["AAPL"]))

        result = engine.sell_stock("AAPL", 100)
        assert result.success is False
