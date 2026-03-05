"""Tests for csp.trading.utils — shared trading utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from csp.trading.utils import (
    _arun,
    is_option_symbol,
    update_candidate_from_snapshot,
    build_execution_components,
)


class TestIsOptionSymbol:
    def test_stock_tickers_return_false(self):
        assert not is_option_symbol("AAPL")
        assert not is_option_symbol("MSFT")
        assert not is_option_symbol("TSLA")
        assert not is_option_symbol("BRK.B")

    def test_option_symbols_return_true(self):
        assert is_option_symbol("AAPL260320P00220000")
        assert is_option_symbol("MSFT260320C00400000")
        assert is_option_symbol("TSLA260117P00150000")

    def test_short_digit_string_returns_false(self):
        # Short symbols with digits are not options
        assert not is_option_symbol("A1234")
        assert not is_option_symbol("X2Y")

    def test_empty_string_returns_false(self):
        assert not is_option_symbol("")

    def test_long_no_digits_returns_false(self):
        # Long but no digits — not OCC format
        assert not is_option_symbol("ABCDEFGHIJKL")


class TestUpdateCandidateFromSnapshot:
    def test_updates_bid_ask_mid(self):
        candidate = SimpleNamespace(bid=0, ask=0, mid=0)
        snap = {"bid": 1.50, "ask": 2.00}

        update_candidate_from_snapshot(candidate, snap)

        assert candidate.bid == 1.50
        assert candidate.ask == 2.00
        assert candidate.mid == 1.75

    def test_updates_greek_attributes(self):
        candidate = SimpleNamespace(
            bid=0, ask=0, mid=0,
            delta=0, implied_volatility=0, volume=0, open_interest=0,
        )
        snap = {
            "bid": 1.0, "ask": 2.0,
            "delta": -0.30,
            "implied_volatility": 0.45,
            "volume": 150,
            "open_interest": 5000,
        }

        update_candidate_from_snapshot(candidate, snap)

        assert candidate.delta == -0.30
        assert candidate.implied_volatility == 0.45
        assert candidate.volume == 150
        assert candidate.open_interest == 5000

    def test_none_values_not_overwritten(self):
        candidate = SimpleNamespace(
            bid=0, ask=0, mid=0,
            delta=-0.25, implied_volatility=0.40,
        )
        snap = {"bid": 1.0, "ask": 2.0, "delta": None, "implied_volatility": None}

        update_candidate_from_snapshot(candidate, snap)

        # delta and IV should stay unchanged
        assert candidate.delta == -0.25
        assert candidate.implied_volatility == 0.40

    def test_missing_keys_use_zero(self):
        candidate = SimpleNamespace(bid=5.0, ask=6.0, mid=5.5)
        snap = {}  # empty snapshot

        update_candidate_from_snapshot(candidate, snap)

        assert candidate.bid == 0.0
        assert candidate.ask == 0.0
        assert candidate.mid == 0.0

    def test_handles_none_bid_ask(self):
        candidate = SimpleNamespace(bid=0, ask=0, mid=0)
        snap = {"bid": None, "ask": None}

        update_candidate_from_snapshot(candidate, snap)

        assert candidate.bid == 0.0
        assert candidate.ask == 0.0
        assert candidate.mid == 0.0


class TestBuildExecutionComponents:
    def test_returns_stepped_and_router(self):
        execution = MagicMock()
        config = MagicMock()
        config.entry_order_type = "stepped_limit"
        config.entry_start_price = "mid"
        config.entry_step_pct = 0.10
        config.entry_max_steps = 3
        config.entry_step_interval = 5
        config.exit_step_pct = 0.10
        config.exit_max_steps = 3
        config.exit_step_interval = 5

        stepped, router = build_execution_components(
            execution=execution,
            config=config,
            snapshot_fetcher=MagicMock(),
            vprint=lambda msg: None,
        )

        from csp.trading.stepped_executor import SteppedOrderExecutor
        from csp.trading.exit_router import ExitRouter

        assert isinstance(stepped, SteppedOrderExecutor)
        assert isinstance(router, ExitRouter)

    def test_router_uses_stepped_executor(self):
        execution = MagicMock()
        config = MagicMock()
        config.entry_order_type = "stepped_limit"
        config.entry_start_price = "mid"
        config.entry_step_pct = 0.10
        config.entry_max_steps = 3
        config.entry_step_interval = 5
        config.exit_step_pct = 0.10
        config.exit_max_steps = 3
        config.exit_step_interval = 5

        stepped, router = build_execution_components(
            execution=execution,
            config=config,
            snapshot_fetcher=MagicMock(),
            vprint=lambda msg: None,
        )

        assert router.stepped_executor is stepped


class TestArun:
    async def test_runs_sync_function(self):
        def add(a, b):
            return a + b

        result = await _arun(add, 2, 3)
        assert result == 5

    async def test_passes_kwargs(self):
        def greet(name="world"):
            return f"hello {name}"

        result = await _arun(greet, name="test")
        assert result == "hello test"
