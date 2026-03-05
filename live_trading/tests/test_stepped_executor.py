"""Tests for SteppedOrderExecutor and execution/risk config dataclasses.

Covers:
- ExecutionConfig: defaults, from_strategy_config adapter
- RiskConfig: defaults, from_strategy_config adapter
- SteppedEntry: fill step 0, fill after cancel, fill during cancel race,
  all steps exhausted, order submission failure, refetch with valid/zero-bid/missing,
  validate_fn abort, update_fn called, entry_start_price="bid", floor respected
- SteppedExit: fill step 0, fill after cancel, all exhausted, auto-fetch snapshot,
  ask=0 abort, exit_start_price="ask", ceiling respected
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

from csp.config import StrategyConfig
from csp.trading.execution_config import ExecutionConfig, RiskConfig
from csp.trading.models import OrderResult
from csp.trading.stepped_executor import SteppedOrderExecutor


# ── Helpers ────────────────────────────────────────────────────


def _make_exec_config(**overrides):
    defaults = dict(
        entry_order_type="stepped",
        entry_start_price="mid",
        entry_step_pct=0.25,
        entry_max_steps=2,
        entry_step_interval=0,  # no sleep in tests
        entry_refetch_snapshot=False,
        exit_start_price="mid",
        exit_step_pct=0.25,
        exit_max_steps=2,
        exit_step_interval=0,
        exit_refetch_snapshot=False,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _make_execution_mock(fill_on_step=0):
    """Build a mock ExecutionEngine that fills on the given step.

    Steps before fill_on_step return 'new' (unfilled).
    """
    mock = MagicMock()
    mock.sell_to_open.return_value = OrderResult(
        success=True, order_id="ord-001", message="OK",
    )
    mock.buy_to_close.return_value = OrderResult(
        success=True, order_id="ord-002", message="OK",
    )
    mock.cancel_order.return_value = True

    _call_count = {"sell": 0, "buy": 0}

    def _order_status_side_effect(order_id):
        # Count calls per order to determine which step we're on
        if order_id == "ord-001":
            _call_count["sell"] += 1
            call_num = _call_count["sell"]
        else:
            _call_count["buy"] += 1
            call_num = _call_count["buy"]

        # Each step calls get_order_status twice (once after wait, once after cancel)
        # Except the fill step which returns on the first call.
        # step 0: calls 1,2; step 1: calls 3,4; step N: calls 2N+1, 2N+2
        step = (call_num - 1) // 2

        if step >= fill_on_step:
            return {
                "status": "filled",
                "filled_avg_price": "1.45",
                "filled_qty": "1",
            }
        return {"status": "new", "filled_avg_price": None, "filled_qty": "0"}

    mock.get_order_status.side_effect = _order_status_side_effect
    return mock


# ── ExecutionConfig tests ──────────────────────────────────────


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.entry_order_type == "stepped"
        assert cfg.entry_start_price == "mid"
        assert cfg.entry_step_pct == 0.25
        assert cfg.entry_max_steps == 4
        assert cfg.entry_step_interval == 3
        assert cfg.exit_start_price == "mid"
        assert cfg.exit_max_steps == 4

    def test_from_strategy_config(self):
        sc = StrategyConfig(
            ticker_universe=["AAPL"],
            entry_start_price="bid",
            entry_max_steps=6,
            exit_start_price="ask",
        )
        ec = ExecutionConfig.from_strategy_config(sc)
        assert ec.entry_start_price == "bid"
        assert ec.entry_max_steps == 6
        assert ec.exit_start_price == "ask"


# ── RiskConfig tests ───────────────────────────────────────────


class TestRiskConfig:
    def test_defaults(self):
        rc = RiskConfig()
        assert rc.delta_stop_multiplier == 2.0
        assert rc.delta_absolute_stop == 0.40
        assert rc.enable_delta_stop is False
        assert rc.enable_early_exit is True

    def test_from_strategy_config(self):
        sc = StrategyConfig(
            ticker_universe=["AAPL"],
            delta_absolute_stop=0.50,
            enable_vix_spike_stop=False,
        )
        rc = RiskConfig.from_strategy_config(sc)
        assert rc.delta_absolute_stop == 0.50
        assert rc.enable_vix_spike_stop is False


# ── SteppedEntry tests ─────────────────────────────────────────


class TestSteppedEntry:
    async def test_fill_on_step_0(self):
        """Order fills immediately on step 0."""
        cfg = _make_exec_config(entry_max_steps=2)
        mock_exec = _make_execution_mock(fill_on_step=0)
        executor = SteppedOrderExecutor(mock_exec, cfg)

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        order_result, filled_price = result
        assert order_result.order_id == "ord-001"
        assert filled_price == 1.45
        # Should have called sell_to_open once
        assert mock_exec.sell_to_open.call_count == 1

    async def test_fill_on_step_1_after_cancel(self):
        """Step 0 is unfilled, cancelled, step 1 fills."""
        cfg = _make_exec_config(entry_max_steps=2)
        mock_exec = _make_execution_mock(fill_on_step=1)
        executor = SteppedOrderExecutor(mock_exec, cfg)

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        _, filled_price = result
        assert filled_price == 1.45
        # sell_to_open called twice (step 0 + step 1)
        assert mock_exec.sell_to_open.call_count == 2
        assert mock_exec.cancel_order.call_count >= 1

    async def test_fill_during_cancel_race(self):
        """Order fills between cancel request and re-check."""
        cfg = _make_exec_config(entry_max_steps=1)
        mock_exec = MagicMock()
        mock_exec.sell_to_open.return_value = OrderResult(
            success=True, order_id="ord-001", message="OK",
        )
        mock_exec.cancel_order.return_value = True

        # First get_order_status: 'new' (not filled)
        # Second get_order_status (after cancel): 'filled' (race condition)
        mock_exec.get_order_status.side_effect = [
            {"status": "new", "filled_avg_price": None, "filled_qty": "0"},
            {"status": "filled", "filled_avg_price": "1.55", "filled_qty": "1"},
        ]

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        _, filled_price = result
        assert filled_price == 1.55

    async def test_all_steps_exhausted(self):
        """Returns None when all steps fail to fill."""
        cfg = _make_exec_config(entry_max_steps=1)
        mock_exec = _make_execution_mock(fill_on_step=99)  # never fills
        executor = SteppedOrderExecutor(mock_exec, cfg)

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is None

    async def test_order_submission_failure(self):
        """Returns None when sell_to_open fails."""
        cfg = _make_exec_config()
        mock_exec = MagicMock()
        mock_exec.sell_to_open.return_value = OrderResult(
            success=False, order_id=None, message="Insufficient buying power",
        )

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is None

    async def test_refetch_valid_continues(self):
        """After refetch with valid data, stepping continues."""
        cfg = _make_exec_config(
            entry_max_steps=2,
            entry_refetch_snapshot=True,
        )

        mock_exec = _make_execution_mock(fill_on_step=1)

        async def mock_fetcher(symbols):
            return {
                symbols[0]: {
                    "bid": 1.48, "ask": 1.68, "mid": 1.58,
                    "delta": -0.26, "implied_volatility": 0.32,
                    "volume": 100, "open_interest": 500,
                }
            }

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)
        assert result is not None

    async def test_refetch_zero_bid_aborts(self):
        """Aborts when refetch returns bid=0."""
        cfg = _make_exec_config(
            entry_max_steps=2,
            entry_refetch_snapshot=True,
        )

        mock_exec = _make_execution_mock(fill_on_step=99)

        async def mock_fetcher(symbols):
            return {symbols[0]: {"bid": 0, "ask": 1.68}}

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)
        assert result is None

    async def test_refetch_missing_symbol_aborts(self):
        """Aborts when refetch doesn't return the symbol."""
        cfg = _make_exec_config(
            entry_max_steps=2,
            entry_refetch_snapshot=True,
        )

        mock_exec = _make_execution_mock(fill_on_step=99)

        async def mock_fetcher(symbols):
            return {}  # symbol not in result

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)
        assert result is None

    async def test_validate_fn_abort(self):
        """Aborts when validate_fn returns False after refetch."""
        cfg = _make_exec_config(
            entry_max_steps=2,
            entry_refetch_snapshot=True,
        )

        mock_exec = _make_execution_mock(fill_on_step=99)

        async def mock_fetcher(symbols):
            return {symbols[0]: {"bid": 1.48, "ask": 1.68}}

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_entry(
            "SYM", 1, bid=1.50, ask=1.70, mid=1.60,
            validate_fn=lambda snap: False,
        )
        assert result is None

    async def test_update_fn_called(self):
        """update_fn is called with the snapshot after refetch."""
        cfg = _make_exec_config(
            entry_max_steps=2,
            entry_refetch_snapshot=True,
        )

        mock_exec = _make_execution_mock(fill_on_step=1)
        updated = {}

        async def mock_fetcher(symbols):
            return {symbols[0]: {"bid": 1.48, "ask": 1.68, "delta": -0.30}}

        def update_fn(snap):
            updated["delta"] = snap["delta"]

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_entry(
            "SYM", 1, bid=1.50, ask=1.70, mid=1.60,
            update_fn=update_fn, validate_fn=lambda snap: True,
        )
        assert result is not None
        assert updated["delta"] == -0.30

    async def test_entry_start_bid(self):
        """When entry_start_price='bid', starts at bid instead of mid."""
        # Need max_steps >= 2 so the floor (mid - steps*pct*spread) drops to bid
        cfg = _make_exec_config(entry_start_price="bid", entry_max_steps=2)
        mock_exec = _make_execution_mock(fill_on_step=0)
        executor = SteppedOrderExecutor(mock_exec, cfg)

        result = await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        # sell_to_open should have been called with limit_price=1.50 (bid)
        call_kwargs = mock_exec.sell_to_open.call_args
        assert call_kwargs[1]["limit_price"] == 1.50

    async def test_floor_respected(self):
        """Limit price never drops below bid."""
        cfg = _make_exec_config(
            entry_max_steps=10,  # many steps
            entry_step_pct=0.50,  # large steps
        )
        mock_exec = _make_execution_mock(fill_on_step=99)
        executor = SteppedOrderExecutor(mock_exec, cfg)

        await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        # All calls to sell_to_open should have limit_price >= bid
        for call in mock_exec.sell_to_open.call_args_list:
            assert call[1]["limit_price"] >= 1.50

    async def test_step_log_populated(self):
        """last_step_log is populated after execution."""
        cfg = _make_exec_config(entry_max_steps=1)
        mock_exec = _make_execution_mock(fill_on_step=0)
        executor = SteppedOrderExecutor(mock_exec, cfg)

        await executor.execute_entry("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert len(executor.last_step_log) >= 1
        assert "step" in executor.last_step_log[0]
        assert "limit_price" in executor.last_step_log[0]


# ── SteppedExit tests ──────────────────────────────────────────


class TestSteppedExit:
    async def test_fill_on_step_0(self):
        """Exit fills immediately on step 0."""
        cfg = _make_exec_config(exit_max_steps=2)
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "1.65", "filled_qty": "1",
        }

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_exit("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        order_result, filled_price = result
        assert filled_price == 1.65
        assert mock_exec.buy_to_close.call_count == 1

    async def test_fill_after_cancel(self):
        """Step 0 unfilled, cancel, step 1 fills."""
        cfg = _make_exec_config(exit_max_steps=2)
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.cancel_order.return_value = True

        # Step 0: new, new (after cancel). Step 1: filled.
        mock_exec.get_order_status.side_effect = [
            {"status": "new", "filled_avg_price": None, "filled_qty": "0"},
            {"status": "new", "filled_avg_price": None, "filled_qty": "0"},
            {"status": "filled", "filled_avg_price": "1.68", "filled_qty": "1"},
        ]

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_exit("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        _, filled_price = result
        assert filled_price == 1.68

    async def test_all_exhausted(self):
        """Returns None when all exit steps fail."""
        cfg = _make_exec_config(exit_max_steps=1)
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.cancel_order.return_value = True
        mock_exec.get_order_status.return_value = {
            "status": "new", "filled_avg_price": None, "filled_qty": "0",
        }

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_exit("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is None

    async def test_auto_fetch_snapshot(self):
        """When bid/ask not provided, fetches snapshot first."""
        cfg = _make_exec_config(exit_max_steps=0)
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "1.62", "filled_qty": "1",
        }

        async def mock_fetcher(symbols):
            return {symbols[0]: {"bid": 1.50, "ask": 1.70}}

        executor = SteppedOrderExecutor(
            mock_exec, cfg, snapshot_fetcher=mock_fetcher,
        )

        # Pass no bid/ask — executor must fetch them
        result = await executor.execute_exit("SYM", 1)
        assert result is not None
        _, filled_price = result
        assert filled_price == 1.62

    async def test_ask_zero_aborts(self):
        """Aborts when ask is zero."""
        cfg = _make_exec_config()
        executor = SteppedOrderExecutor(MagicMock(), cfg)

        result = await executor.execute_exit("SYM", 1, bid=0.0, ask=0.0, mid=0.0)
        assert result is None

    async def test_exit_start_ask(self):
        """When exit_start_price='ask', starts at ask."""
        # Need max_steps >= 2 so the ceiling (mid + steps*pct*spread) reaches ask
        cfg = _make_exec_config(exit_start_price="ask", exit_max_steps=2)
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "1.70", "filled_qty": "1",
        }

        executor = SteppedOrderExecutor(mock_exec, cfg)
        result = await executor.execute_exit("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        assert result is not None
        call_kwargs = mock_exec.buy_to_close.call_args
        assert call_kwargs[1]["limit_price"] == 1.70

    async def test_ceiling_respected(self):
        """Limit price never exceeds ask."""
        cfg = _make_exec_config(
            exit_max_steps=10,
            exit_step_pct=0.50,
        )
        mock_exec = MagicMock()
        mock_exec.buy_to_close.return_value = OrderResult(
            success=True, order_id="ord-002", message="OK",
        )
        mock_exec.cancel_order.return_value = True
        mock_exec.get_order_status.return_value = {
            "status": "new", "filled_avg_price": None, "filled_qty": "0",
        }

        executor = SteppedOrderExecutor(mock_exec, cfg)
        await executor.execute_exit("SYM", 1, bid=1.50, ask=1.70, mid=1.60)

        for call in mock_exec.buy_to_close.call_args_list:
            assert call[1]["limit_price"] <= 1.70

    async def test_no_snapshot_fetcher_aborts(self):
        """When no prices provided and no fetcher, returns None."""
        cfg = _make_exec_config()
        executor = SteppedOrderExecutor(MagicMock(), cfg, snapshot_fetcher=None)

        result = await executor.execute_exit("SYM", 1)
        assert result is None

    async def test_snapshot_missing_symbol_aborts(self):
        """When snapshot doesn't contain the symbol, returns None."""
        cfg = _make_exec_config()

        async def mock_fetcher(symbols):
            return {}

        executor = SteppedOrderExecutor(
            MagicMock(), cfg, snapshot_fetcher=mock_fetcher,
        )

        result = await executor.execute_exit("SYM", 1)
        assert result is None
