"""Tests for ExitRouter — shared exit order routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from csp.trading.exit_router import ExitRouter, _STEPPED_EXIT_REASONS
from csp.trading.models import ExitReason, OrderResult


def _make_router(stepped_result=None, market_result=None):
    """Build an ExitRouter with mocked dependencies."""
    execution = MagicMock()
    stepped_executor = MagicMock()

    # Stepped executor returns an async result
    stepped_executor.execute_exit = AsyncMock(return_value=stepped_result)

    # ExecutionEngine.buy_to_close is sync (wrapped by _arun)
    execution.buy_to_close = MagicMock(return_value=market_result)

    messages = []
    router = ExitRouter(execution, stepped_executor, vprint=messages.append)
    return router, execution, stepped_executor, messages


class TestExitRouting:
    """Test that exit reasons are routed correctly."""

    async def test_early_exit_uses_stepped(self):
        result = OrderResult(success=True, order_id="ord-1", message="filled")
        router, _, stepped, _ = _make_router(stepped_result=(result, 0.75))

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EARLY_EXIT,
        )

        assert success
        stepped.execute_exit.assert_awaited_once_with("AAPL260320P00220000", 1)

    async def test_expiry_uses_stepped(self):
        result = OrderResult(success=True, order_id="ord-2", message="filled")
        router, _, stepped, _ = _make_router(stepped_result=(result, 0.10))

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EXPIRY,
        )

        assert success
        stepped.execute_exit.assert_awaited_once()

    async def test_delta_stop_uses_market(self):
        market_result = OrderResult(success=True, order_id="mkt-1", message="filled")
        router, execution, stepped, _ = _make_router(market_result=market_result)

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.DELTA_STOP,
        )

        assert success
        stepped.execute_exit.assert_not_awaited()
        execution.buy_to_close.assert_called_once()

    async def test_vix_spike_uses_market(self):
        market_result = OrderResult(success=True, order_id="mkt-2", message="filled")
        router, execution, stepped, _ = _make_router(market_result=market_result)

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.VIX_SPIKE,
        )

        assert success
        stepped.execute_exit.assert_not_awaited()
        execution.buy_to_close.assert_called_once()

    async def test_stock_drop_uses_market(self):
        market_result = OrderResult(success=True, order_id="mkt-3", message="filled")
        router, _, stepped, _ = _make_router(market_result=market_result)

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.STOCK_DROP,
        )

        assert success
        stepped.execute_exit.assert_not_awaited()

    async def test_all_stop_loss_reasons_use_market(self):
        """Every ExitReason NOT in _STEPPED_EXIT_REASONS should route to market."""
        for reason in ExitReason:
            if reason in _STEPPED_EXIT_REASONS:
                continue
            market_result = OrderResult(success=True, order_id="mkt", message="ok")
            router, execution, stepped, _ = _make_router(market_result=market_result)

            await router.execute_exit("SYM", 1, reason)

            stepped.execute_exit.assert_not_awaited()
            execution.buy_to_close.assert_called_once()


class TestSteppedWithFallback:
    """Test the stepped → market fallback path."""

    async def test_stepped_fills_no_fallback(self):
        result = OrderResult(success=True, order_id="ord-1", message="filled")
        router, execution, _, _ = _make_router(stepped_result=(result, 0.50))

        success = await router.execute_exit(
            "AAPL260320P00220000", 2, ExitReason.EARLY_EXIT,
        )

        assert success
        execution.buy_to_close.assert_not_called()

    async def test_stepped_exhausted_falls_back_to_market(self):
        """When stepped returns None, should fall back to market order."""
        market_result = OrderResult(success=True, order_id="mkt-1", message="filled")
        router, execution, stepped, messages = _make_router(
            stepped_result=None,
            market_result=market_result,
        )

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EXPIRY,
        )

        assert success
        stepped.execute_exit.assert_awaited_once()
        execution.buy_to_close.assert_called_once()
        assert any("falling back" in m.lower() for m in messages)

    async def test_stepped_exhausted_market_also_fails(self):
        """When stepped returns None AND market order fails, returns False."""
        market_result = OrderResult(success=False, order_id=None, message="rejected")
        router, _, _, _ = _make_router(
            stepped_result=None,
            market_result=market_result,
        )

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EARLY_EXIT,
        )

        assert not success


class TestRecordFnCallback:
    """Test that record_fn is called correctly."""

    async def test_stepped_exit_calls_record_fn(self):
        result = OrderResult(success=True, order_id="ord-99", message="filled")
        router, _, _, _ = _make_router(stepped_result=(result, 1.25))

        recorded = {}
        def record_fn(order_id, filled_price):
            recorded["order_id"] = order_id
            recorded["filled_price"] = filled_price

        await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EARLY_EXIT,
            record_fn=record_fn,
        )

        assert recorded["order_id"] == "ord-99"
        assert recorded["filled_price"] == 1.25

    async def test_market_exit_calls_record_fn_with_zero_price(self):
        market_result = OrderResult(success=True, order_id="mkt-5", message="filled")
        router, _, _, _ = _make_router(market_result=market_result)

        recorded = {}
        def record_fn(order_id, filled_price):
            recorded["order_id"] = order_id
            recorded["filled_price"] = filled_price

        await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.DELTA_STOP,
            record_fn=record_fn,
        )

        assert recorded["order_id"] == "mkt-5"
        assert recorded["filled_price"] == 0.0

    async def test_no_record_fn_does_not_raise(self):
        result = OrderResult(success=True, order_id="ord-1", message="filled")
        router, _, _, _ = _make_router(stepped_result=(result, 0.50))

        # Should not raise even without record_fn
        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EARLY_EXIT,
        )
        assert success

    async def test_record_fn_not_called_on_failure(self):
        market_result = OrderResult(success=False, order_id=None, message="rejected")
        router, _, _, _ = _make_router(market_result=market_result)

        called = False
        def record_fn(order_id, filled_price):
            nonlocal called
            called = True

        await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.DELTA_STOP,
            record_fn=record_fn,
        )

        assert not called

    async def test_fallback_market_calls_record_fn(self):
        """When stepped exhausts and market fills, record_fn still called."""
        market_result = OrderResult(success=True, order_id="mkt-fb", message="filled")
        router, _, _, _ = _make_router(
            stepped_result=None,
            market_result=market_result,
        )

        recorded = {}
        def record_fn(order_id, filled_price):
            recorded["order_id"] = order_id
            recorded["filled_price"] = filled_price

        await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.EARLY_EXIT,
            record_fn=record_fn,
        )

        assert recorded["order_id"] == "mkt-fb"
        assert recorded["filled_price"] == 0.0


class TestMarketOrderEdgeCases:
    """Edge cases for market order execution."""

    async def test_market_order_returns_none(self):
        """If buy_to_close returns None entirely."""
        router, execution, _, _ = _make_router(market_result=None)

        success = await router.execute_exit(
            "AAPL260320P00220000", 1, ExitReason.DELTA_STOP,
        )

        assert not success

    async def test_quantity_passed_through(self):
        market_result = OrderResult(success=True, order_id="mkt", message="ok")
        router, execution, _, _ = _make_router(market_result=market_result)

        await router.execute_exit(
            "AAPL260320P00220000", 3, ExitReason.DELTA_STOP,
        )

        execution.buy_to_close.assert_called_once_with(
            option_symbol="AAPL260320P00220000",
            quantity=3,
            limit_price=None,
        )
