"""Unit tests for ExecutionEngine (mocked Alpaca trading client)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from csp.config import StrategyConfig
from csp.trading.execution import ExecutionEngine

from tests.conftest import make_alpaca_order


@pytest.fixture
def config():
    return StrategyConfig(ticker_universe=["AAPL"])


@pytest.fixture
def mock_alpaca():
    """Mock AlpacaClientManager with a mock trading_client."""
    mgr = SimpleNamespace(
        trading_client=MagicMock(),
        paper=True,
    )
    return mgr


@pytest.fixture
def engine(mock_alpaca, config):
    return ExecutionEngine(mock_alpaca, config)


# ── sell_to_open ────────────────────────────────────────────────


class TestSellToOpen:
    def test_limit_success(self, engine, mock_alpaca):
        mock_order = make_alpaca_order(order_id="sell-123", status="accepted", filled_avg_price=1.50)
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        result = engine.sell_to_open("AAPL260220P00220000", quantity=1, limit_price=1.50)
        assert result.success is True
        assert result.order_id == "sell-123"
        assert mock_alpaca.trading_client.submit_order.called

    def test_market_success(self, engine, mock_alpaca):
        mock_order = make_alpaca_order(order_id="sell-456", status="accepted")
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        result = engine.sell_to_open("AAPL260220P00220000", quantity=1)
        assert result.success is True
        assert result.order_id == "sell-456"

    def test_api_error_returns_failure(self, engine, mock_alpaca):
        mock_alpaca.trading_client.submit_order.side_effect = Exception("API error")

        result = engine.sell_to_open("AAPL260220P00220000")
        assert result.success is False
        assert result.order_id is None
        assert "API error" in result.message


# ── buy_to_close ────────────────────────────────────────────────


class TestBuyToClose:
    def test_limit_success(self, engine, mock_alpaca):
        mock_order = make_alpaca_order(order_id="buy-123", side="buy", status="accepted")
        mock_alpaca.trading_client.submit_order.return_value = mock_order

        result = engine.buy_to_close("AAPL260220P00220000", quantity=1, limit_price=0.80)
        assert result.success is True
        assert result.order_id == "buy-123"

    def test_api_error_returns_failure(self, engine, mock_alpaca):
        mock_alpaca.trading_client.submit_order.side_effect = Exception("Insufficient funds")

        result = engine.buy_to_close("AAPL260220P00220000")
        assert result.success is False
        assert "Insufficient funds" in result.message


# ── get_order_status ────────────────────────────────────────────


class TestGetOrderStatus:
    def test_found(self, engine, mock_alpaca):
        mock_order = make_alpaca_order(order_id="order-789", status="filled", filled_avg_price=1.50)
        mock_alpaca.trading_client.get_order_by_id.return_value = mock_order

        status = engine.get_order_status("order-789")
        assert status is not None
        assert status['status'] == "filled"
        assert status['filled_avg_price'] == "1.5"

    def test_not_found_returns_none(self, engine, mock_alpaca):
        mock_alpaca.trading_client.get_order_by_id.side_effect = Exception("Not found")
        assert engine.get_order_status("unknown") is None


# ── cancel_order ────────────────────────────────────────────────


class TestCancelOrder:
    def test_success(self, engine, mock_alpaca):
        mock_alpaca.trading_client.cancel_order_by_id.return_value = None
        assert engine.cancel_order("order-123") is True

    def test_failure_returns_false(self, engine, mock_alpaca):
        mock_alpaca.trading_client.cancel_order_by_id.side_effect = Exception("Cannot cancel")
        assert engine.cancel_order("order-123") is False


# ── get_positions ───────────────────────────────────────────────


class TestGetPositions:
    def test_returns_list(self, engine, mock_alpaca):
        from tests.conftest import make_alpaca_position
        positions = [
            make_alpaca_position(symbol="AAPL260220P00220000", qty=-1),
        ]
        mock_alpaca.trading_client.get_all_positions.return_value = positions

        result = engine.get_positions()
        assert len(result) == 1
        assert result[0]['symbol'] == "AAPL260220P00220000"
        assert result[0]['qty'] == "-1"

    def test_handles_error(self, engine, mock_alpaca):
        mock_alpaca.trading_client.get_all_positions.side_effect = Exception("Connection error")
        assert engine.get_positions() == []
