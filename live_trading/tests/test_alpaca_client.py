"""Unit tests for AlpacaClientManager (static methods + mocked clients)."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from csp.clients import AlpacaClientManager

from tests.conftest import make_alpaca_position, make_alpaca_account


# ── OCC symbol parsing (pure static, no mocks) ─────────────────


class TestParseStrikeFromSymbol:
    def test_standard_symbol(self):
        assert AlpacaClientManager.parse_strike_from_symbol("AAPL260220P00220000") == 220.0

    def test_fractional_strike(self):
        assert AlpacaClientManager.parse_strike_from_symbol("SPY260220P00450500") == 450.5

    def test_low_strike(self):
        assert AlpacaClientManager.parse_strike_from_symbol("F260220P00012000") == 12.0

    def test_high_strike(self):
        assert AlpacaClientManager.parse_strike_from_symbol("AMZN260220P01850000") == 1850.0

    def test_call_symbol(self):
        assert AlpacaClientManager.parse_strike_from_symbol("TSLA260220C00250000") == 250.0

    def test_invalid_symbol_returns_zero(self):
        assert AlpacaClientManager.parse_strike_from_symbol("INVALID") == 0.0


class TestParseExpirationFromSymbol:
    def test_standard_symbol(self):
        result = AlpacaClientManager.parse_expiration_from_symbol("AAPL260220P00220000")
        assert result == date(2026, 2, 20)

    def test_call_symbol(self):
        result = AlpacaClientManager.parse_expiration_from_symbol("TSLA260315C00250000")
        assert result == date(2026, 3, 15)

    def test_invalid_symbol_returns_none(self):
        result = AlpacaClientManager.parse_expiration_from_symbol("INVALID")
        assert result is None


# ── Mocked client methods ───────────────────────────────────────


class TestGetAccountInfo:
    @patch.dict("os.environ", {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"})
    def test_returns_account_dict(self):
        mgr = AlpacaClientManager.__new__(AlpacaClientManager)
        mgr.paper = True
        mgr.api_key = "test"
        mgr.secret_key = "test"
        mgr._data_client = None

        mock_account = make_alpaca_account(cash=100000, buying_power=200000, portfolio_value=150000)
        mock_trading = MagicMock()
        mock_trading.get_account.return_value = mock_account
        mgr._trading_client = mock_trading

        info = mgr.get_account_info()
        assert info['cash'] == 100000
        assert info['buying_power'] == 200000
        assert info['portfolio_value'] == 150000


class TestGetShortCollateral:
    @patch.dict("os.environ", {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"})
    def test_sums_short_positions(self):
        mgr = AlpacaClientManager.__new__(AlpacaClientManager)
        mgr.paper = True
        mgr.api_key = "test"
        mgr.secret_key = "test"
        mgr._data_client = None

        positions = [
            make_alpaca_position(symbol="AAPL260220P00220000", qty=-1, side="short"),
            make_alpaca_position(symbol="MSFT260220P00400000", qty=-2, side="short"),
        ]
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = positions
        mgr._trading_client = mock_trading

        collateral = mgr.get_short_collateral()
        # AAPL: 1 * 220 * 100 = 22000
        # MSFT: 2 * 400 * 100 = 80000
        assert collateral == 102000.0


class TestComputeAvailableCapital:
    @patch.dict("os.environ", {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"})
    def test_cash_minus_collateral(self):
        mgr = AlpacaClientManager.__new__(AlpacaClientManager)
        mgr.paper = True
        mgr.api_key = "test"
        mgr.secret_key = "test"
        mgr._data_client = None

        mock_account = make_alpaca_account(cash=100000, buying_power=200000, portfolio_value=150000)
        positions = [
            make_alpaca_position(symbol="AAPL260220P00220000", qty=-1, side="short"),
        ]
        mock_trading = MagicMock()
        mock_trading.get_account.return_value = mock_account
        mock_trading.get_all_positions.return_value = positions
        mgr._trading_client = mock_trading

        available = mgr.compute_available_capital()
        assert available == 100000 - 22000  # cash - collateral


class TestInitValidation:
    def test_raises_without_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="credentials"):
                AlpacaClientManager(paper=True)
