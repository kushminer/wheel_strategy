"""Tests for PositionProxy dataclass and factory methods."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from csp.trading.models import PositionProxy
from csp.trading.risk import RiskManager
from csp.trading.execution_config import RiskConfig


class TestPositionProxy:
    def test_defaults(self):
        proxy = PositionProxy(
            symbol="AAPL",
            option_symbol="AAPL260320P00220000",
            quantity=-1,
            strike=220.0,
            expiration=date.today() + timedelta(days=5),
        )
        assert proxy.symbol == "AAPL"
        assert proxy.entry_delta == 0.0
        assert proxy.entry_premium == 0.0
        assert proxy.days_held == 0

    def test_calculate_pnl_profit(self):
        proxy = PositionProxy(
            symbol="AAPL",
            option_symbol="AAPL260320P00220000",
            quantity=-1,
            strike=220.0,
            expiration=date.today(),
            entry_premium=1.50,
        )
        # Sold at 1.50, buy back at 0.50 → profit $100 per contract
        pnl = proxy.calculate_pnl(0.50)
        assert pnl == 100.0

    def test_calculate_pnl_loss(self):
        proxy = PositionProxy(
            symbol="AAPL",
            option_symbol="AAPL260320P00220000",
            quantity=-2,
            strike=220.0,
            expiration=date.today(),
            entry_premium=1.50,
        )
        # Sold at 1.50, buy back at 2.50 → loss $200 per contract * 2
        pnl = proxy.calculate_pnl(2.50)
        assert pnl == -200.0

    def test_calculate_pnl_zero(self):
        proxy = PositionProxy(
            symbol="AAPL",
            option_symbol="AAPL260320P00220000",
            quantity=-1,
            strike=220.0,
            expiration=date.today(),
            entry_premium=0.0,
        )
        assert proxy.calculate_pnl(0.0) == 0.0


class TestFromAlpacaAndMetadata:
    def test_builds_correct_proxy(self):
        alpaca_pos = SimpleNamespace(
            symbol="AAPL260320P00220000",
            qty="-1",
        )
        meta = {
            "underlying": "AAPL",
            "entry_delta": -0.25,
            "entry_iv": 0.30,
            "entry_vix": 18.0,
            "entry_stock_price": 230.0,
            "entry_premium": 1.50,
            "entry_daily_return": 0.0015,
            "dte_at_entry": 7,
            "entry_order_id": "order-123",
            "entry_date": (datetime.now() - timedelta(days=2)).isoformat(),
        }

        with patch("csp.clients.AlpacaClientManager.parse_strike_from_symbol", return_value=220.0), \
             patch("csp.clients.AlpacaClientManager.parse_expiration_from_symbol",
                   return_value=date.today() + timedelta(days=5)):
            proxy = PositionProxy.from_alpaca_and_metadata(alpaca_pos, meta)

        assert proxy.symbol == "AAPL"
        assert proxy.option_symbol == "AAPL260320P00220000"
        assert proxy.quantity == -1
        assert proxy.strike == 220.0
        assert proxy.entry_delta == -0.25
        assert proxy.entry_premium == 1.50
        assert proxy.days_held == 2
        assert proxy.current_dte == 5
        assert proxy.collateral_required == 22000.0

    def test_missing_metadata_uses_defaults(self):
        alpaca_pos = SimpleNamespace(
            symbol="MSFT260320P00300000",
            qty="-1",
        )
        meta = {}  # empty metadata

        with patch("csp.clients.AlpacaClientManager.parse_strike_from_symbol", return_value=300.0), \
             patch("csp.clients.AlpacaClientManager.parse_expiration_from_symbol",
                   return_value=date.today()):
            proxy = PositionProxy.from_alpaca_and_metadata(alpaca_pos, meta)

        assert proxy.symbol == ""
        assert proxy.entry_delta == 0
        assert proxy.entry_premium == 0


class TestFromCcStore:
    def test_builds_correct_proxy(self):
        cc_data = {
            "option_symbol": "AAPL260320C00240000",
            "strike": 240.0,
            "expiration": (date.today() + timedelta(days=3)).isoformat(),
            "entry_premium": 2.00,
            "quantity": 1,
            "entry_date": (datetime.now() - timedelta(days=1)).isoformat(),
        }
        pos_data = {
            "cost_basis": 230.0,
        }

        proxy = PositionProxy.from_cc_store("AAPL", cc_data, pos_data)

        assert proxy.symbol == "AAPL"
        assert proxy.option_symbol == "AAPL260320C00240000"
        assert proxy.strike == 240.0
        assert proxy.entry_premium == 2.00
        assert proxy.entry_stock_price == 230.0
        assert proxy.days_held == 1
        assert proxy.current_dte == 3


class TestRiskManagerCompatibility:
    def test_works_with_risk_manager(self):
        """PositionProxy should be accepted by RiskManager (duck typing)."""
        config = RiskConfig(
            enable_delta_stop=True,
            enable_delta_absolute_stop=True,
            enable_stock_drop_stop=False,
            enable_vix_spike_stop=False,
        )
        rm = RiskManager(config)

        proxy = PositionProxy(
            symbol="AAPL",
            option_symbol="AAPL260320P00220000",
            quantity=-1,
            strike=220.0,
            expiration=date.today() + timedelta(days=5),
            entry_delta=-0.20,
            entry_stock_price=230.0,
            entry_premium=1.50,
            entry_vix=18.0,
            entry_daily_return=0.0015,
            days_held=2,
        )

        result = rm.evaluate_position(
            position=proxy,
            current_delta=-0.25,
            current_stock_price=228.0,
            current_vix=19.0,
            current_premium=1.00,
        )

        assert not result.should_exit
