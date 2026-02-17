"""Shared fixtures and factory functions for CSP strategy tests."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from csp.config import StrategyConfig
from csp.trading.models import ExitReason, RiskCheckResult, OrderResult
from csp.data.options import OptionContract


# ─── Configuration Fixtures ─────────────────────────────────────────


@pytest.fixture
def config():
    """Minimal StrategyConfig with small universe (no Wikipedia fetch)."""
    return StrategyConfig(
        ticker_universe=["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"],
        starting_cash=100_000,
        paper_trading=True,
        num_tickers=5,
    )


# ─── Factory Functions ──────────────────────────────────────────────


def make_option_contract(**overrides) -> OptionContract:
    """Factory for OptionContract with sensible defaults."""
    defaults = dict(
        symbol="AAPL260220P00220000",
        underlying="AAPL",
        contract_type="put",
        strike=220.0,
        expiration=date.today() + timedelta(days=5),
        dte=5,
        bid=1.50,
        ask=1.70,
        mid=1.60,
        stock_price=230.0,
        entry_time=None,
        delta=-0.25,
        gamma=0.015,
        theta=-0.10,
        vega=0.08,
        implied_volatility=0.30,
        open_interest=500,
        volume=100,
        days_since_strike=30,
    )
    defaults.update(overrides)
    return OptionContract(**defaults)


def make_position_proxy(**overrides) -> SimpleNamespace:
    """Factory for position proxy objects (as used by RiskManager)."""
    defaults = dict(
        symbol="AAPL",
        option_symbol="AAPL260220P00220000",
        quantity=-1,
        strike=220.0,
        expiration=date.today() + timedelta(days=5),
        entry_delta=-0.25,
        entry_iv=0.30,
        entry_vix=18.0,
        entry_stock_price=230.0,
        entry_premium=1.50,
        entry_daily_return=0.0015,
        dte_at_entry=7,
        entry_order_id="test-order-123",
        current_dte=5,
        days_held=2,
        collateral_required=22000.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_order_result(success=True, **overrides) -> OrderResult:
    """Factory for OrderResult."""
    defaults = dict(
        success=success,
        order_id="order-abc-123" if success else None,
        message="Order submitted" if success else "Order failed",
        order_details={"id": "order-abc-123", "status": "accepted"} if success else None,
    )
    defaults.update(overrides)
    return OrderResult(**defaults)


def make_risk_check_result(should_exit=False, exit_reason=None, **overrides) -> RiskCheckResult:
    """Factory for RiskCheckResult."""
    defaults = dict(
        should_exit=should_exit,
        exit_reason=exit_reason,
        details="All checks passed" if not should_exit else f"Triggered: {exit_reason}",
        current_values={},
    )
    defaults.update(overrides)
    return RiskCheckResult(**defaults)


def make_alpaca_position(symbol="AAPL260220P00220000", qty=-1, side="short",
                         current_price=1.20, avg_entry_price=1.50):
    """Mock Alpaca position object."""
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        side=SimpleNamespace(value=side),
        current_price=str(current_price),
        avg_entry_price=str(avg_entry_price),
        market_value=str(float(current_price) * abs(qty) * 100),
        unrealized_pl=str((float(avg_entry_price) - float(current_price)) * abs(qty) * 100),
    )


def make_alpaca_account(cash=100000, buying_power=200000, portfolio_value=150000):
    """Mock Alpaca account object."""
    return SimpleNamespace(
        cash=str(cash),
        buying_power=str(buying_power),
        portfolio_value=str(portfolio_value),
        status="ACTIVE",
        trading_blocked=False,
        options_trading_level=2,
    )


def make_alpaca_order(order_id="order-123", symbol="AAPL260220P00220000",
                      side="sell", qty=1, status="filled", filled_avg_price=1.50):
    """Mock Alpaca order object."""
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=SimpleNamespace(value=side),
        qty=str(qty),
        filled_qty=str(qty) if status == "filled" else "0",
        type=SimpleNamespace(value="limit"),
        status=SimpleNamespace(value=status),
        limit_price=str(filled_avg_price),
        filled_avg_price=str(filled_avg_price) if status == "filled" else None,
    )


def make_price_series(base=230.0, n=60, trend="up", seed=42):
    """Generate synthetic price series for equity filter tests."""
    rng = np.random.default_rng(seed)
    if trend == "up":
        prices = base * (1 + np.cumsum(rng.normal(0.001, 0.01, n)))
    elif trend == "down":
        prices = base * (1 + np.cumsum(rng.normal(-0.001, 0.01, n)))
    else:  # flat
        prices = base * (1 + np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range(end=date.today(), periods=n)
    return pd.Series(prices, index=idx, name="TEST")
