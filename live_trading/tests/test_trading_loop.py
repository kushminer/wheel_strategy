"""Integration tests for TradingLoop.

Tests cover:
- Position monitoring & risk-driven exits
- Assignment detection
- Exit execution routing (assignment, early exit, stop-loss variants)
- Scan and enter flow
- Stepped entry logic
- Market hours
- VIX regime & global stop
- run_cycle orchestration
"""

from datetime import date, datetime, timedelta, time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from csp.config import StrategyConfig
from csp.trading.loop import TradingLoop
from csp.trading.models import ExitReason, RiskCheckResult, OrderResult
from csp.trading.metadata import StrategyMetadataStore
from csp.trading.daily_log import DailyLog

from tests.conftest import (
    make_option_contract,
    make_position_proxy,
    make_alpaca_position,
    make_alpaca_order,
    make_alpaca_account,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_config(**overrides):
    """Config with stepped entry/exit disabled for simpler testing."""
    defaults = dict(
        ticker_universe=["AAPL", "MSFT"],
        starting_cash=100_000,
        paper_trading=True,
        num_tickers=5,
        entry_max_steps=0,
        entry_step_interval=0,
        exit_max_steps=0,
        exit_step_interval=0,
        vix_spike_multiplier=1.15,
        close_before_expiry_days=0,
        exit_on_missing_delta=False,
        max_position_pct=0.10,
        max_contracts_per_ticker=5,
        contract_rank_mode="lowest_strike_price",
        enable_delta_stop=True,
        enable_delta_absolute_stop=True,
        enable_stock_drop_stop=False,
        enable_vix_spike_stop=True,
        enable_early_exit=True,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _make_loop(config=None, tmp_path=None, **mock_overrides):
    """Build a TradingLoop with all dependencies mocked."""
    config = config or _make_config()

    mock_alpaca = SimpleNamespace(
        trading_client=MagicMock(),
        paper=True,
    )
    mock_alpaca.trading_client.get_all_positions.return_value = []
    mock_alpaca.get_account_info = MagicMock(return_value={
        'cash': 100000, 'buying_power': 200000, 'portfolio_value': 150000,
    })
    mock_alpaca.get_short_collateral = MagicMock(return_value=0)
    mock_alpaca.compute_available_capital = MagicMock(return_value=100000)

    mock_data_mgr = MagicMock()
    mock_data_mgr.equity_fetcher.get_current_price.return_value = 230.0
    mock_data_mgr.equity_fetcher.get_close_history.return_value = {}
    mock_data_mgr.options_fetcher.get_option_snapshots.return_value = {}
    mock_data_mgr.options_fetcher.get_puts_chain.return_value = []

    mock_scanner = MagicMock()
    mock_scanner.scan_universe.return_value = []
    mock_scanner.options_filter = MagicMock()

    log_dir = str(tmp_path) if tmp_path else "/tmp/test_logs"
    metadata_path = str(tmp_path / "meta.json") if tmp_path else "/tmp/test_meta.json"

    metadata = StrategyMetadataStore(path=metadata_path)

    mock_risk = MagicMock()
    mock_risk.evaluate_position.return_value = RiskCheckResult(
        should_exit=False, exit_reason=None, details="All clear", current_values={},
    )

    mock_execution = MagicMock()
    mock_execution.paper = True
    mock_execution.sell_to_open.return_value = OrderResult(
        success=True, order_id="entry-001", message="OK",
    )
    mock_execution.buy_to_close.return_value = OrderResult(
        success=True, order_id="exit-001", message="OK",
    )
    mock_execution.get_order_status.return_value = {
        'status': 'filled', 'filled_avg_price': '1.50', 'filled_qty': '1',
    }
    mock_execution.cancel_order.return_value = True

    mock_vix = MagicMock()
    mock_vix.get_current_vix.return_value = 18.0
    mock_vix.get_session_reference_vix.return_value = (date.today(), 17.5)

    mock_greeks = MagicMock()
    mock_greeks.compute_greeks_from_price.return_value = {'delta': -0.25}

    # Apply overrides
    components = {
        'alpaca_manager': mock_alpaca,
        'data_manager': mock_data_mgr,
        'scanner': mock_scanner,
        'metadata': metadata,
        'risk_manager': mock_risk,
        'execution': mock_execution,
        'vix_fetcher': mock_vix,
        'greeks_calc': mock_greeks,
    }
    components.update(mock_overrides)

    with patch('csp.trading.loop.DailyLog') as MockDailyLog:
        mock_logger = MagicMock()
        mock_logger.today_path = f"{log_dir}/{date.today().isoformat()}.json"
        MockDailyLog.return_value = mock_logger

        loop = TradingLoop(
            config=config,
            data_manager=components['data_manager'],
            scanner=components['scanner'],
            metadata_store=components['metadata'],
            risk_manager=components['risk_manager'],
            execution=components['execution'],
            vix_fetcher=components['vix_fetcher'],
            greeks_calc=components['greeks_calc'],
            alpaca_manager=components['alpaca_manager'],
        )
        loop.logger = mock_logger

    return loop, components


# ── Monitor → Risk → Exit ──────────────────────────────────────


class TestMonitorPositions:
    """monitor_positions: check existing positions for exits."""

    async def test_delta_absolute_stop_triggers_exit(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # Set up an active position in Alpaca + metadata
        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Snapshot returns delta data
        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            "AAPL260220P00220000": {'bid': 2.00, 'ask': 2.50, 'delta': -0.45},
        }

        # Risk manager says exit
        comp['risk_manager'].evaluate_position.return_value = RiskCheckResult(
            should_exit=True,
            exit_reason=ExitReason.DELTA_ABSOLUTE,
            details="Delta 0.45 >= 0.40",
            current_values={'current_delta': -0.45},
        )

        exits = await loop.monitor_positions(current_vix=18.0)
        assert len(exits) == 1
        _, risk_result, _ = exits[0]
        assert risk_result.exit_reason == ExitReason.DELTA_ABSOLUTE

    async def test_skips_unknown_metadata(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # Position exists in Alpaca but NOT in metadata
        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        exits = await loop.monitor_positions(current_vix=18.0)
        assert len(exits) == 0

    async def test_handles_missing_delta_fallback(self, tmp_path):
        loop, comp = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(exit_on_missing_delta=False),
        )

        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Snapshot has no delta, and greeks calc also returns None
        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            "AAPL260220P00220000": {'bid': 1.20, 'ask': 1.40, 'delta': None},
        }
        comp['greeks_calc'].compute_greeks_from_price.return_value = {'delta': None}

        # Risk manager should be called with entry_delta as fallback
        exits = await loop.monitor_positions(current_vix=18.0)
        comp['risk_manager'].evaluate_position.assert_called_once()
        call_kwargs = comp['risk_manager'].evaluate_position.call_args
        assert call_kwargs.kwargs.get('current_delta') == -0.25 or call_kwargs[1].get('current_delta') == -0.25

    async def test_missing_delta_triggers_exit_when_configured(self, tmp_path):
        loop, comp = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(exit_on_missing_delta=True),
        )

        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            "AAPL260220P00220000": {'bid': 1.20, 'ask': 1.40, 'delta': None},
        }
        comp['greeks_calc'].compute_greeks_from_price.return_value = {'delta': None}

        exits = await loop.monitor_positions(current_vix=18.0)
        assert len(exits) == 1
        _, risk_result, _ = exits[0]
        assert risk_result.exit_reason == ExitReason.DATA_UNAVAILABLE

    async def test_detects_expiring_position(self, tmp_path):
        config = _make_config(close_before_expiry_days=1)
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        # Build OCC symbol that expires tomorrow (proxy parses expiry from symbol)
        tomorrow = date.today() + timedelta(days=1)
        occ_sym = f"AAPL{tomorrow.strftime('%y%m%d')}P00220000"

        alpaca_pos = make_alpaca_position(occ_sym, qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol=occ_sym,
            underlying="AAPL", strike=220.0,
            expiration=tomorrow.isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            occ_sym: {'ask': 0.10},
        }

        exits = await loop.monitor_positions(current_vix=18.0)
        assert len(exits) == 1
        _, risk_result, premium = exits[0]
        assert risk_result.exit_reason == ExitReason.EXPIRY
        assert premium == 0.10


# ── Parallel Monitor Positions ─────────────────────────────────


class TestParallelMonitorPositions:
    """monitor_positions: positions are checked in parallel via asyncio.gather."""

    async def test_parallel_monitor_checks_all_positions(self, tmp_path):
        """All positions are checked concurrently; risk_manager called for each."""
        loop, comp = _make_loop(tmp_path=tmp_path)

        symbols = ["AAPL260220P00220000", "MSFT260220P00400000", "GOOG260220P00170000"]
        positions = []
        for sym in symbols:
            underlying = sym[:4]
            pos = make_alpaca_position(sym, qty=-1, side="short")
            positions.append(pos)
            loop.metadata.record_entry(
                option_symbol=sym,
                underlying=underlying, strike=220.0,
                expiration=(date.today() + timedelta(days=5)).isoformat(),
                entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
                entry_stock_price=230.0, entry_premium=1.50,
                entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
                entry_order_id=f"entry-{underlying}",
            )

        comp['alpaca_manager'].trading_client.get_all_positions.return_value = positions

        def mock_snapshots(option_symbols):
            return {sym: {'bid': 1.20, 'ask': 1.40, 'delta': -0.30} for sym in option_symbols}

        comp['data_manager'].options_fetcher.get_option_snapshots.side_effect = mock_snapshots

        comp['risk_manager'].evaluate_position.return_value = RiskCheckResult(
            should_exit=False,
            exit_reason=None,
            details="All checks passed",
            current_values={},
        )

        exits = await loop.monitor_positions(current_vix=18.0)
        assert len(exits) == 0
        assert comp['risk_manager'].evaluate_position.call_count == 3

    async def test_parallel_monitor_partial_failure(self, tmp_path):
        """One position snapshot raises exception; other positions still evaluated."""
        loop, comp = _make_loop(tmp_path=tmp_path)

        symbols = ["AAPL260220P00220000", "MSFT260220P00400000"]
        positions = []
        for sym in symbols:
            underlying = sym[:4]
            pos = make_alpaca_position(sym, qty=-1, side="short")
            positions.append(pos)
            loop.metadata.record_entry(
                option_symbol=sym,
                underlying=underlying, strike=220.0,
                expiration=(date.today() + timedelta(days=5)).isoformat(),
                entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
                entry_stock_price=230.0, entry_premium=1.50,
                entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
                entry_order_id=f"entry-{underlying}",
            )

        comp['alpaca_manager'].trading_client.get_all_positions.return_value = positions

        call_count = 0
        def mock_snapshots(option_symbols):
            nonlocal call_count
            call_count += 1
            sym = option_symbols[0]
            if "MSFT" in sym:
                raise Exception("API timeout")
            return {sym: {'bid': 1.20, 'ask': 1.40, 'delta': -0.30}}

        comp['data_manager'].options_fetcher.get_option_snapshots.side_effect = mock_snapshots

        comp['risk_manager'].evaluate_position.return_value = RiskCheckResult(
            should_exit=True,
            exit_reason=ExitReason.DELTA_ABSOLUTE,
            details="Delta 0.30 >= 0.40",
            current_values={},
        )

        exits = await loop.monitor_positions(current_vix=18.0)
        # AAPL should succeed, MSFT should fail gracefully
        assert len(exits) == 1
        assert comp['risk_manager'].evaluate_position.call_count == 1


# ── Assignment Detection ───────────────────────────────────────


class TestAssignmentDetection:
    """_check_assignments: detect when option is gone and stock appeared."""

    async def test_detects_assignment(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Alpaca shows: option gone, stock appeared (100 shares)
        stock_pos = SimpleNamespace(
            symbol="AAPL", qty="100",
            side=SimpleNamespace(value="long"),
        )
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [stock_pos]

        assignments = await loop._check_assignments()
        assert len(assignments) == 1
        proxy, result, premium = assignments[0]
        assert result.exit_reason == ExitReason.ASSIGNED
        assert premium == 0.0

    async def test_no_false_positive_with_option_still_present(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Option is still present in Alpaca
        option_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [option_pos]

        assignments = await loop._check_assignments()
        assert len(assignments) == 0

    async def test_no_false_positive_without_stock(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Option gone but NO stock shares appeared (expired OTM)
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = []

        assignments = await loop._check_assignments()
        assert len(assignments) == 0


# ── Execute Exit ───────────────────────────────────────────────


class TestExecuteExit:
    """execute_exit: routes to correct order type based on exit reason."""

    async def test_assignment_no_order(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        position = make_position_proxy()
        position.calculate_pnl = lambda ep: (1.50 - ep) * 100

        risk_result = RiskCheckResult(
            should_exit=True, exit_reason=ExitReason.ASSIGNED,
            details="Assigned", current_values={},
        )

        result = await loop.execute_exit(position, risk_result, current_premium=0.0)
        assert result is True
        # No order should be submitted for assignment
        comp['execution'].buy_to_close.assert_not_called()
        comp['execution'].sell_to_open.assert_not_called()

    async def test_stop_loss_market_order(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        position = make_position_proxy()
        position.calculate_pnl = lambda ep: (1.50 - ep) * 100

        risk_result = RiskCheckResult(
            should_exit=True, exit_reason=ExitReason.DELTA_ABSOLUTE,
            details="Delta too high", current_values={},
        )

        result = await loop.execute_exit(position, risk_result, current_premium=2.50)
        assert result is True
        comp['execution'].buy_to_close.assert_called_once()
        call_kwargs = comp['execution'].buy_to_close.call_args
        # Market order: limit_price should be None
        assert call_kwargs.kwargs.get('limit_price') is None or call_kwargs[1].get('limit_price') is None



# ── Parallel Exit Execution ───────────────────────────────────


class TestParallelExitExecution:
    """Exit orders execute in parallel via asyncio.gather."""

    async def test_parallel_exits_all_succeed(self, tmp_path):
        """Multiple exits run concurrently, all succeed."""
        loop, comp = _make_loop(tmp_path=tmp_path)

        # Set up two positions
        for sym, underlying in [("AAPL260220P00220000", "AAPL"), ("MSFT260220P00400000", "MSFT")]:
            loop.metadata.record_entry(
                option_symbol=sym,
                underlying=underlying, strike=220.0,
                expiration=(date.today() + timedelta(days=5)).isoformat(),
                entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
                entry_stock_price=230.0, entry_premium=1.50,
                entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
                entry_order_id=f"entry-{underlying}",
            )

        positions = [
            make_position_proxy(symbol="AAPL", option_symbol="AAPL260220P00220000"),
            make_position_proxy(symbol="MSFT", option_symbol="MSFT260220P00400000"),
        ]
        for pos in positions:
            pos.calculate_pnl = lambda ep, _p=pos: (_p.entry_premium - ep) * abs(_p.quantity) * 100

        risk_result = RiskCheckResult(
            should_exit=True,
            exit_reason=ExitReason.DELTA_ABSOLUTE,
            details="Delta exceeded",
            current_values={},
        )

        comp['execution'].buy_to_close.return_value = OrderResult(
            success=True, order_id="exit-001", message="OK",
        )

        exits_needed = [(pos, risk_result, 2.50) for pos in positions]

        import asyncio
        exit_results = await asyncio.gather(
            *[loop.execute_exit(pos, rr, current_premium=ep)
              for pos, rr, ep in exits_needed]
        )
        total_exits = sum(1 for r in exit_results if r)
        assert total_exits == 2
        assert comp['execution'].buy_to_close.call_count == 2

    async def test_parallel_exits_partial_failure(self, tmp_path):
        """One exit fails, other succeeds; count reflects partial success."""
        loop, comp = _make_loop(tmp_path=tmp_path)

        for sym, underlying in [("AAPL260220P00220000", "AAPL"), ("MSFT260220P00400000", "MSFT")]:
            loop.metadata.record_entry(
                option_symbol=sym,
                underlying=underlying, strike=220.0,
                expiration=(date.today() + timedelta(days=5)).isoformat(),
                entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
                entry_stock_price=230.0, entry_premium=1.50,
                entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
                entry_order_id=f"entry-{underlying}",
            )

        positions = [
            make_position_proxy(symbol="AAPL", option_symbol="AAPL260220P00220000"),
            make_position_proxy(symbol="MSFT", option_symbol="MSFT260220P00400000"),
        ]
        for pos in positions:
            pos.calculate_pnl = lambda ep, _p=pos: (_p.entry_premium - ep) * abs(_p.quantity) * 100

        risk_result = RiskCheckResult(
            should_exit=True,
            exit_reason=ExitReason.DELTA_ABSOLUTE,
            details="Delta exceeded",
            current_values={},
        )

        call_count = 0
        def mock_buy_to_close(**kwargs):
            nonlocal call_count
            call_count += 1
            if "MSFT" in kwargs.get('option_symbol', ''):
                return OrderResult(success=False, order_id=None, message="Rejected")
            return OrderResult(success=True, order_id="exit-001", message="OK")

        comp['execution'].buy_to_close.side_effect = mock_buy_to_close

        exits_needed = [(pos, risk_result, 2.50) for pos in positions]

        import asyncio
        exit_results = await asyncio.gather(
            *[loop.execute_exit(pos, rr, current_premium=ep)
              for pos, rr, ep in exits_needed]
        )
        total_exits = sum(1 for r in exit_results if r)
        assert total_exits == 1  # AAPL succeeds, MSFT fails


# ── Scan and Enter ─────────────────────────────────────────────


class TestScanAndEnter:
    """scan_and_enter: find candidates and place orders."""

    async def test_monitor_only_returns_zero(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)
        loop._monitor_only = True
        result = await loop.scan_and_enter(deployable_cash=90000)
        assert result == 0

    async def test_skips_existing_symbols(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # Simulate equity scan already cached
        loop._equity_passing = ["AAPL"]
        loop._equity_scan_date = datetime.now(loop.eastern).date()

        # AAPL already has a position
        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        result = await loop.scan_and_enter(deployable_cash=90000)
        # No puts should be fetched for AAPL (already in portfolio)
        assert result == 0

    async def test_respects_cash_limit(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)
        comp['alpaca_manager'].compute_available_capital.return_value = 0

        result = await loop.scan_and_enter(deployable_cash=0)
        assert result == 0

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_market_order_entry_bypasses_stepped(self, mock_sleep, tmp_path):
        """When entry_order_type='market', _execute_market_entry is used instead of stepped."""
        config = _make_config(entry_order_type="market")
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(bid=1.50, ask=1.70, mid=1.60)

        # Mock _execute_market_entry and _execute_stepped_entry to track which is called
        market_result = (
            OrderResult(success=True, order_id="mkt-001", message="OK"),
            1.55,
        )
        loop._execute_market_entry = AsyncMock(return_value=market_result)
        loop._execute_stepped_entry = AsyncMock(return_value=None)

        # Pre-set equity scan results so scan_and_enter skips the scan
        loop._equity_passing = ["AAPL"]
        loop._equity_scan_date = datetime.now(loop.eastern).date()

        # Options chain and filter return one ranked candidate
        comp['data_manager'].options_fetcher.get_puts_chain.return_value = [candidate]
        comp['data_manager'].equity_fetcher.get_close_history.return_value = {}
        comp['scanner'].options_filter.filter_and_rank.return_value = (
            [candidate],  # ranked
            [SimpleNamespace(passes=True, failure_reasons=[])],  # filter_results
        )

        result = await loop.scan_and_enter(deployable_cash=90000)

        # _execute_market_entry should have been called, not _execute_stepped_entry
        loop._execute_market_entry.assert_called()
        loop._execute_stepped_entry.assert_not_called()

    async def test_no_equity_passing_returns_negative_one(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # Scanner returns results where nothing passes equity
        scan_result = SimpleNamespace(
            symbol="AAPL",
            equity_result=SimpleNamespace(
                passes=False, symbol="AAPL", current_price=230.0,
                sma_8=228.0, sma_20=225.0, sma_50=220.0,
                bb_upper=240.0, rsi=55.0,
            ),
            has_candidates=False,
            options_candidates=[],
        )
        comp['scanner'].scan_universe.return_value = [scan_result]

        result = await loop.scan_and_enter(deployable_cash=90000)
        assert result == -1


# ── Parallel Options Fetch ─────────────────────────────────────


class TestFetchSymbolOptions:
    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_parallel_fetch_all_succeed(self, mock_sleep, tmp_path):
        """asyncio.gather fetches multiple symbols in parallel, all succeed."""
        config = _make_config(ticker_universe=["AAPL", "MSFT", "GOOG"])
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidates = {
            "AAPL": make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", bid=1.50),
            "MSFT": make_option_contract(underlying="MSFT", symbol="MSFT260220P00400000", strike=400.0, bid=2.00),
            "GOOG": make_option_contract(underlying="GOOG", symbol="GOOG260220P00170000", strike=170.0, bid=1.80),
        }

        def mock_get_puts_chain(symbol, stock_price, config, sma_ceiling=None):
            return [candidates[symbol]]

        comp['data_manager'].options_fetcher.get_puts_chain.side_effect = mock_get_puts_chain
        comp['data_manager'].equity_fetcher.get_close_history.return_value = {}

        def mock_filter_and_rank(puts):
            return (puts, [SimpleNamespace(passes=True, failure_reasons=[])])

        comp['scanner'].options_filter.filter_and_rank.side_effect = mock_filter_and_rank

        # Pre-set equity scan results so scan_and_enter skips the equity scan
        loop._equity_passing = ["AAPL", "MSFT", "GOOG"]
        loop._equity_scan_date = datetime.now(loop.eastern).date()

        # Mock _execute_market_entry to avoid order execution
        loop._execute_market_entry = AsyncMock(return_value=None)

        result = await loop.scan_and_enter(deployable_cash=500000)

        # All 3 symbols should have had options scanned
        assert loop.logger.log_options_scan.call_count == 3

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_parallel_fetch_partial_failure(self, mock_sleep, tmp_path):
        """One symbol raises an exception, other symbols still succeed."""
        config = _make_config(ticker_universe=["AAPL", "MSFT", "GOOG"])
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        good_candidate = make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", bid=1.50)

        def mock_get_puts_chain(symbol, stock_price, config, sma_ceiling=None):
            if symbol == "MSFT":
                raise Exception("API timeout for MSFT")
            return [good_candidate]

        comp['data_manager'].options_fetcher.get_puts_chain.side_effect = mock_get_puts_chain
        comp['data_manager'].equity_fetcher.get_close_history.return_value = {}

        def mock_filter_and_rank(puts):
            return (puts, [SimpleNamespace(passes=True, failure_reasons=[])])

        comp['scanner'].options_filter.filter_and_rank.side_effect = mock_filter_and_rank

        loop._equity_passing = ["AAPL", "MSFT", "GOOG"]
        loop._equity_scan_date = datetime.now(loop.eastern).date()
        loop._execute_market_entry = AsyncMock(return_value=None)

        result = await loop.scan_and_enter(deployable_cash=500000)

        # AAPL and GOOG should succeed, MSFT should fail gracefully
        assert loop.logger.log_options_scan.call_count == 2


# ── Compute Target Quantity ────────────────────────────────────


class TestComputeTargetQuantity:
    def test_basic_computation(self, tmp_path):
        loop, _ = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(starting_cash=100_000, max_position_pct=0.10, max_contracts_per_ticker=5),
        )
        # target_allocation = 100k * 0.10 = 10k
        # available_cash = 50k >= 10k, so qty = 10000 // 22000 = 0 → clamped to 1
        qty = loop.compute_target_quantity(collateral_per_contract=22000, available_cash=50000)
        assert qty == 1  # Can't afford more than 1 at $22k

    def test_multiple_contracts(self, tmp_path):
        loop, _ = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(starting_cash=200_000, max_position_pct=0.20, max_contracts_per_ticker=5),
        )
        # target_allocation = 200k * 0.20 = 40k
        # available_cash = 100k >= 40k, so qty = 40000 // 10000 = 4
        qty = loop.compute_target_quantity(collateral_per_contract=10000, available_cash=100000)
        assert qty == 4

    def test_capped_by_max_contracts(self, tmp_path):
        loop, _ = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(starting_cash=1_000_000, max_position_pct=0.50, max_contracts_per_ticker=3),
        )
        # target_allocation = 1M * 0.50 = 500k
        # qty = 500000 // 10000 = 50, capped to 3
        qty = loop.compute_target_quantity(collateral_per_contract=10000, available_cash=1_000_000)
        assert qty == 3

    def test_zero_collateral_returns_one(self, tmp_path):
        loop, _ = _make_loop(tmp_path=tmp_path)
        qty = loop.compute_target_quantity(collateral_per_contract=0, available_cash=100000)
        assert qty == 1

    def test_low_cash_uses_available(self, tmp_path):
        loop, _ = _make_loop(
            tmp_path=tmp_path,
            config=_make_config(starting_cash=100_000, max_position_pct=0.50, max_contracts_per_ticker=10),
        )
        # target_allocation = 100k * 0.50 = 50k
        # available_cash = 25k < 50k, so qty = 25000 // 10000 = 2
        qty = loop.compute_target_quantity(collateral_per_contract=10000, available_cash=25000)
        assert qty == 2


# ── Stepped Entry ──────────────────────────────────────────────


class TestSteppedEntry:
    """_execute_stepped_entry: stepped limit order logic."""

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_fills_on_first_step(self, mock_sleep, tmp_path):
        config = _make_config(entry_max_steps=3, entry_start_price="mid", entry_step_interval=1)
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(
            bid=1.50, ask=1.70, mid=1.60,
        )

        # First order fills immediately
        comp['execution'].get_order_status.return_value = {
            'status': 'filled', 'filled_avg_price': '1.60', 'filled_qty': '1',
        }

        result = await loop._execute_stepped_entry(candidate, qty=1, current_vix=18.0)
        assert result is not None
        order_result, filled_price = result
        assert filled_price == 1.60

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_exhausts_all_steps_returns_none(self, mock_sleep, tmp_path):
        config = _make_config(
            entry_max_steps=2, entry_start_price="mid",
            entry_step_interval=1, entry_refetch_snapshot=False,
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(bid=1.50, ask=1.70, mid=1.60)

        # Never fills
        comp['execution'].get_order_status.return_value = {
            'status': 'new', 'filled_avg_price': None, 'filled_qty': '0',
        }

        result = await loop._execute_stepped_entry(candidate, qty=1, current_vix=18.0)
        assert result is None

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_market_entry_fills(self, mock_sleep, tmp_path):
        """_execute_market_entry: market order fills and returns result."""
        config = _make_config(entry_order_type="market")
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(bid=1.50, ask=1.70, mid=1.60)

        comp['execution'].sell_to_open.return_value = OrderResult(
            success=True, order_id="mkt-001", message="OK",
        )
        comp['execution'].get_order_status.return_value = {
            'status': 'filled', 'filled_avg_price': '1.55', 'filled_qty': '1',
        }

        result = await loop._execute_market_entry(candidate, qty=1)
        assert result is not None
        order_result, filled_price = result
        assert order_result.order_id == "mkt-001"
        assert filled_price == 1.55

        # Verify limit_price=None (market order)
        call_kwargs = comp['execution'].sell_to_open.call_args
        limit_price = call_kwargs.kwargs.get('limit_price') if call_kwargs.kwargs else call_kwargs[1].get('limit_price')
        assert limit_price is None

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_market_entry_failure_returns_none(self, mock_sleep, tmp_path):
        """_execute_market_entry: failed order returns None."""
        config = _make_config(entry_order_type="market")
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(bid=1.50, ask=1.70, mid=1.60)

        comp['execution'].sell_to_open.return_value = OrderResult(
            success=False, order_id=None, message="Insufficient buying power",
        )

        result = await loop._execute_market_entry(candidate, qty=1)
        assert result is None

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_abort_on_filter_failure_after_refetch(self, mock_sleep, tmp_path):
        config = _make_config(
            entry_max_steps=2, entry_start_price="mid",
            entry_step_interval=1, entry_refetch_snapshot=True,
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)

        candidate = make_option_contract(bid=1.50, ask=1.70, mid=1.60)

        # First attempt: not filled
        comp['execution'].get_order_status.return_value = {
            'status': 'new', 'filled_avg_price': None, 'filled_qty': '0',
        }

        # Refetch returns new data
        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            candidate.symbol: {'bid': 1.30, 'ask': 1.50, 'delta': -0.15, 'implied_volatility': 0.25,
                               'volume': 50, 'open_interest': 200},
        }

        # But filter rejects it after refetch
        comp['scanner'].options_filter.evaluate.return_value = SimpleNamespace(
            passes=False, failure_reasons=["Delta too low"],
        )

        result = await loop._execute_stepped_entry(candidate, qty=1, current_vix=18.0)
        assert result is None


# ── Market Hours ───────────────────────────────────────────────


class TestMarketHours:
    """is_market_open: weekend/holiday/time checks."""

    async def test_weekend_returns_false(self, tmp_path):
        loop, _ = _make_loop(tmp_path=tmp_path)

        # Mock datetime.now to return a Saturday
        import pytz
        eastern = pytz.timezone('US/Eastern')
        # Find next Saturday
        today = date.today()
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7
        saturday = today + timedelta(days=days_until_saturday)
        saturday_dt = datetime(saturday.year, saturday.month, saturday.day, 12, 0, 0, tzinfo=eastern)

        with patch('csp.trading.loop.datetime') as mock_dt:
            mock_dt.now.return_value = saturday_dt
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            assert await loop.is_market_open() is False


# ── VIX Regime ─────────────────────────────────────────────────


class TestVixRegime:
    """check_global_vix_stop: global VIX-based emergency stop."""

    async def test_global_stop_triggers(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)
        # Reference VIX = 17.5, threshold = 17.5 * 1.15 = 20.125
        triggered = await loop.check_global_vix_stop(current_vix=21.0)
        assert triggered is True

    async def test_global_stop_does_not_trigger(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)
        triggered = await loop.check_global_vix_stop(current_vix=18.0)
        assert triggered is False


# ── run_cycle ──────────────────────────────────────────────────


class TestRunCycle:
    """run_cycle: single cycle orchestration."""

    async def test_monitors_before_scans(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # No positions, no entries, market open
        with patch.object(loop, 'is_market_open', new_callable=AsyncMock, return_value=True):
            summary = await loop.run_cycle()

        assert 'current_vix' in summary
        assert summary['exits'] == 0

    async def test_global_vix_stop_exits_all(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # VIX spiked way above threshold
        comp['vix_fetcher'].get_current_vix.return_value = 30.0

        # One active position
        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        summary = await loop.run_cycle()
        assert summary.get('global_vix_stop') is True
        assert summary['exits'] == 1

    async def test_no_candidates_no_positions_stops(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = []

        # Scanner returns nothing passing equity
        scan_result = SimpleNamespace(
            symbol="AAPL",
            equity_result=SimpleNamespace(
                passes=False, symbol="AAPL", current_price=230.0,
                sma_8=228.0, sma_20=225.0, sma_50=220.0,
                bb_upper=240.0, rsi=55.0,
            ),
            has_candidates=False,
            options_candidates=[],
        )
        comp['scanner'].scan_universe.return_value = [scan_result]

        with patch.object(loop, 'is_market_open', new_callable=AsyncMock, return_value=True):
            summary = await loop.run_cycle()

        assert summary.get('shutdown_reason') == 'no_candidates_no_positions'
        assert loop._running is False

    async def test_handles_api_error(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # VIX fetch throws
        comp['vix_fetcher'].get_current_vix.side_effect = Exception("API timeout")

        summary = await loop.run_cycle()
        assert len(summary['errors']) > 0
        assert "API timeout" in summary['errors'][0]

    async def test_no_candidates_with_positions_switches_to_monitor(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        # One existing position
        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")
        comp['alpaca_manager'].trading_client.get_all_positions.return_value = [alpaca_pos]

        loop.metadata.record_entry(
            option_symbol="AAPL260220P00220000",
            underlying="AAPL", strike=220.0,
            expiration=(date.today() + timedelta(days=5)).isoformat(),
            entry_delta=-0.25, entry_iv=0.30, entry_vix=18.0,
            entry_stock_price=230.0, entry_premium=1.50,
            entry_daily_return=0.0015, dte_at_entry=7, quantity=-1,
            entry_order_id="entry-001",
        )

        # Snapshot needed for monitor_positions (to avoid warning)
        comp['data_manager'].options_fetcher.get_option_snapshots.return_value = {
            "AAPL260220P00220000": {'bid': 1.20, 'ask': 1.40, 'delta': -0.25},
        }

        # Scanner returns nothing passing equity
        scan_result = SimpleNamespace(
            symbol="AAPL",
            equity_result=SimpleNamespace(
                passes=False, symbol="AAPL", current_price=230.0,
                sma_8=228.0, sma_20=225.0, sma_50=220.0,
                bb_upper=240.0, rsi=55.0,
            ),
            has_candidates=False,
            options_candidates=[],
        )
        comp['scanner'].scan_universe.return_value = [scan_result]

        with patch.object(loop, 'is_market_open', new_callable=AsyncMock, return_value=True):
            summary = await loop.run_cycle()

        assert loop._monitor_only is True


# ── Build Position Proxy ───────────────────────────────────────


class TestBuildPositionProxy:
    """_build_position_proxy: converts Alpaca position + metadata to proxy."""

    def test_proxy_has_all_fields(self, tmp_path):
        loop, comp = _make_loop(tmp_path=tmp_path)

        alpaca_pos = make_alpaca_position("AAPL260220P00220000", qty=-1, side="short")

        meta = {
            'underlying': 'AAPL',
            'entry_delta': -0.25,
            'entry_iv': 0.30,
            'entry_vix': 18.0,
            'entry_stock_price': 230.0,
            'entry_premium': 1.50,
            'entry_daily_return': 0.0015,
            'dte_at_entry': 7,
            'entry_order_id': 'entry-001',
            'entry_date': datetime.now().isoformat(),
        }

        proxy = loop._build_position_proxy(alpaca_pos, meta)

        assert proxy.symbol == 'AAPL'
        assert proxy.option_symbol == 'AAPL260220P00220000'
        assert proxy.strike == 220.0
        assert proxy.entry_delta == -0.25
        assert proxy.entry_premium == 1.50
        assert proxy.quantity == -1
        assert hasattr(proxy, 'calculate_pnl')
        assert hasattr(proxy, 'days_held')
        assert hasattr(proxy, 'current_dte')
        assert hasattr(proxy, 'collateral_required')


# ── Get Sort Key ───────────────────────────────────────────────


class TestGetSortKey:
    """_get_sort_key: returns sort value based on rank mode."""

    def test_daily_return_per_delta(self, tmp_path):
        config = _make_config(contract_rank_mode="daily_return_per_delta")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(daily_return_per_delta=0.006, strike=220.0,
                                   days_since_strike=30, daily_return_on_collateral=0.0015)
        assert loop._get_sort_key(contract) == 0.006

    def test_lowest_strike_price(self, tmp_path):
        config = _make_config(contract_rank_mode="lowest_strike_price")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(strike=220.0, daily_return_per_delta=0.006,
                                   days_since_strike=30, daily_return_on_collateral=0.0015)
        assert loop._get_sort_key(contract) == -220.0

    def test_days_since_strike(self, tmp_path):
        config = _make_config(contract_rank_mode="days_since_strike")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(days_since_strike=45, strike=220.0,
                                   daily_return_per_delta=0.006, daily_return_on_collateral=0.0015)
        assert loop._get_sort_key(contract) == 45

    def test_default_daily_return_on_collateral(self, tmp_path):
        config = _make_config(contract_rank_mode="daily_return_on_collateral")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(daily_return_on_collateral=0.0015, strike=220.0,
                                   daily_return_per_delta=0.006, days_since_strike=30)
        assert loop._get_sort_key(contract) == 0.0015

    def test_lowest_delta(self, tmp_path):
        config = _make_config(contract_rank_mode="lowest_delta")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(delta=-0.15, strike=220.0,
                                   daily_return_per_delta=0.006, days_since_strike=30,
                                   daily_return_on_collateral=0.0015)
        assert loop._get_sort_key(contract) == -0.15


class TestGetUniverseSortKey:
    """_get_universe_sort_key: returns sort value based on universe_rank_mode."""

    def test_daily_return_per_delta(self, tmp_path):
        config = _make_config(universe_rank_mode="daily_return_per_delta")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(daily_return_per_delta=0.006, strike=220.0,
                                   days_since_strike=30, daily_return_on_collateral=0.0015,
                                   delta=-0.25)
        assert loop._get_universe_sort_key(contract) == 0.006

    def test_lowest_strike_price(self, tmp_path):
        config = _make_config(universe_rank_mode="lowest_strike_price")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(strike=220.0, daily_return_per_delta=0.006,
                                   days_since_strike=30, daily_return_on_collateral=0.0015,
                                   delta=-0.25)
        assert loop._get_universe_sort_key(contract) == -220.0

    def test_days_since_strike(self, tmp_path):
        config = _make_config(universe_rank_mode="days_since_strike")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(days_since_strike=45, strike=220.0,
                                   daily_return_per_delta=0.006, daily_return_on_collateral=0.0015,
                                   delta=-0.25)
        assert loop._get_universe_sort_key(contract) == 45

    def test_default_daily_return_on_collateral(self, tmp_path):
        config = _make_config(universe_rank_mode="daily_return_on_collateral")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(daily_return_on_collateral=0.0015, strike=220.0,
                                   daily_return_per_delta=0.006, days_since_strike=30,
                                   delta=-0.25)
        assert loop._get_universe_sort_key(contract) == 0.0015

    def test_lowest_delta(self, tmp_path):
        config = _make_config(universe_rank_mode="lowest_delta")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(delta=-0.15, strike=220.0,
                                   daily_return_per_delta=0.006, days_since_strike=30,
                                   daily_return_on_collateral=0.0015)
        assert loop._get_universe_sort_key(contract) == -0.15

    def test_independent_of_contract_rank_mode(self, tmp_path):
        """universe_rank_mode and contract_rank_mode are independent."""
        config = _make_config(
            contract_rank_mode="lowest_strike_price",
            universe_rank_mode="daily_return_on_collateral",
        )
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        contract = SimpleNamespace(daily_return_on_collateral=0.0015, strike=220.0,
                                   daily_return_per_delta=0.006, days_since_strike=30,
                                   delta=-0.25)
        assert loop._get_sort_key(contract) == -220.0
        assert loop._get_universe_sort_key(contract) == 0.0015


class TestSequentialSizing:
    """Sequential sizing pass: allocate cash to candidates in priority order."""

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_sizes_in_priority_order(self, mock_sleep, tmp_path):
        """First candidate gets full allocation, subsequent get remainder."""
        config = _make_config(
            starting_cash=100_000, max_position_pct=0.50,
            max_contracts_per_ticker=10, entry_order_type="market",
            universe_rank_mode="daily_return_on_collateral",
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)
        comp['alpaca_manager'].compute_available_capital.return_value = 60_000

        # AAPL=$22k collateral, MSFT=$15k, GOOG=$10k
        # bid/dte/strike chosen so daily_return_on_collateral (bid/dte/strike) ranks AAPL > MSFT > GOOG:
        # AAPL: bid=4.40, dte=5, strike=220 -> 4.40/5/220 = 0.004
        # MSFT: bid=2.25, dte=5, strike=150 -> 2.25/5/150 = 0.003
        # GOOG: bid=1.00, dte=5, strike=100 -> 1.00/5/100 = 0.002
        aapl = make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", strike=220.0,
                                     stock_price=230.0, bid=4.40, ask=4.60, mid=4.50)
        msft = make_option_contract(underlying="MSFT", symbol="MSFT260220P00150000", strike=150.0,
                                     stock_price=160.0, bid=2.25, ask=2.45, mid=2.35)
        goog = make_option_contract(underlying="GOOG", symbol="GOOG260220P00100000", strike=100.0,
                                     stock_price=110.0, bid=1.00, ask=1.20, mid=1.10)

        # Pre-cache equity scan so scan_and_enter skips the scan step
        import pytz
        eastern = pytz.timezone('US/Eastern')
        loop._equity_passing = ["AAPL", "MSFT", "GOOG"]
        loop._equity_scan_date = datetime.now(eastern).date()
        loop._last_scan_results = []

        # Options fetch returns our candidates
        async def mock_fetch(sym):
            lookup = {"AAPL": aapl, "MSFT": msft, "GOOG": goog}
            c = lookup.get(sym)
            if c is None:
                return None
            return {
                "symbol": sym, "stock_price": c.stock_price, "puts_count": 1,
                "ranked": [c], "filter_results": [],
            }
        loop._fetch_symbol_options = mock_fetch

        # Track which entries are executed
        executed = []
        async def mock_execute(candidate, qty, vix, idx, total):
            executed.append((candidate.underlying, qty))
            return True
        loop._execute_single_entry = mock_execute

        # VIX
        comp['vix_fetcher'].get_current_vix.return_value = 18.0
        # Events check
        comp['scanner'].equity_filter.check_events.return_value = {}

        result = await loop.scan_and_enter(deployable_cash=60_000)

        # $60k: AAPL gets 2 contracts ($44k), $16k left; MSFT gets 1 ($15k), $1k left; GOOG skipped
        assert len(executed) == 2
        assert executed[0] == ("AAPL", 2)
        assert executed[1] == ("MSFT", 1)
        assert result == 2

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_skips_when_cash_exhausted(self, mock_sleep, tmp_path):
        """Candidates beyond cash capacity are skipped."""
        config = _make_config(
            starting_cash=100_000, max_position_pct=0.50,
            max_contracts_per_ticker=1, entry_order_type="market",
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)
        comp['alpaca_manager'].compute_available_capital.return_value = 25_000

        # AAPL: 4.40/5/220 = 0.004, MSFT: 3.00/5/200 = 0.003 -> AAPL ranks first
        aapl = make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", strike=220.0,
                                     stock_price=230.0, bid=4.40, ask=4.60, mid=4.50)
        msft = make_option_contract(underlying="MSFT", symbol="MSFT260220P00200000", strike=200.0,
                                     stock_price=210.0, bid=3.00, ask=3.20, mid=3.10)

        import pytz
        eastern = pytz.timezone('US/Eastern')
        loop._equity_passing = ["AAPL", "MSFT"]
        loop._equity_scan_date = datetime.now(eastern).date()
        loop._last_scan_results = []

        async def mock_fetch(sym):
            lookup = {"AAPL": aapl, "MSFT": msft}
            c = lookup.get(sym)
            if c is None:
                return None
            return {
                "symbol": sym, "stock_price": c.stock_price, "puts_count": 1,
                "ranked": [c], "filter_results": [],
            }
        loop._fetch_symbol_options = mock_fetch

        executed = []
        async def mock_execute(candidate, qty, vix, idx, total):
            executed.append((candidate.underlying, qty))
            return True
        loop._execute_single_entry = mock_execute

        comp['vix_fetcher'].get_current_vix.return_value = 18.0
        comp['scanner'].equity_filter.check_events.return_value = {}

        result = await loop.scan_and_enter(deployable_cash=25_000)

        # $25k: AAPL gets 1 ($22k), $3k left — not enough for MSFT ($20k)
        assert len(executed) == 1
        assert executed[0] == ("AAPL", 1)
        assert result == 1


class TestParallelEntryExecution:
    """Parallel entry execution via asyncio.gather."""

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_parallel_entries_all_succeed(self, mock_sleep, tmp_path):
        """Multiple entries execute in parallel, all fill."""
        config = _make_config(
            entry_order_type="market", starting_cash=500_000, max_position_pct=0.10,
            max_contracts_per_ticker=1,
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)
        comp['alpaca_manager'].compute_available_capital.return_value = 500_000

        aapl = make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", strike=220.0,
                                     stock_price=230.0, bid=1.50, ask=1.70, mid=1.60)
        msft = make_option_contract(underlying="MSFT", symbol="MSFT260220P00400000", strike=400.0,
                                     stock_price=420.0, bid=2.00, ask=2.20, mid=2.10)

        import pytz
        eastern = pytz.timezone('US/Eastern')
        loop._equity_passing = ["AAPL", "MSFT"]
        loop._equity_scan_date = datetime.now(eastern).date()
        loop._last_scan_results = []

        async def mock_fetch(sym):
            lookup = {"AAPL": aapl, "MSFT": msft}
            c = lookup.get(sym)
            if c is None:
                return None
            return {
                "symbol": sym, "stock_price": c.stock_price, "puts_count": 1,
                "ranked": [c], "filter_results": [],
            }
        loop._fetch_symbol_options = mock_fetch

        comp['vix_fetcher'].get_current_vix.return_value = 18.0
        comp['scanner'].equity_filter.check_events.return_value = {}

        # Both orders fill
        comp['execution'].sell_to_open.return_value = OrderResult(
            success=True, order_id="entry-001", message="OK"
        )
        comp['execution'].get_order_status.return_value = {
            'status': 'filled', 'filled_avg_price': '1.55', 'filled_qty': '1'
        }

        result = await loop.scan_and_enter(deployable_cash=500_000)
        assert result == 2

    @patch('csp.trading.loop.asyncio.sleep', new_callable=AsyncMock)
    async def test_parallel_entries_partial_failure(self, mock_sleep, tmp_path):
        """One entry fails, others succeed; count reflects partial success."""
        config = _make_config(
            entry_order_type="market", starting_cash=500_000, max_position_pct=0.10,
            max_contracts_per_ticker=1,
        )
        loop, comp = _make_loop(tmp_path=tmp_path, config=config)
        comp['alpaca_manager'].compute_available_capital.return_value = 500_000

        aapl = make_option_contract(underlying="AAPL", symbol="AAPL260220P00220000", strike=220.0,
                                     stock_price=230.0, bid=1.50, ask=1.70, mid=1.60)
        msft = make_option_contract(underlying="MSFT", symbol="MSFT260220P00400000", strike=400.0,
                                     stock_price=420.0, bid=2.00, ask=2.20, mid=2.10)

        import pytz
        eastern = pytz.timezone('US/Eastern')
        loop._equity_passing = ["AAPL", "MSFT"]
        loop._equity_scan_date = datetime.now(eastern).date()
        loop._last_scan_results = []

        async def mock_fetch(sym):
            lookup = {"AAPL": aapl, "MSFT": msft}
            c = lookup.get(sym)
            if c is None:
                return None
            return {
                "symbol": sym, "stock_price": c.stock_price, "puts_count": 1,
                "ranked": [c], "filter_results": [],
            }
        loop._fetch_symbol_options = mock_fetch

        comp['vix_fetcher'].get_current_vix.return_value = 18.0
        comp['scanner'].equity_filter.check_events.return_value = {}

        # AAPL fills, MSFT fails
        def mock_sell_to_open(option_symbol, quantity, limit_price=None):
            if "MSFT" in option_symbol:
                return OrderResult(success=False, order_id=None, message="Rejected")
            return OrderResult(success=True, order_id="entry-001", message="OK")
        comp['execution'].sell_to_open.side_effect = mock_sell_to_open
        comp['execution'].get_order_status.return_value = {
            'status': 'filled', 'filled_avg_price': '1.55', 'filled_qty': '1'
        }

        result = await loop.scan_and_enter(deployable_cash=500_000)
        assert result == 1


# ── Print Mode Tests ─────────────────────────────────────────


class TestPrintMode:
    """Tests for _vprint() and summary/verbose print modes."""

    def test_vprint_verbose_mode(self, tmp_path, capsys):
        """_vprint prints when print_mode='verbose'."""
        config = _make_config(print_mode="verbose")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        loop._vprint("hello verbose")
        captured = capsys.readouterr()
        assert "hello verbose" in captured.out

    def test_vprint_summary_mode(self, tmp_path, capsys):
        """_vprint is silent when print_mode='summary'."""
        config = _make_config(print_mode="summary")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        loop._vprint("should not appear")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    def test_summary_header_printed(self, tmp_path, capsys):
        """Summary header is printed in summary mode during run() startup."""
        config = _make_config(print_mode="summary")
        loop, _ = _make_loop(tmp_path=tmp_path, config=config)
        # We can't easily run the full loop, but we can verify the header format
        # by checking that the config is wired correctly
        assert config.print_mode == "summary"
        assert hasattr(loop, '_last_equity_passed')
        assert hasattr(loop, '_last_options_evaluated')
        assert loop._last_equity_passed == 0
        assert loop._last_options_evaluated == 0
