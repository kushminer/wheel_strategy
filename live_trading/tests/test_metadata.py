"""Unit tests for StrategyMetadataStore."""

import json
import os
import pytest

from csp.trading.metadata import StrategyMetadataStore


@pytest.fixture
def store(tmp_path):
    """Fresh metadata store in temp directory."""
    return StrategyMetadataStore(path=str(tmp_path / "metadata.json"))


def _record_sample_entry(store, option_symbol="AAPL260220P00220000", underlying="AAPL"):
    """Helper to record a standard entry."""
    store.record_entry(
        option_symbol,
        underlying=underlying,
        strike=220.0,
        expiration="2026-02-20",
        entry_delta=-0.25,
        entry_iv=0.30,
        entry_vix=18.0,
        entry_stock_price=230.0,
        entry_premium=1.50,
        entry_daily_return=0.0015,
        dte_at_entry=7,
        quantity=1,
        entry_order_id="order-123",
    )


class TestRecordAndGet:
    def test_record_entry_then_get(self, store):
        _record_sample_entry(store)
        meta = store.get("AAPL260220P00220000")
        assert meta is not None
        assert meta["underlying"] == "AAPL"
        assert meta["strike"] == 220.0
        assert meta["entry_delta"] == -0.25
        assert meta["exit_date"] is None

    def test_get_unknown_returns_none(self, store):
        assert store.get("UNKNOWN") is None

    def test_record_exit_updates_fields(self, store):
        _record_sample_entry(store)
        store.record_exit(
            "AAPL260220P00220000",
            exit_reason="delta_exceeded_absolute",
            exit_details="Delta hit 0.42",
            exit_order_id="exit-456",
        )
        meta = store.get("AAPL260220P00220000")
        assert meta["exit_date"] is not None
        assert meta["exit_reason"] == "delta_exceeded_absolute"
        assert meta["exit_order_id"] == "exit-456"


class TestGetActive:
    def test_returns_only_active(self, store):
        _record_sample_entry(store, "OPT1", "AAPL")
        _record_sample_entry(store, "OPT2", "MSFT")
        store.record_exit("OPT1", exit_reason="expired")
        active = store.get_active()
        assert "OPT1" not in active
        assert "OPT2" in active

    def test_empty_when_all_exited(self, store):
        _record_sample_entry(store)
        store.record_exit("AAPL260220P00220000", exit_reason="expired")
        assert len(store.get_active()) == 0


class TestHasSymbol:
    def test_true_for_active(self, store):
        _record_sample_entry(store)
        assert store.has_symbol("AAPL") is True

    def test_false_for_exited(self, store):
        _record_sample_entry(store)
        store.record_exit("AAPL260220P00220000", exit_reason="expired")
        assert store.has_symbol("AAPL") is False

    def test_false_for_unknown(self, store):
        assert store.has_symbol("TSLA") is False


class TestPersistence:
    def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "metadata.json")
        store1 = StrategyMetadataStore(path=path)
        _record_sample_entry(store1)

        # Re-instantiate and verify
        store2 = StrategyMetadataStore(path=path)
        assert store2.get("AAPL260220P00220000") is not None

    def test_atomic_write_no_tmp_left(self, store):
        _record_sample_entry(store)
        assert not os.path.exists(store.path + ".tmp")

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = str(tmp_path / "metadata.json")
        with open(path, "w") as f:
            f.write("{bad json!!!")
        store = StrategyMetadataStore(path=path)
        assert len(store.entries) == 0


class TestEdgeCases:
    def test_exit_on_unknown_creates_entry(self, store):
        store.record_exit("UNKNOWN_OPT", exit_reason="manual")
        meta = store.get("UNKNOWN_OPT")
        assert meta is not None
        assert meta["exit_reason"] == "manual"

    def test_multiple_symbols(self, store):
        _record_sample_entry(store, "OPT_AAPL", "AAPL")
        _record_sample_entry(store, "OPT_MSFT", "MSFT")
        _record_sample_entry(store, "OPT_GOOG", "GOOG")
        assert len(store.entries) == 3
        assert store.has_symbol("AAPL") is True
        assert store.has_symbol("MSFT") is True
        assert store.has_symbol("GOOG") is True
