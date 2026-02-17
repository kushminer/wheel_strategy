"""Tests for storage backend abstraction."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from csp.storage import (
    StorageBackend,
    LocalStorage,
    GCSStorage,
    build_storage_backend,
)


# ─── LocalStorage Tests ─────────────────────────────────────────────


class TestLocalStorageRead:
    def test_reads_existing_file(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"key": "value"}')
        backend = LocalStorage()
        assert backend.read(str(p)) == '{"key": "value"}'

    def test_read_missing_raises(self, tmp_path):
        backend = LocalStorage()
        with pytest.raises(FileNotFoundError):
            backend.read(str(tmp_path / "missing.json"))


class TestLocalStorageWrite:
    def test_creates_new_file(self, tmp_path):
        p = tmp_path / "new.json"
        backend = LocalStorage()
        backend.write(str(p), '{"a": 1}')
        assert p.read_text() == '{"a": 1}'

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("old")
        backend = LocalStorage()
        backend.write(str(p), "new")
        assert p.read_text() == "new"

    def test_write_is_atomic_no_tmp_left(self, tmp_path):
        p = tmp_path / "atomic.json"
        backend = LocalStorage()
        backend.write(str(p), "content")
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "atomic.json"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "deep" / "file.json"
        backend = LocalStorage()
        backend.write(str(p), "nested")
        assert p.read_text() == "nested"


class TestLocalStorageExists:
    def test_exists_true(self, tmp_path):
        p = tmp_path / "exists.json"
        p.write_text("data")
        assert LocalStorage().exists(str(p)) is True

    def test_exists_false(self, tmp_path):
        assert LocalStorage().exists(str(tmp_path / "nope.json")) is False


class TestLocalStorageMkdir:
    def test_creates_nested(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        LocalStorage().mkdir(str(target))
        assert target.is_dir()

    def test_idempotent(self, tmp_path):
        target = tmp_path / "idem"
        backend = LocalStorage()
        backend.mkdir(str(target))
        backend.mkdir(str(target))  # no error
        assert target.is_dir()


class TestLocalStorageListPrefix:
    def test_lists_files(self, tmp_path):
        (tmp_path / "a.json").write_text("1")
        (tmp_path / "b.json").write_text("2")
        result = LocalStorage().list_prefix(str(tmp_path))
        assert len(result) == 2
        assert any("a.json" in r for r in result)
        assert any("b.json" in r for r in result)

    def test_empty_directory(self, tmp_path):
        assert LocalStorage().list_prefix(str(tmp_path)) == []

    def test_nonexistent_directory(self, tmp_path):
        assert LocalStorage().list_prefix(str(tmp_path / "nope")) == []


class TestLocalStorageProtocol:
    def test_satisfies_protocol(self):
        assert isinstance(LocalStorage(), StorageBackend)


# ─── GCSStorage Tests (mocked) ──────────────────────────────────────


class TestGCSStorage:
    @patch("csp.storage.gcs_lib", create=True)
    def _make_gcs(self, mock_gcs_module):
        """Helper to build a GCSStorage with mocked google.cloud.storage."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_gcs_module.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google.cloud": MagicMock(), "google.cloud.storage": mock_gcs_module}):
            with patch("csp.storage.GCSStorage.__init__", lambda self, bucket_name, prefix="": None):
                gcs = GCSStorage.__new__(GCSStorage)
                gcs._client = mock_client
                gcs._bucket = mock_bucket
                gcs._prefix = "prod"
                return gcs, mock_bucket, mock_client

    def test_read_existing(self):
        gcs, bucket, _ = self._make_gcs()
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = '{"key": "val"}'
        bucket.blob.return_value = blob

        result = gcs.read("test.json")
        assert result == '{"key": "val"}'
        bucket.blob.assert_called_with("prod/test.json")

    def test_read_missing_raises(self):
        gcs, bucket, _ = self._make_gcs()
        blob = MagicMock()
        blob.exists.return_value = False
        bucket.blob.return_value = blob
        bucket.name = "test-bucket"

        with pytest.raises(FileNotFoundError):
            gcs.read("missing.json")

    def test_write_uploads(self):
        gcs, bucket, _ = self._make_gcs()
        blob = MagicMock()
        bucket.blob.return_value = blob

        gcs.write("data.json", '{"x": 1}')
        blob.upload_from_string.assert_called_once_with('{"x": 1}', content_type="application/json")

    def test_mkdir_is_noop(self):
        gcs, bucket, _ = self._make_gcs()
        gcs.mkdir("logs")  # should not raise or call anything
        bucket.blob.assert_not_called()

    def test_satisfies_protocol(self):
        gcs, _, _ = self._make_gcs()
        assert isinstance(gcs, StorageBackend)


# ─── build_storage_backend Tests ────────────────────────────────────


class TestBuildStorageBackend:
    def test_local_default(self):
        from csp.config import StrategyConfig
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            paper_trading=True,
        )
        backend = build_storage_backend(config)
        assert isinstance(backend, LocalStorage)

    def test_gcs_missing_bucket_raises(self):
        from csp.config import StrategyConfig
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            paper_trading=True,
            storage_backend="gcs",
            gcs_bucket_name=None,
        )
        with pytest.raises(ValueError, match="gcs_bucket_name"):
            build_storage_backend(config)

    def test_local_explicit(self):
        from csp.config import StrategyConfig
        config = StrategyConfig(
            ticker_universe=["AAPL"],
            paper_trading=True,
            storage_backend="local",
        )
        backend = build_storage_backend(config)
        assert isinstance(backend, LocalStorage)
