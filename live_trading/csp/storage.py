"""Storage backend abstraction for local filesystem and GCS."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from csp.config import StrategyConfig


@runtime_checkable
class StorageBackend(Protocol):
    """Interface for reading/writing strategy data."""

    def read(self, path: str) -> str: ...
    def write(self, path: str, data: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def mkdir(self, path: str) -> None: ...
    def list_prefix(self, prefix: str) -> list[str]: ...


class LocalStorage:
    """Filesystem storage backend (default)."""

    def read(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def write(self, path: str, data: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def mkdir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def list_prefix(self, prefix: str) -> list[str]:
        directory = prefix
        if not os.path.isdir(directory):
            return []
        return sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        )


class GCSStorage:
    """Google Cloud Storage backend for Cloud Run deployment."""

    def __init__(self, bucket_name: str, prefix: str = ""):
        from google.cloud import storage as gcs_lib
        self._client = gcs_lib.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/")

    def _blob_path(self, path: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def read(self, path: str) -> str:
        blob = self._bucket.blob(self._blob_path(path))
        if not blob.exists():
            raise FileNotFoundError(f"gs://{self._bucket.name}/{self._blob_path(path)}")
        return blob.download_as_text()

    def write(self, path: str, data: str) -> None:
        blob = self._bucket.blob(self._blob_path(path))
        blob.upload_from_string(data, content_type="application/json")

    def exists(self, path: str) -> bool:
        blob = self._bucket.blob(self._blob_path(path))
        return blob.exists()

    def mkdir(self, path: str) -> None:
        pass  # GCS has no directories

    def list_prefix(self, prefix: str) -> list[str]:
        full_prefix = self._blob_path(prefix)
        if not full_prefix.endswith("/"):
            full_prefix += "/"
        return [
            blob.name[len(self._prefix) + 1:] if self._prefix else blob.name
            for blob in self._client.list_blobs(self._bucket, prefix=full_prefix)
        ]


def build_storage_backend(config: StrategyConfig) -> StorageBackend:
    """Factory: build the right storage backend from config fields."""
    if config.storage_backend == "gcs":
        if not config.gcs_bucket_name:
            raise ValueError("gcs_bucket_name is required when storage_backend='gcs'")
        return GCSStorage(config.gcs_bucket_name, config.gcs_prefix)
    return LocalStorage()
