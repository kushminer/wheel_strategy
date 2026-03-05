"""Databento OPRA data fetcher with chunked parallel downloads and incremental caching."""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import databento as db


def _get_client(api_key: str = None) -> db.Historical:
    key = api_key or os.environ.get("TQQQ_DATABENTO_API_KEY")
    if not key:
        raise ValueError(
            "No API key. Set TQQQ_DATABENTO_API_KEY in .env or pass api_key=."
        )
    return db.Historical(key)


def _build_chunks(start: str, end: str, freq: str = "QS") -> list[tuple[str, str]]:
    boundaries = pd.date_range(start, end, freq=freq).tolist()
    boundaries.append(pd.Timestamp(end))
    # Ensure start is always included
    start_ts = pd.Timestamp(start)
    if not boundaries or boundaries[0] > start_ts:
        boundaries.insert(0, start_ts)
    # Deduplicate and skip zero-length ranges
    pairs = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i].strftime("%Y-%m-%d")
        e = boundaries[i + 1].strftime("%Y-%m-%d")
        if s != e:
            pairs.append((s, e))
    return pairs


def fetch_chunked(
    symbol: str,
    schema: str,
    start: str,
    end: str,
    cache_dir: Path,
    label: str = "data",
    api_key: str = None,
    max_workers: int = 4,
    freq: str = "QS",
) -> pd.DataFrame:
    """
    Fetch Databento OPRA data in parallel quarterly chunks with incremental caching.

    Each chunk is saved to its own parquet file. If a chunk already exists on disk,
    it is skipped. After all chunks are fetched, they are merged into a single file.

    Args:
        symbol: Databento parent symbol, e.g. "TQQQ.OPT"
        schema: Databento schema, e.g. "definition" or "ohlcv-1d"
        start: Start date string, e.g. "2023-04-01"
        end: End date string, e.g. "2026-03-01"
        cache_dir: Root cache directory (chunk subfolder created automatically)
        label: Name prefix for chunk files and subfolder
        api_key: Optional API key override
        max_workers: Number of parallel download threads
        freq: Pandas date_range freq for chunking (default quarterly)

    Returns:
        Merged DataFrame of all chunks.
    """
    merged_path = cache_dir / f"{label}.parquet"

    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        print(f"[{label}] Loaded cached: {len(df):,} rows")
        return df

    client = _get_client(api_key)
    pairs = _build_chunks(start, end, freq)

    chunk_dir = cache_dir / f"{label}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def chunk_path(s, e):
        return chunk_dir / f"{label}_{s}_{e}.parquet"

    missing = [(s, e) for s, e in pairs if not chunk_path(s, e).exists()]
    cached = [(s, e) for s, e in pairs if chunk_path(s, e).exists()]

    if cached:
        print(f"[{label}] {len(cached)} chunks cached, {len(missing)} remaining")

    def fetch_one(s, e):
        df = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            symbols=[symbol],
            stype_in="parent",
            schema=schema,
            start=s,
            end=e,
        ).to_df()
        df.to_parquet(chunk_path(s, e))
        return s, e, len(df)

    if missing:
        print(f"[{label}] Fetching {len(missing)} chunks (workers={max_workers})...")
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_one, s, e): (s, e) for s, e in missing}
            for f in as_completed(futures):
                s, e = futures[f]
                try:
                    _, _, n = f.result()
                    print(f"  {s} → {e}: {n:,} rows")
                except Exception as ex:
                    errors.append((s, e, str(ex)))
                    print(f"  FAILED {s} → {e}: {ex}")

        if errors:
            print(f"\n[{label}] {len(errors)} chunks failed:")
            for s, e, err in errors:
                print(f"  {s} → {e}: {err}")

    # Merge whatever chunks exist
    all_frames = [
        pd.read_parquet(chunk_path(s, e))
        for s, e in pairs
        if chunk_path(s, e).exists()
    ]

    if not all_frames:
        raise RuntimeError(
            f"[{label}] No chunks completed. Check errors above and re-run."
        )

    completed = len(all_frames)
    total = len(pairs)
    df = pd.concat(all_frames)
    df.to_parquet(merged_path)
    print(f"[{label}] Merged {completed}/{total} chunks → {len(df):,} rows")
    return df


def check_cost(
    symbol: str,
    schema: str,
    start: str,
    end: str,
    api_key: str = None,
) -> float:
    """Check cost of a Databento query before fetching."""
    client = _get_client(api_key)
    cost = client.metadata.get_cost(
        dataset="OPRA.PILLAR",
        symbols=[symbol],
        stype_in="parent",
        schema=schema,
        start=start,
        end=end,
    )
    print(f"Cost for {schema} ({start} → {end}): ${cost:.2f}")
    return cost


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    CACHE = Path(__file__).parent / "cache"
    CACHE.mkdir(exist_ok=True)
    SYMBOL = "TQQQ.OPT"

    # Test with a small date range first
    print("=== Cost Check ===")
    check_cost(SYMBOL, "definition", "2025-01-01", "2025-04-01")

    print("\n=== Fetch Test (1 quarter) ===")
    df = fetch_chunked(
        symbol=SYMBOL,
        schema="definition",
        start="2025-01-01",
        end="2025-04-01",
        cache_dir=CACHE,
        label="test_defs",
        max_workers=2,
    )
    print(f"Result: {len(df):,} rows, columns: {list(df.columns)[:8]}")
    print("PASS")
