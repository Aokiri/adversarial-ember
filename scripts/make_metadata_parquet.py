"""Dump EMBER2024 per-sample metadata to Parquet, one file per subset.

Why this exists: the raw *.jsonl hold the only copy of the sample metadata
(sha256, timestamps, label, family, ...). The vectorized X_*.dat / y_*.dat keep
features and labels but NOT the metadata, and there is no join key back to the raw
files other than row order. Before deleting the ~104 GB of raw *.jsonl, this
snapshots the metadata into compact Parquet, row-aligned to the .dat matrices.

Alignment: rows are emitted in exactly the order thrember used to vectorize
(gather_feature_paths -> sorted(os.listdir), then line order within each file), so
row i of {subset}_metadata.parquet matches row i of X_{subset}.dat. After each
subset the script asserts the row count equals X_{subset}.dat's, so a misalignment
aborts BEFORE anything is deleted. thrember's own helpers are reused on purpose:
they are what guarantees the identical ordering and field extraction.

Run in background and watch the log (PowerShell):
    Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" `
        -ArgumentList "scripts\make_metadata_parquet.py"
    Get-Content -Wait .\data\make_metadata_parquet.log
"""
from __future__ import annotations

import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from thrember.model import (
    ORDERED_COLUMNS,
    gather_feature_paths,
    raw_feature_iterator,
    read_metadata_record,
)

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "ember2024"
LOG_PATH: Path = DATA_DIR.parent / "make_metadata_parquet.log"
SUBSETS: list[str] = ["train", "test", "challenge"]  # challenge is preserved but unused in phase 1
DIM: int = 2568  # thrember v3 PE feature dim; only used to check row alignment vs X_*.dat
WORKERS: int = min(os.cpu_count() or 4, 8)  # capped to avoid memory spikes on the ~1 GB files


def _expected_rows(subset: str) -> int | None:
    """Row count implied by X_{subset}.dat's size, or None if that file is absent.

    Args:
        subset: Subset name.

    Returns:
        Number of rows (file size // (DIM * 4 bytes)) or None.
    """
    x_path = DATA_DIR / f"X_{subset}.dat"
    if not x_path.is_file():
        return None
    return x_path.stat().st_size // (DIM * 4)


def _process_file(path: Path) -> pl.DataFrame:
    """Parse one .jsonl into a metadata DataFrame, preserving line order.

    Only the metadata fields are kept; json.loads parses each full line but the big
    feature payload is dropped immediately, so retained memory stays small.

    Args:
        path: Path to a raw feature .jsonl file.

    Returns:
        A DataFrame with whatever metadata columns are present in that file; the
        schema is unified across files by the caller.
    """
    records = [read_metadata_record(line) for line in raw_feature_iterator([path])]
    return pl.DataFrame(records)


def dump_subset(subset: str, log: logging.Logger) -> None:
    """Write {subset}_metadata.parquet, row-aligned to X_{subset}.dat.

    Args:
        subset: One of "train", "test", "challenge".
        log: Logger for progress and errors.

    Raises:
        AssertionError: If the parquet row count does not match X_{subset}.dat.
    """
    t0 = time.perf_counter()
    paths = gather_feature_paths(DATA_DIR, subset)
    log.info("[%s] %d files, %d workers", subset, len(paths), WORKERS)

    frames: list[pl.DataFrame] = []
    total = 0
    with Pool(processes=WORKERS) as pool:
        for i, df in enumerate(pool.imap(_process_file, paths), start=1):  # imap keeps input order
            frames.append(df)
            total += df.height
            log.info("[%s] file %d/%d done rows=%d cumulative=%d",
                     subset, i, len(paths), df.height, total)

    df = pl.concat(frames, how="diagonal_relaxed")  # union columns across files, upcast dtypes
    missing = [c for c in ORDERED_COLUMNS if c not in df.columns]
    if missing:
        log.warning("[%s] columns absent across all files, filled null: %s", subset, missing)
        df = df.with_columns([pl.lit(None).alias(c) for c in missing])
    df = df.select(ORDERED_COLUMNS).with_columns(subset=pl.lit(subset))

    expected = _expected_rows(subset)
    if expected is None:
        log.warning("[%s] X_%s.dat absent; cannot verify alignment", subset, subset)
    else:
        assert df.height == expected, (
            f"[{subset}] row mismatch vs X_{subset}.dat: parquet={df.height} dat={expected}. "
            "NOT row-aligned; do not delete the raw .jsonl."
        )
        log.info("[%s] row count %d matches X_%s.dat", subset, df.height, subset)

    out_path = DATA_DIR / f"{subset}_metadata.parquet"
    df.write_parquet(out_path)
    log.info("[%s] wrote %s (%d rows, %.1f MB) in %.1f s",
             subset, out_path.name, df.height, out_path.stat().st_size / 1e6,
             time.perf_counter() - t0)


def main() -> int:
    """Dump every subset in SUBSETS, logging to file and stdout.

    Returns:
        0 on success, 1 if any subset failed (traceback is logged).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("metadata")
    log.info("data_dir=%s subsets=%s", DATA_DIR, SUBSETS)

    t0 = time.perf_counter()
    try:
        for subset in SUBSETS:
            dump_subset(subset, log)
    except Exception:  # log the full traceback so a background run leaves a trace
        log.exception("FAILED")
        return 1

    log.info("all done in %.1f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
