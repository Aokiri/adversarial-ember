"""One-time heavy pass: build the full-train binned Dataset, then train the baseline.

Safe to re-run: the binary Dataset is load-if-exists, so a second run skips the slow build.
Everything is logged to results/build_train.log and the three report numbers (build time,
build peak RAM, deterministic training time) are printed at the end.

Run in background from the repo root:
    Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" `
        -ArgumentList "scripts\build_baseline.py"
    Get-Content -Wait .\results\build_train.log
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import psutil

import src
from src import data, model

GB = 2**30
REPO = Path(src.__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "ember2024"
MODELS = REPO / "results" / "models"
TABLES = REPO / "results" / "tables"
BINARY = MODELS / "train_full.bin"
MODEL_OUT = MODELS / "baseline_chronological.txt"
CURVE_OUT = TABLES / "validation_curve.json"
LOG_PATH = REPO / "results" / "build_train.log"

_proc = psutil.Process()


class PeakRSS:
    """Background sampler of resident memory; .peak_gb is the max seen since reset()."""

    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval
        self._peak = 0
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop:
            try:
                self._peak = max(self._peak, _proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(self._interval)

    def reset(self) -> None:
        self._peak = _proc.memory_info().rss

    @property
    def peak_gb(self) -> float:
        return self._peak / GB


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("baseline")
    vm = psutil.virtual_memory()
    log.info("RAM total=%.1fGB available=%.1fGB | binary_exists=%s",
             vm.total / GB, vm.available / GB, BINARY.is_file())

    train_idx, dev_idx, split_meta = data.load_split(str(REPO / "results" / "split_chronological"))
    log.info("split loaded: n_fit=%d n_dev=%d type=%s", len(train_idx), len(dev_idx),
             split_meta.get("split_type"))

    X_train, y_train = data.load_subset(DATA_DIR, "train")
    log.info("train memmap: X%s y%s", X_train.shape, y_train.shape)

    peak = PeakRSS()

    # --- build (the slow, one-time pass; loads instantly if already cached) ---
    peak.reset()
    t0 = time.perf_counter()
    dataset = model.build_train_dataset(X_train, y_train, max_bin=model.DEFAULT_MAX_BIN,
                                        binary_path=BINARY)
    build_secs = time.perf_counter() - t0
    build_peak = peak.peak_gb
    log.info("BUILD done: n_data=%d in %.1fs | build peak RSS=%.2fGB (projection was ~3.9GB) | binary=%.2fGB",
             dataset.num_data(), build_secs, build_peak,
             BINARY.stat().st_size / GB if BINARY.is_file() else float("nan"))

    # --- train on fit, early stopping on dev, record the validation curve ---
    curve: dict = {}
    peak.reset()
    t0 = time.perf_counter()
    booster = model.train_baseline(dataset, train_idx, dev_idx, config=None,
                                   early_stopping_rounds=50, log_period=50, evals_result=curve)
    train_secs = time.perf_counter() - t0
    train_peak = peak.peak_gb
    log.info("TRAIN done: best_iteration=%d num_trees=%d in %.1fs (deterministic) | train peak RSS=%.2fGB",
             booster.best_iteration, booster.num_trees(), train_secs, train_peak)

    model.save_model(booster, MODEL_OUT, max_bin=model.DEFAULT_MAX_BIN)
    import json
    CURVE_OUT.write_text(json.dumps(curve, indent=2))
    log.info("saved model -> %s | curve -> %s", MODEL_OUT.name, CURVE_OUT.name)

    log.info("=== REPORT NUMBERS ===")
    log.info("build time            : %.1f s", build_secs)
    log.info("build peak RAM        : %.2f GB (vs ~3.9 GB projected)", build_peak)
    log.info("deterministic train   : %.1f s", train_secs)
    log.info("best_iteration        : %d", booster.best_iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
