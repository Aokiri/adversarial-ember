"""Shared fixtures for the src/model.py tests. Mechanical setup only for testing."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import src
from src import data, model

# Small enough that a full train + predict finishes in seconds, large enough that the class
# balance stays representative of train (~51% benign on this slice).
SLICE_ROWS: int = 30_000
DEV_FRAC: float = 0.1
SEED: int = 42
FAST_ROUNDS: int = 20


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Locate the directory holding the EMBER2024 .dat matrices.

    Returns:
        Path to data/ember2024, resolved from the installed package location so it does not
        depend on the working directory.
    """
    path = Path(src.__file__).resolve().parent.parent / "data" / "ember2024"
    if not (path / "X_train.dat").is_file():
        pytest.skip(f"EMBER2024 vectorized data not found at {path}")
    return path


@pytest.fixture(scope="session")
def train_slice(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Take the first SLICE_ROWS rows of the real train memmap.

    X stays a contiguous memmap view, so no feature data is copied into RAM.

    Returns:
        (X, y) with X of shape (SLICE_ROWS, n_features) and y of shape (SLICE_ROWS,).
    """
    X_full, y_full = data.load_subset(data_dir, "train")
    return X_full[:SLICE_ROWS], np.asarray(y_full[:SLICE_ROWS])


@pytest.fixture(scope="session")
def dev_split(train_slice: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build a stratified fit/dev split over the slice with src.data.make_dev_split.

    Returns:
        (train_idx, dev_idx), integer index arrays into the slice.
    """
    _, y = train_slice
    return data.make_dev_split(y, dev_frac=DEV_FRAC, seed=SEED)


@pytest.fixture(scope="session")
def fast_config() -> dict:
    """Published EMBER2024 params with the round count cut so tests run in seconds.

    Returns:
        A copy of model.EMBER2024_LGBM_PARAMS with fewer iterations and quiet logging.
    """
    config = dict(model.EMBER2024_LGBM_PARAMS)
    config.update({"num_iterations": FAST_ROUNDS, "verbosity": -1})
    return config


@pytest.fixture(scope="session")
def train_dataset(train_slice: tuple[np.ndarray, np.ndarray]) -> lgb.Dataset:
    """Bin the whole slice into a Dataset, in memory with no binary cache.

    Returns:
        A constructed lgb.Dataset over all SLICE_ROWS rows, ready for .subset().
    """
    X, y = train_slice
    return model.build_train_dataset(X, y, max_bin=model.DEFAULT_MAX_BIN)


@pytest.fixture(scope="session")
def trained_booster(
    train_dataset: lgb.Dataset,
    dev_split: tuple[np.ndarray, np.ndarray],
    fast_config: dict,
) -> lgb.Booster:
    """Train one booster on the slice, shared by tests that just need a fitted model.

    Session-scoped, so it is trained once. Tests that need to retrain under altered inputs
    should call model.train_baseline themselves rather than reuse this.

    Returns:
        The Booster returned by model.train_baseline under fast_config.
    """
    train_idx, dev_idx = dev_split
    return model.train_baseline(
        train_dataset,
        train_idx,
        dev_idx,
        config=fast_config,
        early_stopping_rounds=FAST_ROUNDS,
        log_period=0,
    )
