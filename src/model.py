"""LightGBM baseline plumbing for the EMBER2024 feature-space detector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

# Feature indices EMBER2024 declares categorical. Taken from thrember.train_model so the
# baseline stays comparable to the reference setup rather than silently differing from it.
CATEGORICAL_FEATURES: list[int] = [2, 3, 4, 5, 6, 701, 702]

# Published EMBER2024 baseline parameters, copied verbatim from examples/lgbm_config.json
# in FutureComputing4AI/EMBER2024 (the repo thrember is installed from, commit 0ef753e).
EMBER2024_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "task": "train",
    "boosting": "gbdt",
    "num_iterations": 500,
    "learning_rate": 0.1,
    "max_depth": -1,
    "num_leaves": 64,
    "tree_learner": "serial",
    "num_threads": 0,
    "device_type": "cpu",
    "seed": 0,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 0.001,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "bagging_seed": 0,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.9,
    "feature_fraction_seed": 0,
    "first_metric_only": True,
    "max_delta_step": 0,
    "lambda_l1": 0,
    "lambda_l2": 1.0,
    "verbosity": 2,
    "is_unbalance": True,
    "sigmoid": 1.0,
    "boost_from_average": True,
    "metric": ["auc", "binary_logloss", "binary_error"],
}

DEFAULT_MAX_BIN: int = 255

# Bit-reproducible training across thread counts. `deterministic` needs a forced histogram
# method; force_col_wise is the fast one for wide feature sets like EMBER's 2568.
DETERMINISM: dict[str, Any] = {"deterministic": True, "force_col_wise": True}


def _sidecar_path(path: Path) -> Path:
    """Path of the JSON manifest stored next to a binary or model file."""
    return path.with_suffix(path.suffix + ".json")


def _write_sidecar(path: Path, meta: dict[str, Any]) -> None:
    """Write the JSON manifest belonging to path."""
    _sidecar_path(path).write_text(json.dumps(meta, indent=2))


def _read_sidecar(path: Path) -> dict[str, Any]:
    """Read the JSON manifest belonging to path, or an empty dict if absent."""
    sidecar = _sidecar_path(path)
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {}


def build_train_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_bin: int = DEFAULT_MAX_BIN,
    binary_path: str | Path | None = None,
) -> lgb.Dataset:
    """Build, or load, the binned Dataset covering every train row.

    Args:
        X_train: Full contiguous train memmap of shape (N, n_features). Handed to LightGBM
            whole so it bins straight from the mapping; it is never fancy-indexed.
        y_train: Train labels of shape (N,).
        max_bin: Bin count, kept at 255 to match the EMBER2024 reference.
        binary_path: Where the binned Dataset is cached. If the file exists it is loaded
            instead of re-binning; otherwise the Dataset is built and saved there.

    Returns:
        A constructed Dataset over all N train rows, ready to be split with .subset().

    Raises:
        ValueError: If the cached binary was binned with a different max_bin. LightGBM
            refuses to reopen a binary under another max_bin, so this fails loudly rather
            than letting the abort surface from the C library.

    Note:
        Bin boundaries are derived over the full train set, dev rows included. That is an
        accepted and documented decision for this project, not an oversight.
    """
    path = Path(binary_path) if binary_path is not None else None

    if path is not None and path.is_file():
        stored = _read_sidecar(path).get("max_bin")
        if stored is not None and int(stored) != int(max_bin):
            raise ValueError(
                f"{path.name} was binned with max_bin={stored}; LightGBM cannot reopen it "
                f"as max_bin={max_bin}. Pass max_bin={stored} or delete the cached binary."
            )
        # The binary carries its own max_bin and LightGBM aborts on a mismatch, so the
        # value has to be passed back in explicitly here.
        return lgb.Dataset(str(path), params={"max_bin": max_bin}).construct()

    dataset = lgb.Dataset(
        X_train,
        label=np.asarray(y_train),
        params={"max_bin": max_bin},
        categorical_feature=CATEGORICAL_FEATURES,
    ).construct()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_binary(str(path))
        _write_sidecar(
            path,
            {
                "max_bin": int(max_bin),
                "n_rows": int(dataset.num_data()),
                "n_features": int(dataset.num_feature()),
                "categorical_feature": CATEGORICAL_FEATURES,
            },
        )
    return dataset


def train_baseline(
    train_dataset: lgb.Dataset,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    config: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    log_period: int = 50,
) -> lgb.Booster:
    """Train the baseline on the fit rows, early-stopping on dev.

    Args:
        train_dataset: Constructed Dataset over all train rows, from build_train_dataset.
        train_idx: Row indices of the fit part of the split.
        dev_idx: Row indices of the dev part, used only as the early-stopping validation set.
        config: LightGBM parameters; defaults to the published EMBER2024 set. 
        early_stopping_rounds: Rounds without improvement before training stops.
        log_period: Iterations between evaluation logs; 0 silences them.

    Returns:
        The trained Booster. best_iteration is set when early stopping fired, and is 0
        otherwise, which callers must treat as "use every iteration".

    Note:
        Both halves are .subset() views over the shared binned data, so no feature matrix is
        copied. The test set is unreachable from this function by construction.
    """
    params = dict(EMBER2024_LGBM_PARAMS if config is None else config)
    params.update(DETERMINISM)  # always deterministic, even under a custom config
    # num_iterations is an alias of num_boost_round; pass it explicitly so LightGBM cannot
    # end up with two disagreeing round counts.
    num_boost_round = int(params.pop("num_iterations", 500))

    fit_set = train_dataset.subset(np.asarray(train_idx))
    dev_set = train_dataset.subset(np.asarray(dev_idx))

    callbacks: list[Any] = [
        lgb.early_stopping(
            stopping_rounds=early_stopping_rounds,
            first_metric_only=bool(params.get("first_metric_only", False)),
        )
    ]
    if log_period:
        callbacks.append(lgb.log_evaluation(period=log_period))

    return lgb.train(
        params,
        train_set=fit_set,
        num_boost_round=num_boost_round,
        valid_sets=[dev_set],
        valid_names=["dev"],
        callbacks=callbacks,
    )


def predict_scores(
    booster: lgb.Booster,
    X: np.ndarray,
    idx: np.ndarray | None = None,
    batch_size: int = 20_000,
    num_iteration: int | None = None,
) -> np.ndarray:
    """Score rows in bounded batches straight off the memmap.

    Args:
        booster: Trained model.
        X: Feature memmap of shape (N, n_features).
        idx: Row indices to score; None scores every row in order.
        batch_size: Rows materialized per batch. 20k x 2568 float32 is ~205 MB.
        num_iteration: Iteration count to predict with. None falls back to the booster's
            best_iteration, or all iterations when early stopping never fired.

    Returns:
        Positive-class scores of shape (len(idx),), aligned elementwise with idx.
    """
    rows = np.arange(X.shape[0]) if idx is None else np.asarray(idx)
    if num_iteration is None:
        num_iteration = booster.best_iteration or None

    scores = np.empty(rows.shape[0], dtype=np.float64)
    for start in range(0, rows.shape[0], batch_size):
        batch = rows[start : start + batch_size]
        # Indexing ONE batch is bounded; doing this over a whole split would be ~43 GB.
        block = np.ascontiguousarray(X[batch])
        scores[start : start + batch.shape[0]] = booster.predict(
            block, num_iteration=num_iteration
        )
    return scores


def save_model(
    booster: lgb.Booster,
    path: str | Path,
    config: dict[str, Any] | None = None,
    max_bin: int = DEFAULT_MAX_BIN,
) -> None:
    """Persist the booster at its best iteration, plus what is needed to reproduce it.

    Args:
        booster: Trained model.
        path: Destination for the LightGBM text model; a .json manifest is written beside it.
        config: Parameters the model was trained with; defaults to the published set.
        max_bin: Bin count used to build the Dataset, recorded so the binary and the model
            cannot drift apart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_iteration = booster.best_iteration or None
    booster.save_model(str(path), num_iteration=best_iteration)
    _write_sidecar(
        path,
        {
            "best_iteration": best_iteration,
            "max_bin": int(max_bin),
            "config": EMBER2024_LGBM_PARAMS if config is None else config,
        },
    )


def load_model(path: str | Path) -> tuple[lgb.Booster, dict[str, Any]]:
    """Load a saved booster together with its manifest.

    Args:
        path: Path passed to save_model.

    Returns:
        (booster, manifest), where manifest holds best_iteration, max_bin and the config.
    """
    path = Path(path)
    return lgb.Booster(model_file=str(path)), _read_sidecar(path)
