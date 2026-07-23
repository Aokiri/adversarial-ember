"""Anchor tests for src/model.py."""
import lightgbm as lgb
import numpy as np

from src import model


def test_training_never_fits_on_dev_rows(
    train_slice: tuple[np.ndarray, np.ndarray],
    dev_split: tuple[np.ndarray, np.ndarray],
    fast_config: dict,
) -> None:
    """Flipping dev labels leaves the fitted trees unchanged; dev is validation only."""
    X, y = train_slice
    train_idx, dev_idx = dev_split

    y_corrupt = y.copy()
    y_corrupt[dev_idx] = 1 - y_corrupt[dev_idx]

    clean = model.train_baseline(model.build_train_dataset(X, y), train_idx, dev_idx,
                                 fast_config, log_period=0)
    corrupt = model.train_baseline(model.build_train_dataset(X, y_corrupt), train_idx, dev_idx,
                                   fast_config, log_period=0)

    # Early stopping picks best_iteration from dev and truncates the booster there, so the two
    # keep different tree counts. Tree 1 is built from fit rows only and both retain it; if dev
    # had leaked into the fit, flipping its labels would change that first tree.
    clean_t1 = model.predict_scores(clean, X, num_iteration=1)
    corrupt_t1 = model.predict_scores(corrupt, X, num_iteration=1)
    np.testing.assert_array_equal(clean_t1, corrupt_t1)


def test_predict_scores_is_invariant_to_batch_size(
    train_slice: tuple[np.ndarray, np.ndarray],
    dev_split: tuple[np.ndarray, np.ndarray],
    trained_booster: lgb.Booster,
) -> None:
    """Batch size only bounds memory; the scores it returns must not depend on it."""
    X, _ = train_slice
    _, dev_idx = dev_split

    many_batches = model.predict_scores(trained_booster, X, dev_idx, batch_size=512)
    single_batch = model.predict_scores(trained_booster, X, dev_idx, batch_size=len(dev_idx) + 1)
    np.testing.assert_array_equal(many_batches, single_batch)


def test_saved_model_round_trips(
    train_slice: tuple[np.ndarray, np.ndarray],
    dev_split: tuple[np.ndarray, np.ndarray],
    trained_booster: lgb.Booster,
    tmp_path,
) -> None:
    """A saved-then-reloaded booster reproduces the same scores and reports its manifest."""
    X, _ = train_slice
    _, dev_idx = dev_split
    path = tmp_path / "model.txt"

    model.save_model(trained_booster, path, max_bin=model.DEFAULT_MAX_BIN)
    reloaded, manifest = model.load_model(path)

    before = model.predict_scores(trained_booster, X, dev_idx)
    after = model.predict_scores(reloaded, X, dev_idx)
    np.testing.assert_array_equal(before, after)
    assert manifest["best_iteration"] == (trained_booster.best_iteration or None)
    assert manifest["max_bin"] == model.DEFAULT_MAX_BIN
    
    
def test_binary_cache_round_trips(
    train_slice: tuple[np.ndarray, np.ndarray],
    dev_split: tuple[np.ndarray, np.ndarray],
    fast_config: dict,
    tmp_path,
) -> None:
    """A model from the binary cache matches the raw build, with the load path exercised."""
    X, y = train_slice
    train_idx, dev_idx = dev_split
    binary_path = tmp_path / "train.bin"

    # This build is the raw baseline and also writes the binary.
    fresh = model.build_train_dataset(X, y, max_bin=model.DEFAULT_MAX_BIN, binary_path=binary_path)
    assert binary_path.is_file()

    # A different X: if the cache path regressed to rebuilding, cached would train on these
    # zeros and diverge, so loading the binary is required for the scores to match.
    cached = model.build_train_dataset(np.zeros_like(X), y, max_bin=model.DEFAULT_MAX_BIN,
                                       binary_path=binary_path)

    b_fresh = model.train_baseline(fresh, train_idx, dev_idx, fast_config, log_period=0)
    b_cached = model.train_baseline(cached, train_idx, dev_idx, fast_config, log_period=0)
    np.testing.assert_array_equal(
        model.predict_scores(b_fresh, X, dev_idx),
        model.predict_scores(b_cached, X, dev_idx),
    )