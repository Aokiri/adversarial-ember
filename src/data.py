import json 
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# thrember v3 PE feature dim. Verified == thrember.PEFeatureExtractor().dim (2568).
# Hardcoded so the rest of the code never imports thrember.
DIM: int = 2568
BENIGN: int = 0  # verified label encoding on disk

def load_subset(data_dir: str | Path, subset: str) -> tuple[np.memmap, np.memmap]:
    """Open one EMBER2024 subset as a lazy X memmap and a checked y array.

    Args:
        data_dir: Directory holding X_{subset}.dat and y_{subset}.dat.
        subset: One of "train", "test" (phase 1 does not use "challenge").

    Returns:
        (X, y): X is a read-only memmap of shape (N, DIM),
        y is an int32 array of shape (N,).

    Raises:
        FileNotFoundError: If either .dat file is missing.
        ValueError: If X and y row counts disagree, or y holds labels outside {0, 1}.

    Note:
        y is read as int32 (verified from the on-disk bytes). The {0, 1} check is what
        catches the float32/int32 landmine on thrember.
    """
    data_dir = Path(data_dir)
    X_path, y_path = data_dir / f"X_{subset}.dat", data_dir / f"y_{subset}.dat"
    if not X_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"missing {X_path} or {y_path}")
    
    X = np.memmap(X_path, dtype=np.float32, mode = 'r').reshape(-1, DIM)
    y = np.array(np.memmap(y_path, dtype=np.int32, mode = 'r')) # ~18MB into RAM once
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"row mismatch: X={X.shape[0]} y={y.shape[0]}")
    
    labels = set(np.unique(y).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"unexpected labels {labels}; y likely misread (dtype float32 to int32)")
    
    return X, y


def make_dev_split(y_train: np.ndarray, dev_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random dev split, computed on indices only.

    Args:
        y_train: Train labels, shape (N,). Only y is needed, never X.
        dev_frac: Fraction of train held out as dev (e.g. 0.1).
        seed: RNG seed for a reproducible split.

    Returns:
        (train_idx, dev_idx): sorted integer index arrays into the train rows, each
        preserving the 50/50 class balance.

    Note:
        sklearn only sees y and a length-only placeholder, so the 48 GB X is never
        materialized. Indices are sorted for sequential-ish memmap reads; row order
        is irrelevant to LightGBM.
    """
    y = np.asarray(y_train)
    placeholder = np.zeros(len(y), dtype=np.int8) # split() reads arg 1 only for n_samples
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)
    train_idx, dev_idx = next(splitter.split(placeholder, y))
    
    return np.sort(train_idx), np.sort(dev_idx)
    

def make_chronological_dev_split(train_week: np.ndarray, dev_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-respecting dev split: dev is the most recent dev_frac of train.

    Args:
        train_week: Sortable time key per train row, shape (N,) (e.g. week index or
            submission date), aligned to the X_train rows.
        dev_frac: Fraction held out as dev, taken from the latest end of time.

    Returns:
        (train_idx, dev_idx): sorted integer index arrays; dev holds the latest rows.

    Note:
        Holding out the latest weeks mimics the train->test temporal gap and keeps
        future data out of fit.
    """
    w = np.asarray(train_week)
    n_dev = int(round(dev_frac * len(w)))
    assert n_dev > 0, "n_dev must be greater than 0"
    order = np.argsort(w, kind="stable")
    
    return np.sort(order[:-n_dev]), np.sort(order[-n_dev:])
    

def save_split(path: str | Path, train_idx: np.ndarray, dev_idx: np.ndarray, seed: int, meta: dict) -> None:
    """Persist split indices and a manifest under results/.

    Args:
        path: Output path without extension; .npz and .json are written next to it.
        train_idx: Train-minus-dev indices.
        dev_idx: Dev indices.
        seed: Seed used to build the split, recorded in the manifest.
        meta: Extra manifest fields (feature version, benign count, split type, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), train_idx=train_idx, dev_idx=dev_idx)
    manifest = {"seed": seed, "n_train": len(train_idx), "n_dev": len(dev_idx), **meta}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    
    
def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load indices and manifest written by save_split.

    Args:
        path: Same path (without extension) passed to save_split.

    Returns:
        (train_idx, dev_idx, manifest).
    """
    path = Path(path)
    data = np.load(path.with_suffix(".npz"))
    manifest = json.loads(path.with_suffix(".json").read_text())
    return data["train_idx"], data["dev_idx"], manifest
    
    
def describe(y_train: np.ndarray, y_test: np.ndarray, train_idx: np.ndarray, dev_idx: np.ndarray) -> dict:
    """Summarize the split and assert no train/dev leakage.

    Args:
        y_train: Train labels, shape (N_train,).
        y_test: Test labels, shape (N_test,).
        train_idx: Train-minus-dev indices into y_train.
        dev_idx: Dev indices into y_train.

    Returns:
        A dict with per-split sizes, class counts, and benign counts.

    Raises:
        AssertionError: If any index appears in both train and dev.
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    overlap = np.intersect1d(train_idx, dev_idx)
    assert overlap.size == 0, f"train/dev overlap: {overlap.size} shared indices"
    
    y_fit, y_dev = y_train[train_idx], y_train[dev_idx]
    def counts(y: np.ndarray) -> dict:
        # Helper function for counts.
        u, c = np.unique(y, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))
    
    return {
        "n_train": len(train_idx), "n_dev": len(dev_idx), "n_test": len(y_test),
        "counts_fit": counts(y_fit), "counts_dev": counts(y_dev), "counts_test": counts(y_test),
        "benign_fit": int((y_fit == BENIGN).sum()),
        "benign_dev": int((y_dev == BENIGN).sum()),
        "benign_test": int((y_test == BENIGN).sum()),
    }
    
if __name__ == "__main__":
    print("Usage: import data.py and use any of its intern functions")