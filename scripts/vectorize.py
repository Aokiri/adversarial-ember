import thrember
from pathlib import Path

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "ember2024"

if __name__ == '__main__':
    print(f"data_dir: {DATA_DIR}")
    thrember.create_vectorized_features(DATA_DIR, label_type="label")
    print("Vectorization complete")