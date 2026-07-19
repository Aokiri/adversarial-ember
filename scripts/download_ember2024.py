import thrember
from pathlib import Path

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "ember2024"

if __name__ == '__main__':
    print(f"Starting EMBER2024 PE download (all splits) to {DATA_DIR}")
    thrember.download_dataset(DATA_DIR, file_type="PE")
    print("Download complete")