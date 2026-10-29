"""Download Budapest flats-for-rent dataset via kagglehub into data/raw/."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET_SLUG = "gaborvecsei/flats-to-rent-at-budapest"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset(force: bool = False) -> Path:
    """Download the Kaggle dataset and copy tabular files into data/raw/.

    Returns
    -------
    Path
        Directory containing the local raw files (data/raw).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_csvs = list(RAW_DIR.glob("*.csv"))
    if existing_csvs and not force:
        print(f"Raw data already present: {[p.name for p in existing_csvs]}")
        return RAW_DIR

    print(f"Downloading {DATASET_SLUG} via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Kagglehub cache path: {cache_path}")

    copied = []
    for src in cache_path.rglob("*"):
        if not src.is_file():
            continue
        # Prefer tabular files; skip large image assets by default
        if src.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".md"}:
            continue
        dest = RAW_DIR / src.name
        shutil.copy2(src, dest)
        copied.append(dest.name)

    if not copied:
        # Fallback: copy everything that is not an image
        for src in cache_path.rglob("*"):
            if src.is_file() and src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                dest = RAW_DIR / src.name
                shutil.copy2(src, dest)
                copied.append(dest.name)

    print(f"Copied to {RAW_DIR}: {copied}")
    return RAW_DIR


def find_primary_csv(raw_dir: Path | None = None) -> Path:
    """Return the primary tabular CSV path under data/raw/."""
    raw_dir = raw_dir or RAW_DIR
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Run: python -m data.download"
        )
    # Prefer filenames that look like listing tables
    preferred = [
        p
        for p in csvs
        if any(k in p.name.lower() for k in ("flat", "rent", "budapest", "listing", "data"))
    ]
    return preferred[0] if preferred else csvs[0]


if __name__ == "__main__":
    download_dataset()
    print("Primary CSV:", find_primary_csv())