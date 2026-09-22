#!/usr/bin/env python3
"""
Download the Jigsaw Toxic Comment Classification Challenge dataset into ./data.

The dataset is not committed to the repository (it is ~50 MB gzipped and subject to
Kaggle's competition rules). This script fetches it via the Kaggle API.

Prerequisites
-------------
1. Get the dataset from huggingface (or via the Kaggle API). You need:
    A Kaggle account that has accepted the competition rules:
   https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules
2. A Kaggle API token placed at ~/.kaggle/kaggle.json (Account -> Create New API Token).
3. `pip install kaggle` (already listed if you installed requirements.txt with extras).

Usage
-----
    python download_data.py

The script places the following gzipped CSVs in ./data:
    train.csv.gz, test.csv.gz, test_labels.csv.gz

If the Kaggle API is unavailable, download the files manually from the competition
"Data" tab, gzip them, and place them in ./data with the names above.
"""
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

COMP = "jigsaw-toxic-comment-classification-challenge"
FILES = ["train.csv", "test.csv", "test_labels.csv"]
DATA_DIR = Path(__file__).parent / "data"


def already_present() -> bool:
    return all((DATA_DIR / f"{f}.gz").exists() for f in FILES)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    if already_present():
        print("All data files already present in ./data — nothing to do.")
        return 0

    try:
        import kaggle  # noqa: F401
    except Exception:
        print(
            "The `kaggle` package is not installed, or ~/.kaggle/kaggle.json is missing.\n"
            "Install it (`pip install kaggle`), add your API token, accept the competition\n"
            "rules, then re-run. See the module README for manual-download instructions.",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading competition files for '{COMP}' ...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMP, "-p", str(DATA_DIR)],
        check=True,
    )

    # Kaggle delivers a single zip; extract, then gzip the CSVs we need.
    import zipfile

    zip_path = DATA_DIR / f"{COMP}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()

    for f in FILES:
        csv_path = DATA_DIR / f
        # Some files arrive as nested zips (e.g. train.csv.zip).
        nested = DATA_DIR / f"{f}.zip"
        if nested.exists():
            with zipfile.ZipFile(nested) as z:
                z.extractall(DATA_DIR)
            nested.unlink()
        if csv_path.exists():
            gz_path = DATA_DIR / f"{f}.gz"
            with open(csv_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            csv_path.unlink()
            print(f"  wrote {gz_path.relative_to(DATA_DIR.parent)}")

    if already_present():
        print("Done. Data is in ./data.")
        return 0
    print("Some files are still missing — please download manually.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
