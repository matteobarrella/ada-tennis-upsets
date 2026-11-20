"""Stage 1: obtain the data.

The ATP match files 2000-2024 are already stored in ../data, so by default this
script only checks that all 25 files are there. Files that are missing are
downloaded from the archival mirror of Jeff Sackmann's tennis_atp repository,
pinned to one commit so that the download is reproducible.

Why a mirror: the original repository (JeffSackmann/tennis_atp) was taken down
by its owner in September 2026. The mirror redistributes the same files under the
same licence (CC BY-NC-SA 4.0).

Run from any folder:  python code/01_get_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

MIRROR = "Aneeshers/tennis-sackmann-archive"
COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
BASE_URL = f"https://raw.githubusercontent.com/{MIRROR}/{COMMIT}/atp"
YEARS = range(2000, 2025)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    total = 0
    for y in YEARS:
        name = f"atp_matches_{y}.csv"
        path = DATA_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urlretrieve(f"{BASE_URL}/{name}", path)
        total += len(pd.read_csv(path, low_memory=False))
    print(f"{len(list(YEARS))} files in {DATA_DIR}, {total:,} match rows")


if __name__ == "__main__":
    main()
