#!/usr/bin/env python3
"""Fetch public source inputs. The supplied bundle already includes these bytes."""
from pathlib import Path
import argparse
import io
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing raw data explicitly")
    args = parser.parse_args()
    output = ROOT / "data/raw"
    output.mkdir(parents=True, exist_ok=True)
    direct = {
        "yield_monthly.csv": "https://drive.google.com/uc?export=download&id=1CdczrQXT6e07ayIczuATtDfYgJN2MLhQ",
        "fred_md_2026_07.csv": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-07-md.csv",
    }
    for name, url in direct.items():
        path = output / name
        if path.exists() and not args.force:
            print(f"Already present: {name}")
            continue
        body = fetch(url)
        if body.lstrip().startswith(b"<"):
            raise ValueError(f"Source returned HTML rather than CSV for {name}")
        path.write_bytes(body)
        print(f"Saved {name}")
    path = output / "fred_md_2025_07.csv"
    if not path.exists() or args.force:
        url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip"
        with zipfile.ZipFile(io.BytesIO(fetch(url))) as archive:
            path.write_bytes(archive.read("FRED-MD_2025m07.csv"))
        print("Saved fred_md_2025_07.csv")


if __name__ == "__main__":
    main()
