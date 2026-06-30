#!/usr/bin/env python
"""Download the Kaggle 'Theft Detection Scheme in Smart Grids' dataset.

Requires Kaggle API credentials (~/.kaggle/kaggle.json) and the dataset slug.
Set it in configs/smartgrid.yaml (dataset.kaggle_dataset: "owner/slug") or pass
--dataset. The first CSV found is copied to data/smartgrid/theft.csv.
"""
import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

import yaml

from fedcity.config import CONFIG_DIR, REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Kaggle dataset slug 'owner/slug'")
    args = ap.parse_args()

    cfg = yaml.safe_load((CONFIG_DIR / "smartgrid.yaml").read_text())
    slug = args.dataset or cfg["dataset"].get("kaggle_dataset", "")
    if not slug:
        raise SystemExit(
            "No Kaggle dataset slug. Set dataset.kaggle_dataset in configs/smartgrid.yaml "
            "or pass --dataset owner/slug (and place credentials at ~/.kaggle/kaggle.json)."
        )

    out_dir = REPO_ROOT / "data" / "smartgrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(out_dir), unzip=True)

    csvs = sorted(out_dir.glob("**/*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV found after download in {out_dir}")
    target = out_dir / "theft.csv"
    if csvs[0] != target:
        shutil.copy(csvs[0], target)
    print(f"Smart-grid data ready at {target} (source: {csvs[0].name})")


if __name__ == "__main__":
    main()
