#!/usr/bin/env python
"""Fetch the Breast Cancer Wisconsin Diagnostic dataset (UCI id=17) and cache a CSV.

ucimlrepo downloads on demand, so this just warms the cache and writes a local
copy for inspection. Requires network access on first run.
"""
import _bootstrap  # noqa: F401  (adds src/ to path)

from pathlib import Path

from fedcity.config import REPO_ROOT


def main():
    from ucimlrepo import fetch_ucirepo

    out_dir = REPO_ROOT / "data" / "healthcare"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = fetch_ucirepo(id=17)
    x = repo.data.features
    y = repo.data.targets
    df = x.copy()
    df["target"] = y.values.ravel()
    csv = out_dir / "breast_cancer_wdbc.csv"
    df.to_csv(csv, index=False)
    print(f"Saved {len(df)} rows x {x.shape[1]} features -> {csv}")


if __name__ == "__main__":
    main()
