#!/usr/bin/env python
"""Regenerate the partition heterogeneity figures (thesis Figs 3.3-3.8).

Produces one stacked label-distribution plot per Dirichlet alpha in {0.1,0.3,0.5,1,5}
plus an IID baseline, for the chosen domain (or synthetic).
"""
import _bootstrap  # noqa: F401

import argparse

from fedcity.config import REPO_ROOT, load_config
from fedcity.datasets.loaders import load_dataset
from fedcity.datasets.visualize import make_partition_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="healthcare")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (no download)")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.domain, {"use_synthetic": args.synthetic} if args.synthetic else None)
    bundle = load_dataset(cfg)
    out_dir = args.out or (REPO_ROOT / "experiments" / args.domain / "partitions")

    figs = make_partition_figures(
        bundle.y_train, args.clients, out_dir, seed=cfg["seed"], n_classes=bundle.n_classes
    )
    print(f"Wrote {len(figs)} partition figures to {out_dir}:")
    for f in figs:
        print("  ", f)


if __name__ == "__main__":
    main()
