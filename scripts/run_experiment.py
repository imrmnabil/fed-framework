#!/usr/bin/env python
"""Run the thesis FL experiment: a single cell or a full per-domain sweep.

Examples
--------
  # quick end-to-end check (synthetic data, tiny everything)
  python scripts/run_experiment.py --config healthcare --smoke --synthetic

  # one cell: FedAdam on Dirichlet(0.3)
  python scripts/run_experiment.py --config healthcare \
      --strategy fedadam --partition dirichlet --alpha 0.3

  # full sweep reproducing Table 3.2 + Figs 3.12-3.14 + centralized comparison
  python scripts/run_experiment.py --config healthcare --sweep
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

from fedcity.config import REPO_ROOT, load_config
from fedcity.centralized import train_centralized
from fedcity.datasets.loaders import load_dataset
from fedcity import report
from fedcity.runner import FederatedSimulation

# Default partition sets per domain (thesis §3.7/§3.8/§3.9).
DOMAIN_PARTITIONS = {
    "healthcare": [{"kind": "iid"}, {"kind": "dirichlet", "alpha": 0.3}, {"kind": "dirichlet", "alpha": 5}],
    "smartgrid": [{"kind": "dirichlet", "alpha": 0.3}, {"kind": "dirichlet", "alpha": 5}],
    "traffic": [{"kind": "dirichlet", "alpha": 0.3}, {"kind": "dirichlet", "alpha": 5}],
    "synthetic": [{"kind": "dirichlet", "alpha": 0.3}, {"kind": "iid"}],
}
STRATEGIES = ["fedavg", "fedopt", "fedadam"]


def _base_cfg(args) -> dict:
    overrides = {}
    if args.synthetic:
        overrides["use_synthetic"] = True
    if args.smoke:
        overrides["smoke"] = True
    if args.selector:
        overrides.setdefault("federated", {})["selector"] = args.selector
    if args.no_fedpaq:
        overrides.setdefault("federated", {}).setdefault("fedpaq", {})["enabled"] = False
    if args.rounds:
        overrides.setdefault("federated", {})["rounds"] = args.rounds
    if args.clients:
        overrides.setdefault("federated", {})["n_clients"] = args.clients
    return load_config(args.config, overrides or None)


def run_single(args, run_id):
    cfg = _base_cfg(args)
    cfg["partition"] = {"kind": args.partition, **({"alpha": args.alpha} if args.partition == "dirichlet" else {})}
    cfg["federated"]["strategy"] = args.strategy
    print(f"== {cfg['domain']} | {args.strategy} | {report.partition_label(cfg['partition'])} ==")
    res = FederatedSimulation(cfg).run(verbose=True)
    domain_dir = REPO_ROOT / cfg["output"]["dir"] / cfg["domain"]
    out = domain_dir / run_id / "single"
    report.save_json(res, out / f"{args.strategy}_{args.partition}{args.alpha if args.partition=='dirichlet' else ''}.json")
    report.update_latest(domain_dir, run_id)
    print(f"\nfinal_acc={res['final_accuracy']:.4f}  best={res['best_accuracy']:.4f}  "
          f"conv@{res['rounds_to_threshold']}  comm={res['comm_total_mb']:.2f}MB")
    print(f"saved to {out}")


def run_sweep(args, run_id):
    cfg0 = _base_cfg(args)
    domain = cfg0["domain"]
    bundle = load_dataset(cfg0)          # load once, reuse across all cells
    partitions = DOMAIN_PARTITIONS.get(domain, DOMAIN_PARTITIONS["synthetic"])
    domain_dir = REPO_ROOT / cfg0["output"]["dir"] / domain
    out_dir = domain_dir / run_id

    all_results = []
    by_partition: dict[str, list[dict]] = {}
    for part in partitions:
        plabel = report.partition_label(part)
        for strat in STRATEGIES:
            cfg = _base_cfg(args)
            cfg["partition"] = part
            cfg["federated"]["strategy"] = strat
            print(f"\n== {domain} | {strat} | {plabel} ==")
            res = FederatedSimulation(cfg, bundle=bundle).run(verbose=not args.quiet)
            all_results.append(res)
            by_partition.setdefault(plabel, []).append(res)
            report.save_json(res, out_dir / "runs" / f"{strat}_{plabel.replace('/','-')}.json")

    # accuracy curves per partition (Figs 3.12-3.19)
    for plabel, results in by_partition.items():
        report.plot_accuracy_curves(
            results, f"{domain}: federated accuracy ({plabel})",
            out_dir / "curves" / f"accuracy_{plabel.replace('(','').replace(')','').replace('=','').replace('.','_')}.png",
        )

    # centralized baseline + tables (Tables 3.2/3.4)
    print(f"\n== {domain} | centralized baseline ==")
    cl = train_centralized(cfg0, bundle)
    report.save_json(cl, out_dir / "centralized.json")
    report.write_table_csv(all_results, out_dir / "results_table.csv")
    md = report.write_table_markdown(all_results, cl, out_dir / "results_table.md")
    report.update_latest(domain_dir, run_id)
    print("\n" + md)
    print(f"All outputs under {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="healthcare", help="domain or config path")
    ap.add_argument("--strategy", default="fedadam", choices=STRATEGIES)
    ap.add_argument("--partition", default="dirichlet", choices=["iid", "dirichlet"])
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--selector", default=None, choices=["random", "auction"])
    ap.add_argument("--sweep", action="store_true", help="run the full strategy×partition matrix")
    ap.add_argument("--no-fedpaq", action="store_true", help="disable FedPAQ quantization")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (no downloads)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.sweep:
        run_sweep(args, run_id)
    else:
        run_single(args, run_id)


if __name__ == "__main__":
    main()
