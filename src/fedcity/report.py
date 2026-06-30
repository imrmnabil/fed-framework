"""Reporting: accuracy curves (Figs 3.12-3.19) and result tables (Tables 3.2/3.4)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def partition_label(part: dict) -> str:
    if part.get("kind") == "iid":
        return "IID"
    return f"Dir(a={part.get('alpha')})"


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def plot_accuracy_curves(results: list[dict], title: str, out_path: str | Path):
    """One curve per strategy: eval accuracy vs communication round."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        accs = [h["eval_acc"] for h in res["history"]]
        ax.plot(range(1, len(accs) + 1), accs, label=res["strategy"], linewidth=1.6)
    ax.set_xlabel("communication round")
    ax.set_ylabel("federated evaluation accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def accuracy_table(results: list[dict]) -> tuple[list[str], list[str], dict]:
    """Build a {strategy -> {partition_label -> final_acc}} table from sweep results."""
    strategies, partitions, cells = [], [], {}
    for res in results:
        strat = res["strategy"]
        plabel = partition_label(res["partition"])
        if strat not in strategies:
            strategies.append(strat)
        if plabel not in partitions:
            partitions.append(plabel)
        cells.setdefault(strat, {})[plabel] = res["final_accuracy"]
    return strategies, partitions, cells


def write_table_csv(results: list[dict], out_path: str | Path):
    strategies, partitions, cells = accuracy_table(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", *partitions])
        for s in strategies:
            w.writerow([s, *[f"{cells[s].get(p, float('nan')):.4f}" for p in partitions]])
    return out_path


def write_table_markdown(results: list[dict], centralized: dict | None, out_path: str | Path) -> str:
    strategies, partitions, cells = accuracy_table(results)
    lines = ["| Strategy | " + " | ".join(partitions) + " |",
             "|" + "---|" * (len(partitions) + 1)]
    for s in strategies:
        row = [f"{cells[s].get(p, float('nan')):.4f}" for p in partitions]
        lines.append(f"| {s} | " + " | ".join(row) + " |")

    md = "### Validation accuracy by strategy × partition\n\n" + "\n".join(lines) + "\n"

    if results:
        fl_mb = sum(r["comm_total_mb"] for r in results) / len(results)
        md += f"\n**Avg federated comm/run:** {fl_mb:.2f} MB\n"
    if centralized:
        cl_mb = centralized["comm_mb"]
        md += (
            f"\n**Centralized:** acc={centralized['accuracy']:.4f}, "
            f"comm={cl_mb:.2f} MB, time={centralized['train_time_s']:.1f}s\n"
        )
        if results:
            red = (cl_mb - fl_mb) / cl_mb * 100 if cl_mb else 0.0
            md += f"\n**Comm reduction (FL vs CL):** {red:.2f}%\n"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    return md
