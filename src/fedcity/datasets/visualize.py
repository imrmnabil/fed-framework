"""Label-distribution visualizations for the partitioning schemes (Figs 3.3-3.8)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .partition import class_distribution_matrix, make_partition


def plot_label_distribution(
    labels: np.ndarray,
    client_indices: list[np.ndarray],
    title: str,
    out_path: str | Path,
    n_classes: int | None = None,
):
    """Stacked horizontal bar chart of per-client class composition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = class_distribution_matrix(labels, client_indices, n_classes)
    n_clients, n_cls = mat.shape

    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(n_clients)
    cmap = plt.get_cmap("tab20" if n_cls <= 20 else "viridis")
    ys = np.arange(n_clients)
    for k in range(n_cls):
        ax.barh(ys, mat[:, k], left=left, color=cmap(k % cmap.N), height=0.8)
        left += mat[:, k]
    ax.set_xlabel("number of samples")
    ax.set_ylabel("client")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_partition_figures(
    labels: np.ndarray,
    n_clients: int,
    out_dir: str | Path,
    alphas=(0.1, 0.3, 0.5, 1.0, 5.0),
    seed: int = 42,
    n_classes: int | None = None,
):
    """Regenerate the thesis heterogeneity figures: one per alpha + IID baseline."""
    out_dir = Path(out_dir)
    produced = []
    for a in alphas:
        ci = make_partition(labels, n_clients, kind="dirichlet", alpha=a, seed=seed)
        p = plot_label_distribution(
            labels, ci, f"Dirichlet partition (alpha = {a})",
            out_dir / f"partition_dirichlet_alpha_{a}.png", n_classes,
        )
        produced.append(p)
    ci = make_partition(labels, n_clients, kind="iid", seed=seed)
    produced.append(
        plot_label_distribution(labels, ci, "IID partition", out_dir / "partition_iid.png", n_classes)
    )
    return produced
