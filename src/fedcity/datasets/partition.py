"""Client data partitioning (thesis §3.2.1).

Two schemes:
  * ``iid_partition`` — random, even split (homogeneous baseline, Fig. 3.8).
  * ``dirichlet_partition`` — label-skewed non-IID split controlled by alpha
    (Eq. 3.2, Figs 3.3-3.7). Lower alpha => more heterogeneous.

Both return ``list[np.ndarray]`` of sample indices, one array per client.
"""
from __future__ import annotations

import numpy as np


def iid_partition(labels: np.ndarray, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Shuffle all samples and split as evenly as possible across clients."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [np.sort(part) for part in np.array_split(idx, n_clients)]


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_per_client: int = 1,
) -> list[np.ndarray]:
    """Partition by drawing per-class client proportions from Dir(alpha).

    For each class ``k``, sample ``p ~ Dir(alpha * 1_{n_clients})`` and split that
    class's indices among clients by ``p`` (Hsu et al. 2019). This realises the
    thesis's Eq. 3.2 control over heterogeneity: small alpha concentrates each
    class on few clients, large alpha approaches a uniform (IID-like) spread.

    ``min_per_client`` retries the draw (with a perturbed seed) until every client
    holds at least this many samples, avoiding empty clients on extreme skew.
    """
    labels = np.asarray(labels).ravel()
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    for attempt in range(100):
        client_idx: list[list[int]] = [[] for _ in range(n_clients)]
        for k in classes:
            k_idx = np.where(labels == k)[0]
            rng.shuffle(k_idx)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            # cut points scaled to this class's count
            cuts = (np.cumsum(proportions) * len(k_idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(k_idx, cuts)):
                client_idx[c].extend(part.tolist())

        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_per_client:
            break
        rng = np.random.default_rng(seed + attempt + 1)

    return [np.sort(np.array(c, dtype=int)) for c in client_idx]


def make_partition(
    labels: np.ndarray, n_clients: int, *, kind: str, alpha: float | None = None, seed: int = 42
) -> list[np.ndarray]:
    """Dispatch on ``kind`` ('iid' or 'dirichlet')."""
    if kind == "iid":
        return iid_partition(labels, n_clients, seed=seed)
    if kind == "dirichlet":
        if alpha is None:
            raise ValueError("dirichlet partition requires alpha")
        return dirichlet_partition(labels, n_clients, alpha, seed=seed)
    raise ValueError(f"unknown partition kind: {kind!r}")


def class_distribution_matrix(
    labels: np.ndarray, client_indices: list[np.ndarray], n_classes: int | None = None
) -> np.ndarray:
    """Return a (n_clients, n_classes) matrix of per-client class counts.

    Used by the partition-visualization figures (3.3-3.8).
    """
    labels = np.asarray(labels).ravel()
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    mat = np.zeros((len(client_indices), n_classes), dtype=int)
    for c, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        vals, counts = np.unique(labels[idx], return_counts=True)
        for v, cnt in zip(vals, counts):
            mat[c, int(v)] = cnt
    return mat
