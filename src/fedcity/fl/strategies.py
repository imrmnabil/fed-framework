"""Server-side aggregation strategies (thesis §3.6): FedAvg, FedOpt, FedAdam.

All three follow the Reddi et al. (2021) "FedOpt" template on the sample-weighted
aggregated client update (pseudo-gradient) ``Δ_t = Σ_k (n_k/n) Δ_k`` where each
client sends ``Δ_k = w_local_k − w_global``:

  FedAvg :  w ← w + Δ_t
  FedOpt :  m ← β1·m + Δ_t ;                       w ← w + η_g·m         (server momentum)
  FedAdam:  m ← β1·m + (1−β1)·Δ_t ;
            v ← β2·v + (1−β2)·Δ_t² ;               w ← w + η_g·m/(√v+τ)  (adaptive)

Kept as pure functions over ``list[np.ndarray]`` so they are unit-testable and
reusable by both the in-process engine and the optional Flower/gRPC strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STRATEGIES = ("fedavg", "fedopt", "fedadam")


def aggregate_deltas(
    client_updates: list[tuple[list[np.ndarray], int]]
) -> list[np.ndarray]:
    """Sample-weighted average of per-client update tensors → Δ_t."""
    total = sum(n for _, n in client_updates)
    if total == 0:
        raise ValueError("no samples across selected clients")
    agg = [np.zeros_like(t, dtype=np.float64) for t in client_updates[0][0]]
    for delta, n in client_updates:
        w = n / total
        for i, t in enumerate(delta):
            agg[i] += w * np.asarray(t, dtype=np.float64)
    return agg


@dataclass
class AggregatorState:
    """Server-optimizer moment buffers (lazily shaped to the model on first use)."""
    m: list[np.ndarray] | None = None
    v: list[np.ndarray] | None = None

    def ensure(self, ref: list[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(t, dtype=np.float64) for t in ref]
            self.v = [np.zeros_like(t, dtype=np.float64) for t in ref]


def apply_server_update(
    strategy: str,
    global_w: list[np.ndarray],
    agg_delta: list[np.ndarray],
    state: AggregatorState,
    *,
    server_lr: float = 0.01,
    beta_1: float = 0.9,
    beta_2: float = 0.99,
    tau: float = 1e-3,
) -> list[np.ndarray]:
    """Apply the chosen strategy's update rule; mutates ``state`` in place."""
    strategy = strategy.lower()
    gw = [np.asarray(w, dtype=np.float64) for w in global_w]

    if strategy == "fedavg":
        return [(w + d).astype(np.float32) for w, d in zip(gw, agg_delta)]

    state.ensure(gw)
    if strategy == "fedopt":
        new = []
        for i, (w, d) in enumerate(zip(gw, agg_delta)):
            state.m[i] = beta_1 * state.m[i] + d
            new.append((w + server_lr * state.m[i]).astype(np.float32))
        return new

    if strategy == "fedadam":
        new = []
        for i, (w, d) in enumerate(zip(gw, agg_delta)):
            state.m[i] = beta_1 * state.m[i] + (1 - beta_1) * d
            state.v[i] = beta_2 * state.v[i] + (1 - beta_2) * np.square(d)
            step = server_lr * state.m[i] / (np.sqrt(state.v[i]) + tau)
            new.append((w + step).astype(np.float32))
        return new

    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
