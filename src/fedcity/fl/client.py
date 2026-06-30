"""Local client: periodic local training + update (delta) computation (thesis §3.4).

A client owns its data partition and a simulated *profile* (data size, network
strength, battery, claimed price) that the AUCTION agent observes for selection.
Training reuses a single shared Keras model owned by the engine (set weights →
fit τ local epochs → read weights) to keep memory flat across 20 clients on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientProfile:
    """Quality/cost features observed by the AUCTION agent (thesis §3.5)."""
    cid: int
    n_samples: int
    network_strength: float   # [0,1] simulated link quality
    battery: float            # [0,1] simulated battery level
    price: float              # claimed payment b_i (cost to select)
    last_loss: float = 1.0    # most recent local training loss (quality signal)
    last_acc: float = 0.0

    def feature_vector(self, max_samples: int) -> np.ndarray:
        return np.array(
            [
                self.n_samples / max(1, max_samples),
                self.network_strength,
                self.battery,
                self.price,
                self.last_loss,
                self.last_acc,
            ],
            dtype=np.float32,
        )


N_CLIENT_FEATURES = 6


class LocalClient:
    def __init__(self, cid: int, x: np.ndarray, y: np.ndarray, profile: ClientProfile):
        self.cid = cid
        self.x = x
        self.y = y
        self.profile = profile

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def local_train(
        self, model, global_weights: list[np.ndarray], local_epochs: int, batch_size: int
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train τ local epochs from ``global_weights``; return (Δ, n_samples, metrics)."""
        model.set_weights(global_weights)
        hist = model.fit(
            self.x, self.y,
            epochs=local_epochs, batch_size=batch_size, verbose=0,
        )
        new_weights = model.get_weights()
        delta = [nw - gw for nw, gw in zip(new_weights, global_weights)]

        loss = float(hist.history["loss"][-1])
        acc = float(hist.history.get("accuracy", [0.0])[-1])
        self.profile.last_loss = loss
        self.profile.last_acc = acc
        return delta, self.n_samples, {"loss": loss, "accuracy": acc}


def make_profiles(clients_data: list[tuple[np.ndarray, np.ndarray]], seed: int) -> list[ClientProfile]:
    """Generate reproducible simulated profiles for a set of client partitions."""
    rng = np.random.default_rng(seed + 777)
    profiles = []
    for cid, (_, y) in enumerate(clients_data):
        profiles.append(
            ClientProfile(
                cid=cid,
                n_samples=len(y),
                network_strength=float(rng.uniform(0.3, 1.0)),
                battery=float(rng.uniform(0.2, 1.0)),
                price=float(rng.uniform(0.1, 1.0)),
            )
        )
    return profiles
