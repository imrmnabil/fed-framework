"""Centralized baseline: train on pooled data, for FL-vs-CL comparison."""
from __future__ import annotations

import time

import numpy as np

from ..datasets.loaders import DatasetBundle
from ..models import build_model


def train_centralized(cfg: dict, bundle: DatasetBundle, epochs: int | None = None) -> dict:
    """Train one model on the full training set; return accuracy/loss/time/comm.

    Centralized communication cost is modelled as the one-time transfer of the
    raw dataset to the server (thesis compares this against federated upload+download).
    """
    model = build_model(cfg, bundle.input_shape, bundle.n_outputs, bundle.task)
    epochs = epochs or cfg.get("centralized", {}).get("epochs", max(10, cfg["federated"]["rounds"]))
    batch_size = cfg["federated"].get("local_batch_size", 32)

    t0 = time.perf_counter()
    model.fit(bundle.x_train, bundle.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    train_time = time.perf_counter() - t0

    loss, acc = model.evaluate(bundle.x_val, bundle.y_val, verbose=0)
    data_bytes = int(bundle.x_train.nbytes + bundle.x_val.nbytes)
    return {
        "domain": cfg.get("domain"),
        "mode": "centralized",
        "epochs": epochs,
        "accuracy": float(acc),
        "loss": float(loss),
        "train_time_s": train_time,
        "comm_mb": data_bytes / (1024 * 1024),
    }
