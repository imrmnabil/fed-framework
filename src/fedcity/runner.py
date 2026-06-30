"""In-process federated-learning engine implementing the thesis methodology.

One round = AUCTION/random selection within budget (partial participation) →
client failures (p=0.2) → τ local epochs per surviving client (periodic averaging)
→ quantized update upload (FedPAQ) → server aggregation (FedAvg/FedOpt/FedAdam) →
centralized validation eval → reward back to the AUCTION agent.

This sequential, single-model engine is the default (fast + deterministic on CPU).
A Flower/gRPC path for true distributed runs lives in ``fl/flower_app.py`` and
reuses the same aggregation, quantization, and client code.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .config import load_config
from .datasets.loaders import DatasetBundle, load_dataset
from .datasets.partition import make_partition
from .fl.auction import make_selector
from .fl.client import LocalClient, make_profiles
from .fl.failures import apply_failures
from .fl.metrics import CommAccountant, RoundRecord, rounds_to_threshold
from .fl.quantize import float_nbytes, quantize_update
from .fl.strategies import AggregatorState, aggregate_deltas, apply_server_update
from .models import build_model


class FederatedSimulation:
    def __init__(self, cfg: dict, bundle: DatasetBundle | None = None):
        self.cfg = cfg
        self.fed = cfg["federated"]
        self.seed = cfg["seed"]
        self.bundle = bundle or load_dataset(cfg)
        self.rng = np.random.default_rng(self.seed)
        self._set_global_seed()

        # partition the training data across clients
        part = cfg.get("partition", {"kind": "iid"})
        self.client_indices = make_partition(
            self.bundle.y_train,
            self.fed["n_clients"],
            kind=part["kind"],
            alpha=part.get("alpha"),
            seed=self.seed,
        )
        clients_data = [
            (self.bundle.x_train[idx], self.bundle.y_train[idx]) for idx in self.client_indices
        ]
        profiles = make_profiles(clients_data, self.seed)
        self.clients = [
            LocalClient(cid, x, y, profiles[cid]) for cid, (x, y) in enumerate(clients_data)
        ]
        self.profiles = profiles
        self.max_samples = max(1, max(c.n_samples for c in self.clients))

        # one shared global model, server optimizer state, selector
        self.model = build_model(cfg, self.bundle.input_shape, self.bundle.n_outputs, self.bundle.task)
        self.global_weights = self.model.get_weights()
        self.agg_state = AggregatorState()
        self.selector = make_selector(cfg, self.seed)
        self.comm = CommAccountant()

    def _set_global_seed(self):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)

    @staticmethod
    def _resolve_server_opt(fed: dict) -> dict:
        """Per-strategy server-optimizer hyperparameters, falling back to flat values."""
        opt = {
            "server_lr": fed["server_lr"],
            "beta_1": fed["beta_1"],
            "beta_2": fed["beta_2"],
            "tau": fed["tau"],
        }
        opt.update(fed.get("server_opt", {}).get(fed["strategy"], {}))
        return opt

    def _evaluate(self) -> tuple[float, float]:
        self.model.set_weights(self.global_weights)
        loss, acc = self.model.evaluate(self.bundle.x_val, self.bundle.y_val, verbose=0)
        return float(loss), float(acc)

    def run(self, verbose: bool = True) -> dict:
        fed = self.fed
        n_clients = fed["n_clients"]
        k = max(1, round(fed["fraction_fit"] * n_clients))
        s = fed["fedpaq"]["levels"] if fed["fedpaq"]["enabled"] else None
        opt = self._resolve_server_opt(fed)
        history: list[RoundRecord] = []

        prev_loss, prev_acc = self._evaluate()
        t0 = time.perf_counter()

        for rnd in range(1, fed["rounds"] + 1):
            selected = self.selector.select(rnd, self.profiles, self.max_samples, k)
            survivors = apply_failures(selected, fed["client_failure_prob"], self.rng)

            # server -> client download (full float32 global model per survivor)
            self.comm.add(download=len(survivors) * float_nbytes(self.global_weights))

            updates, upload = [], 0
            for cid in survivors:
                delta, n, _ = self.clients[cid].local_train(
                    self.model, self.global_weights, fed["local_epochs"], fed["local_batch_size"]
                )
                q_delta, nbytes = quantize_update(delta, s, self.rng)
                updates.append((q_delta, n))
                upload += nbytes
            self.comm.add(upload=upload)

            agg_delta = aggregate_deltas(updates)
            self.global_weights = apply_server_update(
                fed["strategy"], self.global_weights, agg_delta, self.agg_state, **opt,
            )

            loss, acc = self._evaluate()
            self.selector.observe_reward(rnd, prev_acc, acc, self.profiles)
            prev_acc = acc

            history.append(
                RoundRecord(rnd, loss, acc, len(survivors), self.comm.upload_bytes, self.comm.download_bytes)
            )
            if verbose:
                print(
                    f"  round {rnd:>3}/{fed['rounds']}  "
                    f"acc={acc:.4f} loss={loss:.4f} "
                    f"clients={len(survivors)}/{len(selected)} "
                    f"comm={self.comm.total_mb:.2f}MB",
                    flush=True,
                )

        train_time = time.perf_counter() - t0
        acc_hist = [r.eval_acc for r in history]
        threshold = self.cfg["eval"]["convergence_threshold"]
        return {
            "domain": self.cfg.get("domain"),
            "mode": "federated",
            "strategy": fed["strategy"],
            "selector": fed.get("selector"),
            "partition": self.cfg.get("partition", {"kind": "iid"}),
            "rounds": fed["rounds"],
            "final_accuracy": acc_hist[-1] if acc_hist else 0.0,
            "best_accuracy": max(acc_hist) if acc_hist else 0.0,
            "rounds_to_threshold": rounds_to_threshold(acc_hist, threshold),
            "convergence_threshold": threshold,
            "train_time_s": train_time,
            "comm_upload_mb": self.comm.upload_bytes / (1024 * 1024),
            "comm_download_mb": self.comm.download_bytes / (1024 * 1024),
            "comm_total_mb": self.comm.total_mb,
            "fedpaq_levels": s,
            "history": [asdict(r) for r in history],
        }


def run_federated(domain_or_cfg, overrides: dict | None = None, verbose: bool = True) -> dict:
    """Convenience entry: load a config (or use a dict) and run one federated experiment."""
    cfg = domain_or_cfg if isinstance(domain_or_cfg, dict) else load_config(domain_or_cfg, overrides)
    if isinstance(domain_or_cfg, dict) and overrides:
        from .config import _deep_merge

        cfg = _deep_merge(cfg, overrides)
    return FederatedSimulation(cfg).run(verbose=verbose)
