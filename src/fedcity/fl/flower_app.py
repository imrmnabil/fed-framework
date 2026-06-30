"""Optional Flower/gRPC path (thesis §3.4.1) — real distributed transport.

The primary, fast, fully-featured engine is ``fedcity.runner``. This module wires
the same TF models + data partitions into Flower's gRPC stack so the experiment can
also run as a genuine distributed system: an in-process Ray simulation
(``run_flower_simulation``) or separate server/client processes over gRPC
(``start_server_app`` / ``start_client_app``). Aggregation uses Flower's built-in
FedAvg/FedAvgM/FedAdam, which implement the same rules as ``fl.strategies``.
"""
from __future__ import annotations

import numpy as np

from ..config import load_config
from ..datasets.loaders import load_dataset
from ..datasets.partition import make_partition
from ..models import build_model


def _partition_clients(cfg, bundle):
    part = cfg.get("partition", {"kind": "iid"})
    idx = make_partition(
        bundle.y_train, cfg["federated"]["n_clients"],
        kind=part["kind"], alpha=part.get("alpha"), seed=cfg["seed"],
    )
    return [(bundle.x_train[i], bundle.y_train[i]) for i in idx]


def make_numpy_client(cfg, bundle, x, y):
    """Build a Flower NumPyClient for one partition."""
    from flwr.client import NumPyClient

    model = build_model(cfg, bundle.input_shape, bundle.n_outputs, bundle.task)
    fed = cfg["federated"]

    class TFClient(NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x, y, epochs=fed["local_epochs"], batch_size=fed["local_batch_size"], verbose=0)
            return model.get_weights(), len(y), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, acc = model.evaluate(bundle.x_val, bundle.y_val, verbose=0)
            return float(loss), len(bundle.y_val), {"accuracy": float(acc)}

    return TFClient()


def build_flower_strategy(cfg, init_weights):
    """Map our strategy name to a Flower strategy (gRPC path)."""
    from flwr.common import ndarrays_to_parameters
    from flwr.server import strategy as fl_strategy

    fed = cfg["federated"]
    opt = {**{"server_lr": fed["server_lr"], "beta_1": fed["beta_1"], "beta_2": fed["beta_2"],
              "tau": fed["tau"]}, **fed.get("server_opt", {}).get(fed["strategy"], {})}
    init = ndarrays_to_parameters(init_weights)
    common = dict(
        fraction_fit=fed["fraction_fit"],
        min_fit_clients=max(2, round(fed["fraction_fit"] * fed["n_clients"])),
        min_available_clients=fed["n_clients"],
        initial_parameters=init,
    )
    name = fed["strategy"]
    if name == "fedavg":
        return fl_strategy.FedAvg(**common)
    if name == "fedopt":
        return fl_strategy.FedAvgM(server_momentum=fed["beta_1"], server_learning_rate=opt["server_lr"], **common)
    if name == "fedadam":
        return fl_strategy.FedAdam(eta=opt["server_lr"], beta_1=opt["beta_1"],
                                   beta_2=opt["beta_2"], tau=opt["tau"], **common)
    raise ValueError(f"unknown strategy {name!r}")


def run_flower_simulation(domain_or_cfg, overrides=None, num_rounds=None):
    """Run the experiment via Flower's Ray simulation engine (verified on flwr 1.32)."""
    import flwr as fl

    cfg = domain_or_cfg if isinstance(domain_or_cfg, dict) else load_config(domain_or_cfg, overrides)
    bundle = load_dataset(cfg)
    clients_data = _partition_clients(cfg, bundle)

    def client_fn(cid):  # legacy signature still accepted in 1.32
        x, y = clients_data[int(cid)]
        return make_numpy_client(cfg, bundle, x, y).to_client()

    init_model = build_model(cfg, bundle.input_shape, bundle.n_outputs, bundle.task)
    strategy = build_flower_strategy(cfg, init_model.get_weights())
    rounds = num_rounds or cfg["federated"]["rounds"]
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["federated"]["n_clients"],
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


# --- real gRPC entrypoints (run on separate machines/processes) ------------- #
def start_server_app(domain="healthcare", address="0.0.0.0:8080", overrides=None):
    import flwr as fl

    cfg = load_config(domain, overrides)
    bundle = load_dataset(cfg)
    init_model = build_model(cfg, bundle.input_shape, bundle.n_outputs, bundle.task)
    strategy = build_flower_strategy(cfg, init_model.get_weights())
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["rounds"]),
        strategy=strategy,
    )


def start_client_app(cid: int, domain="healthcare", address="127.0.0.1:8080", overrides=None):
    import flwr as fl

    cfg = load_config(domain, overrides)
    bundle = load_dataset(cfg)
    clients_data = _partition_clients(cfg, bundle)
    x, y = clients_data[int(cid)]
    client = make_numpy_client(cfg, bundle, x, y)
    fl.client.start_client(server_address=address, client=client.to_client())
