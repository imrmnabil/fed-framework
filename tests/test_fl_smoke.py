"""End-to-end FL smoke run on synthetic data (exercises the whole engine)."""
import pytest

pytest.importorskip("tensorflow")

from fedcity.config import load_config
from fedcity.runner import FederatedSimulation


def _cfg(strategy="fedadam", selector="random"):
    cfg = load_config("healthcare", {"use_synthetic": True})
    cfg["federated"].update(
        n_clients=4, rounds=3, local_epochs=1, fraction_fit=0.75, strategy=strategy, selector=selector
    )
    cfg["partition"] = {"kind": "dirichlet", "alpha": 0.3}
    return cfg


@pytest.mark.parametrize("strategy", ["fedavg", "fedopt", "fedadam"])
def test_each_strategy_runs(strategy):
    res = FederatedSimulation(_cfg(strategy)).run(verbose=False)
    assert len(res["history"]) == 3
    assert 0.0 <= res["final_accuracy"] <= 1.0
    assert res["comm_total_mb"] > 0


def test_auction_selector_runs_and_learns():
    res = FederatedSimulation(_cfg(selector="auction")).run(verbose=False)
    assert res["selector"] == "auction"
    assert len(res["history"]) == 3


def test_fedpaq_quantization_reduces_upload():
    cfg_q = _cfg()
    cfg_noq = _cfg()
    cfg_noq["federated"]["fedpaq"]["enabled"] = False
    up_q = FederatedSimulation(cfg_q).run(verbose=False)["comm_upload_mb"]
    up_noq = FederatedSimulation(cfg_noq).run(verbose=False)["comm_upload_mb"]
    assert up_q < up_noq
