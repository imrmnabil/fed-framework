"""Keras model builders, one per domain (thesis Tables 3.1, 3.3, 3.5)."""
from __future__ import annotations

from .healthcare import build_healthcare_model
from .smartgrid import build_smartgrid_model
from .traffic import build_traffic_model

_BUILDERS = {
    "healthcare": build_healthcare_model,
    "smartgrid": build_smartgrid_model,
    "traffic": build_traffic_model,
    "synthetic": None,  # resolved at call time from task/shape
}


def build_model(cfg: dict, input_shape, n_outputs: int, task: str):
    """Build + compile the model for ``cfg['domain']``.

    For the synthetic domain, pick a builder by task (binary->healthcare DNN,
    multiclass->traffic CNN) so smoke runs exercise a real architecture.
    """
    domain = cfg.get("domain", "healthcare")
    if cfg.get("dataset", {}).get("name") == "synthetic" or domain == "synthetic":
        domain = "traffic" if task == "multiclass" else "healthcare"
    builder = _BUILDERS[domain]
    return builder(input_shape, n_outputs, cfg.get("model", {}))
