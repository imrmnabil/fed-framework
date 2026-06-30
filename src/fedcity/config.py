"""Configuration loading: merge a domain config over ``configs/base.yaml``."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(domain_or_path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load and merge a config.

    ``domain_or_path`` may be a domain name (``healthcare``/``smartgrid``/``traffic``),
    a bare filename in ``configs/``, or an explicit path. ``base.yaml`` is always the
    foundation. ``overrides`` (e.g. CLI flags) are deep-merged last.
    """
    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())

    p = Path(domain_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        domain_cfg = yaml.safe_load(p.read_text())
    else:
        name = p.stem if p.suffix else domain_or_path
        domain_cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())

    cfg = _deep_merge(base, domain_cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    _apply_smoke(cfg)
    return cfg


def _apply_smoke(cfg: dict) -> None:
    """Shrink an experiment for a fast end-to-end check when ``smoke`` is set."""
    if not cfg.get("smoke"):
        return
    fed = cfg["federated"]
    fed["n_clients"] = min(fed["n_clients"], 4)
    fed["rounds"] = min(fed["rounds"], 3)
    fed["local_epochs"] = 1
    fed["auction"]["warmup_rounds"] = 1
    fed["auction"]["budget"] = min(fed["auction"]["budget"], fed["n_clients"])
    ds = cfg.get("dataset", {})
    if cfg.get("domain") == "traffic":
        ds["subsample_fraction"] = min(ds.get("subsample_fraction", 1.0), 0.03)
