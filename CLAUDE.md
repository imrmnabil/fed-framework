# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FedCity is a runnable implementation of the thesis *"Optimizing Smart City Functions
through Federated Learning"* (`docs/Thesis_190217_200227_.pdf`, Ch. 3). It compares
**federated vs. centralized** learning across three domains — healthcare, smart grid,
traffic — under controlled data heterogeneity, with FedPAQ (comm efficiency), AUCTION
(DRL client selection), and FedAvg/FedOpt/FedAdam aggregation. Code is organized to map
1:1 onto thesis sections (see the table in `README.md`); docstrings cite the section
(e.g. "§3.4", "Table 3.5") for each component.

## Commands

All commands assume the `uv`-managed venv at `.venv` (Python 3.12).

```bash
# Setup (one time)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .

# Fast end-to-end sanity check (synthetic data, no downloads, ~1 min)
.venv/bin/python scripts/run_experiment.py --config healthcare --sweep --synthetic --smoke

# One experiment cell
.venv/bin/python scripts/run_experiment.py --config healthcare --strategy fedadam --partition dirichlet --alpha 0.3

# Full sweep: strategy × partition matrix + centralized baseline + tables/curves
.venv/bin/python scripts/run_experiment.py --config <healthcare|smartgrid|traffic> --sweep

# Tests
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_strategies.py -q            # one file
.venv/bin/python -m pytest tests/test_strategies.py::test_name    # one test
```

There is no linter/formatter configured; match the existing style (`from __future__ import
annotations`, module docstrings with thesis citations, type hints).

### Key run flags (`scripts/run_experiment.py`)

`--sweep` (full matrix) · `--strategy {fedavg,fedopt,fedadam}` · `--partition {iid,dirichlet}
--alpha A` · `--selector {random,auction}` · `--no-fedpaq` (ablation) · `--synthetic` (no
downloads) · `--smoke` (tiny fast run) · `--rounds N` / `--clients N`.

`--smoke` and `--synthetic` are how you exercise real code paths without GPUs or dataset
downloads — prefer them for verification. CPU knobs (`local_batch_size`, traffic
`subsample_fraction`) live in the YAML configs.

## Architecture

### Config layering
Every run starts from `configs/base.yaml`, deep-merged with a domain config
(`configs/{healthcare,smartgrid,traffic}.yaml`), then CLI overrides last. `config.py`
`load_config()` does the merge; `_apply_smoke()` shrinks the run when `smoke: true`.
Nearly all hyperparameters (clients, rounds, FedPAQ levels, AUCTION budget, per-strategy
server-optimizer settings) are config-driven, not hardcoded.

### Two execution engines, one set of components
- **`runner.py` `FederatedSimulation`** is the default: a fast, deterministic,
  single-process simulation with one shared Keras model. This is where the round loop
  lives and where you'll do most work. One round = selection → failures → τ local epochs
  → quantized upload → server aggregation → centralized eval → reward to AUCTION agent.
- **`fl/flower_app.py`** is an optional real-distributed path over Flower/gRPC (Ray sim or
  separate processes). It reuses the same models, partitions, and aggregation semantics.

When changing FL behavior, the reusable pieces below are shared by both engines — keep
them engine-agnostic (pure functions / small classes over `list[np.ndarray]`).

### FL components (`src/fedcity/fl/`)
- **`strategies.py`** — server aggregation as pure functions following the Reddi et al.
  FedOpt template over the sample-weighted pseudo-gradient `Δ_t`. Critical subtlety:
  `apply_server_update`'s `plain_avg_idx` set forces BatchNorm running buffers
  (`moving_mean`/`moving_variance`) to be FedAvg-averaged rather than pushed through the
  adaptive optimizer — otherwise `moving_variance` goes negative and BN yields NaN.
  `runner._non_trainable_indices` computes that set from the model.
- **`quantize.py`** — FedPAQ / QSGD-style **unbiased** stochastic quantization. Also owns
  byte accounting: upload cost comes from the quantized payload, download from full
  float32. Changing quantization means keeping `quantize_array` unbiased AND
  `quantized_nbytes` consistent with it.
- **`client.py`** — `LocalClient` (set weights → fit τ epochs → return delta) plus
  `ClientProfile` (data size, network, battery, price, recent loss/acc) — the feature
  vector the AUCTION agent observes. `N_CLIENT_FEATURES` must match `feature_vector`.
- **`auction/`** — DRL client selection. `selector.py` exposes the swappable
  `RandomSelector` vs `AuctionSelector` (same `select`/`observe_reward` API); `agent.py`
  is a REINFORCE policy-gradient agent; `env.py` builds the state and computes reward
  (`Δaccuracy − cost_weight · cost`). The agent trains only during `warmup_rounds`.
- **`failures.py`** — per-round client drop at `client_failure_prob` (§3.7.3).
- **`metrics.py`** — `RoundRecord`, `CommAccountant`, `rounds_to_threshold`.

### Data + models
- **`datasets/loaders.py`** returns a `DatasetBundle` (train/val arrays, `input_shape`,
  `task` ∈ {binary, multiclass}, `n_outputs`). Heavy imports (TF, ucimlrepo) are lazy so
  partitioning and unit tests run without them. Healthcare is now **Breast Cancer WDBC
  (UCI id=17)**, top-10 ANOVA-F features — note the module docstring still says CDC
  Diabetes and is stale.
- **`datasets/partition.py`** — IID vs Dirichlet(α) label partitioning (lower α = more
  heterogeneous). `visualize.py` renders the label-distribution figures.
- **`models/`** — one Keras builder per domain (`build_model` dispatches on
  `cfg['domain']`; synthetic picks by task). Param counts are asserted against the thesis
  tables in `tests/test_models.py`, so architecture edits will break those tests by design.

### Scripts and output layout
Scripts (`scripts/run_experiment.py`, `make_partition_figures.py`, `download_*.py`) import
`_bootstrap` first to put `src/` on the path without an editable install. Each run writes
to a timestamped `experiments/<domain>/<YYYYMMDDHHMMSS>/` folder with an
`experiments/<domain>/latest` symlink; `report.py` writes `results_table.{csv,md}`,
`curves/*.png`, and per-run JSON. Reruns never overwrite earlier results.

## Conventions worth preserving
- Determinism: everything is seeded from `cfg['seed']` (global TF seed + per-purpose
  offset RNGs, e.g. `seed + 777` for profiles, `seed + 13` for random selection). Keep new
  randomness seeded the same way so smoke runs stay reproducible.
- Fair strategy comparison: `federated.server_opt` in `base.yaml` gives per-strategy
  server-lr defaults (FedAvg is implicitly lr=1; FedAdam needs a small lr). Don't collapse
  these to a single flat value.
- Tests double as the spec for the non-obvious invariants (quantizer unbiasedness + byte
  accounting, aggregation update rules, model param counts, failure rates, a full FL smoke
  run per strategy). Run them after touching `fl/` or `models/`.
