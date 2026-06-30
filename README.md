# FedCity — Federated Learning for Smart-City Functions

A full, runnable implementation of the methodology in the thesis *"Optimizing Smart
City Functions through Federated Learning: Implementation and Framework Development"*
(`docs/Thesis_190217_200227_.pdf`, Chapter 3).

It compares **federated vs. centralized** learning across three smart-city domains —
**healthcare**, **smart grid (energy)**, and **traffic** — under controlled data
heterogeneity, with the thesis's communication-efficiency (**FedPAQ**), quality-aware
client selection (**AUCTION**), and server aggregation (**FedAvg / FedOpt / FedAdam**).

---

## What is implemented (maps 1:1 to the thesis)

| Thesis § | Component | Where |
|---|---|---|
| §3.2.1, Eq. 3.2, Figs 3.3–3.8 | Dirichlet + IID partitioning & label-distribution figures | [partition.py](src/fedcity/datasets/partition.py), [visualize.py](src/fedcity/datasets/visualize.py) |
| §3.2/3.7/3.8/3.9 | Dataset loaders (CDC Diabetes, smart-grid theft, GTSRB) | [loaders.py](src/fedcity/datasets/loaders.py) |
| Tables 3.1/3.3/3.5 | Keras models (DNN / logistic-reg / deep CNN) — **param counts verified** | [models/](src/fedcity/models/) |
| §3.4 | FedPAQ: periodic local averaging + quantized updates + partial participation | [client.py](src/fedcity/fl/client.py), [quantize.py](src/fedcity/fl/quantize.py), [runner.py](src/fedcity/runner.py) |
| §3.4.1 | gRPC / Flower transport (optional real-distributed path) | [flower_app.py](src/fedcity/fl/flower_app.py) |
| §3.5 | AUCTION DRL client selection (policy-gradient agent) | [fl/auction/](src/fedcity/fl/auction/) |
| §3.6 | FedAvg / FedOpt / FedAdam aggregation | [strategies.py](src/fedcity/fl/strategies.py) |
| §3.7.3 | Client failures (p = 0.2 per round) | [failures.py](src/fedcity/fl/failures.py) |
| Ch. 4 | Metrics: accuracy/loss curves, convergence rate, time, comm cost | [metrics.py](src/fedcity/fl/metrics.py), [report.py](src/fedcity/report.py) |
| §3.7.4/3.8.4/3.9.4 | Centralized baselines | [centralized/train.py](src/fedcity/centralized/train.py) |

### Experiment matrix

| Domain | Dataset | Model | Clients | Rounds | Partitions |
|---|---|---|---|---|---|
| Healthcare | CDC Diabetes Health Indicators (UCI 891) | DNN, 10→256→128→1 | 20 | 30 | IID, Dir(0.3), Dir(5) |
| Smart Grid | Theft Detection (Kaggle) | LogReg NN, (104,1)→1 | 20 | 100 | Dir(0.3), Dir(5) |
| Traffic | GTSRB (43 classes) | CNN, 5 conv blocks → softmax-43 | 20 | 100 | Dir(0.3), Dir(5) |

Each cell runs FedAvg / FedOpt / FedAdam, with AUCTION-vs-random selection and a
centralized baseline for comparison.

---

## Setup

The project targets **Python 3.12** (TensorFlow + Flower compatible). The repo was
bootstrapped with [`uv`](https://docs.astral.sh/uv/), which also manages the interpreter:

```bash
# install uv if needed:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```

All commands below assume `.venv/bin/python` (or activate with `source .venv/bin/activate`).

---

## Quick start

```bash
# 1) End-to-end sanity check — synthetic data, no downloads, ~1 min
.venv/bin/python scripts/run_experiment.py --config healthcare --sweep --synthetic --smoke

# 2) Partition heterogeneity figures (thesis Figs 3.3–3.8)
.venv/bin/python scripts/make_partition_figures.py --domain healthcare

# 3) One real cell: FedAdam on Dirichlet(0.3)
.venv/bin/python scripts/run_experiment.py --config healthcare --strategy fedadam --partition dirichlet --alpha 0.3

# 4) Full real sweep -> tables (3.2/3.4) + curves (3.12–3.19) + centralized comparison
.venv/bin/python scripts/run_experiment.py --config healthcare --sweep
```

Outputs land under `experiments/<domain>/`: `results_table.{csv,md}`, `curves/*.png`,
`runs/*.json`, `centralized.json`, `partitions/*.png`.

### Datasets

```bash
.venv/bin/python scripts/download_healthcare.py            # UCI, automatic
.venv/bin/python scripts/download_smartgrid.py --dataset owner/slug   # needs ~/.kaggle/kaggle.json
.venv/bin/python scripts/download_traffic.py               # GTSRB (~263 MB)
```

- **Healthcare** downloads automatically via `ucimlrepo`.
- **Smart grid** needs a Kaggle slug + credentials; set `dataset.kaggle_dataset` in
  [configs/smartgrid.yaml](configs/smartgrid.yaml).
- **Traffic (GTSRB)** is large; **a GPU is recommended** for the full CNN. On CPU use
  the subsample knob (`dataset.subsample_fraction`, default 0.15) in
  [configs/traffic.yaml](configs/traffic.yaml).

---

## Key flags

| Flag | Effect |
|---|---|
| `--sweep` | run the full strategy × partition matrix + centralized + reports |
| `--strategy {fedavg,fedopt,fedadam}` | single-cell aggregation strategy |
| `--partition {iid,dirichlet} --alpha A` | data heterogeneity |
| `--selector {random,auction}` | random partial participation vs AUCTION DRL |
| `--no-fedpaq` | disable update quantization (ablation) |
| `--synthetic` | run on synthetic data (no downloads) |
| `--smoke` | shrink clients/rounds/data for a fast check |
| `--rounds N` / `--clients N` | override round / client counts |

CPU knobs also live in the YAML configs (`local_batch_size`, traffic `subsample_fraction`).

---

## Design notes

- **Two execution engines.** The default [`runner.py`](src/fedcity/runner.py) is a
  fast, deterministic, single-process simulation that gives full control over FedPAQ
  quantization, AUCTION selection, client failures, and byte-level communication
  accounting — ideal on CPU. The same models/partitions also run on Flower's real
  **gRPC** stack (Ray simulation or separate server/client processes) via
  [`flower_app.py`](src/fedcity/fl/flower_app.py), satisfying the thesis's §3.4.1
  Flower/gRPC transport for genuine distributed runs.
- **FedPAQ** (§3.4) = τ local epochs (periodic averaging) + QSGD-style **unbiased**
  update quantization (`levels=s`) + partial participation (`fraction_fit`). Upload
  cost is accounted from the quantized payload; download from the full float32 model.
- **AUCTION** (§3.5) is a real policy-gradient (REINFORCE + baseline) agent that scores
  clients from a status vector (data size, network, battery, price, recent
  loss/accuracy) and selects a budget-sized subset; reward = validation gain − cost.
- **Fair aggregation comparison.** FedAvg is implicitly server-lr = 1 while FedAdam's
  adaptive denominator needs a small lr; per-strategy `server_opt` defaults in
  [configs/base.yaml](configs/base.yaml) keep the comparison sound.

---

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers partition coverage/heterogeneity, quantizer unbiasedness + byte accounting,
aggregation update rules, **model param-counts vs the thesis tables**, failure rates,
and an end-to-end FL smoke run for every strategy + the AUCTION selector + the FedPAQ
ablation.

---

## Status & notes

- Healthcare runs reproduce the thesis's **FedAdam ≥ FedOpt ≥ FedAvg** ordering. The
  CDC dataset is ~14% positive, so accuracies sit near the thesis's reported values
  (e.g. FedAdam/Dir(0.3) ≈ 0.86); see `experiments/healthcare/results_table.md`.
- The traffic CNN at full GTSRB scale is GPU-recommended; the CPU path subsamples.
- **Open items / assumptions** (documented in the plan): the exact 10-feature selection
  for CDC Diabetes uses ANOVA F-score top-k; the smart-grid (104,1) input aggregates the
  daily series into 104 equal time-bins; FedPAQ `levels` (s) and local epochs (τ) are
  exposed in config with sensible defaults (tune to hit specific comm-cost targets).
