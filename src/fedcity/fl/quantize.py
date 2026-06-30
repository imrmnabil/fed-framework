"""FedPAQ low-precision quantization of model updates (thesis §3.4).

Implements QSGD-style stochastic quantization (Alistarh et al. 2017): for a tensor
``v`` with L2 norm ``‖v‖``, each entry is randomly rounded to one of ``s`` evenly
spaced levels in ``[0, ‖v‖]`` such that the result is **unbiased** (E[Q(v)] = v).
This is exactly the compression FedPAQ applies to the local update before upload.

``quantized_nbytes`` accounts the wire size: one float32 norm per tensor plus
``ceil(log2(s)) + 1`` bits per entry (level index + sign). With ``levels=None``
quantization is disabled and updates cost the full float32 size.
"""
from __future__ import annotations

import math

import numpy as np


def quantize_array(v: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Stochastically quantize ``v`` to ``s`` levels; returns the dequantized tensor."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v.ravel())
    if norm == 0 or s is None:
        return v.astype(np.float32)

    ratios = np.abs(v) / norm                 # in [0, 1]
    lower = np.floor(ratios * s)              # lower level index
    prob_up = ratios * s - lower              # P(round up)
    xi = lower + (rng.random(v.shape) < prob_up)   # stochastic rounding
    q = np.sign(v) * norm * xi / s
    return q.astype(np.float32)


def bits_per_entry(s: int) -> int:
    """Level-index bits + 1 sign bit."""
    return max(1, math.ceil(math.log2(s))) + 1


def quantized_nbytes(arrays: list[np.ndarray], s: int | None) -> int:
    """Wire size (bytes) for a list of update tensors under quantization level ``s``."""
    total = 0
    for a in arrays:
        n = int(np.asarray(a).size)
        if s is None:
            total += n * 4                     # float32, no compression
        else:
            total += math.ceil(n * bits_per_entry(s) / 8) + 4   # +4 bytes for the norm
    return total


def float_nbytes(arrays: list[np.ndarray]) -> int:
    """Uncompressed float32 size (bytes) — used for the server->client download leg."""
    return int(sum(np.asarray(a).size for a in arrays) * 4)


def quantize_update(
    deltas: list[np.ndarray], s: int | None, rng: np.random.Generator
) -> tuple[list[np.ndarray], int]:
    """Quantize each tensor of an update; return (dequantized update, upload bytes)."""
    if s is None:
        return [np.asarray(d, dtype=np.float32) for d in deltas], float_nbytes(deltas)
    q = [quantize_array(d, s, rng) for d in deltas]
    return q, quantized_nbytes(deltas, s)
