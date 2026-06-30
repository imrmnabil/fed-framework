"""Evaluation + accounting helpers (thesis Ch. 4 metrics)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundRecord:
    rnd: int
    eval_loss: float
    eval_acc: float
    n_selected: int
    upload_bytes: int
    download_bytes: int


@dataclass
class CommAccountant:
    """Accumulates federated upload (client→server) + download (server→client) bytes."""
    upload_bytes: int = 0
    download_bytes: int = 0

    def add(self, upload: int = 0, download: int = 0):
        self.upload_bytes += int(upload)
        self.download_bytes += int(download)

    @property
    def total_bytes(self) -> int:
        return self.upload_bytes + self.download_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


def rounds_to_threshold(acc_history: list[float], threshold: float) -> int | None:
    """First 1-indexed round whose eval accuracy reaches ``threshold`` (convergence rate)."""
    for i, a in enumerate(acc_history, start=1):
        if a >= threshold:
            return i
    return None


def mb(num_bytes: float) -> float:
    return num_bytes / (1024 * 1024)


def reduction_pct(fl_mb: float, cl_mb: float) -> float:
    """Communication-cost reduction of FL vs centralized, as a percentage (thesis §3.7.3)."""
    if cl_mb == 0:
        return 0.0
    return (cl_mb - fl_mb) / cl_mb * 100.0
