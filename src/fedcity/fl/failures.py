"""Client-failure simulation (thesis §3.7.3): each selected client may drop per round."""
from __future__ import annotations

import numpy as np


def apply_failures(
    selected: list[int], prob: float, rng: np.random.Generator
) -> list[int]:
    """Drop each selected client with probability ``prob``.

    Guarantees at least one survivor so a round always has a participant.
    """
    if prob <= 0 or len(selected) == 0:
        return list(selected)
    survivors = [c for c in selected if rng.random() >= prob]
    if not survivors:
        survivors = [rng.choice(selected)]
    return survivors
