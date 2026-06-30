"""AUCTION environment helpers (thesis §3.5).

The "environment" is the FL task itself: the agent observes client status, selects
a subset within budget B, the selected clients train, and the resulting validation
improvement is returned as reward. These helpers build the state matrix and the
reward signal; the round dynamics live in the engine (``runner.py``).
"""
from __future__ import annotations

import numpy as np


def build_state(profiles, max_samples: int) -> np.ndarray:
    """Stack per-client feature vectors into an (N_clients, F) state matrix."""
    return np.stack([p.feature_vector(max_samples) for p in profiles], axis=0)


def compute_reward(
    prev_acc: float,
    new_acc: float,
    selected_profiles,
    cost_weight: float,
) -> float:
    """Reward = validation-accuracy gain − cost_weight · normalized selection cost.

    Encourages selecting clients that most improve the global model while
    respecting the budget (cheaper, higher-quality clients).
    """
    gain = new_acc - prev_acc
    if selected_profiles:
        norm_cost = float(np.mean([p.price for p in selected_profiles]))
    else:
        norm_cost = 0.0
    return float(gain - cost_weight * norm_cost)
