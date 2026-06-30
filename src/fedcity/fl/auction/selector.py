"""Pluggable client-selection interface (thesis §3.5).

``RandomSelector`` is the FedAvg-style random partial-participation baseline.
``AuctionSelector`` wraps the DRL agent: it selects within budget, and (during
warm-up) updates the policy from the round's reward. Both expose the same
``select`` / ``observe_reward`` API so the engine can swap them for ablation.
"""
from __future__ import annotations

import numpy as np

from .env import build_state, compute_reward


class RandomSelector:
    name = "random"

    def __init__(self, budget: int, seed: int = 42):
        self.budget = budget
        self._rng = np.random.default_rng(seed + 13)

    def select(self, rnd: int, profiles, max_samples: int, k: int) -> list[int]:
        k = min(k, self.budget, len(profiles))
        return sorted(int(c) for c in self._rng.choice(len(profiles), size=k, replace=False))

    def observe_reward(self, *args, **kwargs):
        return None


class AuctionSelector:
    name = "auction"

    def __init__(self, agent, budget: int, warmup_rounds: int, cost_weight: float):
        self.agent = agent
        self.budget = budget
        self.warmup_rounds = warmup_rounds
        self.cost_weight = cost_weight
        self._last_state = None
        self._last_selected = None

    def select(self, rnd: int, profiles, max_samples: int, k: int) -> list[int]:
        state = build_state(profiles, max_samples)
        selected = self.agent.select(state, k=min(k, self.budget, len(profiles)))
        self._last_state = state
        self._last_selected = selected
        return selected

    def observe_reward(self, rnd: int, prev_acc: float, new_acc: float, profiles):
        """Feed the round outcome back to the agent (only while warming up)."""
        if self._last_state is None or rnd > self.warmup_rounds:
            return None
        sel_profiles = [profiles[c] for c in self._last_selected]
        reward = compute_reward(prev_acc, new_acc, sel_profiles, self.cost_weight)
        return self.agent.update(self._last_state, self._last_selected, reward)


def make_selector(cfg: dict, seed: int):
    """Build the selector named by ``cfg['federated']['selector']``."""
    fed = cfg["federated"]
    budget = fed["auction"]["budget"]
    kind = fed.get("selector", "random")
    if kind == "random":
        return RandomSelector(budget, seed=seed)
    if kind == "auction":
        from .agent import AuctionAgent
        from ..client import N_CLIENT_FEATURES

        agent = AuctionAgent(
            n_features=N_CLIENT_FEATURES,
            budget=budget,
            policy_lr=fed["auction"]["policy_lr"],
            seed=seed,
        )
        return AuctionSelector(
            agent, budget, fed["auction"]["warmup_rounds"], fed["auction"]["cost_weight"]
        )
    raise ValueError(f"unknown selector {kind!r}")
