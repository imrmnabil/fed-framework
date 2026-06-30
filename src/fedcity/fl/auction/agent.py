"""DRL policy for AUCTION client selection (thesis §3.5).

A score network (the *encoder*) maps each client's status vector to a scalar logit;
a Plackett-Luce top-k sampler (the *decoder*) draws a budget-sized subset via
Gumbel-perturbed scores. The policy is trained with REINFORCE using a moving-average
return baseline (the critic), where the reward is the validation gain minus cost
(see ``env.compute_reward``). This mirrors AUCTION's RL formulation: observe client
status → select subset within budget → receive FL-performance reward → update policy.
"""
from __future__ import annotations

import numpy as np


class AuctionAgent:
    def __init__(
        self,
        n_features: int,
        budget: int,
        policy_lr: float = 1e-3,
        baseline_momentum: float = 0.9,
        seed: int = 42,
    ):
        import tensorflow as tf

        self.budget = budget
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum
        self._rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)

        self.score_net = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(n_features,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="auction_policy",
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)

    # -- selection (decoder) ------------------------------------------------- #
    def select(self, state: np.ndarray, k: int | None = None) -> list[int]:
        """Sample a subset of size ``k`` (default budget) via Gumbel-top-k on scores."""
        k = min(k or self.budget, state.shape[0])
        scores = self.score_net(state, training=False).numpy().ravel()
        gumbel = -np.log(-np.log(self._rng.random(scores.shape) + 1e-12) + 1e-12)
        perturbed = scores + gumbel
        selected = np.argsort(-perturbed)[:k]
        return sorted(int(i) for i in selected)

    # -- learning (REINFORCE w/ baseline) ------------------------------------ #
    def update(self, state: np.ndarray, selected: list[int], reward: float) -> float:
        """One policy-gradient step toward selections that earned above-baseline reward."""
        import tensorflow as tf

        advantage = reward - self.baseline
        self.baseline = (
            self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward
        )

        state_t = tf.convert_to_tensor(state, dtype=tf.float32)
        sel_idx = tf.constant(selected, dtype=tf.int32)
        with tf.GradientTape() as tape:
            scores = tf.reshape(self.score_net(state_t, training=True), [-1])
            log_probs = tf.nn.log_softmax(scores)               # Plackett-Luce surrogate
            log_prob_selected = tf.reduce_sum(tf.gather(log_probs, sel_idx))
            loss = -advantage * log_prob_selected
        grads = tape.gradient(loss, self.score_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.score_net.trainable_variables))
        return float(loss.numpy())
