import numpy as np

from fedcity.fl.failures import apply_failures


def test_no_failures_when_prob_zero():
    sel = [0, 1, 2, 3]
    assert apply_failures(sel, 0.0, np.random.default_rng(0)) == sel


def test_always_at_least_one_survivor():
    for seed in range(20):
        out = apply_failures([0, 1, 2], 1.0, np.random.default_rng(seed))
        assert len(out) >= 1


def test_failure_rate_is_approximately_prob():
    rng = np.random.default_rng(0)
    sel = list(range(1000))
    survivors = apply_failures(sel, 0.2, rng)
    dropped = 1 - len(survivors) / len(sel)
    assert 0.15 < dropped < 0.25
