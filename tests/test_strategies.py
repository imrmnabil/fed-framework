import numpy as np

from fedcity.fl.strategies import AggregatorState, aggregate_deltas, apply_server_update


def test_aggregate_deltas_is_sample_weighted():
    a = [np.array([1.0, 1.0])]
    b = [np.array([3.0, 3.0])]
    agg = aggregate_deltas([(a, 1), (b, 3)])     # weights 1/4, 3/4
    np.testing.assert_allclose(agg[0], [2.5, 2.5])


def test_fedavg_adds_delta_to_global():
    gw = [np.array([0.0, 0.0])]
    delta = [np.array([0.5, -0.5])]
    new = apply_server_update("fedavg", gw, delta, AggregatorState())
    np.testing.assert_allclose(new[0], [0.5, -0.5])


def test_fedadam_moves_toward_target():
    """Repeated positive pseudo-gradient should steadily increase weights."""
    gw = [np.zeros(3, dtype=np.float32)]
    state = AggregatorState()
    delta = [np.ones(3, dtype=np.float32)]
    for _ in range(20):
        gw = apply_server_update("fedadam", gw, delta, state, server_lr=0.1)
    assert np.all(gw[0] > 0.5)


def test_fedopt_momentum_accumulates():
    gw = [np.zeros(2)]
    state = AggregatorState()
    delta = [np.array([1.0, 1.0])]
    step1 = apply_server_update("fedopt", [g.copy() for g in gw], delta, AggregatorState(), server_lr=1.0)
    # with momentum, later steps move further than the first
    s2 = AggregatorState()
    g = [np.zeros(2)]
    for _ in range(3):
        g = apply_server_update("fedopt", g, delta, s2, server_lr=1.0)
    assert g[0][0] > step1[0][0]


def test_plain_avg_idx_bypasses_server_optimizer():
    """BN-buffer tensors flagged via ``plain_avg_idx`` are FedAvg-averaged (w+Δ)
    instead of taking the adaptive step that can drive a moving_variance negative."""
    gw = [np.zeros(2), np.full(2, 1.0)]          # tensor 1 stands in for moving_variance
    delta = [np.ones(2), np.full(2, -0.3)]
    state = AggregatorState()
    new = apply_server_update(
        "fedadam", gw, delta, state, server_lr=0.05, plain_avg_idx={1},
    )
    np.testing.assert_allclose(new[1], [0.7, 0.7])               # exactly w + Δ
    assert not np.allclose(new[0], gw[0])                        # tensor 0 still optimized
    assert np.allclose(state.m[1], 0.0) and np.allclose(state.v[1], 0.0)  # buffer untouched


def test_unknown_strategy_raises():
    try:
        apply_server_update("nope", [np.zeros(2)], [np.zeros(2)], AggregatorState())
    except ValueError:
        return
    raise AssertionError("expected ValueError")
