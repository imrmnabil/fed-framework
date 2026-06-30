import numpy as np
import pytest

from fedcity.datasets.partition import (
    class_distribution_matrix,
    dirichlet_partition,
    iid_partition,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4, 800)


def _coverage_ok(parts, n):
    allidx = np.concatenate(parts)
    return len(allidx) == n and len(np.unique(allidx)) == n


def test_iid_covers_all_and_is_balanced(labels):
    parts = iid_partition(labels, 10, seed=1)
    assert _coverage_ok(parts, len(labels))
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_covers_all_exactly_once(labels):
    parts = dirichlet_partition(labels, 10, alpha=0.5, seed=1)
    assert _coverage_ok(parts, len(labels))


def test_dirichlet_no_empty_clients(labels):
    parts = dirichlet_partition(labels, 8, alpha=0.1, seed=2, min_per_client=1)
    assert all(len(p) >= 1 for p in parts)


def test_lower_alpha_is_more_heterogeneous(labels):
    """Smaller alpha => more skew: higher mean per-client class concentration."""
    def concentration(alpha):
        parts = dirichlet_partition(labels, 12, alpha=alpha, seed=7)
        mat = class_distribution_matrix(labels, parts, 4).astype(float)
        props = mat / np.clip(mat.sum(1, keepdims=True), 1, None)
        return props.max(axis=1).mean()       # how dominant the top class is per client

    assert concentration(0.1) > concentration(5.0)


def test_distribution_matrix_sums_to_total(labels):
    parts = dirichlet_partition(labels, 6, alpha=1.0, seed=3)
    mat = class_distribution_matrix(labels, parts, 4)
    assert mat.sum() == len(labels)
