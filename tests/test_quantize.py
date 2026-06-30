import numpy as np

from fedcity.fl.quantize import (
    bits_per_entry,
    float_nbytes,
    quantize_array,
    quantize_update,
    quantized_nbytes,
)


def test_quantizer_is_unbiased():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(5000).astype(np.float32)
    est = np.mean([quantize_array(v, 128, np.random.default_rng(i)) for i in range(400)], axis=0)
    # E[Q(v)] = v. The aggregate signed bias has tiny variance (~1/sqrt(N*dim)),
    # so it isolates *bias* from per-element rounding noise.
    assert abs(float(np.mean(est - v))) < 2e-3
    # per-element errors are also bounded (SEM-limited, not biased)
    assert np.mean(np.abs(est - v)) < 5e-2


def test_zero_vector_quantizes_to_zero():
    v = np.zeros(100, dtype=np.float32)
    assert np.all(quantize_array(v, 256, np.random.default_rng(0)) == 0)


def test_bits_per_entry():
    assert bits_per_entry(256) == 9      # 8 level bits + 1 sign
    assert bits_per_entry(2) == 2


def test_quantization_reduces_bytes():
    arrs = [np.ones((1000,), dtype=np.float32)]
    assert quantized_nbytes(arrs, 256) < float_nbytes(arrs)
    assert quantized_nbytes(arrs, None) == float_nbytes(arrs)


def test_quantize_update_returns_bytes():
    deltas = [np.random.randn(50, 50).astype(np.float32), np.random.randn(50).astype(np.float32)]
    q, nbytes = quantize_update(deltas, 256, np.random.default_rng(0))
    assert len(q) == 2
    assert nbytes == quantized_nbytes(deltas, 256)
    q_none, nbytes_none = quantize_update(deltas, None, np.random.default_rng(0))
    assert nbytes_none == float_nbytes(deltas)
