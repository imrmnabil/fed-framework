"""Model param-count checks against thesis Tables 3.1, 3.3, 3.5."""
import pytest

pytest.importorskip("tensorflow")

from fedcity.models import build_healthcare_model, build_smartgrid_model, build_traffic_model


def test_smartgrid_logreg_has_105_params():
    # Table 3.3: (104,1) -> Flatten -> Dense(1) = 104 weights + 1 bias
    model = build_smartgrid_model((104, 1), 1, {})
    assert model.count_params() == 105


def test_traffic_cnn_matches_table_3_5():
    # Sum of all layer params in Table 3.5
    model = build_traffic_model((32, 32, 3), 43, {})
    assert model.count_params() == 8_022_699
    # first conv block: 3x3x3x64 + 64 = 1792
    assert model.layers[0].count_params() == 1792


def test_healthcare_dnn_structure():
    model = build_healthcare_model((10,), 1, {})
    # 10->256 (BN) ->128 (BN) ->1 ; trainable params per Table 3.1
    assert model.count_params() == 37_377
    assert model.output_shape[-1] == 1
