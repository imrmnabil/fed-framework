"""Healthcare DNN — thesis Table 3.1.

Input(10) -> [Dense(256,relu,L2) -> BN -> Dropout(0.3)]
          -> [Dense(128,relu,L2) -> BN -> Dropout(0.3)] -> Dense(1, sigmoid)
"""
from __future__ import annotations

from ._common import compile_model


def build_healthcare_model(input_shape, n_outputs: int, hp: dict):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    hidden = hp.get("hidden_units", [256, 128])
    dropout = hp.get("dropout", 0.3)
    l2 = hp.get("l2", 0.01)
    lr = hp.get("lr", 0.001)

    model = tf.keras.Sequential(name="healthcare_dnn")
    model.add(tf.keras.Input(shape=input_shape))
    for units in hidden:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    act = "sigmoid" if n_outputs == 1 else "softmax"
    model.add(layers.Dense(n_outputs, activation=act))
    return compile_model(model, n_outputs, lr)
