"""Smart-grid logistic-regression NN — thesis Table 3.3.

Input(104,1) -> Flatten -> Dense(1, sigmoid)   (105 trainable params)
"""
from __future__ import annotations

from ._common import compile_model


def build_smartgrid_model(input_shape, n_outputs: int, hp: dict):
    import tensorflow as tf
    from tensorflow.keras import layers

    lr = hp.get("lr", 0.001)
    act = "sigmoid" if n_outputs == 1 else "softmax"

    model = tf.keras.Sequential(name="smartgrid_logreg")
    model.add(tf.keras.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_outputs, activation=act))
    return compile_model(model, n_outputs, lr)
