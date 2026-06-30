"""Traffic CNN — thesis Table 3.5.

Five Conv blocks (64->1024), each Conv2D(3x3,same,relu)+BN+MaxPool(2,2)+Dropout(0.25);
Flatten; Dense(1024/512/256/128, relu); Dropout(0.5); Dense(43, softmax).
"""
from __future__ import annotations

from ._common import compile_model


def build_traffic_model(input_shape, n_outputs: int, hp: dict):
    import tensorflow as tf
    from tensorflow.keras import layers

    conv_filters = hp.get("conv_filters", [64, 128, 256, 512, 1024])
    dense_units = hp.get("dense_units", [1024, 512, 256, 128])
    conv_dropout = hp.get("conv_dropout", 0.25)
    dense_dropout = hp.get("dense_dropout", 0.5)
    lr = hp.get("lr", 0.001)

    model = tf.keras.Sequential(name="traffic_cnn")
    model.add(tf.keras.Input(shape=input_shape))
    for f in conv_filters:
        model.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(conv_dropout))
    model.add(layers.Flatten())
    for u in dense_units:
        model.add(layers.Dense(u, activation="relu"))
    model.add(layers.Dropout(dense_dropout))
    act = "sigmoid" if n_outputs == 1 else "softmax"
    model.add(layers.Dense(n_outputs, activation=act))
    return compile_model(model, n_outputs, lr)
