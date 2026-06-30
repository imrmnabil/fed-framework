"""Shared compile helper."""
from __future__ import annotations


def compile_model(model, n_outputs: int, lr: float):
    import tensorflow as tf

    if n_outputs == 1:
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model
