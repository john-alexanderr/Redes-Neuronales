"""Lazy TensorFlow/Keras helpers for the required neural-network model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def compute_training_class_weights(y_train: pd.Series | np.ndarray) -> dict[int, float]:
    """Calculate balanced binary class weights from training labels only."""
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Training labels must be a non-empty one-dimensional array.")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("Training labels must contain only binary labels 0 and 1.")
    classes = np.array([0, 1], dtype=np.int32)
    if not set(np.unique(labels).tolist()) == {0, 1}:
        raise ValueError("Both classes must be present in training data to calculate class weights.")

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def build_mlp(*, input_features: int, seed: int = 42):
    """Build the assignment MLP, importing TensorFlow only when this is called."""
    if input_features <= 0:
        raise ValueError("input_features must be a positive integer.")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install the project dependencies to build the Keras model.") from error

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def make_early_stopping(*, patience: int = 5):
    """Return EarlyStopping on validation PR-AUC with best-weight restoration."""
    if patience < 0:
        raise ValueError("patience must be non-negative.")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install the project dependencies to train the Keras model.") from error

    return tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
