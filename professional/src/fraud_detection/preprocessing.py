"""Leakage-safe feature scaling for pre-partitioned data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScaledFeatures:
    """Scaled matrices and the training-fitted scaler."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    scaler: StandardScaler


def scale_splits(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> ScaledFeatures:
    """Fit ``StandardScaler`` on training features and transform all partitions."""
    if not all(isinstance(frame, pd.DataFrame) for frame in (X_train, X_validation, X_test)):
        raise TypeError("All feature partitions must be pandas DataFrames.")
    if X_train.empty or X_validation.empty or X_test.empty:
        raise ValueError("Train, validation, and test feature partitions must be non-empty.")
    if not (
        X_train.columns.equals(X_validation.columns)
        and X_train.columns.equals(X_test.columns)
    ):
        raise ValueError("Feature partitions must have identical columns in identical order.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    validation_scaled = scaler.transform(X_validation)
    test_scaled = scaler.transform(X_test)
    return ScaledFeatures(
        X_train=train_scaled,
        X_validation=validation_scaled,
        X_test=test_scaled,
        scaler=scaler,
    )
