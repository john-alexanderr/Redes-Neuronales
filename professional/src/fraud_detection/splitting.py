"""Chronological data splitting that keeps tied timestamps together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """Features and labels for ordered train, validation, and test partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    def row_proportions(self, total_rows: int) -> dict[str, float]:
        """Return realized row proportions for this split."""
        if total_rows <= 0:
            raise ValueError("total_rows must be positive.")
        return {
            "train": len(self.X_train) / total_rows,
            "validation": len(self.X_validation) / total_rows,
            "test": len(self.X_test) / total_rows,
        }


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    time_column: str = "Time",
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split by time alone, choosing nearby boundaries between timestamp groups.

    Target values are carried into the resulting partitions but never influence
    ordering or boundary selection. Tied timestamps are never divided between
    partitions. Requested fractions are approximate when a timestamp group
    straddles a desired row-count boundary.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a pandas DataFrame and y must be a pandas Series.")
    if time_column not in X.columns:
        raise ValueError(f"Time column {time_column!r} is missing from X.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have equal lengths and matching indexes.")
    if len(X) < 3:
        raise ValueError("At least three rows are required for non-empty partitions.")
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1.")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction plus validation_fraction must be less than 1.")

    try:
        times = X[time_column].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Time column {time_column!r} must be numeric.") from error
    if not np.isfinite(times).all():
        raise ValueError(f"Time column {time_column!r} must contain only finite values.")

    sorted_positions = np.argsort(times, kind="stable")
    sorted_times = times[sorted_positions]
    group_ends = np.flatnonzero(
        np.r_[sorted_times[1:] != sorted_times[:-1], True]
    ) + 1
    legal_boundaries = group_ends[:-1]
    if len(legal_boundaries) < 2:
        raise ValueError(
            "At least three distinct timestamp groups are required to create "
            "non-empty train, validation, and test partitions without splitting ties."
        )

    train_target = len(X) * train_fraction
    validation_target = len(X) * (train_fraction + validation_fraction)
    train_boundary_options = legal_boundaries[:-1]
    train_boundary = int(
        train_boundary_options[np.argmin(np.abs(train_boundary_options - train_target))]
    )
    later_boundaries = legal_boundaries[legal_boundaries > train_boundary]
    validation_boundary = int(
        later_boundaries[np.argmin(np.abs(later_boundaries - validation_target))]
    )

    train_positions = sorted_positions[:train_boundary]
    validation_positions = sorted_positions[train_boundary:validation_boundary]
    test_positions = sorted_positions[validation_boundary:]

    return ChronologicalSplit(
        X_train=X.iloc[train_positions].copy(),
        X_validation=X.iloc[validation_positions].copy(),
        X_test=X.iloc[test_positions].copy(),
        y_train=y.iloc[train_positions].copy(),
        y_validation=y.iloc[validation_positions].copy(),
        y_test=y.iloc[test_positions].copy(),
    )
