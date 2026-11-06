"""Probability extraction, validation threshold selection, and binary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Thresholded metrics plus ranking metrics for a binary classifier."""

    threshold: float
    accuracy: float
    average_precision: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


def _validated_labels_and_probabilities(
    y_true: pd.Series | np.ndarray,
    positive_probabilities: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64).reshape(-1)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same non-zero length.")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Positive-class probabilities must be finite values in [0, 1].")
    return labels.astype(np.int32, copy=False), probabilities


def positive_class_probabilities(
    estimator: object,
    X: pd.DataFrame | np.ndarray,
    *,
    positive_label: int = 1,
) -> np.ndarray:
    """Return ``predict_proba`` values for the positive class, independent of class order."""
    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
        raise TypeError("The fitted estimator must expose predict_proba and classes_.")

    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(X):
        raise ValueError("predict_proba returned an unexpected shape.")
    if probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba columns do not match estimator.classes_.")

    matching_class = np.flatnonzero(classes == positive_label)
    if len(matching_class) == 0:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return _validated_labels_and_probabilities(
        np.zeros(probabilities.shape[0], dtype=np.int32),
        probabilities[:, int(matching_class[0])],
    )[1]


def select_f1_threshold(
    y_validation: pd.Series | np.ndarray,
    positive_probabilities: np.ndarray | list[float],
) -> float:
    """Choose the validation threshold that maximizes F1 (ties use the lowest threshold)."""
    labels, probabilities = _validated_labels_and_probabilities(
        y_validation, positive_probabilities
    )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Validation threshold selection requires both classes to be present.")

    precisions, recalls, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced from validation probabilities.")
    denominators = precisions[:-1] + recalls[:-1]
    f1_values = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        denominators,
        out=np.zeros_like(denominators),
        where=denominators > 0,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    positive_probabilities: np.ndarray | list[float],
    *,
    threshold: float,
) -> BinaryClassificationMetrics:
    """Calculate sklearn Average Precision, ROC-AUC, and thresholded metrics.

    Average Precision is left undefined when there are no positive examples.
    ROC-AUC is left undefined whenever either class is absent.
    """
    labels, probabilities = _validated_labels_and_probabilities(
        y_true, positive_probabilities
    )
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value in [0, 1].")

    predictions = (probabilities >= threshold).astype(np.int32)
    average_precision = (
        float(average_precision_score(labels, probabilities))
        if np.any(labels == 1)
        else float("nan")
    )
    roc_auc = (
        float(roc_auc_score(labels, probabilities))
        if set(np.unique(labels).tolist()) == {0, 1}
        else float("nan")
    )
    return BinaryClassificationMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(labels, predictions)),
        average_precision=average_precision,
        roc_auc=roc_auc,
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]),
    )
