"""Small, interpretable baselines for contextualizing the neural network."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_majority_baseline() -> DummyClassifier:
    """Return a classifier that always predicts the most frequent training class."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(*, random_state: int = 42) -> Pipeline:
    """Return a balanced logistic-regression benchmark with train-fitted scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
