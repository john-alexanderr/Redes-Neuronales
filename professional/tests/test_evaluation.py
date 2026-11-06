import numpy as np
import pytest

from fraud_detection.evaluation import (
    evaluate_binary_classifier,
    positive_class_probabilities,
    select_f1_threshold,
)


class ReverseClassOrderEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.1, 0.9]])[: len(X)]


def test_positive_probabilities_follow_estimator_class_order() -> None:
    probabilities = positive_class_probabilities(ReverseClassOrderEstimator(), np.zeros((2, 1)))

    np.testing.assert_allclose(probabilities, [0.8, 0.1])


def test_validation_threshold_maximizes_f1_for_the_candidates() -> None:
    y_validation = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.9])

    threshold = select_f1_threshold(y_validation, probabilities)

    assert threshold == pytest.approx(0.35)


def test_threshold_selection_requires_both_validation_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        select_f1_threshold(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_metrics_include_average_precision_roc_auc_and_fixed_threshold_counts() -> None:
    metrics = evaluate_binary_classifier(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    np.testing.assert_array_equal(metrics.confusion_matrix, [[2, 0], [0, 2]])


def test_roc_auc_is_undefined_when_test_labels_contain_one_class() -> None:
    metrics = evaluate_binary_classifier(
        np.array([0, 0, 0]),
        np.array([0.1, 0.2, 0.3]),
        threshold=0.5,
    )

    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.average_precision)
    np.testing.assert_array_equal(metrics.confusion_matrix, [[3, 0], [0, 0]])


def test_evaluation_rejects_probabilities_outside_zero_to_one() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        evaluate_binary_classifier(np.array([0, 1]), np.array([-0.1, 0.9]), threshold=0.5)
