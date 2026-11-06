import numpy as np
import pandas as pd
import pytest

from fraud_detection.preprocessing import scale_splits


def test_scaler_is_fitted_on_training_features_only() -> None:
    X_train = pd.DataFrame({"amount": [0.0, 2.0], "signal": [10.0, 30.0]})
    X_validation = pd.DataFrame({"amount": [100.0], "signal": [300.0]})
    X_test = pd.DataFrame({"amount": [-100.0], "signal": [-300.0]})

    scaled = scale_splits(X_train, X_validation, X_test)

    np.testing.assert_allclose(scaled.scaler.mean_, [1.0, 20.0])
    np.testing.assert_allclose(scaled.X_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
    assert not np.allclose(scaled.X_validation.mean(axis=0), [0.0, 0.0])
    np.testing.assert_array_equal(X_train["amount"].to_numpy(), [0.0, 2.0])


def test_scaler_rejects_feature_order_or_schema_mismatch() -> None:
    X_train = pd.DataFrame({"first": [0.0, 1.0], "second": [1.0, 2.0]})
    X_validation = X_train[["second", "first"]]
    X_test = X_train.copy()

    with pytest.raises(ValueError, match="identical columns in identical order"):
        scale_splits(X_train, X_validation, X_test)


def test_scaler_rejects_empty_partition() -> None:
    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    X_validation = pd.DataFrame({"feature": [3.0]})
    X_test = pd.DataFrame({"feature": []}, dtype=np.float64)

    with pytest.raises(ValueError, match="must be non-empty"):
        scale_splits(X_train, X_validation, X_test)
