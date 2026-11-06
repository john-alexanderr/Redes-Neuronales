import numpy as np
import pandas as pd
import pytest

from fraud_detection.splitting import chronological_split


def make_ordered_data() -> tuple[pd.DataFrame, pd.Series]:
    times = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 14]
    X = pd.DataFrame({"Time": times, "signal": np.arange(len(times), dtype=np.float64)})
    y = pd.Series(np.resize(np.array([0, 1], dtype=np.int32), len(times)), index=X.index)
    return X, y


def test_chronological_split_is_ordered_non_overlapping_and_keeps_tied_times() -> None:
    X, y = make_ordered_data()

    split = chronological_split(X, y)

    assert split.X_train["Time"].max() < split.X_validation["Time"].min()
    assert split.X_validation["Time"].max() < split.X_test["Time"].min()
    for timestamp in X["Time"].unique():
        memberships = sum(
            timestamp in partition["Time"].to_numpy()
            for partition in (split.X_train, split.X_validation, split.X_test)
        )
        assert memberships == 1
    assert split.y_train.index.equals(split.X_train.index)
    assert split.y_validation.index.equals(split.X_validation.index)
    assert split.y_test.index.equals(split.X_test.index)


def test_chronological_split_reports_realized_proportions() -> None:
    X, y = make_ordered_data()

    split = chronological_split(X, y)

    proportions = split.row_proportions(len(X))
    assert proportions["train"] == pytest.approx(0.65)
    assert proportions["validation"] == pytest.approx(0.15)
    assert proportions["test"] == pytest.approx(0.20)


def test_chronological_boundaries_do_not_depend_on_labels() -> None:
    X, y = make_ordered_data()

    original_split = chronological_split(X, y)
    changed_label_split = chronological_split(X, 1 - y)

    assert original_split.X_train.index.equals(changed_label_split.X_train.index)
    assert original_split.X_validation.index.equals(changed_label_split.X_validation.index)
    assert original_split.X_test.index.equals(changed_label_split.X_test.index)


def test_chronological_split_requires_three_distinct_timestamp_groups() -> None:
    X = pd.DataFrame({"Time": [1, 1, 1], "signal": [0, 1, 2]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="three distinct timestamp groups"):
        chronological_split(X, y)


def test_chronological_split_rejects_misaligned_indexes() -> None:
    X, y = make_ordered_data()
    y.index = y.index + 1

    with pytest.raises(ValueError, match="matching indexes"):
        chronological_split(X, y)
