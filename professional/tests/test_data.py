import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import FEATURE_COLUMNS, load_credit_card_csv, validate_credit_card_schema


def make_valid_frame(rows: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: np.arange(rows, dtype=np.float64) for column in FEATURE_COLUMNS}
    )
    frame["Class"] = np.resize(np.array([0, 1], dtype=np.int32), rows)
    return frame


def test_schema_accepts_expected_numeric_binary_columns() -> None:
    validate_credit_card_schema(make_valid_frame())


def test_schema_rejects_missing_required_column() -> None:
    frame = make_valid_frame().drop(columns="V28")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_credit_card_schema(frame)


def test_schema_rejects_non_numeric_feature() -> None:
    frame = make_valid_frame()
    frame["V1"] = ["bad"] * len(frame)

    with pytest.raises(TypeError, match="non-numeric"):
        validate_credit_card_schema(frame)


def test_schema_rejects_non_binary_target() -> None:
    frame = make_valid_frame()
    frame.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="binary labels"):
        validate_credit_card_schema(frame)


def test_schema_rejects_missing_and_infinite_values() -> None:
    missing_value_frame = make_valid_frame()
    missing_value_frame.loc[0, "Amount"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_credit_card_schema(missing_value_frame)

    infinite_value_frame = make_valid_frame()
    infinite_value_frame.loc[0, "Time"] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        validate_credit_card_schema(infinite_value_frame)


def test_load_credit_card_csv_validates_a_local_csv(tmp_path) -> None:
    csv_path = tmp_path / "creditcard.csv"
    expected = make_valid_frame()
    expected.to_csv(csv_path, index=False)

    loaded = load_credit_card_csv(csv_path)

    assert loaded.shape == expected.shape
    assert loaded.columns.tolist() == expected.columns.tolist()


def test_load_credit_card_csv_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_credit_card_csv(tmp_path / "missing.csv")
