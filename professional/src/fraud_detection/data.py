"""Dataset download, loading, and schema validation utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

DATASET_HANDLE = "mlg-ulb/creditcardfraud"
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ("Time",) + tuple(f"V{index}" for index in range(1, 29)) + ("Amount",)


def validate_credit_card_schema(frame: pd.DataFrame) -> None:
    """Raise a clear error when a frame does not match the expected dataset schema."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected the credit-card dataset as a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The dataset must contain at least one row.")
    if frame.columns.has_duplicates:
        raise ValueError("The dataset contains duplicate column names.")

    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    non_numeric = [
        column
        for column in FEATURE_COLUMNS + (TARGET_COLUMN,)
        if not is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric dataset columns, found non-numeric: {non_numeric}")

    required_data = frame.loc[:, list(FEATURE_COLUMNS + (TARGET_COLUMN,))]
    if required_data.isna().any().any():
        raise ValueError("Required dataset columns must not contain missing values.")

    values = required_data.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Required dataset columns must contain only finite values.")

    target_values = set(frame[TARGET_COLUMN].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 (legitimate) and 1 (fraud).")


def load_credit_card_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a local copy of ``creditcard.csv``."""
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Credit-card CSV was not found at: {path}")

    frame = pd.read_csv(path)
    validate_credit_card_schema(frame)
    return frame


def download_credit_card_csv(destination: str | Path) -> Path:
    """Download Kaggle's public CSV and stage it at ``destination``.

    KaggleHub may require internet access and user consent. This function never
    reads, requests, or writes credentials; configure KaggleHub outside the code
    if Kaggle requires authentication.
    """
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Install the project dependencies to use KaggleHub.") from error

    downloaded_path_text = kagglehub.dataset_download(DATASET_HANDLE, path="creditcard.csv")
    downloaded_path = Path(downloaded_path_text)
    if not downloaded_path.exists():
        raise FileNotFoundError(
            f"KaggleHub returned a path that does not exist: {downloaded_path_text!r}"
        )
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"KaggleHub returned a path that is not a file: {downloaded_path_text!r}"
        )

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != destination_path.resolve():
        shutil.copy2(downloaded_path, destination_path)
    if not destination_path.exists() or not destination_path.is_file():
        raise FileNotFoundError(f"The dataset copy was not created at: {destination_path}")
    return destination_path
