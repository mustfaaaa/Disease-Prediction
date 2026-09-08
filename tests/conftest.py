"""Shared fixtures. The dataset is downloaded once and cached in data/raw."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C          # noqa: E402
from src.data_loader import load_clean, split_xy  # noqa: E402


@pytest.fixture(scope="session")
def clean_data():
    df, report = load_clean()
    return df, report


@pytest.fixture(scope="session")
def xy(clean_data):
    df, _ = clean_data
    return split_xy(df)


@pytest.fixture(scope="session")
def artefacts_present() -> bool:
    return (C.MODELS_DIR / "final_model.pkl").exists()


@pytest.fixture(scope="session")
def valid_record() -> dict:
    """A record inside every documented domain; values are not a real patient."""
    return {
        "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240, "fbs": 0,
        "restecg": 0, "thalach": 145, "exang": 1, "oldpeak": 1.5, "slope": 2,
        "ca": 1, "thal": 7,
    }
