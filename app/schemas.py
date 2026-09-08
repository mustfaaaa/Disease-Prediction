"""Request/response schemas for the prediction API.

Validation is layered on purpose: Pydantic checks the shape (field presence,
types, no stray fields), and ``src.inference.validate_record`` then checks the
clinical domain of every value against the same schema the model was trained
on.  Client-side checks in the browser are a convenience only - the server
never trusts them.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


class PredictRequest(BaseModel):
    """One patient record, in the raw units the UCI dataset uses."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: float | None = Field(default=None, description="Age in years")
    sex: float | None = Field(default=None, description="0 = female, 1 = male")
    cp: float | None = Field(default=None, description="Chest pain type 1-4")
    trestbps: float | None = Field(default=None, description="Resting BP, mm Hg")
    chol: float | None = Field(default=None, description="Serum cholesterol, mg/dl")
    fbs: float | None = Field(default=None, description="Fasting blood sugar >120, 0/1")
    restecg: float | None = Field(default=None, description="Resting ECG 0-2")
    thalach: float | None = Field(default=None, description="Max heart rate, bpm")
    exang: float | None = Field(default=None, description="Exercise angina, 0/1")
    oldpeak: float | None = Field(default=None, description="ST depression, mm")
    slope: float | None = Field(default=None, description="ST slope 1-3")
    ca: float | None = Field(default=None, description="Vessels coloured 0-3")
    thal: float | None = Field(default=None, description="Thallium scan 3/6/7")
    threshold: float | None = Field(
        default=None, ge=0.05, le=0.95,
        description="Optional decision-threshold override for what-if exploration; "
                    "the exported model's own threshold is used when omitted.",
    )

    def features(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=False)
        data.pop("threshold", None)
        return {k: v for k, v in data.items() if k in C.FEATURE_ORDER}


class ErrorResponse(BaseModel):
    error: str
    message: str
    fields: dict[str, str] = Field(default_factory=dict)
