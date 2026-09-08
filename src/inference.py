"""Inference pipeline (Phase 21).

    USER INPUT -> VALIDATION -> PREPROCESSING -> MODEL -> PROBABILITY
              -> CLASSIFICATION -> EXPLANATION

The artefacts are loaded once into a module-level singleton, so the API does
not touch disk per request.  Nothing in this module produces a number that did
not come out of the exported model.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from . import config as C
from .utils import read_json


class ModelNotTrainedError(RuntimeError):
    """Raised when the artefacts are missing - the app must tell the user to train."""


class ValidationError(ValueError):
    """Raised when the submitted record cannot be turned into a model input."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


REQUIRED_ARTEFACTS = ("final_model.pkl", "local_explainer.pkl",
                      "metadata.json", "metrics.json")


# --------------------------------------------------------------------------
def validate_record(payload: dict, *, allow_missing: bool = False) -> pd.DataFrame:
    """Server-side validation.  Client-side checks are a convenience, not a gate.

    Returns a single-row DataFrame with columns in the exact training order.
    """
    errors: dict[str, str] = {}
    row: dict[str, float] = {}

    unknown = set(payload) - set(C.FEATURE_ORDER)
    for key in sorted(unknown):
        errors[key] = "Unknown field - not part of this model's input schema."

    for spec in C.FEATURES:
        name, label = spec["name"], spec["label"]
        raw = payload.get(name, None)

        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            if allow_missing:
                row[name] = float("nan")
                continue
            errors[name] = f"{label} is required."
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue

        if math.isnan(value) or math.isinf(value):
            errors[name] = f"{label} must be a finite number."
            continue

        if spec["role"] == "numeric":
            lo, hi = spec["min"], spec["max"]
            if not (lo <= value <= hi):
                errors[name] = (f"{label} must be between {lo} and {hi} "
                                f"{spec['unit'] or ''}".strip() + ".")
                continue
            if spec.get("step") == 1 and abs(value - round(value)) > 1e-9:
                errors[name] = f"{label} must be a whole number."
                continue
        else:
            allowed = [float(c["value"]) for c in spec["choices"]]
            if value not in allowed:
                readable = ", ".join(
                    f"{c['value']} ({c['label']})" for c in spec["choices"])
                errors[name] = f"{label} must be one of: {readable}."
                continue
        row[name] = value

    if errors:
        raise ValidationError(errors)
    return pd.DataFrame([row], columns=C.FEATURE_ORDER).astype(float)


# --------------------------------------------------------------------------
@dataclass
class PredictionResult:
    classification: str
    predicted_label: int
    probability: float
    threshold: float
    confidence_band: str
    explanation: dict
    model: dict

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "predicted_label": self.predicted_label,
            "probability": self.probability,
            "probability_percent": round(self.probability * 100, 1),
            "threshold": self.threshold,
            "threshold_percent": round(self.threshold * 100, 1),
            "margin_from_threshold": round(self.probability - self.threshold, 4),
            "confidence_band": self.confidence_band,
            "explanation": self.explanation,
            "model": self.model,
            "disclaimer": (
                "Educational use only. This is a model-estimated probability "
                "from a classifier trained on a small public research dataset. "
                "It is not a medical diagnosis, not medical advice, and not a "
                "substitute for assessment by a qualified clinician."
            ),
        }


class Predictor:
    """Loads the exported artefacts once and serves predictions."""

    def __init__(self, models_dir: Path | None = None):
        self.dir = Path(models_dir or C.MODELS_DIR)
        missing = [f for f in REQUIRED_ARTEFACTS if not (self.dir / f).exists()]
        if missing:
            raise ModelNotTrainedError(
                f"Missing model artefacts in {self.dir}: {', '.join(missing)}. "
                "Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(self.dir / "final_model.pkl")
        self.explainer = joblib.load(self.dir / "local_explainer.pkl")
        self.metadata = read_json(self.dir / "metadata.json")
        self.metrics = read_json(self.dir / "metrics.json")
        self.threshold = float(
            self.metadata["final_model"]["decision_threshold"])
        self.model_name = self.metadata["final_model"]["name"]

    # ------------------------------------------------------------------
    @property
    def model_card(self) -> dict:
        fm = self.metadata["final_model"]
        test = self.metrics["final_test_metrics_chosen_threshold"]
        return {
            "name": fm["name"],
            "estimator": fm["estimator"],
            "dataset": self.metadata["dataset"]["name"],
            "dataset_rows": self.metadata["dataset"]["rows"],
            "input_features": self.metadata["dataset"]["features"],
            "encoded_features": self.metrics["explainability"]["encoded_feature_count"],
            "classes": self.metadata["dataset"]["classes"],
            "threshold": fm["decision_threshold"],
            "calibration": fm["calibration"],
            "explanation_method": fm["local_explanation_method"],
            "test_roc_auc": test["roc_auc"],
            "test_pr_auc": test["pr_auc"],
            "test_accuracy": test["accuracy"],
            "test_sensitivity": test["sensitivity"],
            "test_specificity": test["specificity"],
            "trained_utc": self.metadata["generated_utc"],
        }

    # ------------------------------------------------------------------
    def predict(self, payload: dict, *, top_k: int = 6,
                threshold: float | None = None) -> PredictionResult:
        X = validate_record(payload)
        probability = float(self.pipeline.predict_proba(X)[0, 1])
        thr = float(self.threshold if threshold is None else threshold)
        label = int(probability >= thr)
        explanation = self.explainer.explain(X, top_k=top_k)

        distance = abs(probability - thr)
        band = ("near the decision threshold" if distance < 0.10
                else "clearly on one side of the threshold" if distance > 0.25
                else "moderately separated from the threshold")

        return PredictionResult(
            classification=C.TARGET_LABELS[label],
            predicted_label=label,
            probability=round(probability, 4),
            threshold=round(thr, 4),
            confidence_band=band,
            explanation=explanation,
            model=self.model_card,
        )


# --------------------------------------------------------------------------
_predictor: Predictor | None = None
_lock = threading.Lock()


def get_predictor(reload: bool = False) -> Predictor:
    """Process-wide singleton: the model is deserialised at most once."""
    global _predictor
    if _predictor is None or reload:
        with _lock:
            if _predictor is None or reload:
                _predictor = Predictor()
    return _predictor


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json

    p = get_predictor()
    sample = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
              "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
              "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
    print(json.dumps(p.predict(sample).to_dict(), indent=2))
