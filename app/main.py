"""FastAPI service for the heart-disease risk classification demo.

The model is deserialised once at start-up (``src.inference.get_predictor``) and
reused for every request - no per-request disk access.  Errors are converted to
plain, user-facing JSON; no Python traceback ever reaches the browser.

Run with:  python -m uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C           # noqa: E402
from src.data_loader import feature_dictionary  # noqa: E402
from src.inference import (           # noqa: E402
    ModelNotTrainedError, ValidationError, get_predictor,
)
from src.utils import jsonable        # noqa: E402
from app.schemas import PredictRequest  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("disease-prediction")

STATIC_DIR = Path(__file__).parent / "static"

_startup: dict = {"model_loaded": False, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Deserialise the model once, at start-up, not per request."""
    try:
        p = get_predictor()
        _startup.update(model_loaded=True, error=None)
        log.info("Loaded %s (threshold %.2f) from %s",
                 p.model_name, p.threshold, C.MODELS_DIR)
    except ModelNotTrainedError as exc:
        _startup.update(model_loaded=False, error=str(exc))
        log.error("Model artefacts unavailable: %s", exc)
    yield


app = FastAPI(
    title="Disease Risk Prediction API",
    version="1.0.0",
    description=(
        "Educational machine-learning demo. Classifies structured cardiology "
        "records from the UCI Heart Disease (Cleveland) dataset. Not a medical "
        "device and not a diagnostic tool."
    ),
    lifespan=lifespan,
)


# ------------------------------------------------------------------ errors
def _json_error(status: int, error: str, message: str, fields: dict | None = None):
    return JSONResponse(status_code=status,
                        content={"error": error, "message": message,
                                 "fields": fields or {}})


@app.exception_handler(RequestValidationError)
async def _pydantic_error(request: Request, exc: RequestValidationError):
    fields: dict[str, str] = {}
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        key = loc[-1] if loc else "request"
        spec = C.FEATURE_BY_NAME.get(key)
        label = spec["label"] if spec else key
        if err.get("type") == "extra_forbidden":
            fields[key] = f"'{key}' is not part of this model's input schema."
        elif err.get("type", "").startswith("missing"):
            fields[key] = f"{label} is required."
        else:
            fields[key] = f"{label}: {err.get('msg', 'invalid value')}."
    return _json_error(400, "invalid_request",
                       "Some fields need attention before the model can run.",
                       fields)


@app.exception_handler(ValidationError)
async def _domain_error(request: Request, exc: ValidationError):
    return _json_error(400, "invalid_request",
                       "Some fields need attention before the model can run.",
                       exc.errors)


@app.exception_handler(ModelNotTrainedError)
async def _no_model(request: Request, exc: ModelNotTrainedError):
    return _json_error(503, "model_unavailable",
                       "The prediction model is not available on this server. "
                       "Train it with `python -m src.train`, then restart.")


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return _json_error(500, "server_error",
                       "Something went wrong while processing the request. "
                       "Please try again.")


# -------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {
        "status": "ok" if _startup["model_loaded"] else "degraded",
        "model_loaded": _startup["model_loaded"],
        "detail": _startup["error"],
    }


@app.get("/api/schema")
def schema():
    """Everything the browser needs to render the form - straight from the
    training-time schema, so the UI can never drift from the model."""
    return jsonable({
        "features": C.FEATURES,
        "groups": C.GROUP_ORDER,
        "feature_order": C.FEATURE_ORDER,
        "labels": C.TARGET_LABELS,
        "target_description": C.TARGET_DESCRIPTION,
        "dictionary": feature_dictionary(),
        "dataset": {"name": C.DATASET_NAME, "page": C.DATASET_PAGE},
    })


@app.get("/api/model")
def model_card():
    p = get_predictor()
    return jsonable({
        "card": p.model_card,
        "metadata": p.metadata,
        "threshold_rationale": p.metadata["final_model"]["threshold_rationale"],
        "selection": p.metrics["selection"],
    })


@app.get("/api/metrics")
def metrics():
    """Real numbers from the last training run - the performance page reads
    this and renders it; nothing there is written by hand."""
    p = get_predictor()
    m = p.metrics
    return jsonable({
        "comparison_table": m["comparison_table"],
        "final_model": m["final_model"],
        "test_default": m["final_test_metrics_default_threshold"],
        "test_chosen": m["final_test_metrics_chosen_threshold"],
        "curves": m["curves"],
        "threshold_analysis": m["threshold_analysis"],
        "threshold_sweep": m["threshold_sweep"],
        "calibration": m["calibration"],
        "explainability": {
            "permutation_importance": m["explainability"]["permutation_importance"],
            "coefficients": m["explainability"].get("coefficients", [])[:14],
            "impurity_importance": m["explainability"].get("impurity_importance", [])[:14],
            "shap_summary": m["explainability"].get("shap_summary", {}),
            "permutation_note": m["explainability"]["permutation_note"],
        },
        "error_analysis": m["error_analysis"],
        "imbalance_study": m["imbalance_study"],
        "early_stopping": m["early_stopping"],
        "eda": m["eda"],
        "selection": m["selection"],
        "dataset": p.metadata["dataset"],
        "split": p.metadata["split"],
        "environment": p.metadata["environment"],
        "per_model_details": m["per_model_details"],
    })


@app.post("/api/predict")
def predict(payload: PredictRequest):
    started = time.perf_counter()
    p = get_predictor()
    result = p.predict(payload.features(), threshold=payload.threshold)
    body = result.to_dict()
    body["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return jsonable(body)


@app.get("/api/example")
def example():
    """A real record from the held-out test set, so the demo can be tried
    without inventing plausible-looking clinical values."""
    import random
    import pandas as pd

    path = C.PREDICTIONS / "test_predictions.csv"
    if not path.exists():
        return _json_error(404, "not_found", "No held-out predictions on disk yet.")
    df = pd.read_csv(path)
    row = df.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    return jsonable({
        "record": {f: float(row[f]) for f in C.FEATURE_ORDER},
        "true_label": C.TARGET_LABELS[int(row["true_label"])],
        "note": ("Loaded from the held-out test split. The recorded outcome is "
                 "shown only after you run the model, so the comparison is fair."),
    })


# ----------------------------------------------------------------- static
@app.get("/")
@app.head("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
