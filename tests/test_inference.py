"""Exported-artefact, validation and inference tests (Phases 20-21)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from src import config as C
from src.inference import (
    Predictor, ValidationError, get_predictor, validate_record,
)

pytestmark = pytest.mark.usefixtures("artefacts_present")


def _skip_without_artefacts(present):
    if not present:
        pytest.skip("run `python -m src.train` first")


# ------------------------------------------------------------- validation
def test_valid_record_becomes_a_single_ordered_row(valid_record):
    X = validate_record(valid_record)
    assert list(X.columns) == C.FEATURE_ORDER
    assert len(X) == 1
    assert X.iloc[0]["age"] == 57


def test_missing_field_is_rejected_with_a_field_level_message(valid_record):
    payload = dict(valid_record)
    del payload["chol"]
    with pytest.raises(ValidationError) as exc:
        validate_record(payload)
    assert "chol" in exc.value.errors
    assert "required" in exc.value.errors["chol"].lower()


def test_out_of_range_numeric_is_rejected(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record({**valid_record, "age": 250})
    assert "Age" in exc.value.errors["age"]
    assert "between" in exc.value.errors["age"]


def test_undocumented_category_is_rejected(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record({**valid_record, "cp": 9})
    assert "cp" in exc.value.errors


def test_non_numeric_input_is_rejected(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record({**valid_record, "trestbps": "high"})
    assert "must be a number" in exc.value.errors["trestbps"]


def test_unknown_field_is_rejected(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record({**valid_record, "smoker": 1})
    assert "smoker" in exc.value.errors


def test_fractional_value_rejected_where_the_schema_says_whole_numbers(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record({**valid_record, "age": 57.4})
    assert "whole number" in exc.value.errors["age"]
    # oldpeak has step 0.1, so a decimal there must be accepted
    validate_record({**valid_record, "oldpeak": 2.3})


def test_every_error_message_uses_the_interface_label_not_the_column_code():
    with pytest.raises(ValidationError) as exc:
        validate_record({})
    for name, message in exc.value.errors.items():
        assert C.FEATURE_BY_NAME[name]["label"] in message


# --------------------------------------------------------------- artefacts
def test_all_declared_artefacts_exist(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    for f in ("final_model.pkl", "preprocessing_pipeline.pkl", "local_explainer.pkl",
              "metadata.json", "metrics.json", "feature_info.json"):
        assert (C.MODELS_DIR / f).exists(), f"missing artefact {f}"


def test_model_reloads_and_reproduces_its_own_predictions(artefacts_present, xy):
    _skip_without_artefacts(artefacts_present)
    X, _ = xy
    a = joblib.load(C.MODELS_DIR / "final_model.pkl").predict_proba(X)[:, 1]
    b = joblib.load(C.MODELS_DIR / "final_model.pkl").predict_proba(X)[:, 1]
    assert np.array_equal(a, b)


def test_exported_metadata_describes_the_exported_model(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(C.MODELS_DIR / "final_model.pkl")
    assert meta["final_model"]["estimator"] == type(model.named_steps["clf"]).__name__
    assert 0.05 <= meta["final_model"]["decision_threshold"] <= 0.95
    assert meta["dataset"]["rows"] == 303
    assert meta["split"]["random_seed"] == C.RANDOM_SEED


def test_exported_metrics_are_within_valid_ranges(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    metrics = json.loads((C.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    for row in metrics["comparison_table"]:
        for key in ("cv_roc_auc_mean", "cv_pr_auc_mean", "test_roc_auc",
                    "test_accuracy", "test_sensitivity", "test_specificity"):
            assert 0.0 <= row[key] <= 1.0, f"{row['model']}.{key} out of range"
    assert metrics["final_model"] in {r["model"] for r in metrics["comparison_table"]}


def test_confusion_matrix_totals_match_the_test_split(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    metrics = json.loads((C.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    cm = metrics["final_test_metrics_chosen_threshold"]["confusion_matrix"]
    assert sum(cm.values()) == meta["split"]["test_rows"]


# --------------------------------------------------------------- inference
def test_predictor_returns_a_complete_result(artefacts_present, valid_record):
    _skip_without_artefacts(artefacts_present)
    out = get_predictor().predict(valid_record).to_dict()
    assert out["classification"] in ("Positive", "Negative")
    assert 0.0 <= out["probability"] <= 1.0
    assert out["predicted_label"] == int(out["probability"] >= out["threshold"])
    assert out["explanation"]["items"], "no feature attributions returned"
    assert "not a medical diagnosis" in out["disclaimer"].lower()


def test_prediction_is_deterministic(artefacts_present, valid_record):
    _skip_without_artefacts(artefacts_present)
    p = get_predictor()
    assert p.predict(valid_record).probability == p.predict(valid_record).probability


def test_predictor_is_a_singleton_so_the_model_loads_once(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    assert get_predictor() is get_predictor()


def test_threshold_override_changes_only_the_classification(artefacts_present, valid_record):
    _skip_without_artefacts(artefacts_present)
    p = get_predictor()
    low = p.predict(valid_record, threshold=0.05)
    high = p.predict(valid_record, threshold=0.95)
    assert low.probability == high.probability
    assert low.predicted_label >= high.predicted_label


def test_explanations_are_ranked_and_use_interface_labels(artefacts_present, valid_record):
    _skip_without_artefacts(artefacts_present)
    items = get_predictor().predict(valid_record, top_k=6).explanation["items"]
    shares = [abs(i["contribution"]) for i in items]
    assert shares == sorted(shares, reverse=True)
    labels = {f["label"] for f in C.FEATURES}
    for item in items:
        assert item["label"] in labels
        assert item["direction"] in ("increases", "decreases", "neutral")
        assert item["display_value"]


def test_explanation_wording_avoids_causal_claims(artefacts_present, valid_record):
    _skip_without_artefacts(artefacts_present)
    note = get_predictor().predict(valid_record).explanation["note"].lower()
    assert "model" in note
    for banned in ("causes", "caused by", "diagnos"):
        assert banned not in note


def test_two_clinically_opposite_records_move_the_estimate_in_opposite_directions(
        artefacts_present, valid_record):
    """Not a medical claim - a sanity check that the model responds to inputs at
    all, using the direction the dataset's own outcome rates show."""
    _skip_without_artefacts(artefacts_present)
    p = get_predictor()
    higher = {**valid_record, "cp": 4, "exang": 1, "ca": 3, "thal": 7, "oldpeak": 4.0}
    lower = {**valid_record, "cp": 2, "exang": 0, "ca": 0, "thal": 3, "oldpeak": 0.0}
    assert p.predict(higher).probability > p.predict(lower).probability


def test_missing_artefacts_raise_a_clear_error(tmp_path):
    from src.inference import ModelNotTrainedError
    with pytest.raises(ModelNotTrainedError) as exc:
        Predictor(models_dir=tmp_path)
    assert "src.train" in str(exc.value)


def test_model_card_reports_real_metrics(artefacts_present):
    _skip_without_artefacts(artefacts_present)
    card = get_predictor().model_card
    metrics = json.loads((C.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert card["test_roc_auc"] == metrics["final_test_metrics_chosen_threshold"]["roc_auc"]
    assert card["dataset_rows"] == 303
    assert card["input_features"] == len(C.FEATURE_ORDER)
