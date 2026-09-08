"""HTTP-layer tests (Phase 24). Exercises the real FastAPI app end to end."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import config as C

pytest.importorskip("httpx", reason="TestClient needs httpx")


@pytest.fixture(scope="module")
def client(artefacts_present):
    if not artefacts_present:
        pytest.skip("run `python -m src.train` first")
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Cardiac Risk Model Explorer" in res.text
    assert "Educational use only" in res.text


def test_static_assets_are_served(client):
    for path in ("/static/styles.css", "/static/app.js"):
        assert client.get(path).status_code == 200


def test_schema_endpoint_drives_the_form(client):
    body = client.get("/api/schema").json()
    assert len(body["features"]) == len(C.FEATURE_ORDER)
    assert body["groups"] == C.GROUP_ORDER
    for feat in body["features"]:
        assert feat["label"] and feat["group"] in body["groups"]
        if feat["role"] == "numeric":
            assert feat["min"] < feat["max"]
        else:
            assert len(feat["choices"]) >= 2


def test_model_endpoint_exposes_the_card_and_selection_rule(client):
    body = client.get("/api/model").json()
    assert body["card"]["name"]
    assert 0 < body["card"]["threshold"] < 1
    assert "rule" in body["selection"] and body["selection"]["selected"]


def test_metrics_endpoint_returns_every_block_the_ui_renders(client):
    body = client.get("/api/metrics").json()
    for key in ("comparison_table", "test_chosen", "curves", "threshold_analysis",
                "threshold_sweep", "calibration", "explainability", "error_analysis",
                "imbalance_study", "early_stopping", "selection", "dataset", "split"):
        assert key in body, f"/api/metrics is missing {key}"
    assert len(body["comparison_table"]) >= 4
    assert body["explainability"]["permutation_importance"]


def test_predict_happy_path(client, valid_record):
    res = client.post("/api/predict", json=valid_record)
    assert res.status_code == 200
    body = res.json()
    assert body["classification"] in ("Positive", "Negative")
    assert 0 <= body["probability"] <= 1
    assert body["probability_percent"] == round(body["probability"] * 100, 1)
    assert body["threshold"] == body["model"]["threshold"]
    assert body["explanation"]["items"]
    assert body["latency_ms"] >= 0


def test_predict_is_consistent_across_requests(client, valid_record):
    a = client.post("/api/predict", json=valid_record).json()
    b = client.post("/api/predict", json=valid_record).json()
    assert a["probability"] == b["probability"]


def test_predict_rejects_a_missing_field(client, valid_record):
    payload = dict(valid_record)
    del payload["thal"]
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "invalid_request"
    assert "thal" in body["fields"]


def test_predict_rejects_an_out_of_range_value(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "chol": 5000})
    assert res.status_code == 400
    assert "chol" in res.json()["fields"]


def test_predict_rejects_an_undocumented_category(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "thal": 4})
    assert res.status_code == 400
    assert "thal" in res.json()["fields"]


def test_predict_rejects_unknown_fields(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "bmi": 27})
    assert res.status_code == 400
    assert "bmi" in res.json()["fields"]


def test_predict_rejects_a_wrong_type(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "age": "fifty"})
    assert res.status_code == 400
    assert "age" in res.json()["fields"]


def test_predict_rejects_an_out_of_bounds_threshold(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "threshold": 1.5})
    assert res.status_code == 400
    assert "threshold" in res.json()["fields"]


def test_threshold_override_is_honoured(client, valid_record):
    body = client.post("/api/predict", json={**valid_record, "threshold": 0.9}).json()
    assert body["threshold"] == 0.9
    assert body["predicted_label"] == int(body["probability"] >= 0.9)


def test_error_responses_never_leak_a_traceback(client, valid_record):
    res = client.post("/api/predict", json={**valid_record, "cp": 42})
    assert res.status_code == 400
    text = res.text
    for leak in ("Traceback", "File \"", "src/inference.py", "src\\\\inference.py"):
        assert leak not in text


def test_example_endpoint_returns_a_usable_record(client):
    res = client.get("/api/example")
    assert res.status_code == 200
    body = res.json()
    assert set(body["record"]) == set(C.FEATURE_ORDER)
    assert body["true_label"] in ("Positive", "Negative")
    # and the record it hands back must be accepted by /api/predict
    assert client.post("/api/predict", json=body["record"]).status_code == 200
