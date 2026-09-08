"""Model-zoo and metric tests (Phases 7-15)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config as C
from src.evaluate import (
    calibration_points, classification_metrics, curves, pick_threshold,
    specificity_score, threshold_sweep,
)
from src.models import XGBOOST_AVAILABLE, XGBOOST_ERROR, model_zoo


# ------------------------------------------------------------------ models
def test_all_four_required_algorithm_families_are_present():
    zoo = model_zoo()
    assert "Logistic Regression" in zoo
    assert any(k.startswith("SVM") for k in zoo)
    assert "Random Forest" in zoo
    assert "XGBoost" in zoo, f"XGBoost unavailable: {XGBOOST_ERROR}"


def test_xgboost_is_importable():
    assert XGBOOST_AVAILABLE, f"XGBoost failed to import: {XGBOOST_ERROR}"


@pytest.mark.parametrize("name", list(model_zoo()))
def test_each_model_trains_and_produces_calibratable_probabilities(name, xy):
    X, y = xy
    pipe = model_zoo()[name]["pipeline"]
    pipe.fit(X.iloc[:200], y.iloc[:200])
    proba = pipe.predict_proba(X.iloc[200:])
    assert proba.shape == (len(X) - 200, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", list(model_zoo()))
def test_each_model_beats_chance_under_cross_validation(name, xy):
    X, y = xy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    scores = cross_val_score(model_zoo()[name]["pipeline"], X, y, cv=cv, scoring="roc_auc")
    assert scores.mean() > 0.7, f"{name} scored {scores.mean():.3f} ROC-AUC"


@pytest.mark.parametrize("name", list(model_zoo()))
def test_search_grids_are_small_enough_to_be_honest(name):
    """A giant grid on 242 training rows overfits the search itself."""
    spec = model_zoo()[name]
    size = int(np.prod([len(v) for v in spec["grid"].values()]))
    assert 1 < size <= 512, f"{name} grid has {size} combinations"


def test_scaling_is_requested_exactly_where_it_matters():
    zoo = model_zoo()
    assert zoo["Logistic Regression"]["needs_scaling"] is True
    assert zoo["SVM (RBF)"]["needs_scaling"] is True
    assert zoo["Random Forest"]["needs_scaling"] is False
    assert zoo["XGBoost"]["needs_scaling"] is False


# ----------------------------------------------------------------- metrics
def test_metrics_match_a_hand_computed_confusion_matrix():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([.9, .8, .7, .2, .1, .2, .3, .6, .4, .05])
    m = classification_metrics(y_true, y_prob, 0.5)
    # tp = 3 (.9 .8 .7), fn = 1 (.2), fp = 1 (.6), tn = 5
    assert m["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == 0.8
    assert m["precision"] == 0.75
    assert m["recall"] == 0.75
    assert m["sensitivity"] == m["recall"]
    assert m["specificity"] == round(5 / 6, 4)
    assert m["f1"] == 0.75


def test_specificity_helper_agrees_with_the_metric_bundle():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    assert specificity_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_perfect_and_random_scores_are_at_the_expected_extremes():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = classification_metrics(y, np.array([.1, .2, .3, .7, .8, .9]))
    assert perfect["roc_auc"] == 1.0 and perfect["accuracy"] == 1.0
    inverted = classification_metrics(y, np.array([.9, .8, .7, .3, .2, .1]))
    assert inverted["roc_auc"] == 0.0


def test_threshold_sweep_is_monotone_in_the_right_directions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.35 + rng.normal(0.35, 0.18, 400), 0.001, 0.999)
    sweep = threshold_sweep(y, p)
    sens = [r["sensitivity"] for r in sweep]
    spec = [r["specificity"] for r in sweep]
    assert all(a >= b - 1e-9 for a, b in zip(sens, sens[1:])), "sensitivity must not rise with threshold"
    assert all(a <= b + 1e-9 for a, b in zip(spec, spec[1:])), "specificity must not fall with threshold"


def test_threshold_choice_respects_the_sensitivity_floor():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0.001, 0.999)
    out = pick_threshold(y, p, objective="min_sensitivity", min_sensitivity=0.85)
    assert out["chosen_metrics"]["sensitivity"] >= 0.85
    assert out["default_threshold"] == 0.5
    assert 0.05 <= out["chosen_threshold"] <= 0.95
    assert out["best_f1_metrics"]["f1"] >= out["default_metrics"]["f1"]


def test_calibration_points_and_brier_are_consistent():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.array([0.0] * 50 + [1.0] * 50)
    cal = calibration_points(y, perfect, n_bins=10)
    assert cal["brier"] == 0.0
    assert cal["expected_calibration_error"] == 0.0
    assert sum(cal["bin_counts"]) == 100

    useless = np.full(100, 0.5)
    assert calibration_points(y, useless)["brier"] == 0.25


def test_curve_export_is_json_friendly_and_bounded():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    c = curves(y, p)
    for axis in (c["roc"]["fpr"], c["roc"]["tpr"], c["pr"]["recall"], c["pr"]["precision"]):
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in axis)
    assert len(c["roc"]["fpr"]) == len(c["roc"]["tpr"])
    assert len(c["pr"]["recall"]) == len(c["pr"]["precision"])
