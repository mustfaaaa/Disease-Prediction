"""Metric computation, threshold search and calibration (Phases 13-15)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, precision_recall_curve, recall_score,
    roc_auc_score, roc_curve,
)


def specificity_score(y_true, y_pred) -> float:
    """True-negative rate.  Reported alongside recall because in screening a
    model that flags everyone scores perfect recall and useless specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All Phase-13 metrics for one (labels, probabilities, threshold) triple."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def curves(y_true, y_prob) -> dict:
    """ROC / PR curve points, thinned for transport to the browser."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    def thin(a, b, keep=120):
        idx = np.unique(np.linspace(0, len(a) - 1, min(keep, len(a))).astype(int))
        return [round(float(v), 4) for v in a[idx]], [round(float(v), 4) for v in b[idx]]

    fpr_t, tpr_t = thin(fpr, tpr)
    rec_t, prec_t = thin(rec, prec)
    return {
        "roc": {"fpr": fpr_t, "tpr": tpr_t},
        "pr": {"recall": rec_t, "precision": prec_t},
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def threshold_sweep(y_true, y_prob, grid=None) -> list[dict]:
    """Metrics across a threshold grid.  Computed on validation folds only."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    out = []
    for t in grid:
        m = classification_metrics(y_true, y_prob, float(t))
        m.pop("confusion_matrix")
        out.append(m)
    return out


def pick_threshold(y_true, y_prob, objective: str = "f1",
                   min_sensitivity: float = 0.85) -> dict:
    """Choose an operating threshold from *validation* predictions.

    Two candidate rules are computed so the choice can be justified rather
    than assumed:

    ``f1``            - threshold maximising F1 (balanced view).
    ``min_sensitivity``- lowest-cost threshold that still reaches the requested
                         sensitivity, reflecting that in cardiac screening a
                         missed case (false negative) is more costly than an
                         unnecessary follow-up (false positive).
    """
    sweep = threshold_sweep(y_true, y_prob)
    best_f1 = max(sweep, key=lambda r: (r["f1"], r["sensitivity"]))
    feasible = [r for r in sweep if r["sensitivity"] >= min_sensitivity]
    best_sens = (max(feasible, key=lambda r: (r["specificity"], r["f1"]))
                 if feasible else best_f1)
    chosen = best_f1 if objective == "f1" else best_sens
    return {
        "objective": objective,
        "min_sensitivity_target": min_sensitivity,
        "default_threshold": 0.5,
        "default_metrics": classification_metrics(y_true, y_prob, 0.5),
        "best_f1_threshold": best_f1["threshold"],
        "best_f1_metrics": best_f1,
        "min_sensitivity_threshold": best_sens["threshold"],
        "min_sensitivity_metrics": best_sens,
        "chosen_threshold": chosen["threshold"],
        "chosen_metrics": chosen,
        "sweep": sweep,
    }


def calibration_points(y_true, y_prob, n_bins: int = 10) -> dict:
    """Reliability-diagram points using equal-width probability bins."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    pred, obs, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred.append(round(float(y_prob[m].mean()), 4))
        obs.append(round(float(y_true[m].mean()), 4))
        counts.append(int(m.sum()))
    ece = (sum(c * abs(p - o) for c, p, o in zip(counts, pred, obs)) / sum(counts)
           if counts else 0.0)
    return {
        "mean_predicted": pred,
        "observed_frequency": obs,
        "bin_counts": counts,
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "expected_calibration_error": round(float(ece), 4),
        "n_bins": n_bins,
    }
