"""Model-evaluation figures (Phases 13-18) written to ``outputs/figures``."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .eda import PALETTE, _save


def roc_pr_curves(curve_data: dict, roc_auc: float, pr_auc: float,
                  base_rate: float, model_name: str) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(curve_data["roc"]["fpr"], curve_data["roc"]["tpr"],
            color=PALETTE["pos"], lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["muted"], label="Chance (0.500)")
    ax.set_xlabel("False positive rate  (1 - specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title("ROC curve - held-out test set")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    roc_path = _save(fig, "10_roc_curve.png")

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(curve_data["pr"]["recall"], curve_data["pr"]["precision"],
            color=PALETTE["pos"], lw=2, label=f"{model_name} (AP = {pr_auc:.3f})")
    ax.axhline(base_rate, ls="--", lw=1, color=PALETTE["muted"],
               label=f"Base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve - held-out test set")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    pr_path = _save(fig, "11_pr_curve.png")
    return roc_path, pr_path


def confusion_figure(cm: dict, threshold: float, model_name: str) -> str:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.3, 3.8))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.35)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{int(mat[i, j])}", ha="center",
                    va="center", fontsize=9,
                    color="white" if mat[i, j] > mat.max() * 0.6 else PALETTE["ink"])
    ax.set_xticks([0, 1], ["Predicted\nNegative", "Predicted\nPositive"])
    ax.set_yticks([0, 1], ["Actual\nNegative", "Actual\nPositive"])
    ax.set_title(f"Confusion matrix - {model_name}\n(test set, threshold = {threshold:.2f})",
                 fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, "12_confusion_matrix.png")


def calibration_figure(raw: dict, calibrated: dict | None, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["muted"], label="Perfect calibration")
    ax.plot(raw["mean_predicted"], raw["observed_frequency"], "o-",
            color=PALETTE["pos"], lw=1.8, ms=5,
            label=f"{model_name} (Brier = {raw['brier']:.3f})")
    if calibrated:
        ax.plot(calibrated["mean_predicted"], calibrated["observed_frequency"], "s-",
                color=PALETTE["neg"], lw=1.8, ms=5,
                label=f"After calibration (Brier = {calibrated['brier']:.3f})")
    ax.set_xlabel("Mean model-estimated probability")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title("Calibration curve - cross-validated training predictions", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, "13_calibration_curve.png")


def threshold_figure(sweep: list[dict], chosen: float, default: float = 0.5) -> str:
    t = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, colour, label in (
        ("sensitivity", PALETTE["pos"], "Sensitivity (recall)"),
        ("specificity", PALETTE["neg"], "Specificity"),
        ("f1", "#4C7C5B", "F1"),
        ("precision", "#8E6BA8", "Precision"),
    ):
        ax.plot(t, [r[key] for r in sweep], lw=1.8, color=colour, label=label)
    ax.axvline(default, ls=":", lw=1.2, color=PALETTE["muted"])
    ax.text(default, 1.03, "default 0.50", ha="center", fontsize=7.5, color=PALETTE["muted"])
    ax.axvline(chosen, ls="--", lw=1.4, color=PALETTE["ink"])
    ax.text(chosen, 1.09, f"chosen {chosen:.2f}", ha="center", fontsize=7.5)
    ax.set_xlabel("Decision threshold on model-estimated probability")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.16)
    ax.set_title("Threshold sweep on cross-validated training predictions", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="lower center")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, "14_threshold_sweep.png")


def model_comparison_figure(rows: list[dict]) -> str:
    names = [r["model"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(names))
    width = 0.26
    series = [("cv_roc_auc_mean", "cv_roc_auc_std", "CV ROC-AUC", PALETTE["pos"]),
              ("cv_pr_auc_mean", "cv_pr_auc_std", "CV PR-AUC", PALETTE["neg"]),
              ("test_roc_auc", None, "Test ROC-AUC", "#4C7C5B")]
    for i, (mkey, skey, label, colour) in enumerate(series):
        vals = [r[mkey] for r in rows]
        errs = [r[skey] for r in rows] if skey else None
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3,
               color=colour, label=label, error_kw=dict(lw=1, ecolor=PALETTE["muted"]))
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison - repeated stratified CV (train) vs held-out test",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, "15_model_comparison.png")


def importance_figure(perm: list[dict], model_name: str, top: int = 13) -> str:
    rows = perm[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    vals = [r["mean_drop_in_roc_auc"] for r in rows]
    errs = [r["std"] for r in rows]
    ax.barh([r["label"] for r in rows], vals, xerr=errs, height=0.62,
            color=PALETTE["neg"], error_kw=dict(lw=1, ecolor=PALETTE["muted"]))
    ax.set_xlabel("Mean drop in test ROC-AUC when the column is shuffled")
    ax.set_title(f"Permutation importance - {model_name}", fontsize=9)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, "16_permutation_importance.png")


def coefficient_figure(items: list[dict], model_name: str, top: int = 14) -> str:
    rows = items[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vals = [r["coefficient"] for r in rows]
    colours = [PALETTE["pos"] if v > 0 else PALETTE["neg"] for v in vals]
    ax.barh([r["label"] for r in rows], vals, color=colours, height=0.62)
    ax.axvline(0, lw=1, color=PALETTE["ink"])
    ax.set_xlabel("Coefficient (log-odds); positive raises the model's estimate")
    ax.set_title(f"Model coefficients - {model_name}", fontsize=9)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, "17_coefficients.png")
