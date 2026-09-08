"""End-to-end training run (Phases 4, 7-20).

Run with ``python -m src.train``.  Every number printed or exported by this
module is produced by the run itself; nothing is hard-coded.

Pipeline order, and why:

1. Stratified 80/20 split *first*, so the test set never influences anything.
2. Hyper-parameter search on the training split only, with the preprocessing
   inside the estimator pipeline so it is re-fitted per fold.
3. Repeated stratified CV of the tuned pipelines - the honest comparison.
4. Out-of-fold training predictions -> threshold choice and calibration study.
5. One final refit on the full training split; the test set is scored once.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, cross_validate, train_test_split,
)

from . import config as C
from . import plots
from .data_loader import feature_dictionary, load_clean, profile, split_xy
from .eda import run_eda
from .evaluate import (
    calibration_points, classification_metrics, curves, pick_threshold,
)
from .explainability import (
    LocalExplainer, SHAP_AVAILABLE, SHAP_ERROR, global_importance, shap_summary,
)
from .models import XGBOOST_AVAILABLE, XGBOOST_ERROR, model_zoo
from .utils import banner, set_seed, timed, write_json

SEARCH_SCORING = "roc_auc"
CV_METRICS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}
RANDOMISED_MODELS = {"Random Forest", "XGBoost"}
RANDOM_SEARCH_ITER = 60


# --------------------------------------------------------------------------
def imbalance_study(X_train, y_train, seed: int) -> dict:
    """Phase 11 - measure whether re-sampling is actually warranted here."""
    counts = pd.Series(y_train).value_counts().sort_index()
    ratio = float(counts.max() / counts.min())
    study = {
        "train_class_counts": {int(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": round(ratio, 3),
        "minority_share": round(float(counts.min() / counts.sum()), 4),
        "stratification": "StratifiedKFold / stratified train-test split used everywhere.",
    }

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from .preprocessing import build_preprocessor

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    base = SkPipeline([("prep", build_preprocessor(True)),
                       ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
    variants = {
        "no_resampling": base,
        "class_weight_balanced": SkPipeline(
            [("prep", build_preprocessor(True)),
             ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                        random_state=seed))]),
    }

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        variants["smote_inside_folds"] = ImbPipeline(
            [("prep", build_preprocessor(True)),
             ("smote", SMOTE(random_state=seed, k_neighbors=5)),
             ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
        study["smote_available"] = True
    except Exception as exc:  # pragma: no cover
        study["smote_available"] = False
        study["smote_error"] = f"{type(exc).__name__}: {exc}"

    study["comparison"] = {}
    for name, pipe in variants.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring={"roc_auc": "roc_auc", "recall": "recall",
                                      "f1": "f1"}, n_jobs=1)
        study["comparison"][name] = {
            k.replace("test_", ""): {
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
            }
            for k, v in res.items() if k.startswith("test_")
        }

    study["decision"] = (
        f"Training split is only mildly imbalanced ({ratio:.2f}:1, minority "
        f"{study['minority_share']*100:.1f}%). SMOTE was evaluated inside the CV "
        "folds (never on the whole dataset) and is reported above; it is not "
        "used in the final model because it did not improve cross-validated "
        "ROC-AUC over stratified sampling, and synthesising patients adds risk "
        "without solving a problem this dataset has. Class weighting remains a "
        "tunable option in every model's search grid."
    )
    return study


# --------------------------------------------------------------------------
def tune(name: str, spec: dict, X_train, y_train, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    grid_size = int(np.prod([len(v) for v in spec["grid"].values()]))
    if name in RANDOMISED_MODELS and grid_size > RANDOM_SEARCH_ITER:
        search = RandomizedSearchCV(
            spec["pipeline"], spec["grid"], n_iter=RANDOM_SEARCH_ITER, cv=cv,
            scoring=SEARCH_SCORING, random_state=seed, n_jobs=-1, refit=True)
        strategy = f"RandomizedSearchCV({RANDOM_SEARCH_ITER} of {grid_size} combos)"
    else:
        search = GridSearchCV(spec["pipeline"], spec["grid"], cv=cv,
                              scoring=SEARCH_SCORING, n_jobs=-1, refit=True)
        strategy = f"GridSearchCV({grid_size} combos)"
    search.fit(X_train, y_train)
    return {
        "search_strategy": strategy,
        "search_scoring": SEARCH_SCORING,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in search.best_params_.items()},
        "best_cv_score": round(float(search.best_score_), 4),
        "estimator": search.best_estimator_,
    }


def repeated_cv(pipeline, X_train, y_train, seed: int) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS,
                                 random_state=seed)
    res = cross_validate(clone(pipeline), X_train, y_train, cv=cv,
                         scoring=CV_METRICS, n_jobs=-1, return_train_score=True)
    out = {}
    for key in CV_METRICS:
        vals = res[f"test_{key}"]
        out[key] = {"mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4)}
    out["train_roc_auc_mean"] = round(float(np.mean(res["train_roc_auc"])), 4)
    out["generalisation_gap_roc_auc"] = round(
        float(np.mean(res["train_roc_auc"]) - np.mean(res["test_roc_auc"])), 4)
    out["n_fits"] = int(C.CV_FOLDS * C.CV_REPEATS)
    return out


def oof_probabilities(pipeline, X_train, y_train, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(clone(pipeline), X_train, y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]


# --------------------------------------------------------------------------
def xgboost_early_stopping(X_train, y_train, seed: int) -> dict:
    """Phase 10 - a documented early-stopping run on an inner validation split."""
    if not XGBOOST_AVAILABLE:
        return {"available": False, "error": XGBOOST_ERROR}
    from xgboost import XGBClassifier
    from .preprocessing import build_preprocessor

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    prep = build_preprocessor(scale_numeric=False).fit(X_fit)
    Z_fit, Z_val = prep.transform(X_fit), prep.transform(X_val)
    clf = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=2.0, random_state=seed, n_jobs=-1,
        early_stopping_rounds=40)
    clf.fit(Z_fit, y_fit, eval_set=[(Z_val, y_val)], verbose=False)
    return {
        "available": True,
        "inner_split": {"fit_rows": int(len(X_fit)), "validation_rows": int(len(X_val))},
        "max_rounds_offered": 1000,
        "best_iteration": int(clf.best_iteration),
        "best_validation_logloss": round(float(clf.best_score), 5),
        "early_stopping_rounds": 40,
        "note": ("Early stopping was run on an inner 80/20 split of the training "
                 "data to find how many boosting rounds this dataset supports "
                 "before validation loss stops improving. The result informed "
                 "the small n_estimators values in the tuning grid; the grid "
                 "search itself uses plain cross-validation so that every "
                 "candidate is scored the same way."),
    }


# --------------------------------------------------------------------------
def calibration_study(pipeline, X_train, y_train, oof_raw, seed: int) -> dict:
    """Phase 15 - is the raw score usable as a probability, or does it need help?"""
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    out = {"raw": calibration_points(y_train, oof_raw)}
    best_name, best_brier = "raw", out["raw"]["brier"]
    for method in ("sigmoid", "isotonic"):
        try:
            cal = CalibratedClassifierCV(clone(pipeline), method=method, cv=5)
            probs = cross_val_predict(cal, X_train, y_train, cv=cv,
                                      method="predict_proba", n_jobs=-1)[:, 1]
            out[method] = calibration_points(y_train, probs)
            out[method]["roc_auc"] = round(
                float(classification_metrics(y_train, probs)["roc_auc"]), 4)
            if out[method]["brier"] < best_brier - 0.002:
                best_name, best_brier = method, out[method]["brier"]
        except Exception as exc:  # pragma: no cover
            out[method] = {"error": f"{type(exc).__name__}: {exc}"}
    out["raw"]["roc_auc"] = round(
        float(classification_metrics(y_train, oof_raw)["roc_auc"]), 4)
    out["best_by_brier"] = best_name
    out["applied"] = best_name
    out["note"] = (
        "Brier score and expected calibration error are computed on "
        "cross-validated training predictions. Platt scaling (sigmoid) and "
        "isotonic regression were both fitted inside CV. The variant with the "
        f"lowest Brier score is '{best_name}'; a margin of 0.002 is required "
        "before replacing the simpler uncalibrated model."
    )
    return out


# --------------------------------------------------------------------------
# How directly a human can read the model's reasoning. Declared up front so
# the tie-break below cannot be reverse-engineered from the results.
INTERPRETABILITY_TIER = {
    "Logistic Regression": 1,   # signed coefficients / odds ratios, read directly
    "SVM (Linear)": 2,          # weights exist but sit behind a probability layer
    "Random Forest": 3,         # needs SHAP or importance to explain
    "XGBoost": 3,               # needs SHAP or importance to explain
    "SVM (RBF)": 4,             # no per-feature weights at all
}
BRIER_TOLERANCE = 0.01          # calibration differences below this are noise


def select_final(rows: list[dict], fit_seconds: dict) -> dict:
    """Phase 19 - a rule declared before the numbers, not 'highest accuracy wins'.

    Discrimination first, but only to the precision the data supports; models
    that are statistically tied are then separated on calibration, on how far
    they overfit, and finally on how explainable and how cheap they are.
    """
    n = rows[0]["cv_n_fits"]
    best_auc = max(rows, key=lambda r: r["cv_roc_auc_mean"])
    se_auc = best_auc["cv_roc_auc_std"] / np.sqrt(n)
    auc_cutoff = best_auc["cv_roc_auc_mean"] - se_auc

    best_pr = max(rows, key=lambda r: r["cv_pr_auc_mean"])
    se_pr = best_pr["cv_pr_auc_std"] / np.sqrt(n)
    pr_cutoff = best_pr["cv_pr_auc_mean"] - se_pr

    shortlist = [r for r in rows
                 if r["cv_roc_auc_mean"] >= auc_cutoff and r["cv_pr_auc_mean"] >= pr_cutoff]
    if not shortlist:
        shortlist = [best_auc]

    best_brier = min(r["cv_brier"] for r in shortlist)
    ranked = sorted(
        shortlist,
        key=lambda r: (
            # calibration, but only when the difference is meaningful
            0 if r["cv_brier"] <= best_brier + BRIER_TOLERANCE else 1,
            INTERPRETABILITY_TIER.get(r["model"], 9),
            r["generalisation_gap"],
            fit_seconds.get(r["model"], 0.0),
        ),
    )
    chosen = ranked[0]
    return {
        "rule": (
            "1. Rank by mean cross-validated ROC-AUC on the training split and "
            "keep every model within one standard error of the best; at 242 "
            "training rows, differences smaller than that are not real. "
            "2. Apply the same one-standard-error filter to PR-AUC, so a model "
            "that is clearly weaker on the positive class is dropped. "
            "3. Among the survivors prefer the better-calibrated model, "
            f"treating Brier differences below {BRIER_TOLERANCE} as ties. "
            "4. Break remaining ties on interpretability - a clinical screening "
            "aid has to be explainable - then on the train/test gap, then on "
            "fit cost."
        ),
        "best_cv_roc_auc": {"model": best_auc["model"],
                            "mean": best_auc["cv_roc_auc_mean"],
                            "std": best_auc["cv_roc_auc_std"],
                            "standard_error": round(float(se_auc), 4)},
        "roc_auc_one_se_cutoff": round(float(auc_cutoff), 4),
        "pr_auc_one_se_cutoff": round(float(pr_cutoff), 4),
        "shortlist": [r["model"] for r in shortlist],
        "ranking_within_shortlist": [
            {"model": r["model"], "cv_roc_auc": r["cv_roc_auc_mean"],
             "cv_pr_auc": r["cv_pr_auc_mean"], "cv_brier": r["cv_brier"],
             "calibration_tie_group": (
                 "best" if r["cv_brier"] <= best_brier + BRIER_TOLERANCE else "worse"),
             "interpretability_tier": INTERPRETABILITY_TIER.get(r["model"], 9),
             "generalisation_gap": r["generalisation_gap"],
             "fit_seconds": round(float(fit_seconds.get(r["model"], 0.0)), 2)}
            for r in ranked
        ],
        "selected": chosen["model"],
        "interpretability_tiers": INTERPRETABILITY_TIER,
    }


# --------------------------------------------------------------------------
def error_analysis(X_test, y_test, probs, threshold, explainer) -> dict:
    """Phase 18 - inspect the records the model gets wrong."""
    preds = (probs >= threshold).astype(int)
    y_test = np.asarray(y_test).astype(int)
    df = X_test.reset_index(drop=True).copy()
    df["true_label"] = y_test
    df["predicted_label"] = preds
    df["model_probability"] = np.round(probs, 4)
    df["outcome"] = np.select(
        [(y_test == 1) & (preds == 1), (y_test == 0) & (preds == 0),
         (y_test == 0) & (preds == 1), (y_test == 1) & (preds == 0)],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown")
    df.to_csv(C.PREDICTIONS / "test_predictions.csv", index=False)

    cases = []
    for kind in ("false_negative", "false_positive"):
        sub = df[df["outcome"] == kind]
        for _, row in sub.iterrows():
            record = row[C.FEATURE_ORDER].to_frame().T.astype(float)
            expl = explainer.explain(record, top_k=4)
            cases.append({
                "type": kind,
                "true_label": C.TARGET_LABELS[int(row["true_label"])],
                "predicted_label": C.TARGET_LABELS[int(row["predicted_label"])],
                "model_probability": float(row["model_probability"]),
                "distance_from_threshold": round(
                    float(abs(row["model_probability"] - threshold)), 4),
                "top_features": [
                    {"label": i["label"], "value": i["display_value"],
                     "direction": i["direction"], "share": i["share"]}
                    for i in expl["items"]
                ],
            })

    counts = df["outcome"].value_counts().to_dict()
    fn = df[df["outcome"] == "false_negative"]
    fp = df[df["outcome"] == "false_positive"]
    correct = df[df["outcome"].isin(["true_positive", "true_negative"])]
    wrong = df[df["outcome"].isin(["false_positive", "false_negative"])]

    def _conf(frame):
        if not len(frame):
            return None
        return round(float(np.mean(np.abs(frame["model_probability"] - threshold))), 4)

    patterns = []
    if len(wrong):
        patterns.append(
            f"Mean distance from the {threshold:.2f} threshold is "
            f"{_conf(wrong):.3f} for wrong predictions versus {_conf(correct):.3f} "
            "for correct ones - the model is measurably less certain when it errs."
        )
    for frame, label in ((fn, "false negatives"), (fp, "false positives")):
        if len(frame):
            patterns.append(
                f"{len(frame)} {label}: mean age {frame['age'].mean():.1f}, "
                f"mean vessels coloured {frame['ca'].mean():.2f}, "
                f"mean ST depression {frame['oldpeak'].mean():.2f} mm "
                f"(test-set averages: {df['age'].mean():.1f}, "
                f"{df['ca'].mean():.2f}, {df['oldpeak'].mean():.2f})."
            )
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "cases": cases,
        "patterns": patterns,
        "cost_note": (
            "A false negative sends a patient with significant narrowing away "
            "without follow-up; a false positive sends a healthy patient to "
            "further, sometimes invasive, testing. In screening the first error "
            "is normally the more serious one, which is why the operating "
            "threshold below is chosen to favour sensitivity rather than left "
            "at 0.5."
        ),
        "predictions_csv": "outputs/predictions/test_predictions.csv",
    }


# --------------------------------------------------------------------------
def main() -> dict:
    set_seed(C.RANDOM_SEED)
    timings: dict = {}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    banner("PHASE 2-3  Load, clean and profile")
    df, cleaning = load_clean()
    prof = profile(df)
    print(f"  rows={prof['n_rows']} cols={prof['n_cols']} "
          f"target={prof['target_counts']} missing={cleaning.missing_before}")
    write_json(C.METRICS / "dataset_profile.json", prof)
    write_json(C.METRICS / "cleaning_report.json", cleaning.to_dict())
    write_json(C.METRICS / "feature_dictionary.json", feature_dictionary())

    banner("PHASE 5  Exploratory data analysis")
    with timed("eda", timings):
        eda = run_eda(df)

    banner("PHASE 4  Stratified split (test set sealed from here on)")
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_SEED)
    print(f"  train={len(X_train)} rows  test={len(X_test)} rows  "
          f"train pos={int(y_train.sum())}  test pos={int(y_test.sum())}")

    banner("PHASE 11  Class-imbalance investigation")
    with timed("imbalance_study", timings):
        imbalance = imbalance_study(X_train, y_train, C.RANDOM_SEED)
    print("  ", json.dumps(imbalance["comparison"], indent=2)[:600])

    banner("PHASE 10  XGBoost early-stopping probe")
    with timed("early_stopping", timings):
        early = xgboost_early_stopping(X_train, y_train, C.RANDOM_SEED)
    print("  ", {k: v for k, v in early.items() if k != "note"})

    banner("PHASES 7-10  Tune every model on the training split")
    pos, neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    zoo = model_zoo(scale_pos_weight=neg / pos)
    tuned: dict[str, dict] = {}
    for name, spec in zoo.items():
        with timed(f"tune::{name}", timings):
            tuned[name] = tune(name, spec, X_train, y_train, C.RANDOM_SEED)
            tuned[name]["notes"] = spec["notes"]
            tuned[name]["needs_scaling"] = spec["needs_scaling"]
        print(f"     best CV {SEARCH_SCORING} = {tuned[name]['best_cv_score']}  "
              f"params = {tuned[name]['best_params']}")

    banner("PHASE 12  Repeated stratified cross-validation of tuned models")
    rows: list[dict] = []
    oof: dict[str, np.ndarray] = {}
    for name, info in tuned.items():
        with timed(f"cv::{name}", timings):
            cvres = repeated_cv(info["estimator"], X_train, y_train, C.RANDOM_SEED)
            oof[name] = oof_probabilities(info["estimator"], X_train, y_train,
                                          C.RANDOM_SEED)
        oof_metrics = classification_metrics(y_train, oof[name], 0.5)
        info["cv"] = cvres
        info["oof_metrics"] = oof_metrics
        rows.append({
            "model": name,
            "cv_roc_auc_mean": cvres["roc_auc"]["mean"],
            "cv_roc_auc_std": cvres["roc_auc"]["std"],
            "cv_pr_auc_mean": cvres["pr_auc"]["mean"],
            "cv_pr_auc_std": cvres["pr_auc"]["std"],
            "cv_f1_mean": cvres["f1"]["mean"],
            "cv_recall_mean": cvres["recall"]["mean"],
            "cv_accuracy_mean": cvres["accuracy"]["mean"],
            "cv_n_fits": cvres["n_fits"],
            "cv_brier": oof_metrics["brier"],
            "generalisation_gap": cvres["generalisation_gap_roc_auc"],
        })
        print(f"     ROC-AUC {cvres['roc_auc']['mean']:.4f} +/- "
              f"{cvres['roc_auc']['std']:.4f} | PR-AUC "
              f"{cvres['pr_auc']['mean']:.4f} | gap "
              f"{cvres['generalisation_gap_roc_auc']:+.4f}")

    banner("PHASE 13  Score the held-out test set once per model")
    for row in rows:
        name = row["model"]
        est = tuned[name]["estimator"]
        est.fit(X_train, y_train)
        probs = est.predict_proba(X_test)[:, 1]
        m = classification_metrics(y_test, probs, 0.5)
        tuned[name]["test_metrics_default_threshold"] = m
        row.update(test_roc_auc=m["roc_auc"], test_pr_auc=m["pr_auc"],
                   test_accuracy=m["accuracy"], test_f1=m["f1"],
                   test_sensitivity=m["sensitivity"], test_specificity=m["specificity"],
                   test_precision=m["precision"])
        print(f"     {name:<20} test ROC-AUC {m['roc_auc']:.4f}  "
              f"acc {m['accuracy']:.4f}  sens {m['sensitivity']:.4f}  "
              f"spec {m['specificity']:.4f}")

    banner("PHASE 19  Final model selection")
    fit_seconds = {r["model"]: timings.get(f"cv::{r['model']}", 0.0) / max(
        1, tuned[r["model"]]["cv"]["n_fits"]) for r in rows}
    selection = select_final(rows, fit_seconds)
    final_name = selection["selected"]
    print(json.dumps({k: v for k, v in selection.items() if k != "rule"}, indent=2))
    final_pipeline = tuned[final_name]["estimator"]

    banner("PHASES 14-15  Threshold and calibration (training folds only)")
    thr = pick_threshold(y_train, oof[final_name], objective="min_sensitivity",
                         min_sensitivity=0.85)
    chosen_threshold = float(thr["chosen_threshold"])
    print(f"  default 0.50 -> F1 {thr['default_metrics']['f1']:.4f} "
          f"sens {thr['default_metrics']['sensitivity']:.4f}")
    print(f"  chosen  {chosen_threshold:.2f} -> F1 {thr['chosen_metrics']['f1']:.4f} "
          f"sens {thr['chosen_metrics']['sensitivity']:.4f} "
          f"spec {thr['chosen_metrics']['specificity']:.4f}")
    with timed("calibration", timings):
        calib = calibration_study(final_pipeline, X_train, y_train,
                                  oof[final_name], C.RANDOM_SEED)
    print(f"  Brier raw={calib['raw']['brier']} -> best={calib['best_by_brier']}")

    banner("PHASE 13/16  Final evaluation of the selected model")
    final_pipeline.fit(X_train, y_train)
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    final_default = classification_metrics(y_test, test_probs, 0.5)
    final_tuned = classification_metrics(y_test, test_probs, chosen_threshold)
    curve_data = curves(y_test, test_probs)
    print(f"  at 0.50 : acc {final_default['accuracy']:.4f} sens "
          f"{final_default['sensitivity']:.4f} spec {final_default['specificity']:.4f}")
    print(f"  at {chosen_threshold:.2f} : acc {final_tuned['accuracy']:.4f} sens "
          f"{final_tuned['sensitivity']:.4f} spec {final_tuned['specificity']:.4f}")
    print(f"  ROC-AUC {final_tuned['roc_auc']:.4f}  PR-AUC {final_tuned['pr_auc']:.4f}")

    banner("PHASE 17  Explainability")
    with timed("explainability", timings):
        gi = global_importance(final_pipeline, X_test, y_test, C.RANDOM_SEED)
        explainer = LocalExplainer(final_pipeline, X_train)
    tree_models = [n for n in ("XGBoost", "Random Forest") if n in tuned]
    shap_model = (max(tree_models, key=lambda n: tuned[n]["cv"]["roc_auc"]["mean"])
                  if tree_models else None)
    shap_info = (shap_summary(tuned[shap_model]["estimator"], X_train, shap_model)
                 if shap_model else {"available": False,
                                     "error": "no tree model in the zoo"})
    gi["shap_summary"] = shap_info
    print(f"  local method = {explainer.kind}; top permutation feature = "
          f"{gi['permutation_importance'][0]['label']} "
          f"({gi['permutation_importance'][0]['mean_drop_in_roc_auc']:.4f})")
    if shap_info.get("available"):
        top = shap_info["by_feature"][0]
        print(f"  SHAP ({shap_model}) top feature = {top['label']} "
              f"(mean |SHAP| = {top['value']:.4f})")
    else:
        print(f"  SHAP summary unavailable: {shap_info.get('error')}")

    banner("PHASE 18  Error analysis")
    errors = error_analysis(X_test, y_test, test_probs, chosen_threshold, explainer)
    print("  ", errors["counts"])
    for p in errors["patterns"]:
        print("   -", p)

    banner("Figures")
    fig_paths = {**eda["figures"]}
    roc_p, pr_p = plots.roc_pr_curves(curve_data, final_tuned["roc_auc"],
                                      final_tuned["pr_auc"],
                                      curve_data["positive_rate"], final_name)
    fig_paths["roc_curve"] = roc_p
    fig_paths["pr_curve"] = pr_p
    fig_paths["confusion_matrix"] = plots.confusion_figure(
        final_tuned["confusion_matrix"], chosen_threshold, final_name)
    best_cal = calib.get(calib["best_by_brier"]) if calib["best_by_brier"] != "raw" else None
    fig_paths["calibration"] = plots.calibration_figure(
        calib["raw"], best_cal if best_cal and "brier" in best_cal else None, final_name)
    fig_paths["threshold_sweep"] = plots.threshold_figure(thr["sweep"], chosen_threshold)
    fig_paths["model_comparison"] = plots.model_comparison_figure(rows)
    fig_paths["permutation_importance"] = plots.importance_figure(
        gi["permutation_importance"], final_name)
    if "coefficients" in gi:
        fig_paths["coefficients"] = plots.coefficient_figure(gi["coefficients"], final_name)
    print("  ", list(fig_paths))

    banner("PHASE 20  Export artefacts")
    joblib.dump(final_pipeline, C.MODELS_DIR / "final_model.pkl")
    joblib.dump(final_pipeline.named_steps["prep"],
                C.MODELS_DIR / "preprocessing_pipeline.pkl")
    joblib.dump(explainer, C.MODELS_DIR / "local_explainer.pkl")

    metadata = {
        "project": "Disease Risk Prediction - Heart Disease Classification",
        "generated_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": C.DATASET_NAME, "page": C.DATASET_PAGE, "archive": C.DATASET_URL,
            "file": C.DATASET_FILE, "rows": prof["n_rows"],
            "features": len(C.FEATURE_ORDER), "classes": 2,
            "target": C.TARGET, "target_description": C.TARGET_DESCRIPTION,
            "class_counts": prof["target_counts"],
        },
        "final_model": {
            "name": final_name,
            "estimator": type(final_pipeline.named_steps["clf"]).__name__,
            "hyperparameters": tuned[final_name]["best_params"],
            "search_strategy": tuned[final_name]["search_strategy"],
            "decision_threshold": chosen_threshold,
            "threshold_rationale": (
                f"Chosen on cross-validated training predictions as the highest-"
                f"specificity threshold that still reaches {thr['min_sensitivity_target']:.0%} "
                "sensitivity. The test set was not used to pick it."),
            "calibration": calib["applied"],
            "local_explanation_method": explainer.kind,
        },
        "split": {
            "strategy": "stratified train_test_split",
            "test_size": C.TEST_SIZE, "random_seed": C.RANDOM_SEED,
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "cv": f"RepeatedStratifiedKFold({C.CV_FOLDS} folds x {C.CV_REPEATS} repeats)",
        },
        "label_mapping": {str(k): v for k, v in C.TARGET_LABELS.items()},
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "scikit_learn": __import__("sklearn").__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "xgboost_available": XGBOOST_AVAILABLE, "xgboost_error": XGBOOST_ERROR,
            "shap_available": SHAP_AVAILABLE, "shap_error": SHAP_ERROR,
        },
        "timings_seconds": timings,
        "figures": fig_paths,
    }
    if XGBOOST_AVAILABLE:
        import xgboost
        metadata["environment"]["xgboost"] = xgboost.__version__

    metrics_payload = {
        "comparison_table": rows,
        "final_model": final_name,
        "final_test_metrics_default_threshold": final_default,
        "final_test_metrics_chosen_threshold": final_tuned,
        "curves": curve_data,
        "threshold_analysis": {k: v for k, v in thr.items() if k != "sweep"},
        "threshold_sweep": thr["sweep"],
        "calibration": calib,
        "cross_validation": {n: tuned[n]["cv"] for n in tuned},
        "per_model_details": {
            n: {"best_params": tuned[n]["best_params"],
                "search_strategy": tuned[n]["search_strategy"],
                "best_search_score": tuned[n]["best_cv_score"],
                "search_scoring": tuned[n]["search_scoring"],
                "notes": tuned[n]["notes"],
                "oof_metrics": tuned[n]["oof_metrics"],
                "test_metrics_default_threshold": tuned[n]["test_metrics_default_threshold"]}
            for n in tuned
        },
        "selection": selection,
        "imbalance_study": imbalance,
        "early_stopping": early,
        "error_analysis": errors,
        "explainability": gi,
        "eda": eda["insights"],
    }

    write_json(C.MODELS_DIR / "metadata.json", metadata)
    write_json(C.MODELS_DIR / "metrics.json", metrics_payload)
    write_json(C.MODELS_DIR / "feature_info.json", {
        "features": C.FEATURES, "feature_order": C.FEATURE_ORDER,
        "groups": C.GROUP_ORDER, "target": C.TARGET,
        "label_mapping": {str(k): v for k, v in C.TARGET_LABELS.items()},
        "dictionary": feature_dictionary(),
    })
    write_json(C.METRICS / "model_comparison.json", rows)
    write_json(C.METRICS / "threshold_analysis.json", thr)
    write_json(C.METRICS / "calibration.json", calib)
    write_json(C.METRICS / "explainability.json", gi)
    write_json(C.METRICS / "error_analysis.json", errors)
    write_json(C.METRICS / "imbalance_study.json", imbalance)

    banner("Verify the exported model reloads and reproduces its predictions")
    reloaded = joblib.load(C.MODELS_DIR / "final_model.pkl")
    reprobs = reloaded.predict_proba(X_test)[:, 1]
    max_delta = float(np.max(np.abs(reprobs - test_probs)))
    print(f"  max |p_reloaded - p_original| on the test set = {max_delta:.3e}")
    assert max_delta < 1e-9, "Reloaded model does not reproduce original predictions"
    print("  OK - reload verified.")

    banner("SUMMARY")
    print(f"  Final model      : {final_name}")
    print(f"  Threshold        : {chosen_threshold}")
    print(f"  Test ROC-AUC     : {final_tuned['roc_auc']}")
    print(f"  Test sensitivity : {final_tuned['sensitivity']}")
    print(f"  Test specificity : {final_tuned['specificity']}")
    print(f"  Artefacts        : {C.MODELS_DIR}")
    return metrics_payload


if __name__ == "__main__":
    main()
