"""Explainability (Phase 17).

Two levels are provided:

* **Global** - what the model relies on overall: linear coefficients,
  impurity-based importance, model-agnostic permutation importance, and a
  SHAP summary for the best tree model.
* **Local** - why *this* record received *this* score.

Wording note used throughout the project: these quantify influence on the
model's own output.  They are not statements about medical causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None
    SHAP_AVAILABLE = False
    SHAP_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
def encoded_names(pipeline) -> list[str]:
    return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]


def base_feature_of(encoded: str) -> str:
    """Map an encoded column back to the raw dataset feature it came from."""
    token = encoded.split("__", 1)[-1]
    if token in C.FEATURE_BY_NAME:
        return token
    for name in C.FEATURE_ORDER:
        if token.startswith(name + "_"):
            return name
    return token


def aggregate_to_features(names: list[str], values) -> dict[str, float]:
    """Sum encoded-column values back onto the original 13 dataset features."""
    out: dict[str, float] = {n: 0.0 for n in C.FEATURE_ORDER}
    for name, val in zip(names, np.asarray(values, dtype=float).ravel()):
        key = base_feature_of(name)
        out[key] = out.get(key, 0.0) + float(val)
    return out


def _ranked(d: dict[str, float]) -> list[dict]:
    return [
        {"feature": k,
         "label": C.FEATURE_BY_NAME[k]["label"] if k in C.FEATURE_BY_NAME else k,
         "value": round(float(v), 5)}
        for k, v in sorted(d.items(), key=lambda t: -abs(t[1]))
    ]


def display_value(spec: dict, value) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    if spec["role"] == "numeric":
        text = f"{float(value):g}"
        return f"{text} {spec['unit']}" if spec["unit"] and spec["unit"] != "count" else text
    for choice in spec.get("choices", []):
        if float(choice["value"]) == float(value):
            return choice["label"]
    return f"{float(value):g}"


# --------------------------------------------------------------------------
def global_importance(pipeline, X_test: pd.DataFrame, y_test, seed: int) -> dict:
    """Coefficients / impurity importance plus permutation importance."""
    names = encoded_names(pipeline)
    clf = pipeline.named_steps["clf"]
    result: dict = {
        "encoded_feature_count": len(names),
        "method": [],
        "shap_available": SHAP_AVAILABLE,
        "shap_error": SHAP_ERROR,
    }

    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_, dtype=float).ravel()
        result["method"].append("coefficients")
        result["coefficients"] = [
            {"encoded": n, "feature": base_feature_of(n), "label": C.friendly_name(n),
             "coefficient": round(float(c), 4),
             "odds_ratio": round(float(np.exp(c)), 4)}
            for n, c in sorted(zip(names, coefs), key=lambda t: -abs(t[1]))
        ]
        result["coefficient_note"] = (
            "Numeric features are standardised before fitting, so coefficients "
            "are comparable across features. An odds ratio above 1 means the "
            "model raises its estimated odds as that encoded feature increases."
        )
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float).ravel()
        result["method"].append("impurity_importance")
        result["impurity_importance"] = [
            {"encoded": n, "feature": base_feature_of(n), "label": C.friendly_name(n),
             "importance": round(float(v), 5)}
            for n, v in sorted(zip(names, imp), key=lambda t: -t[1])
        ]
        result["impurity_importance_by_feature"] = _ranked(
            aggregate_to_features(names, imp))

    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc",
        n_repeats=30, random_state=seed, n_jobs=1,
    )
    result["method"].append("permutation_importance")
    result["permutation_importance"] = [
        {"feature": f, "label": C.FEATURE_BY_NAME[f]["label"],
         "mean_drop_in_roc_auc": round(float(m), 5), "std": round(float(s), 5)}
        for f, m, s in sorted(
            zip(X_test.columns, perm.importances_mean, perm.importances_std),
            key=lambda t: -t[1])
    ]
    result["permutation_note"] = (
        "Permutation importance shuffles one raw input column of the held-out "
        "test set and measures how much test ROC-AUC drops. A larger drop means "
        "the model leaned on that column more."
    )
    return result


def shap_summary(pipeline, X: pd.DataFrame, model_name: str) -> dict:
    """Mean |SHAP| per feature for a tree model, aggregated to raw features."""
    if not SHAP_AVAILABLE:
        return {"available": False, "error": SHAP_ERROR}
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return {"available": False,
                "error": f"{model_name} is not a tree model; TreeSHAP does not apply."}
    try:
        names = encoded_names(pipeline)
        Z = np.asarray(pipeline.named_steps["prep"].transform(X), dtype=float)
        values = np.asarray(shap.TreeExplainer(clf).shap_values(Z, check_additivity=False))
        if values.ndim == 3:
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        return {
            "available": True,
            "model": model_name,
            "explainer": "shap.TreeExplainer (exact TreeSHAP)",
            "rows_explained": int(len(X)),
            "by_encoded_feature": [
                {"label": C.friendly_name(n), "mean_abs_shap": round(float(v), 5)}
                for n, v in sorted(zip(names, mean_abs), key=lambda t: -t[1])[:15]
            ],
            "by_feature": _ranked(aggregate_to_features(names, mean_abs)),
            "note": ("Mean absolute SHAP value per feature across the training "
                     "split, in log-odds units. It measures how much each input "
                     "moves this model's output, not medical causation."),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
class LocalExplainer:
    """Per-prediction attribution, pickled and shipped alongside the model.

    Three strategies, chosen automatically from the fitted estimator:

    ``linear_coefficients``
        ``w_j * (x_j - mean_j)`` on the encoded matrix - additive in log-odds
        and exact for logistic regression.
    ``tree_shap``
        Exact TreeSHAP values for Random Forest / XGBoost.
    ``ablation``
        Model-agnostic fallback (used for the SVMs, whose probability layer
        hides the margin): replace one raw input with its training-set
        reference value and measure how far the model-estimated probability
        moves.  13 extra forward passes, works for any estimator.
    """

    def __init__(self, pipeline, X_background: pd.DataFrame):
        self.pipeline = pipeline
        self.names = encoded_names(pipeline)
        clf = pipeline.named_steps["clf"]
        prep = pipeline.named_steps["prep"]

        Z = np.asarray(prep.transform(X_background), dtype=float)
        self.baseline = Z.mean(axis=0)

        # Reference patient for the ablation method: median numeric value,
        # most common category - computed on the training split only.
        self.reference: dict[str, float] = {}
        for feat in C.FEATURES:
            col = X_background[feat["name"]].dropna()
            self.reference[feat["name"]] = (
                float(col.median()) if feat["role"] == "numeric"
                else float(col.mode().iloc[0])
            )

        self._coef = None
        self._shap = None
        if hasattr(clf, "coef_"):
            self._coef = np.asarray(clf.coef_, dtype=float).ravel()
            self.kind = "linear_coefficients"
        elif hasattr(clf, "feature_importances_") and SHAP_AVAILABLE:
            try:
                self._shap = shap.TreeExplainer(clf)
                self.kind = "tree_shap"
            except Exception:
                self.kind = "ablation"
        else:
            self.kind = "ablation"

    # ------------------------------------------------------------------
    def explain(self, X_row: pd.DataFrame, top_k: int = 6) -> dict:
        X_row = X_row[C.FEATURE_ORDER]
        if self.kind == "linear_coefficients":
            z = self._encode(X_row)
            per_feature = aggregate_to_features(
                self.names, self._coef * (z - self.baseline))
            units = "log-odds shift vs. an average training record"
        elif self.kind == "tree_shap":
            z = self._encode(X_row)
            sv = np.asarray(self._shap.shap_values(z.reshape(1, -1),
                                                   check_additivity=False))
            sv = sv[0, :, -1] if sv.ndim == 3 else sv.reshape(1, -1)[0]
            per_feature = aggregate_to_features(self.names, sv)
            units = "SHAP value (log-odds contribution)"
        else:
            per_feature = self._ablation(X_row)
            units = "change in model-estimated probability vs. a reference record"

        raw = X_row.iloc[0]
        ranked = sorted(per_feature.items(), key=lambda t: -abs(t[1]))[:top_k]
        total = sum(abs(v) for v in per_feature.values()) or 1.0
        items = []
        for feat, val in ranked:
            spec = C.FEATURE_BY_NAME[feat]
            observed = raw[feat]
            items.append({
                "feature": feat,
                "label": spec["label"],
                "value": None if pd.isna(observed) else float(observed),
                "display_value": display_value(spec, observed),
                "contribution": round(float(val), 5),
                "direction": ("increases" if val > 0
                              else "decreases" if val < 0 else "neutral"),
                "share": round(abs(float(val)) / total, 4),
            })
        return {
            "method": self.kind,
            "units": units,
            "items": items,
            "note": ("These values show how strongly each input moved this "
                     "model's own output. They describe the model's behaviour, "
                     "not a medical cause."),
        }

    # ------------------------------------------------------------------
    def _encode(self, X_row: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.pipeline.named_steps["prep"].transform(X_row), dtype=float)[0]

    def _ablation(self, X_row: pd.DataFrame) -> dict[str, float]:
        base_p = float(self.pipeline.predict_proba(X_row)[0, 1])
        variants = pd.concat([X_row] * len(C.FEATURE_ORDER), ignore_index=True)
        for i, feat in enumerate(C.FEATURE_ORDER):
            variants.loc[i, feat] = self.reference[feat]
        probs = self.pipeline.predict_proba(variants)[:, 1]
        return {feat: float(base_p - probs[i])
                for i, feat in enumerate(C.FEATURE_ORDER)}
