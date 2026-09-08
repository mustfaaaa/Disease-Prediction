"""Model zoo and hyper-parameter search spaces (Phases 7-11).

Each entry pairs an estimator with the preprocessing variant it needs and a
deliberately compact search grid.  With n=303 an enormous grid mostly buys
optimism: the best score of a huge grid on 5x5 CV is itself a biased estimate,
so grids stay small and centred on values that are sensible for this data size.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config as C
from .preprocessing import build_preprocessor

try:  # XGBoost is optional at import time so the rest still runs without it.
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = f"{type(exc).__name__}: {exc}"


def _pipe(estimator, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(scale_numeric=scale_numeric)),
         ("clf", estimator)]
    )


def model_zoo(scale_pos_weight: float = 1.0) -> dict:
    """Build every candidate model with its grid.

    ``scale_pos_weight`` is computed from the *training* class counts only and
    is passed to XGBoost; the other learners use ``class_weight='balanced'``
    as a tuning option instead.
    """
    seed = C.RANDOM_SEED
    zoo: dict[str, dict] = {}

    # ---------------------------------------------------- Phase 7: baseline
    zoo["Logistic Regression"] = {
        "pipeline": _pipe(
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed),
            scale_numeric=True,
        ),
        "grid": {
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            # scikit-learn >= 1.8 expresses the penalty as a mixing ratio:
            # 0.0 = pure L2 (ridge), 1.0 = pure L1 (lasso).
            "clf__l1_ratio": [0.0, 1.0],
            "clf__class_weight": [None, "balanced"],
        },
        "needs_scaling": True,
        "notes": "Regularised linear baseline; coefficients are directly readable.",
    }

    # -------------------------------------------------------- Phase 8: SVM
    # SVC has no native probabilities. Wrapping it in CalibratedClassifierCV
    # fits Platt scaling on internal folds - the supported replacement for the
    # deprecated SVC(probability=True) - so predict_proba is available for
    # ROC-AUC, threshold search and the UI's probability readout.
    zoo["SVM (Linear)"] = {
        "pipeline": _pipe(
            CalibratedClassifierCV(
                SVC(kernel="linear", random_state=seed), method="sigmoid",
                ensemble=False, cv=5),
            scale_numeric=True,
        ),
        "grid": {
            "clf__estimator__C": [0.01, 0.1, 1.0, 10.0],
            "clf__estimator__class_weight": [None, "balanced"],
        },
        "needs_scaling": True,
        "notes": "Maximum-margin linear separator; scaling is mandatory. "
                 "Probabilities come from Platt scaling on internal folds.",
    }
    zoo["SVM (RBF)"] = {
        "pipeline": _pipe(
            CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=seed), method="sigmoid",
                ensemble=False, cv=5),
            scale_numeric=True,
        ),
        "grid": {
            "clf__estimator__C": [0.1, 1.0, 10.0, 100.0],
            "clf__estimator__gamma": ["scale", 0.01, 0.05, 0.1],
            "clf__estimator__class_weight": [None, "balanced"],
        },
        "needs_scaling": True,
        "notes": "Non-linear kernel; C and gamma jointly control the margin. "
                 "Probabilities come from Platt scaling on internal folds.",
    }

    # ---------------------------------------------- Phase 9: Random Forest
    zoo["Random Forest"] = {
        "pipeline": _pipe(
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            scale_numeric=False,
        ),
        "grid": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced"],
        },
        "needs_scaling": False,
        "notes": "Bagged trees; depth and leaf size are the main brakes on "
                 "overfitting at n=303.",
    }

    # --------------------------------------------------- Phase 10: XGBoost
    if XGBOOST_AVAILABLE:
        zoo["XGBoost"] = {
            "pipeline": _pipe(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
            "grid": {
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3, 4],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                "clf__reg_lambda": [1.0, 5.0],
                "clf__min_child_weight": [1, 5],
                "clf__scale_pos_weight": [1.0, round(float(scale_pos_weight), 3)],
            },
            "needs_scaling": False,
            "notes": "Gradient-boosted trees with L2 regularisation; shallow "
                     "depth (2-4) suits a small tabular dataset.",
        }
    return zoo
