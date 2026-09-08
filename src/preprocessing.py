"""Leakage-safe preprocessing (Phases 4 & 6).

The whole transformation lives inside a scikit-learn ``ColumnTransformer`` so
that imputation statistics, scaler means and one-hot categories are learned
from training folds only and re-applied to validation/test data.  Nothing here
ever sees the full dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Return the preprocessing ``ColumnTransformer``.

    Parameters
    ----------
    scale_numeric:
        ``True`` for distance/margin/gradient based learners (Logistic
        Regression, SVM) where feature scale changes the solution.  ``False``
        for tree ensembles, which are invariant to monotone rescaling - it
        keeps their split thresholds in original clinical units, which makes
        the exported trees easier to reason about.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    nominal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False, dtype=np.float64)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, C.NUMERIC_FEATURES),
            ("bin", binary_pipe, C.BINARY_FEATURES),
            ("nom", nominal_pipe, C.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encoded_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Column names of the design matrix produced by a fitted preprocessor."""
    return [str(n) for n in fitted_preprocessor.get_feature_names_out()]


def preprocessing_summary(scale_numeric: bool = True) -> dict:
    """Machine-readable description of the pipeline, for metadata export."""
    return {
        "numeric_features": C.NUMERIC_FEATURES,
        "numeric_steps": ["SimpleImputer(strategy='median')"]
        + (["StandardScaler()"] if scale_numeric else []),
        "binary_features": C.BINARY_FEATURES,
        "binary_steps": ["SimpleImputer(strategy='most_frequent')", "passthrough 0/1"],
        "nominal_features": C.NOMINAL_FEATURES,
        "nominal_steps": [
            "SimpleImputer(strategy='most_frequent')",
            "OneHotEncoder(handle_unknown='ignore')",
        ],
        "leakage_control": (
            "Fitted inside a Pipeline, so every fit happens on training data "
            "only - once per cross-validation fold and once on the final "
            "training split. The held-out test set is transformed, never fitted."
        ),
    }
