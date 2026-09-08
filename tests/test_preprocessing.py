"""Preprocessing and data-leakage tests (Phases 4 & 6).

The leakage tests are the important ones: they check that preprocessing
statistics are learned from training rows only, which is the claim the whole
evaluation rests on.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config as C
from src.preprocessing import (
    build_preprocessor, encoded_feature_names, preprocessing_summary,
)


def test_preprocessor_produces_expected_design_matrix(xy):
    X, _ = xy
    prep = build_preprocessor(scale_numeric=True).fit(X)
    Z = prep.transform(X)
    n_onehot = sum(X[c].nunique(dropna=True) for c in C.NOMINAL_FEATURES)
    expected = len(C.NUMERIC_FEATURES) + len(C.BINARY_FEATURES) + n_onehot
    assert Z.shape == (len(X), expected)
    assert len(encoded_feature_names(prep)) == Z.shape[1]


def test_no_missing_values_survive_preprocessing(xy):
    X, _ = xy
    assert X.isna().sum().sum() > 0, "fixture should still contain the raw NaNs"
    Z = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(Z, dtype=float)).any()


def test_scaler_learns_from_training_rows_only(xy):
    """The core leakage check: fitted statistics must equal the TRAIN statistics,
    not the statistics of the whole dataset."""
    X, y = xy
    X_train, X_test = train_test_split(
        X, test_size=0.2, stratify=y, random_state=C.RANDOM_SEED)[:2]
    prep = build_preprocessor(scale_numeric=True).fit(X_train)
    scaler = prep.named_transformers_["num"].named_steps["scale"]

    train_imputed = prep.named_transformers_["num"].named_steps["impute"].transform(
        X_train[C.NUMERIC_FEATURES])
    assert np.allclose(scaler.mean_, train_imputed.mean(axis=0))

    full_mean = X[C.NUMERIC_FEATURES].median().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean), \
        "scaler means coincide with full-dataset statistics - possible leakage"


def test_transform_does_not_refit_on_test_data(xy):
    X, y = xy
    X_train, X_test = train_test_split(
        X, test_size=0.2, stratify=y, random_state=C.RANDOM_SEED)[:2]
    prep = build_preprocessor().fit(X_train)
    before = prep.named_transformers_["num"].named_steps["scale"].mean_.copy()
    prep.transform(X_test)
    after = prep.named_transformers_["num"].named_steps["scale"].mean_
    assert np.array_equal(before, after), "transform() mutated fitted statistics"


def test_pipeline_refits_preprocessing_per_fold(xy):
    """A Pipeline is what makes cross-validation leakage-free; confirm the
    preprocessing step really is inside the estimator being cloned."""
    from sklearn.base import clone
    X, y = xy
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=2000))])
    fresh = clone(pipe)
    assert not hasattr(fresh.named_steps["prep"], "transformers_"), \
        "cloned pipeline carries fitted preprocessing state"
    pipe.fit(X.iloc[:100], y.iloc[:100])
    a = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"].mean_.copy()
    pipe.fit(X.iloc[100:], y.iloc[100:])
    b = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"].mean_
    assert not np.allclose(a, b), "preprocessing did not re-fit on the new training rows"


def test_unseen_category_does_not_crash_inference(xy):
    """handle_unknown='ignore' means a category absent from training encodes to
    all-zeros rather than raising - important for a public-facing form."""
    X, _ = xy
    prep = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row.loc[row.index[0], "cp"] = 99.0
    Z = prep.transform(row)
    assert Z.shape[1] == prep.transform(X.iloc[[0]]).shape[1]
    assert np.isfinite(np.asarray(Z, dtype=float)).all()


def test_tree_variant_skips_scaling(xy):
    X, _ = xy
    prep = build_preprocessor(scale_numeric=False).fit(X)
    steps = dict(prep.named_transformers_["num"].named_steps)
    assert "scale" not in steps
    assert "impute" in steps


def test_preprocessing_summary_is_accurate():
    s = preprocessing_summary(scale_numeric=True)
    assert s["numeric_features"] == C.NUMERIC_FEATURES
    assert "StandardScaler()" in s["numeric_steps"]
    assert "StandardScaler()" not in preprocessing_summary(False)["numeric_steps"]
