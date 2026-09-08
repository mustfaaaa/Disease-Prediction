"""Dataset loading, cleaning and schema-consistency tests (Phases 2-3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import feature_dictionary, profile


def test_dataset_downloads_and_has_expected_shape(clean_data):
    df, _ = clean_data
    assert len(df) == 303, "Cleveland subset should have 303 records"
    assert list(df.columns) == C.FEATURE_ORDER + [C.TARGET]


def test_target_is_binary_and_matches_published_counts(clean_data):
    df, _ = clean_data
    counts = df[C.TARGET].value_counts().to_dict()
    assert set(counts) == {0, 1}
    assert counts[0] == 164 and counts[1] == 139


def test_no_duplicate_records(clean_data):
    df, report = clean_data
    assert df.duplicated().sum() == 0
    assert report.duplicates_removed == 0


def test_missing_values_are_only_where_uci_documents_them(clean_data):
    df, report = clean_data
    missing = {k: int(v) for k, v in df.isna().sum().items() if v}
    assert missing == {"ca": 4, "thal": 2}
    assert report.missing_before == missing


def test_every_value_sits_inside_its_documented_domain(clean_data):
    df, _ = clean_data
    for spec in C.FEATURES:
        col = df[spec["name"]].dropna()
        if spec["role"] == "numeric":
            assert col.min() >= spec["min"], f"{spec['name']} below documented minimum"
            assert col.max() <= spec["max"], f"{spec['name']} above documented maximum"
        else:
            allowed = {float(c["value"]) for c in spec["choices"]}
            assert set(col.astype(float)) <= allowed, f"{spec['name']} has an undocumented code"


def test_cleaning_report_explains_every_decision(clean_data):
    _, report = clean_data
    assert len(report.notes) >= 4
    joined = " ".join(report.notes).lower()
    for topic in ("target", "missing", "duplicate", "outlier"):
        assert topic in joined, f"cleaning report never mentions {topic}"


def test_profile_reports_real_statistics(clean_data):
    df, _ = clean_data
    prof = profile(df)
    assert prof["n_rows"] == len(df)
    assert prof["target_counts"] == {0: 164, 1: 139}
    assert prof["columns"]["age"]["min"] == float(df["age"].min())
    assert np.isclose(prof["columns"]["chol"]["mean"], df["chol"].mean(), atol=1e-3)


def test_feature_dictionary_covers_every_feature():
    rows = feature_dictionary()
    assert len(rows) == len(C.FEATURES)
    for row in rows:
        assert row["ui_label"] and row["meaning"] and row["domain"]
        assert row["group"] in C.GROUP_ORDER


def test_schema_roles_partition_the_feature_list():
    partition = set(C.NUMERIC_FEATURES) | set(C.BINARY_FEATURES) | set(C.NOMINAL_FEATURES)
    assert partition == set(C.FEATURE_ORDER)
    assert len(C.NUMERIC_FEATURES) + len(C.BINARY_FEATURES) + len(C.NOMINAL_FEATURES) \
        == len(C.FEATURE_ORDER), "a feature is claimed by two roles"


def test_friendly_names_resolve_encoded_columns():
    assert C.friendly_name("num__age") == "Age"
    assert C.friendly_name("nom__cp_4.0") == "Chest Pain Type: Asymptomatic"
    assert C.friendly_name("nom__thal_7.0") == "Thallium Stress Test Result: Reversible defect"


def test_ui_labels_are_human_readable_not_column_codes():
    """Guards the requirement that the interface never shows raw codes like 'cp'."""
    for spec in C.FEATURES:
        assert spec["label"] != spec["name"]
        assert len(spec["label"]) > 2
        assert spec["label"][0].isupper()


def test_clean_csv_written_to_processed(clean_data):
    df, _ = clean_data
    assert C.CLEAN_CSV.exists()
    on_disk = pd.read_csv(C.CLEAN_CSV)
    assert on_disk.shape == df.shape
