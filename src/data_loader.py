"""Dataset acquisition, cleaning and profiling (Phases 2-3).

Downloads the UCI Heart Disease archive once, caches the Cleveland subset to
``data/raw``, and produces a cleaned frame in ``data/processed``.  Every
cleaning decision is recorded in the returned :class:`CleaningReport` so the
report and README can quote real numbers instead of claims.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field, asdict
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Auditable log of what cleaning actually did."""

    rows_in: int = 0
    cols_in: int = 0
    duplicates_removed: int = 0
    missing_before: dict = field(default_factory=dict)
    invalid_values_found: dict = field(default_factory=dict)
    out_of_domain_clipped: dict = field(default_factory=dict)
    rows_out: int = 0
    cols_out: int = 0
    class_counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
def download_raw(force: bool = False) -> pd.DataFrame:
    """Fetch ``processed.cleveland.data`` from UCI, caching it as CSV."""
    if C.RAW_CSV.exists() and not force:
        return pd.read_csv(C.RAW_CSV)

    blob = urlopen(C.DATASET_URL, timeout=120).read()
    digest = hashlib.sha256(blob).hexdigest()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        member = zf.read(C.DATASET_FILE)

    df = pd.read_csv(io.BytesIO(member), names=C.RAW_COLUMNS, na_values="?")
    df.to_csv(C.RAW_CSV, index=False)
    (C.DATA_RAW / "SOURCE.json").write_text(
        json.dumps(
            {
                "dataset": C.DATASET_NAME,
                "page": C.DATASET_PAGE,
                "archive_url": C.DATASET_URL,
                "archive_sha256": digest,
                "member_extracted": C.DATASET_FILE,
                "columns": C.RAW_COLUMNS,
                "rows": int(len(df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df


# --------------------------------------------------------------------------
def profile(df: pd.DataFrame) -> dict:
    """Phase 2 - compute the dataset profile from the actual frame."""
    prof: dict = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "columns": {},
    }
    if C.TARGET in df.columns:
        counts = df[C.TARGET].value_counts().sort_index()
        prof["target_counts"] = {int(k): int(v) for k, v in counts.items()}
        prof["target_share"] = {
            int(k): round(float(v) / len(df), 4) for k, v in counts.items()
        }
        prof["imbalance_ratio"] = round(float(counts.max() / counts.min()), 3)

    for col in df.columns:
        s = df[col]
        entry = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            entry.update(
                min=float(d.min()), max=float(d.max()),
                mean=round(float(d.mean()), 4),
                std=round(float(d.std()), 4),
                median=float(d.median()),
                q1=float(d.quantile(0.25)), q3=float(d.quantile(0.75)),
            )
            if col in C.FEATURE_BY_NAME and C.FEATURE_BY_NAME[col]["role"] != "numeric":
                entry["unique_values"] = sorted(float(v) for v in d.unique())
            elif entry["n_unique"] <= 10:
                entry["unique_values"] = sorted(float(v) for v in d.unique())
        prof["columns"][col] = entry
    return prof


def feature_dictionary() -> list[dict]:
    """Human-readable feature dictionary rendered from the schema."""
    rows = []
    for f in C.FEATURES:
        domain = (
            ", ".join(f"{c['value']} = {c['label']}" for c in f["choices"])
            if f["role"] != "numeric"
            else f"{f['min']} - {f['max']}"
        )
        rows.append(
            {
                "feature": f["name"],
                "ui_label": f["label"],
                "type": f["role"],
                "meaning": f["description"],
                "units": f["unit"] or "-",
                "domain": domain,
                "group": f["group"],
            }
        )
    return rows


# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Phase 3 - cleaning, with each decision justified in ``notes``."""
    rep = CleaningReport(rows_in=int(len(df)), cols_in=int(df.shape[1]))
    out = df.copy()

    # 1. Binarise the multi-class angiographic severity column into the
    #    clinical question we actually pose: any significant narrowing or not.
    if "num" in out.columns:
        rep.notes.append(
            "Target: 'num' (0-4 severity) binarised to 0 / 1. Values 1-4 all "
            "mean >50% narrowing in at least one vessel; the 1-4 sub-levels "
            "have very few samples each (n<40), so a 5-class model would be "
            "unreliable at n=303."
        )
        out[C.TARGET] = (out["num"] > 0).astype(int)
        out = out.drop(columns=["num"])

    # 2. Sentinel '?' already became NaN at read time; record what is missing.
    rep.missing_before = {c: int(v) for c, v in out.isna().sum().items() if v}
    if rep.missing_before:
        rep.notes.append(
            f"Missing values found in {sorted(rep.missing_before)} "
            f"({sum(rep.missing_before.values())} cells, "
            f"{sum(rep.missing_before.values()) / out.size * 100:.2f}% of the frame). "
            "They are NOT dropped here: imputation happens inside the modelling "
            "pipeline so it is fitted on training folds only (leakage control)."
        )

    # 3. Exact duplicates - remove, they would leak between train and test.
    dupes = int(out.duplicated().sum())
    if dupes:
        out = out.drop_duplicates().reset_index(drop=True)
    rep.duplicates_removed = dupes
    rep.notes.append(
        f"Exact duplicate rows: {dupes}. "
        + ("Removed - identical records split across train/test would leak."
           if dupes else "None found, nothing removed.")
    )

    # 4. Values outside the documented UCI code book -> treat as missing.
    for f in C.FEATURES:
        col = f["name"]
        s = out[col]
        if f["role"] == "numeric":
            bad = s.notna() & ((s < f["min"]) | (s > f["max"]))
        else:
            allowed = {float(c["value"]) for c in f["choices"]}
            bad = s.notna() & ~s.astype(float).isin(allowed)
        n_bad = int(bad.sum())
        if n_bad:
            rep.invalid_values_found[col] = {
                "count": n_bad,
                "examples": sorted({float(v) for v in s[bad].unique()})[:5],
            }
            out.loc[bad, col] = np.nan

    if rep.invalid_values_found:
        rep.notes.append(
            "Out-of-codebook values set to NaN (then imputed inside the "
            f"pipeline): {rep.invalid_values_found}."
        )
    else:
        rep.notes.append(
            "No value fell outside its documented UCI domain, so no coercion "
            "was needed."
        )

    # 5. Outliers: kept. In a 303-row clinical cohort an extreme cholesterol
    #    or ST depression is signal, not noise, and dropping it would bias the
    #    model toward the average patient.
    rep.notes.append(
        "Outliers retained. Extreme values here (e.g. chol = 564 mg/dl) are "
        "clinically plausible and informative; deleting them at n=303 would "
        "discard real signal. Robustness is instead handled by scaling and by "
        "regularised / tree-based models."
    )

    out = out[C.FEATURE_ORDER + [C.TARGET]]
    rep.rows_out = int(len(out))
    rep.cols_out = int(out.shape[1])
    counts = out[C.TARGET].value_counts().sort_index()
    rep.class_counts = {int(k): int(v) for k, v in counts.items()}
    return out, rep


# --------------------------------------------------------------------------
def load_clean(force_download: bool = False) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience entry point: download -> clean -> cache -> return."""
    raw = download_raw(force=force_download)
    clean_df, rep = clean(raw)
    clean_df.to_csv(C.CLEAN_CSV, index=False)
    return clean_df, rep


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[C.FEATURE_ORDER].copy(), df[C.TARGET].copy()


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    raw = download_raw()
    df, rep = load_clean()
    print(json.dumps(profile(df), indent=2)[:2000])
    print(json.dumps(rep.to_dict(), indent=2))
