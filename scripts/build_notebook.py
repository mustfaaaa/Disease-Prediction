"""Build and execute notebooks/disease_prediction.ipynb.

The notebook is generated from this script and then executed, so every output
committed in it is a real result rather than pasted text.  Re-run with:

    python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "disease_prediction.ipynb"

md = lambda text: nbf.v4.new_markdown_cell(text.strip())
code = lambda text: nbf.v4.new_code_cell(text.strip())

cells = [
md("""
# Disease Risk Prediction from Medical Data
### Heart disease classification on the UCI Cleveland dataset

This notebook is the readable walk-through of the project. It re-uses the same
modules the application runs on (`src/`), so nothing here is a re-implementation
that could drift from what is deployed.

* Full training run: `python -m src.train`
* Web application:   `python -m uvicorn app.main:app --port 8000`

Every number below is computed when the notebook runs.
"""),

code("""
import json, sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import load_clean, profile, feature_dictionary, split_xy

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)

print("dataset :", C.DATASET_NAME)
print("source  :", C.DATASET_PAGE)
print("seed    :", C.RANDOM_SEED, "| test size:", C.TEST_SIZE)
"""),

md("""
## 1. Dataset selection

Three candidates were profiled before choosing (`scripts/dataset_survey.py`):

| Candidate | Rows x features | Minority class | Missing | Feature types |
|---|---|---|---|---|
| **UCI Heart Disease (Cleveland)** | 303 x 13 | 45.9% | 6 cells in 2 columns | 8 categorical-like + 5 continuous |
| Breast Cancer Wisconsin | 569 x 30 | 37.3% | none | 30 continuous |
| Pima Indians Diabetes | 768 x 8 | 34.9% | 374 hidden zeros in `insulin` alone | 8 continuous |

Cleveland was chosen because it is the only one of the three that:

* covers all four feature groups the brief asks for - symptoms, age, blood
  tests and clinical measurements;
* mixes categorical and continuous inputs, so a `ColumnTransformer` is genuinely
  needed rather than decorative;
* carries real, documented missing values, so imputation has to be handled
  inside the pipeline;
* has clinically named columns that can be shown to a person in a form. Breast
  Cancer's "worst fractal dimension" cannot.

Its weakness is size: 303 rows is small, and every interval below is wide
because of it.
"""),

code("""
df, cleaning = load_clean()
prof = profile(df)

print(f"rows x cols     : {prof['n_rows']} x {prof['n_cols']}")
print(f"target counts   : {prof['target_counts']}")
print(f"class share     : {prof['target_share']}")
print(f"imbalance ratio : {prof['imbalance_ratio']}:1")
print(f"duplicate rows  : {prof['duplicate_rows']}")
print(f"missing cells   : {sum(prof['missing_per_column'].values())}")
df.head()
"""),

md("## 2. Feature dictionary\n\nRendered from the schema in `src/config.py`, which is also what generates the web form."),

code("""
pd.DataFrame(feature_dictionary())[
    ["feature", "ui_label", "type", "units", "domain", "group"]
]
"""),

code("""
stats = pd.DataFrame(prof["columns"]).T[["dtype", "missing", "n_unique", "min", "max", "mean", "std"]]
stats.loc[C.FEATURE_ORDER]
"""),

md("""
## 3. Cleaning decisions

Each decision, and the reason for it, is recorded by the loader rather than
described after the fact.
"""),

code("""
for i, note in enumerate(cleaning.notes, 1):
    print(f"{i}. {note}\\n")
"""),

md("""
## 4. Exploratory analysis

Figures are written to `outputs/figures/`. The numbers behind them are below.
"""),

code("""
from src.eda import run_eda
eda = run_eda(df)

print("Correlation with the outcome, strongest first:")
for k, v in eda["insights"]["target_correlation"].items():
    print(f"  {C.FEATURE_BY_NAME[k]['label']:<40} {v:+.3f}")
"""),

code("""
print("Standardised mean difference (Cohen's d) between outcome groups:")
for k, v in eda["insights"]["numeric_separation"].items():
    print(f"  {C.FEATURE_BY_NAME[k]['label']:<40} d = {v['cohens_d']:+.3f}"
          f"   (neg {v['mean_negative']}, pos {v['mean_positive']})")
"""),

code("""
print(f"Cohort positive rate: {df[C.TARGET].mean():.3f}\\n")
for feat, rates in eda["insights"]["categorical_positive_rates"].items():
    print(f"{C.FEATURE_BY_NAME[feat]['label']}:")
    for level, rate in rates.items():
        print(f"    {level:<32} {rate:.3f}")
"""),

md("""
## 5. Leakage control

Preprocessing lives inside the estimator pipeline. The check below shows the
scaler's statistics come from the training rows, not from the whole dataset.
"""),

code("""
from sklearn.model_selection import train_test_split
from src.preprocessing import build_preprocessor, encoded_feature_names

X, y = split_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_SEED)
print(f"train {len(X_train)} rows ({int(y_train.sum())} positive) | "
      f"test {len(X_test)} rows ({int(y_test.sum())} positive)")

prep = build_preprocessor(scale_numeric=True).fit(X_train)
scaler = prep.named_transformers_["num"].named_steps["scale"]
comparison = pd.DataFrame({
    "fitted_mean": scaler.mean_,
    "train_mean": X_train[C.NUMERIC_FEATURES].fillna(
        X_train[C.NUMERIC_FEATURES].median()).mean().to_numpy(),
    "full_dataset_mean": X[C.NUMERIC_FEATURES].fillna(
        X[C.NUMERIC_FEATURES].median()).mean().to_numpy(),
}, index=C.NUMERIC_FEATURES).round(4)
print()
print(comparison)
print(f"\\nDesign matrix: {len(X.columns)} raw inputs -> "
      f"{len(encoded_feature_names(prep))} encoded columns")
"""),

md("""
## 6. Class imbalance

The training split is 1.18:1, which is mild. SMOTE was still evaluated - applied
**inside** each cross-validation fold, never to the whole dataset - so the
decision not to use it rests on a measurement.
"""),

code("""
from src.train import imbalance_study
study = imbalance_study(X_train, y_train, C.RANDOM_SEED)
pd.DataFrame({
    name: {metric: f"{vals['mean']:.4f} +/- {vals['std']:.4f}"
           for metric, vals in res.items()}
    for name, res in study["comparison"].items()
}).T
"""),

md("""
## 7. Baseline and the three comparison models

A quick untuned pass first, to see where each family starts. The tuned results
from the full run follow in section 8.
"""),

code("""
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.models import model_zoo

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
rows = []
for name, spec in model_zoo().items():
    res = cross_validate(spec["pipeline"], X_train, y_train, cv=cv,
                         scoring=["roc_auc", "average_precision", "f1"], n_jobs=-1)
    rows.append({
        "model": name,
        "roc_auc": f"{res['test_roc_auc'].mean():.4f} +/- {res['test_roc_auc'].std():.4f}",
        "pr_auc": f"{res['test_average_precision'].mean():.4f}",
        "f1": f"{res['test_f1'].mean():.4f}",
        "notes": spec["notes"][:58] + "...",
    })
pd.DataFrame(rows).set_index("model")
"""),

md("""
## 8. Tuned model comparison

Loaded from `models/metrics.json`, written by the full training run. CV columns
are the mean over 25 fits (5-fold x 5 repeats) on the training split; test
columns come from the 61 held-out records at the default 0.50 threshold, so
every model is compared on the same footing.
"""),

code("""
metrics = json.loads((C.MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
table = pd.DataFrame(metrics["comparison_table"]).set_index("model")
table[["cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean", "test_accuracy",
       "test_precision", "test_sensitivity", "test_specificity", "test_f1",
       "test_roc_auc", "test_pr_auc"]].round(4)
"""),

code("""
for name, detail in metrics["per_model_details"].items():
    print(f"{name}")
    print(f"  search     : {detail['search_strategy']}, scored on {detail['search_scoring']}")
    print(f"  best score : {detail['best_search_score']}")
    print(f"  best params: {detail['best_params']}\\n")
"""),

md("""
## 9. Final model selection

The rule was fixed before the numbers were in: rank on cross-validated ROC-AUC,
keep everything within one standard error, then separate the survivors on
calibration, interpretability, generalisation gap and cost.
"""),

code("""
sel = metrics["selection"]
print(sel["rule"], "\\n")
print(f"best CV ROC-AUC : {sel['best_cv_roc_auc']}")
print(f"one-SE cut-offs : ROC-AUC >= {sel['roc_auc_one_se_cutoff']}, "
      f"PR-AUC >= {sel['pr_auc_one_se_cutoff']}")
print(f"shortlist       : {sel['shortlist']}")
print(f"selected        : {sel['selected']}\\n")
pd.DataFrame(sel["ranking_within_shortlist"]).set_index("model")
"""),

md("""
## 10. Threshold

0.50 is a default, not a decision. The operating point was chosen on
cross-validated *training* predictions: the highest-specificity threshold that
still reaches 85% sensitivity. The held-out test set played no part in it.
"""),

code("""
ta = metrics["threshold_analysis"]
pd.DataFrame([
    {"operating_point": "default 0.50", **{k: ta["default_metrics"][k] for k in
        ("threshold", "sensitivity", "specificity", "precision", "f1")}},
    {"operating_point": "best F1", **{k: ta["best_f1_metrics"][k] for k in
        ("threshold", "sensitivity", "specificity", "precision", "f1")}},
    {"operating_point": f"chosen (>= {ta['min_sensitivity_target']:.0%} sensitivity)",
     **{k: ta["chosen_metrics"][k] for k in
        ("threshold", "sensitivity", "specificity", "precision", "f1")}},
]).set_index("operating_point")
"""),

code("""
print("Held-out test performance of the selected model:\\n")
for label, key in (("at the default 0.50", "final_test_metrics_default_threshold"),
                   (f"at the chosen {ta['chosen_threshold']}", "final_test_metrics_chosen_threshold")):
    m = metrics[key]
    print(f"  {label}:")
    for k in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"):
        print(f"      {k:<12} {m[k]:.4f}")
    print(f"      confusion    {m['confusion_matrix']}\\n")
"""),

md("""
## 11. Calibration

"Model-estimated probability" is only a useful phrase if the estimates track
observed frequencies. Brier score and expected calibration error are measured on
cross-validated training predictions; Platt scaling and isotonic regression were
both fitted inside the folds and compared.
"""),

code("""
cal = metrics["calibration"]
rows = []
for variant in ("raw", "sigmoid", "isotonic"):
    v = cal.get(variant, {})
    if "brier" in v:
        rows.append({"variant": variant, "brier": v["brier"],
                     "ECE": v["expected_calibration_error"], "roc_auc": v.get("roc_auc")})
print(f"lowest Brier: {cal['best_by_brier']}   |   applied: {cal['applied']}\\n")
pd.DataFrame(rows).set_index("variant")
"""),

md("""
## 12. Explainability

Three views. All of them describe the model's behaviour - none of them is
evidence that a feature *causes* disease.
"""),

code("""
expl = metrics["explainability"]
print("Permutation importance (drop in held-out ROC-AUC when the column is shuffled):")
display(pd.DataFrame(expl["permutation_importance"])[
    ["label", "mean_drop_in_roc_auc", "std"]].head(8).set_index("label"))

if expl.get("coefficients"):
    print("\\nLargest coefficients of the deployed model (log-odds, standardised inputs):")
    display(pd.DataFrame(expl["coefficients"])[
        ["label", "coefficient", "odds_ratio"]].head(10).set_index("label"))
"""),

code("""
shap_info = expl.get("shap_summary", {})
if shap_info.get("available"):
    print(f"Mean |SHAP| per feature - {shap_info['model']}, "
          f"{shap_info['rows_explained']} training records:")
    display(pd.DataFrame(shap_info["by_feature"]).head(8).set_index("label"))
else:
    print("SHAP summary unavailable:", shap_info.get("error"))
"""),

md("""
## 13. Error analysis

Which records the model gets wrong, and whether the mistakes share anything.
"""),

code("""
err = metrics["error_analysis"]
print("Test-set outcomes:", err["counts"], "\\n")
for p in err["patterns"]:
    print(" -", p)
print("\\n", err["cost_note"])
"""),

code("""
cases = pd.DataFrame([{
    "type": c["type"], "actual": c["true_label"],
    "model_probability": c["model_probability"],
    "distance_from_threshold": c["distance_from_threshold"],
    "strongest_influence": f"{c['top_features'][0]['label']} = {c['top_features'][0]['value']}",
} for c in err["cases"]])
cases
"""),

md("""
## 14. The inference pipeline

The same object the web application uses. Input is validated, transformed by the
saved preprocessing pipeline, scored, thresholded and explained.
"""),

code("""
from src.inference import get_predictor, validate_record, ValidationError

predictor = get_predictor()
record = {"age": 62, "sex": 0, "cp": 4, "trestbps": 140, "chol": 268, "fbs": 0,
          "restecg": 2, "thalach": 160, "exang": 0, "oldpeak": 3.6, "slope": 3,
          "ca": 2, "thal": 3}

result = predictor.predict(record).to_dict()
print(f"classification            : {result['classification']}")
print(f"model-estimated probability: {result['probability_percent']}%")
print(f"decision threshold         : {result['threshold_percent']}%")
print(f"confidence band            : {result['confidence_band']}\\n")
print("Most influential inputs for this record:")
for item in result["explanation"]["items"]:
    print(f"  {item['label']:<42} {item['display_value']:<22} "
          f"{item['direction']:<10} {item['share']:.1%}")
"""),

code("""
# Validation rejects anything outside the documented schema, with a message per field.
for bad in ({**record, "age": 250}, {**record, "cp": 9}, {k: v for k, v in record.items() if k != "thal"}):
    try:
        validate_record(bad)
        print("accepted (unexpected)")
    except ValidationError as exc:
        print(exc.errors)
"""),

md("""
## 15. Where to go next

* `python -m src.train` re-runs everything above and rewrites `models/` and `outputs/`.
* `python -m uvicorn app.main:app --port 8000` serves the interface at
  <http://localhost:8000>, backed by these exact artefacts.
* `python -m pytest tests` runs the test suite.

---

**Medical disclaimer.** Educational use only. This project demonstrates
machine-learning classification on a public dataset. Model outputs are not
medical diagnoses, not medical advice, and not a substitute for evaluation by a
qualified healthcare professional.
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

print("executing notebook ...")
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)

errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"wrote {OUT.relative_to(ROOT)}  |  cells: {len(nb.cells)}  |  errors: {len(errors)}")
for e in errors:
    print(e.get("ename"), e.get("evalue"))
