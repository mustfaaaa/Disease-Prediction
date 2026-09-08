# Disease Risk Prediction from Medical Data

**Heart disease classification on the UCI Cleveland dataset — model, inference
pipeline, and a working web interface.**

> **Educational use only.** This project demonstrates machine-learning
> classification on a public research dataset. Model outputs are **not medical
> diagnoses, not medical advice, and not a substitute for evaluation by a
> qualified healthcare professional.**

---

## 1. Overview

A complete supervised-learning workflow on structured medical data, delivered as
a running application rather than a notebook:

```
UCI archive → clean → EDA → leakage-safe pipeline → 4 algorithm families
   → tuning → repeated CV → threshold + calibration → explainability
   → export → inference service → web interface
```

Four algorithm families (logistic regression, SVM, random forest, XGBoost) are
trained on the same split, tuned the same way, and scored with the same metrics.
One is selected by a rule fixed before the results were known. The selected
model is exported with its preprocessing pipeline, decision threshold and
explainer, then served through a FastAPI endpoint that a browser interface
calls for every prediction.

**Everything numeric in this README, in the report, in the notebook and in the
interface is read from the artefacts of an actual training run.** Nothing is
illustrative.

---

## 2. Features

| Area | What is implemented |
|---|---|
| Data | Automatic download from UCI, SHA-256 recorded, cached locally |
| Profiling | Row/column/dtype/missing/duplicate/range profile, feature dictionary |
| Cleaning | Target binarisation, out-of-codebook coercion, duplicate check — each decision logged with its reason |
| Leakage control | All preprocessing inside a `Pipeline`; test split transformed, never fitted |
| EDA | 6 figures, each answering a question that shapes a later decision |
| Models | Logistic regression, linear SVM, RBF SVM, random forest, XGBoost |
| Tuning | `GridSearchCV` / `RandomizedSearchCV` on the training split only |
| Imbalance | Measured; SMOTE evaluated **inside** CV folds and reported |
| Validation | `RepeatedStratifiedKFold` (5 folds × 5 repeats) = 25 fits per model |
| Metrics | Accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier, confusion matrix |
| Threshold | Swept on out-of-fold *training* predictions; test set never used to choose it |
| Calibration | Reliability curve, Brier, ECE; Platt and isotonic both fitted inside folds |
| Explainability | Coefficients + odds ratios, permutation importance, TreeSHAP, per-prediction attribution |
| Error analysis | Every false positive and false negative inspected individually |
| Export | Model, preprocessing pipeline, explainer, metadata, metrics, feature schema |
| Serving | FastAPI; model deserialised once at start-up |
| Interface | Responsive, accessible single-page app with live model integration |
| Tests | 84 tests covering data, leakage, models, metrics, artefacts, inference and HTTP |

---

## 3. Dataset

**UCI Heart Disease — Cleveland subset** ·
[archive.ics.uci.edu/dataset/45](https://archive.ics.uci.edu/dataset/45/heart+disease)

| | |
|---|---|
| Records | 303 |
| Features | 13 |
| Classes | 2 — 164 negative (54.1%), 139 positive (45.9%) |
| Missing | 6 cells: `ca` (4), `thal` (2) |
| Duplicates | 0 |
| Label | Angiographic status: 0 = <50% narrowing in all major vessels; 1 = >50% in at least one. Derived by binarising the original `num` column (0–4). |

### Why this dataset

Three candidates were profiled with real numbers before choosing
(`scripts/dataset_survey.py`):

| Candidate | Rows × features | Minority | Missing | Feature types |
|---|---|---|---|---|
| **Heart Disease (Cleveland)** | 303 × 13 | 45.9% | 6 cells, 2 columns | 8 categorical-like + 5 continuous |
| Breast Cancer Wisconsin | 569 × 30 | 37.3% | none | 30 continuous |
| Pima Indians Diabetes | 768 × 8 | 34.9% | 374 hidden zeros in `insulin` alone | 8 continuous |

Cleveland was selected because it is the only candidate that:

- covers **all four feature groups the brief names** — symptoms, age, blood
  tests and clinical measurements;
- **mixes categorical and continuous inputs**, so a `ColumnTransformer` with
  one-hot encoding and scaling is genuinely required rather than decorative;
- carries **real documented missing values**, so imputation has to be handled
  inside the pipeline rather than skipped;
- has **clinically meaningful column names** that can be shown to a person in a
  form. Breast Cancer's "worst fractal dimension" cannot be, and Pima is a
  female-only, single-heritage cohort with 49% of `insulin` missing.

Its weakness is size. 303 records is small, and every confidence interval in
this project is wide because of it. That is stated rather than hidden.

### Feature dictionary

| Interface label | Column | Type | Units | Allowed values |
|---|---|---|---|---|
| Age | `age` | numeric | years | 18–100 (observed 29–77) |
| Sex | `sex` | binary | – | 0 = Female, 1 = Male |
| Chest Pain Type | `cp` | nominal | – | 1 Typical angina, 2 Atypical angina, 3 Non-anginal pain, 4 Asymptomatic |
| Exercise-Induced Angina | `exang` | binary | – | 0 = No, 1 = Yes |
| Resting Blood Pressure | `trestbps` | numeric | mm Hg | 80–220 (observed 94–200) |
| Serum Cholesterol | `chol` | numeric | mg/dl | 100–600 (observed 126–564) |
| Fasting Blood Sugar > 120 mg/dl | `fbs` | binary | – | 0 = No, 1 = Yes |
| Resting ECG Result | `restecg` | nominal | – | 0 Normal, 1 ST-T abnormality, 2 LV hypertrophy |
| Maximum Heart Rate Achieved | `thalach` | numeric | bpm | 60–220 (observed 71–202) |
| ST Depression (Exercise vs Rest) | `oldpeak` | numeric | mm | 0–7 (observed 0.0–6.2) |
| Peak Exercise ST Segment Slope | `slope` | nominal | – | 1 Upsloping, 2 Flat, 3 Downsloping |
| Major Vessels Coloured by Fluoroscopy | `ca` | numeric (ordered count) | count | 0–3 |
| Thallium Stress Test Result | `thal` | nominal | – | 3 Normal, 6 Fixed defect, 7 Reversible defect |

---

## 4. ML pipeline

```
303 records
   │
   ├─ clean ─────── binarise target · check duplicates · coerce out-of-codebook values
   │
   ├─ stratified 80/20 split (seed 42) ──► 242 train · 61 test   [test sealed]
   │
   ├─ TRAIN ONLY
   │    ├─ ColumnTransformer inside every estimator pipeline
   │    │     numeric  → median impute → StandardScaler   (LR / SVM only)
   │    │     binary   → mode impute   → passthrough 0/1
   │    │     nominal  → mode impute   → OneHotEncoder    → 22 encoded columns
   │    ├─ hyper-parameter search, 5-fold stratified CV, scored on ROC-AUC
   │    ├─ RepeatedStratifiedKFold(5 × 5) = 25 fits per tuned model
   │    ├─ out-of-fold probabilities → threshold sweep + calibration study
   │    └─ model selection rule
   │
   └─ TEST (scored once) ──► final metrics · curves · confusion matrix
                                  · permutation importance · error analysis
```

Preprocessing is never fitted outside a training fold. `StandardScaler` is
applied for logistic regression and the SVMs, where feature scale changes the
solution, and skipped for the tree ensembles, which are invariant to monotone
rescaling — this keeps their split thresholds in original clinical units.

---

## 5. Algorithms and tuning

| Model | Search | Selected hyper-parameters |
|---|---|---|
| Logistic Regression | GridSearchCV, 24 combinations | `C=0.5`, `l1_ratio=0.0` (ridge), `class_weight='balanced'` |
| SVM (Linear) | GridSearchCV, 8 combinations | `C=0.1`, `class_weight=None` |
| SVM (RBF) | GridSearchCV, 32 combinations | `C=1.0`, `gamma=0.01`, `class_weight='balanced'` |
| Random Forest | RandomizedSearchCV, 60 of 288 | `n_estimators=600`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features='sqrt'`, `class_weight='balanced'` |
| XGBoost | RandomizedSearchCV, 60 of 384 | `n_estimators=200`, `learning_rate=0.03`, `max_depth=2`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=5.0`, `min_child_weight=5`, `scale_pos_weight=1.18` |

Grids are deliberately small. With 242 training rows, the best score of a huge
grid is itself an optimistic estimate.

**XGBoost early stopping.** Run separately on an inner 80/20 split of the
training data (193 fit / 49 validation): of 1000 rounds offered, the best
iteration was **98**, validation log-loss **0.36844**, patience 40. That result
informed the small `n_estimators` values in the grid; the grid search itself
uses plain cross-validation so every candidate is scored identically.

**Class imbalance.** The training split is 1.18:1 (45.9% minority) — mild.
SMOTE was still evaluated, applied inside each CV fold only:

| Strategy | ROC-AUC | Recall | F1 |
|---|---|---|---|
| No resampling | 0.9069 ± 0.0177 | 0.7656 ± 0.0451 | 0.8136 ± 0.0098 |
| `class_weight='balanced'` | 0.9082 ± 0.0184 | 0.8281 ± 0.0542 | 0.8444 ± 0.0264 |
| SMOTE inside folds | 0.9086 ± 0.0177 | 0.8190 ± 0.0512 | 0.8347 ± 0.0199 |

SMOTE gains 0.0017 ROC-AUC over no resampling — a tenth of the fold-to-fold
standard deviation. It is not used: synthesising patients adds risk without
solving a problem this dataset has. Class weighting stays a tunable option in
every grid, and the selected model uses it.

---

## 6. Model comparison

CV columns: mean ± sd over 25 fits (`RepeatedStratifiedKFold`, 5 × 5) on the
training split. Test columns: the 61 held-out records at the default 0.50
threshold, so every model is compared on identical terms.

| Model | CV ROC-AUC | CV PR-AUC | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** *(final)* | 0.898 ± 0.043 | 0.894 ± 0.048 | 0.869 | 0.812 | 0.929 | 0.818 | 0.867 | 0.959 | 0.942 |
| SVM (Linear) | 0.895 ± 0.043 | 0.893 ± 0.045 | 0.836 | 0.765 | 0.929 | 0.758 | 0.839 | 0.956 | 0.939 |
| SVM (RBF) | 0.897 ± 0.041 | 0.893 ± 0.048 | 0.869 | 0.812 | 0.929 | 0.818 | 0.867 | 0.958 | 0.942 |
| Random Forest | 0.893 ± 0.041 | 0.893 ± 0.041 | 0.885 | 0.839 | 0.929 | 0.849 | 0.881 | 0.947 | 0.932 |
| XGBoost | 0.895 ± 0.037 | 0.897 ± 0.036 | 0.869 | 0.833 | 0.893 | 0.849 | 0.862 | 0.944 | 0.941 |

The spread across five models is 0.005 CV ROC-AUC against a fold-to-fold
standard deviation of ~0.04. **They are statistically indistinguishable on
discrimination**, which is the finding — not a defect.

### Why logistic regression was selected

The rule was fixed before the numbers were in:

1. Rank by mean CV ROC-AUC; keep everything within **one standard error** of the
   best (cut-off 0.8890). All five models qualify.
2. Apply the same one-SE filter to PR-AUC (cut-off 0.8893), so a model clearly
   weaker on the positive class is dropped.
3. Among survivors prefer the **better-calibrated** model, treating Brier
   differences below 0.01 as ties.
4. Break remaining ties on **interpretability**, then generalisation gap, then
   fit cost.

| Model | CV ROC-AUC | CV PR-AUC | Brier | Calibration group | Interpretability tier | Train–test gap |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8976 | 0.8938 | 0.1203 | best | 1 | 0.0337 |
| SVM (Linear) | 0.8950 | 0.8935 | 0.1224 | best | 2 | 0.0294 |
| XGBoost | 0.8954 | 0.8966 | 0.1274 | best | 3 | 0.0552 |
| SVM (RBF) | 0.8968 | 0.8932 | 0.1220 | best | 4 | 0.0247 |
| Random Forest | 0.8934 | 0.8935 | 0.1330 | worse | 3 | 0.0823 |

Logistic regression wins on the tie-breakers: best Brier score (0.1203), the
only model whose reasoning is directly readable as signed coefficients and odds
ratios, the smallest generalisation gap among the interpretable models, and the
cheapest to fit. Random forest — which has the best raw test accuracy (0.885) —
is dropped by step 3 for the worst calibration and the largest train–test gap
(0.0823), a reminder of why accuracy alone is a poor selection criterion.

---

## 7. Final results

**Final model:** Logistic Regression · `C=0.5`, ridge penalty,
`class_weight='balanced'` · decision threshold **0.38** · uncalibrated
(recalibration did not improve the Brier score).

### Threshold analysis

Chosen on cross-validated **training** predictions: the highest-specificity
threshold that still reaches 85% sensitivity. The test set played no part.

| Operating point | Threshold | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| Default | 0.50 | 0.811 | 0.886 | 0.857 | 0.833 |
| Best F1 | 0.51 | 0.802 | 0.901 | 0.873 | 0.836 |
| **Selected** (≥85% sensitivity) | **0.38** | 0.865 | 0.756 | 0.750 | 0.803 |

### Held-out test performance (61 records)

| Metric | At 0.50 | At 0.38 (deployed) |
|---|---|---|
| Accuracy | 0.869 | 0.820 |
| Precision | 0.813 | 0.730 |
| Sensitivity | 0.929 | **0.964** |
| Specificity | 0.818 | 0.697 |
| F1 | 0.867 | 0.831 |
| ROC-AUC | 0.959 | 0.959 |
| PR-AUC | 0.942 | 0.942 |
| Brier | 0.089 | 0.089 |

ROC-AUC, PR-AUC and Brier are threshold-independent, so they are identical in
both columns — only the classification decision moves.

**Confusion matrix at 0.38:** TN 23 · FP 10 · FN 1 · TP 27.
The lower threshold converts 1 false negative into 4 extra false positives
relative to 0.50. That is the intended trade in a screening context, and it is
shown to the user in the interface rather than buried.

**Calibration:** Brier 0.1203, expected calibration error 0.0638 on
cross-validated training predictions. Platt scaling (0.1228) and isotonic
regression (0.1213) both scored slightly worse, so the uncalibrated model is
kept — the simpler artefact is preferred unless recalibration improves Brier by
at least 0.002.

### Figures (`outputs/figures/`)

| File | Shows |
|---|---|
| `01_target_distribution.png` | Class balance |
| `02_numeric_distributions.png` | Numeric features by outcome |
| `03_boxplots_by_target.png` | Same, as box plots |
| `04_correlation_heatmap.png` | Feature correlation matrix |
| `05_categorical_outcome_rates.png` | Positive rate per category vs base rate |
| `06_missingness.png` | Missing values per feature |
| `10_roc_curve.png` · `11_pr_curve.png` | Discrimination on the test set |
| `12_confusion_matrix.png` | Test outcomes at 0.38 |
| `13_calibration_curve.png` | Reliability diagram |
| `14_threshold_sweep.png` | Sensitivity/specificity across thresholds |
| `15_model_comparison.png` | CV vs test across all five models |
| `16_permutation_importance.png` · `17_coefficients.png` | Explainability |

---

## 8. Explainability

**Permutation importance** — mean drop in test ROC-AUC when one raw column is
shuffled (30 repeats):

| Feature | Drop in ROC-AUC |
|---|---|
| Major Vessels Coloured by Fluoroscopy | 0.0886 ± 0.0339 |
| Chest Pain Type | 0.0308 ± 0.0111 |
| Thallium Stress Test Result | 0.0139 ± 0.0092 |
| Peak Exercise ST Segment Slope | 0.0126 ± 0.0083 |
| ST Depression (Exercise vs Rest) | 0.0103 ± 0.0043 |

**Coefficients** of the deployed model (log-odds; numeric inputs are
standardised, so magnitudes are comparable):

| Encoded feature | Coefficient | Odds ratio |
|---|---|---|
| Major Vessels Coloured by Fluoroscopy | +1.028 | 2.795 |
| Sex (male) | +0.984 | 2.676 |
| Thallium: Normal | −0.778 | 0.459 |
| Chest Pain Type: Asymptomatic | +0.757 | 2.131 |
| Thallium: Reversible defect | +0.606 | 1.833 |
| ST Slope: Upsloping | −0.558 | 0.572 |

**TreeSHAP** (mean |SHAP| across the 242 training records, XGBoost — included
because gradient-boosted trees have no readable coefficients of their own):
Thallium Stress Test Result 0.731 · Chest Pain Type 0.603 · Major Vessels 0.586
· Sex 0.274 · ST Slope 0.246.

**Per-prediction attribution** is exported with the model and returned by the
API. For the deployed linear model it is `w_j × (x_j − mean_j)` on the encoded
matrix — additive in log-odds and exact. Tree models would use TreeSHAP; the
SVMs fall back to a model-agnostic ablation (replace one input with its training
reference value, measure how far the estimate moves).

> These quantities describe **influence on the model's own output**. They are
> not evidence that a feature causes disease.

### Error analysis

At the 0.38 threshold on 61 test records: 27 true positives, 23 true negatives,
**10 false positives, 1 false negative**.

- Mean distance from the threshold is **0.199 for wrong predictions versus 0.401
  for correct ones** — the model is measurably less certain when it errs, which
  is the behaviour you want from a calibrated score.
- The single false negative: age 60, 0 vessels coloured, ST depression 3.00 mm
  (test-set averages 54.0, 0.93, 1.20) — a record with a normal fluoroscopy
  result, the feature the model leans on hardest.
- The 10 false positives average 0.89 vessels coloured and 0.69 mm ST
  depression, close to the cohort average — genuinely ambiguous records.

A false negative sends a patient with significant narrowing away without
follow-up; a false positive sends a healthy patient to further, sometimes
invasive, testing. In screening the first error is normally the more serious,
which is why the threshold favours sensitivity.

---

## 9. Interface

A single-page application served by the same process as the API. It is **not** a
static demo — every result comes from a `POST /api/predict` round trip against
the exported model.

**Sections:** Home · Prediction · How it works · Model · Performance · About

**Prediction flow**

1. Thirteen inputs, grouped into Patient Information, Symptoms, Vitals & Blood
   Tests, ECG & Exercise Test, and Imaging. The form is **generated from
   `/api/schema`**, so it cannot drift from what the model was trained on.
2. Human labels everywhere — "Chest Pain Type", never `cp` — with units, help
   text, and dropdowns carrying the clinical meaning of each code.
3. Validation on blur and on submit. Failures produce a focusable error summary
   (`role="alert"`) linking to each invalid field, plus an inline message tied to
   the input with `aria-describedby`.
4. Submitting disables the button, shows "Analysing patient data…" over a
   skeleton, and prevents duplicate submissions.
5. The result card shows classification, model-estimated probability, a
   probability meter with the decision threshold marked, distance from the
   threshold, response time, and the six most influential inputs with direction
   and share.
6. "Load a real record" pulls an actual row from the held-out test split, so the
   demo does not require inventing plausible clinical values.
7. Reset clears inputs, errors and the previous result.

**Design decisions**

- **Not red-for-sick / green-for-healthy.** A classification is not a verdict.
  Positive uses a calm orange with a flag icon, negative a blue with a check —
  and colour never carries meaning alone: every state also has an icon and a
  text label.
- Chart series use the blue/orange pair from the data-visualisation palette,
  validated with its own checker against these exact surfaces (lightness band,
  chroma floor, colour-vision-deficiency separation, normal-vision floor and
  contrast all pass in both light and dark).
- Text never wears a data colour. Series hues measure ~3.2:1 on white; direction
  labels use text tokens and let the coloured arrow carry identity.
- Every token pair was checked against WCAG 2.1 AA and adjusted where it failed
  — the muted text token was darkened, and the confusion-matrix ramp was
  re-stepped so a single ink colour clears 4.5:1 on every cell.
- Keyboard navigable end to end, 2px focus rings offset 2px,
  `scroll-padding-top` so focus is never hidden behind the sticky header, 44px+
  touch targets, `prefers-reduced-motion` respected, SVG icons (no emoji).
- Light and dark themes with an explicit toggle that overrides the OS setting in
  both directions; charts re-render with the theme's own validated colours.
- Verified at 375 px, 390 px, 768 px, 1280 px and 1440 px with no horizontal
  overflow; the form collapses to one column, tables scroll inside their own
  container, and the navigation becomes a disclosure menu below 860 px.
- No UI framework and no chart library. All charts are hand-drawn inline SVG,
  so the page has no runtime dependency that can fail to load.

To capture screenshots, run the app and use your browser's screenshot tool; the
generated matplotlib figures in `outputs/figures/` cover the same analytics.

---

## 10. Project structure

```
Disease-Prediction/
├── app/
│   ├── main.py               FastAPI service (model loaded once at start-up)
│   ├── schemas.py            Pydantic request schema
│   └── static/
│       ├── index.html        Semantic single-page structure
│       ├── styles.css        Design system: tokens, components, responsive rules
│       └── app.js            Form generation, validation, charts, rendering
├── data/
│   ├── raw/                  Cached UCI download + SOURCE.json (sha256)
│   └── processed/            heart_clean.csv
├── models/
│   ├── final_model.pkl       Full sklearn Pipeline (preprocessing + estimator)
│   ├── preprocessing_pipeline.pkl
│   ├── local_explainer.pkl   Per-prediction attribution
│   ├── metadata.json         Model card, split, seed, environment, timings
│   ├── metrics.json          Every metric, curve and study from the run
│   └── feature_info.json     Feature schema + dictionary
├── notebooks/
│   └── disease_prediction.ipynb    Executed walk-through (23 code cells, real outputs)
├── outputs/
│   ├── figures/              14 generated figures
│   ├── metrics/              Per-phase JSON exports
│   └── predictions/          test_predictions.csv with per-record outcomes
├── scripts/
│   ├── dataset_survey.py     Phase-1 evidence for the dataset choice
│   └── build_notebook.py     Generates and executes the notebook
├── src/
│   ├── config.py             Paths, seed, feature schema (single source of truth)
│   ├── data_loader.py        Download, profile, clean
│   ├── preprocessing.py      Leakage-safe ColumnTransformer
│   ├── eda.py                Exploratory figures + findings
│   ├── models.py             Model zoo and search grids
│   ├── evaluate.py           Metrics, threshold search, calibration
│   ├── explainability.py     Global and local explanations
│   ├── plots.py              Evaluation figures
│   ├── train.py              End-to-end training run
│   ├── inference.py          Validation → preprocessing → model → explanation
│   └── utils.py              JSON-safe serialisation, timing
├── tests/                    84 tests
├── requirements.txt
├── README.md
└── report.md
```

---

## 11. Installation

Python 3.11+ (built and verified on 3.14.7, Windows 11).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

On macOS/Linux the activation line is `source .venv/bin/activate`.

---

## 12. How to run

Train — downloads the dataset, runs every phase, writes `models/` and
`outputs/` (about 4–5 minutes):

```bash
python -m src.train
```

Serve the application at <http://localhost:8000>:

```bash
python -m uvicorn app.main:app --port 8000
```

Run the tests:

```bash
python -m pytest tests -q
```

Rebuild the executed notebook:

```bash
python scripts/build_notebook.py
```

Re-run the dataset selection survey:

```bash
python scripts/dataset_survey.py
```

---

## 13. API

Interactive docs are served at `/docs` once the app is running.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Whether the model artefacts loaded |
| `GET /api/schema` | Feature schema that generates the form |
| `GET /api/model` | Model card, metadata, selection rule |
| `GET /api/metrics` | Every metric, curve and study the Performance page renders |
| `POST /api/predict` | Classify one record |
| `GET /api/example` | A real record from the held-out test split |

### Example prediction

```bash
curl -s -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d "{\"age\":62,\"sex\":0,\"cp\":4,\"trestbps\":140,\"chol\":268,\"fbs\":0,\"restecg\":2,\"thalach\":160,\"exang\":0,\"oldpeak\":3.6,\"slope\":3,\"ca\":2,\"thal\":3}"
```

Response (abridged; produced by the exported model):

```json
{
  "classification": "Positive",
  "predicted_label": 1,
  "probability": 0.8184,
  "probability_percent": 81.8,
  "threshold": 0.38,
  "margin_from_threshold": 0.4384,
  "confidence_band": "clearly on one side of the threshold",
  "explanation": {
    "method": "linear_coefficients",
    "units": "log-odds shift vs. an average training record",
    "items": [
      {"label": "Major Vessels Coloured by Fluoroscopy", "display_value": "2",
       "contribution": 1.62521, "direction": "increases", "share": 0.3399},
      {"label": "Sex", "display_value": "Female",
       "contribution": -0.671, "direction": "decreases", "share": 0.1403},
      {"label": "Chest Pain Type", "display_value": "Asymptomatic",
       "contribution": 0.58665, "direction": "increases", "share": 0.1227}
    ]
  },
  "model": {"name": "Logistic Regression", "threshold": 0.38, "test_roc_auc": 0.9589},
  "disclaimer": "Educational use only. ..."
}
```

Add an optional `"threshold": 0.5` to score the same record at a different
operating point.

Validation errors return HTTP 400 with a message per field and never expose a
traceback:

```json
{
  "error": "invalid_request",
  "message": "Some fields need attention before the model can run.",
  "fields": {"cp": "Chest Pain Type must be one of: 1 (Typical angina), ..."}
}
```

---

## 14. Testing

`python -m pytest tests -q` → **84 passed**.

| File | Covers |
|---|---|
| `test_data.py` (12) | Download, shape, class counts, duplicates, missing values, domain conformance, cleaning log, profile accuracy, schema partitioning, human-readable labels |
| `test_preprocessing.py` (8) | Design-matrix shape, no surviving NaNs, **scaler fitted on train rows only**, `transform` does not refit, pipeline re-fits per fold, unseen category handling, tree variant skips scaling |
| `test_models_and_metrics.py` (25) | All four families present and importable, each trains and yields valid probabilities, each beats chance under CV, grid sizes bounded, scaling requested where it matters, metrics against a hand-computed confusion matrix, monotonicity of the threshold sweep, sensitivity floor respected, calibration and curve exports |
| `test_inference.py` (22) | Every validation rule, artefact existence, reload determinism, metadata/model agreement, confusion-matrix totals, singleton loading, threshold override, ranked explanations, non-causal wording |
| `test_api.py` (17) | Health, index, static assets, all four GET endpoints, happy-path prediction, consistency, six rejection cases, threshold override, **no traceback in any error body**, example record round-trip |

---

## 15. Limitations

- **303 records from one hospital in 1988.** Small, old, and not representative
  of today's patients or of other populations.
- **~68% male cohort.** Performance on female patients rests on far fewer
  records, and `sex` is the second-largest coefficient in the model.
- **The label is angiographic narrowing above 50%**, not a clinical diagnosis
  and not an outcome such as a cardiac event.
- **61 test records** means every test metric carries a wide confidence
  interval. Differences of a few points between models are noise; the project
  treats them as such.
- **Five models are statistically tied.** The selected one is not demonstrably
  the most accurate — it is the one the pre-declared rule picks among equals.
- **Several inputs require a hospital exercise test, fluoroscopy or a thallium
  scan.** This is not a self-assessment tool.
- `restecg = 1` occurs in only 4 records, so its one-hot column is fitted on
  almost no data.
- The model is trained and evaluated on a single dataset; no external validation
  cohort was available.

---

## 16. Medical disclaimer

**Educational use only.** This application demonstrates machine-learning
classification on a public dataset. Model predictions are not medical diagnoses,
medical advice, or a substitute for evaluation by a qualified healthcare
professional. Do not use it to make, delay or avoid any health decision. If you
have symptoms or concerns about your heart, contact a clinician or your local
emergency service.

The interface uses the phrase **"model-estimated probability"** throughout, and
never "probability of disease".

---

## 17. Future improvements

- Validate externally on the Hungarian, Switzerland and Long Beach VA subsets in
  the same UCI archive — the strongest available test of whether this
  generalises beyond Cleveland.
- Report bootstrap confidence intervals on every test metric, so the width of
  the uncertainty is visible rather than implied.
- Nested cross-validation, so the tuning step is itself inside the evaluation
  loop and the reported CV score is not optimistic.
- A subgroup breakdown by sex and age band, with the sample sizes shown.
- Let the interface move the operating threshold interactively and watch
  sensitivity and specificity trade off live — the API already accepts a
  threshold override.
- Persist predictions with a model version tag to support monitoring.
- Containerise, and pin the dataset by checksum in CI so a silent upstream
  change is caught.
