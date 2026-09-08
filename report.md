# Disease Prediction from Medical Data — Project Report

**Task 4 — Predicting the possibility of disease from structured patient data**
Dataset: UCI Heart Disease (Cleveland) · Final model: Logistic Regression ·
Held-out ROC-AUC 0.959

> **Educational use only.** Model outputs in this project are not medical
> diagnoses, not medical advice, and not a substitute for evaluation by a
> qualified healthcare professional.

Every figure quoted in this report was produced by executing
`python -m src.train` and is read back from `models/metrics.json`,
`models/metadata.json` and `outputs/metrics/`. No value here was estimated,
rounded from memory, or written by hand.

---

## 1. Introduction

Coronary artery disease is diagnosed definitively by angiography — an invasive,
expensive procedure. Long before angiography, a clinician has a much cheaper
record: the patient's age and sex, the character of their chest pain, resting
vitals, a blood panel, a resting ECG, and the results of an exercise stress
test. The question this project asks is a supervised-learning question: **how
much of the angiographic outcome is recoverable from that cheaper record?**

The answer is developed as a working system rather than an analysis. A model is
trained, evaluated honestly, exported, served behind an HTTP API, and driven by
an interface that a non-specialist can use — because a model that cannot be run
by anyone but its author has not really been evaluated.

## 2. Problem statement

Given a structured record of 13 clinical attributes for one patient, produce a
binary classification — angiographic heart disease present or absent — together
with a calibrated, model-estimated probability and an account of which inputs
drove that estimate.

Formally: learn `f: X → [0,1]` where `X ⊂ ℝ¹³` mixes continuous and categorical
attributes, and choose a decision threshold `t` such that `ŷ = 1[f(x) ≥ t]`
reflects the asymmetric cost of the two error types in a screening setting.

## 3. Objective

1. Select one primary dataset on documented, measured grounds.
2. Profile and clean it, recording the reason for every decision.
3. Build a preprocessing pipeline that cannot leak information from the test
   split into training.
4. Train and tune four required algorithm families: logistic regression, SVM,
   random forest, XGBoost.
5. Compare them on cross-validated and held-out metrics that go well beyond
   accuracy.
6. Choose an operating threshold deliberately, and check whether the output
   behaves like a probability.
7. Explain the model globally and per prediction.
8. Analyse the errors.
9. Export the model and serve it through a real inference pipeline.
10. Deliver a professional, accessible interface backed by that pipeline.
11. Test all of it.

## 4. Dataset

**UCI Heart Disease, Cleveland subset** — `processed.cleveland.data` from
<https://archive.ics.uci.edu/dataset/45/heart+disease>. The archive is
downloaded at run time, its SHA-256 recorded in `data/raw/SOURCE.json`, and the
extracted subset cached as CSV.

| Property | Value |
|---|---|
| Records | 303 |
| Attributes used | 13 predictors + 1 target |
| Classes | 2 |
| Class counts | 164 negative (54.13%), 139 positive (45.87%) |
| Imbalance ratio | 1.18 : 1 |
| Missing cells | 6 — `ca` (4), `thal` (2) |
| Exact duplicate rows | 0 |

### 4.1 Why this dataset, with evidence

Three candidates were profiled before the choice was made
(`scripts/dataset_survey.py`, real output):

| Candidate | Rows × features | Minority class | Missing | Low-cardinality / continuous |
|---|---|---|---|---|
| **Heart Disease (Cleveland)** | 303 × 13 | 45.9% | 6 cells in 2 columns | 8 / 5 |
| Breast Cancer Wisconsin | 569 × 30 | 37.3% | 0 | 0 / 30 |
| Pima Indians Diabetes | 768 × 8 | 34.9% | 0 declared, but 374 impossible zeros in `insulin`, 227 in `skin_thickness` | 0 / 8 |

Cleveland is smaller than both alternatives, and that is its real cost. It was
still chosen because it is the only candidate that satisfies the task's own
requirements:

- **Feature coverage.** The brief names symptoms, age, blood-test results and
  clinical measurements. Cleveland has all four: chest pain type and
  exercise-induced angina (symptoms), age, cholesterol and fasting blood sugar
  (blood tests), resting blood pressure, maximum heart rate, ST depression,
  fluoroscopy and thallium scan (clinical measurements). Breast Cancer has one
  kind of feature — cell-nucleus morphometry. Pima has eight continuous
  measurements and no categorical variables at all.
- **Preprocessing is genuinely required.** Eight of the thirteen attributes are
  categorical or binary and five are continuous, so a `ColumnTransformer`
  combining one-hot encoding with scaling is necessary rather than performative.
  On Breast Cancer, "preprocessing" reduces to a single `StandardScaler`.
- **Missing values are real and documented.** Six cells encoded as `?` force the
  imputation question to be answered properly inside the pipeline.
- **The features can be shown to a person.** A form asking for "Chest Pain Type"
  and "Resting Blood Pressure" is meaningful. A form asking for "worst fractal
  dimension" is not, and the brief requires an interactive demonstration.
- **Reproducibility.** A stable, versioned, permanently hosted archive with a
  published code book.

Pima was additionally rejected on ethical grounds for a public demonstration: it
is a single-sex, single-heritage cohort with 48.7% of `insulin` silently missing
as zeros, which invites exactly the kind of over-generalisation this project is
trying to avoid.

### 4.2 Feature dictionary

| Feature | Type | Meaning | Units | Domain | Missing |
|---|---|---|---|---|---|
| `age` | numeric | Patient age | years | 18–100 (observed 29–77) | 0 |
| `sex` | binary | Biological sex as recorded | – | 0 = Female, 1 = Male | 0 |
| `cp` | nominal | Chest pain type at admission | – | 1 Typical angina · 2 Atypical angina · 3 Non-anginal pain · 4 Asymptomatic | 0 |
| `trestbps` | numeric | Resting systolic blood pressure on admission | mm Hg | 80–220 (observed 94–200) | 0 |
| `chol` | numeric | Total serum cholesterol | mg/dl | 100–600 (observed 126–564) | 0 |
| `fbs` | binary | Fasting blood sugar above 120 mg/dl | – | 0 = No, 1 = Yes | 0 |
| `restecg` | nominal | Resting electrocardiographic result | – | 0 Normal · 1 ST-T abnormality · 2 LV hypertrophy | 0 |
| `thalach` | numeric | Maximum heart rate during exercise test | bpm | 60–220 (observed 71–202) | 0 |
| `exang` | binary | Exercise-induced angina | – | 0 = No, 1 = Yes | 0 |
| `oldpeak` | numeric | ST depression, exercise relative to rest | mm | 0–7 (observed 0.0–6.2) | 0 |
| `slope` | nominal | Peak exercise ST segment slope | – | 1 Upsloping · 2 Flat · 3 Downsloping | 0 |
| `ca` | numeric (ordered count) | Major vessels coloured by fluoroscopy | count | 0–3 | 4 |
| `thal` | nominal | Thallium stress test result | – | 3 Normal · 6 Fixed defect · 7 Reversible defect | 2 |
| `target` | binary | **Outcome:** >50% narrowing in ≥1 major vessel | – | 0 = Negative, 1 = Positive | 0 |

`ca` is modelled as an ordered count rather than one-hot encoded: 0 < 1 < 2 < 3
is a real ordering, the relationship with the outcome is monotone, and treating
it as numeric costs three encoded columns fewer on a 242-row training split.

## 5. Data understanding

Descriptive statistics are computed on the actual frame
(`outputs/metrics/dataset_profile.json`). Selected values:

| Feature | Min | Median | Max | Mean | Std |
|---|---|---|---|---|---|
| `age` | 29 | 56 | 77 | 54.44 | 9.04 |
| `trestbps` | 94 | 130 | 200 | 131.69 | 17.60 |
| `chol` | 126 | 241 | 564 | 246.69 | 51.78 |
| `thalach` | 71 | 153 | 202 | 149.61 | 22.88 |
| `oldpeak` | 0.0 | 0.8 | 6.2 | 1.04 | 1.16 |
| `ca` | 0 | 0 | 3 | 0.67 | 0.94 |

Every value in every column falls inside its documented UCI domain: the cleaning
pass found **zero** out-of-codebook entries.

## 6. Data cleaning

Each decision is recorded by the loader, not asserted afterwards
(`outputs/metrics/cleaning_report.json`):

1. **Target binarised.** The original `num` column encodes angiographic severity
   0–4. Values 1–4 all mean >50% narrowing in at least one vessel, and the
   individual sub-levels have fewer than 40 records each — a five-class model at
   n = 303 would be unreliable. Mapped to `target = 1[num > 0]`.
2. **Missing values retained, not dropped.** 6 cells (0.14% of the frame) in
   `ca` and `thal`. Dropping the 6 affected rows would discard 2% of an already
   small dataset; imputing here would leak test statistics into training.
   Instead imputation is deferred to the modelling pipeline, where it is fitted
   per fold.
3. **Duplicates.** 0 exact duplicates found, so none removed. The check matters:
   identical records split across train and test would inflate every metric.
4. **Out-of-codebook values.** Every feature checked against its documented
   domain. None found, so no coercion was applied.
5. **Outliers retained.** A cholesterol of 564 mg/dl and an ST depression of
   6.2 mm are clinically plausible and informative. Deleting them at n = 303
   would bias the model toward the average patient. Robustness is handled by
   standardisation and by regularised or tree-based learners instead.

## 7. Exploratory data analysis

Six figures, each answering a question that shapes a later decision
(`outputs/figures/01`–`06`).

**Class balance** — 164 / 139, minority share 45.87%, ratio 1.18 : 1. Near
balanced, which is what makes heavy re-sampling unnecessary (§11).

**Correlation with the outcome** (point-biserial / Pearson):

| Feature | r | Feature | r |
|---|---|---|---|
| `thal` | +0.526 | `slope` | +0.339 |
| `ca` | +0.460 | `sex` | +0.277 |
| `exang` | +0.432 | `age` | +0.223 |
| `oldpeak` | +0.425 | `restecg` | +0.169 |
| `thalach` | −0.417 | `trestbps` | +0.151 |
| `cp` | +0.414 | `chol` | +0.085 |
| | | `fbs` | +0.025 |

`chol` and `fbs` are almost uncorrelated with the outcome here — a useful
counterweight to the popular intuition that cholesterol is the dominant signal.
They are retained: a weak marginal correlation does not preclude a useful
contribution in a multivariable model.

**Standardised group differences (Cohen's d), positive vs negative:**
`ca` +1.019 · `oldpeak` +0.919 · `thalach` −0.912 · `age` +0.461 ·
`trestbps` +0.303 · `chol` +0.172.

**Positive rate by category** (cohort base rate 45.9%):

- Chest pain: asymptomatic 72.9% (n=144) vs typical angina 30.4% (n=23) —
  counter-intuitive, and one of the strongest categorical signals in the data.
- Thallium: reversible defect 76.1% (n=117), fixed defect 66.7% (n=18), normal
  22.3% (n=166).
- Exercise-induced angina: yes 76.8% (n=99) vs no 30.9% (n=204).
- Sex: male 55.3% (n=206) vs female 25.8% (n=97).
- ST slope: flat 65.0% (n=140) vs upsloping 25.4% (n=142).
- Fasting blood sugar: 48.9% vs 45.3% — essentially no separation.

**Multicollinearity check.** Strongest predictor pairs: `oldpeak ~ slope` 0.578,
`age ~ thalach` 0.394, `thalach ~ slope` 0.386. Nothing near the level that
would destabilise a regularised linear model.

**Caveat found in EDA.** `restecg = 1` occurs in only **4 records**. Its one-hot
column is therefore fitted on almost no data and its coefficient should not be
interpreted.

## 8. Preprocessing and leakage prevention

The split happens first and the test set is not touched again until §13.

```
Stratified train_test_split(test_size=0.20, random_state=42)
    → 242 training rows (111 positive) · 61 test rows (28 positive)
```

All transformation lives inside a `ColumnTransformer` that is the first step of
every estimator `Pipeline`:

| Group | Features | Steps |
|---|---|---|
| Numeric | `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca` | `SimpleImputer(median)` → `StandardScaler` |
| Binary | `sex`, `exang`, `fbs` | `SimpleImputer(most_frequent)` → passthrough |
| Nominal | `cp`, `restecg`, `slope`, `thal` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |

13 raw inputs → **22 encoded columns**.

Because the transformer is inside the pipeline, `cross_validate`,
`GridSearchCV` and `cross_val_predict` all re-fit it on the training portion of
each fold. `StandardScaler` is applied for logistic regression and the SVMs,
where feature scale changes the solution, and omitted for the tree ensembles,
which are invariant to monotone rescaling.

`handle_unknown='ignore'` matters at serving time: a category the training fold
never saw encodes to all zeros instead of raising, so the API degrades rather
than crashing.

**Leakage is asserted by tests, not by claim** (`tests/test_preprocessing.py`):
the fitted scaler means equal the training-split means and differ from the
full-dataset means; `transform()` does not mutate fitted state; a cloned
pipeline carries no fitted preprocessing; re-fitting on different rows changes
the statistics.

## 9. Feature engineering

None beyond encoding, and that is a deliberate decision. With 242 training rows
and 22 encoded columns, the sample-to-parameter ratio is already about 11:1.
Interaction terms, polynomial expansions or ratio features would raise variance
faster than they could reduce bias, and any construction chosen by inspecting
the data would be a soft form of leakage. The one modelling choice made is the
ordinal treatment of `ca` (§4.2), which reduces dimensionality rather than
increasing it.

## 10. Algorithms

| Family | Rationale | Scaling |
|---|---|---|
| **Logistic Regression** | The required baseline, and the standard form of a clinical risk score. Coefficients are directly readable as log-odds and odds ratios. | required |
| **SVM (Linear)** | Maximum-margin alternative to the same linear hypothesis class; different loss, different regularisation behaviour. | required |
| **SVM (RBF)** | Tests whether a non-linear decision boundary buys anything on this data. | required |
| **Random Forest** | Bagged trees; captures interactions without them being specified, and is robust to feature scale. | not required |
| **XGBoost** | Gradient-boosted trees with explicit L1/L2 regularisation; the strongest general-purpose tabular learner. | not required |

The SVMs have no native probability output. They are wrapped in
`CalibratedClassifierCV(..., method='sigmoid', ensemble=False, cv=5)` — Platt
scaling fitted on internal folds, and the supported replacement for the
deprecated `SVC(probability=True)` — so ROC-AUC, the threshold sweep and the
interface's probability readout are all available for them too.

## 11. Class imbalance

The training split is 1.18 : 1 (45.87% minority). That is mild, but the question
was measured rather than assumed. A logistic-regression baseline was evaluated
under three strategies with 5-fold stratified CV, **SMOTE applied inside each
training fold only**:

| Strategy | ROC-AUC | Recall | F1 |
|---|---|---|---|
| No resampling | 0.9069 ± 0.0177 | 0.7656 ± 0.0451 | 0.8136 ± 0.0098 |
| `class_weight='balanced'` | 0.9082 ± 0.0184 | 0.8281 ± 0.0542 | 0.8444 ± 0.0264 |
| SMOTE inside folds | 0.9086 ± 0.0177 | 0.8190 ± 0.0512 | 0.8347 ± 0.0199 |

SMOTE gains 0.0017 ROC-AUC over no resampling — roughly a tenth of the
fold-to-fold standard deviation, i.e. nothing. Class weighting achieves a larger
recall gain (+0.063) at no cost and without synthesising patients, so it is left
as a tunable option in every model's grid; the selected model uses it. SMOTE is
not applied in the final pipeline.

Stratification is used everywhere: in the train/test split, in every `KFold`,
and in `cross_val_predict`.

## 12. Training and cross-validation

**Hyper-parameter search** — training split only, 5-fold stratified CV, scored
on ROC-AUC:

| Model | Strategy | Best parameters | Best search score |
|---|---|---|---|
| Logistic Regression | GridSearchCV, 24 combos | `C=0.5`, `l1_ratio=0.0`, `class_weight='balanced'` | 0.9085 |
| SVM (Linear) | GridSearchCV, 8 combos | `C=0.1`, `class_weight=None` | 0.9036 |
| SVM (RBF) | GridSearchCV, 32 combos | `C=1.0`, `gamma=0.01`, `class_weight='balanced'` | 0.9054 |
| Random Forest | RandomizedSearchCV, 60 of 288 | `n_estimators=600`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features='sqrt'`, `class_weight='balanced'` | 0.8999 |
| XGBoost | RandomizedSearchCV, 60 of 384 | `n_estimators=200`, `learning_rate=0.03`, `max_depth=2`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=5.0`, `min_child_weight=5`, `scale_pos_weight=1.18` | 0.8997 |

Grids are intentionally small. On 242 rows, the maximum of a large grid is
itself an optimistically biased estimate; a compact, well-centred grid is the
more honest instrument.

**XGBoost early stopping.** Run separately on an inner 80/20 split of the
training data (193 fit / 49 validation), 1000 rounds offered, patience 40. Best
iteration **98**, best validation log-loss **0.36844**. This is why the search
grid offers 200–400 estimators rather than thousands. The grid search itself
uses plain cross-validation so all candidates are scored identically.

**Evaluation protocol.** Each tuned pipeline is then re-scored with
`RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` — 25 fits per model — because
a single 5-fold estimate on 242 rows is too noisy to compare five models with.

Out-of-fold probabilities on the training split are produced separately with
`cross_val_predict`; these are the only predictions used to choose the threshold
and to assess calibration.

## 13. Evaluation metrics

Accuracy alone is unsuitable here. A model that predicts "negative" for
everything scores 54.1% accuracy and 0% sensitivity, which in a screening
context is worthless. Seven metrics plus the confusion matrix are therefore
reported:

- **Sensitivity (recall)** — of patients with disease, how many are flagged.
  This is what a screening aid exists to maximise.
- **Specificity** — of patients without disease, how many are correctly cleared.
  This is what stops the aid from flagging everyone.
- **Precision** — of those flagged, how many really have disease. It depends on
  prevalence, so it must be read alongside the base rate.
- **F1** — the harmonic mean of precision and recall.
- **ROC-AUC** — threshold-independent ranking quality.
- **PR-AUC (average precision)** — the same idea focused on the positive class,
  and more informative than ROC-AUC when positives are the minority.
- **Brier score** — squared error of the probability itself; the metric that
  says whether "model-estimated probability" is a meaningful phrase.

## 14. Model comparison

CV columns: mean ± sd over 25 fits on the training split. Test columns: 61
held-out records at the default 0.50 threshold, so every model is compared under
identical conditions.

| Model | CV ROC-AUC | CV PR-AUC | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.898 ± 0.043 | 0.894 ± 0.048 | 0.869 | 0.812 | 0.929 | 0.818 | 0.867 | **0.959** | 0.942 |
| SVM (Linear) | 0.895 ± 0.043 | 0.893 ± 0.045 | 0.836 | 0.765 | 0.929 | 0.758 | 0.839 | 0.956 | 0.939 |
| SVM (RBF) | 0.897 ± 0.041 | 0.893 ± 0.048 | 0.869 | 0.812 | 0.929 | 0.818 | 0.867 | 0.958 | 0.942 |
| Random Forest | 0.893 ± 0.041 | 0.893 ± 0.041 | **0.885** | 0.839 | 0.929 | **0.849** | **0.881** | 0.947 | 0.932 |
| XGBoost | 0.895 ± 0.037 | **0.897** ± 0.036 | 0.869 | 0.833 | 0.893 | **0.849** | 0.862 | 0.944 | 0.941 |

**The most important observation in this project:** the spread across five very
different algorithms is 0.005 CV ROC-AUC, against a fold-to-fold standard
deviation of about 0.04. They are statistically indistinguishable. The RBF
kernel buys nothing over the linear one, and gradient boosting buys nothing over
logistic regression. With 13 well-chosen clinical features and a largely
additive relationship to the outcome, there is little non-linear structure left
for a more flexible learner to find — and 242 rows is not enough to find it
reliably even if it existed.

Note also that the ordering flips between CV and test: logistic regression has
the best test ROC-AUC while random forest has the best test accuracy. On 61
records, that reordering is noise, which is exactly why selection is based on
cross-validation rather than on the test split.

## 15. Threshold analysis

0.50 is a default, not a decision. The sweep is computed on the 242
cross-validated **training** predictions; the test set is not involved.

| Operating point | Threshold | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| Default | 0.50 | 0.811 | 0.886 | 0.857 | 0.833 |
| Maximum F1 | 0.51 | 0.802 | 0.901 | 0.873 | 0.836 |
| **Selected** — highest specificity subject to sensitivity ≥ 0.85 | **0.38** | 0.865 | 0.756 | 0.750 | 0.803 |

Maximum F1 sits essentially at the default and would *reduce* sensitivity,
because F1 weights precision and recall equally — which is not the right weight
for screening. The selected rule states the clinical preference explicitly:
reach at least 85% sensitivity, then take the most specific threshold that still
does so. Moving from 0.50 to 0.38 buys +0.054 sensitivity for −0.130
specificity on the training folds.

That trade is shown to the user in the interface rather than hidden, and the API
accepts a `threshold` override so the alternative can be inspected directly.

## 16. Calibration

Measured on cross-validated training predictions:

| Variant | Brier | Expected calibration error | ROC-AUC |
|---|---|---|---|
| **Uncalibrated (deployed)** | **0.1203** | 0.0638 | 0.9048 |
| Platt scaling (sigmoid) | 0.1228 | 0.0637 | 0.9049 |
| Isotonic regression | 0.1213 | 0.0652 | 0.9000 |

Neither recalibration improves the Brier score, which is unsurprising: logistic
regression optimises log-loss directly and is already close to calibrated. The
rule applied requires an improvement of at least 0.002 Brier before replacing
the simpler artefact, so the uncalibrated model is kept. Isotonic regression on
242 rows would in any case risk overfitting the calibration curve itself.

Brier 0.1203 against a no-skill baseline of 0.248 (predicting the base rate for
everyone) means the estimated probabilities carry real information — which is
what licenses the interface's use of the phrase "model-estimated probability".
On the held-out test set the Brier score is 0.0885.

## 17. Explainability

### Global — permutation importance

Mean drop in held-out ROC-AUC when one raw input column is shuffled, 30 repeats:

| Feature | Drop in ROC-AUC | sd |
|---|---|---|
| Major Vessels Coloured by Fluoroscopy | 0.0886 | 0.0339 |
| Chest Pain Type | 0.0308 | 0.0111 |
| Thallium Stress Test Result | 0.0139 | 0.0092 |
| Peak Exercise ST Segment Slope | 0.0126 | 0.0083 |
| ST Depression (Exercise vs Rest) | 0.0103 | 0.0043 |
| Maximum Heart Rate Achieved | 0.0096 | 0.0077 |
| Exercise-Induced Angina | 0.0092 | 0.0031 |
| Resting Blood Pressure | 0.0073 | 0.0058 |

### Global — coefficients of the deployed model

Numeric inputs are standardised, so magnitudes are directly comparable:

| Encoded feature | Coefficient (log-odds) | Odds ratio |
|---|---|---|
| Major Vessels Coloured by Fluoroscopy | +1.028 | 2.795 |
| Sex (male) | +0.984 | 2.676 |
| Thallium: Normal | −0.778 | 0.459 |
| Chest Pain Type: Asymptomatic | +0.757 | 2.131 |
| Thallium: Reversible defect | +0.606 | 1.833 |
| ST Slope: Upsloping | −0.558 | 0.572 |
| Chest Pain Type: Typical angina | −0.556 | 0.574 |
| Chest Pain Type: Non-anginal pain | −0.554 | 0.575 |

### Global — TreeSHAP

Exact TreeSHAP on the strongest tree model (XGBoost), mean |SHAP| across the 242
training records, in log-odds. Included because gradient-boosted trees have no
readable coefficients of their own:

Thallium Stress Test Result 0.731 · Chest Pain Type 0.603 · Major Vessels 0.586
· Sex 0.274 · ST Slope 0.246 · ST Depression 0.214 · Age 0.158 · Maximum Heart
Rate 0.156.

The three methods disagree on the exact ordering — permutation importance puts
`ca` first, SHAP puts `thal` first — which is itself informative: they measure
different things (loss degradation on held-out data versus average attribution
across training records) and the top three features are the same set either way.

### Local — per prediction

The exported explainer produces a per-record attribution using whichever method
suits the deployed estimator:

- **Linear** (deployed): `w_j × (x_j − mean_j)` on the encoded matrix, summed
  back onto the 13 raw features. Additive in log-odds and exact.
- **Tree**: exact TreeSHAP.
- **Anything else** (e.g. the SVMs behind their probability layer): a
  model-agnostic ablation — replace one input with its training-set reference
  value and measure how far the estimate moves. Thirteen extra forward passes.

> **Wording.** These quantities describe influence on the model's own output.
> The interface says "raised the estimate" and "lowered the estimate", never
> "causes disease". Model explanation is not medical causation.

## 18. Error analysis

At the deployed threshold of 0.38 on 61 held-out records:

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actually negative** | 23 (true negative) | 10 (false positive) |
| **Actually positive** | 1 (false negative) | 27 (true positive) |

Patterns found:

1. **The model knows when it is unsure.** Mean distance from the threshold is
   **0.199 for wrong predictions versus 0.401 for correct ones**. Errors cluster
   near the decision boundary, which is the behaviour a well-calibrated score
   should show.
2. **The single false negative** — age 60, 0 vessels coloured, ST depression
   3.00 mm (test-set averages 54.0, 0.93, 1.20). A record with a normal
   fluoroscopy result, which is the feature the model leans on hardest; the
   elevated ST depression was not enough to overcome it. Model-estimated
   probability 29.2%, so it was not a confident miss.
3. **The ten false positives** average 0.89 vessels coloured and 0.69 mm ST
   depression — close to the cohort average on both. These are genuinely
   ambiguous records rather than a systematic failure mode.
4. **The threshold explains the shape of the errors.** At 0.50 the same model
   produces 27 TN / 6 FP / 2 FN / 26 TP. Moving to 0.38 converts one false
   negative into four extra false positives. That is the intended direction of
   the trade.

**Why both error types matter.** A false negative sends a patient with
significant arterial narrowing away without follow-up — the harm is delayed
treatment of a condition that can be fatal. A false positive sends a healthy
patient to further testing that is costly, anxiety-inducing and sometimes
invasive. Neither is free. In screening the first is normally judged the more
serious, which is the entire justification for a threshold below 0.5 — but at
0.38 this model would send 30% of healthy patients for unnecessary follow-up,
and that cost is stated rather than glossed over.

## 19. Final model

**Logistic Regression** · `C=0.5`, ridge penalty (`l1_ratio=0.0`),
`class_weight='balanced'`, `solver='liblinear'` · decision threshold **0.38** ·
uncalibrated.

### Selection rule, fixed before the results

1. Rank by mean CV ROC-AUC; keep every model within **one standard error** of
   the best (0.8976 − 0.0086 = **0.8890**). All five qualify.
2. Apply the same one-SE filter to PR-AUC (cut-off **0.8893**). All five
   qualify.
3. Among survivors prefer the **better-calibrated** model, treating Brier
   differences below 0.01 as ties.
4. Break remaining ties on **interpretability tier**, then generalisation gap,
   then fit cost.

| Model | CV ROC-AUC | CV PR-AUC | Brier | Calibration group | Interpretability tier | Train–test gap | Fit (s) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8976 | 0.8938 | 0.1203 | best | 1 | 0.0337 | 0.04 |
| SVM (Linear) | 0.8950 | 0.8935 | 0.1224 | best | 2 | 0.0294 | 0.05 |
| XGBoost | 0.8954 | 0.8966 | 0.1274 | best | 3 | 0.0552 | 0.13 |
| SVM (RBF) | 0.8968 | 0.8932 | 0.1220 | best | 4 | 0.0247 | 0.06 |
| Random Forest | 0.8934 | 0.8935 | 0.1330 | worse | 3 | 0.0823 | 0.72 |

Logistic regression wins on the tie-breakers: the best Brier score (0.1203), the
only model whose reasoning is readable directly as signed coefficients and odds
ratios, the smallest train–test gap among the interpretable models, and the
cheapest to fit and to serve.

**Random forest is instructive by contrast.** It has the best raw test accuracy
(0.885) and would have been selected under an accuracy-maximising rule. It is
eliminated at step 3 for the worst calibration (Brier 0.1330) and carries the
largest generalisation gap (0.0823, roughly 2.4× the selected model's) — it fits
the training folds harder without generalising better. That is precisely the
failure mode "do not optimise for accuracy alone" is meant to prevent.

### Held-out performance (61 records the model never saw)

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

ROC-AUC, PR-AUC and Brier are threshold-independent, so only the classification
decision changes between the columns.

Test ROC-AUC (0.959) exceeds cross-validated ROC-AUC (0.898). This is not
evidence of a better model — it is what a 61-record test split looks like. The
CV figure, computed over 25 fits, is the more trustworthy estimate of how the
model would behave on new Cleveland-like data.

## 20. Model export

`python -m src.train` writes:

```
models/
├── final_model.pkl              full sklearn Pipeline (preprocessing + estimator)
├── preprocessing_pipeline.pkl   the fitted ColumnTransformer on its own
├── local_explainer.pkl          per-prediction attribution
├── metadata.json                model card, hyper-parameters, split, seed,
│                                threshold + rationale, environment, timings
├── metrics.json                 every metric, curve, sweep and study
└── feature_info.json            feature schema, dictionary, label mapping
```

The training run verifies the round trip before finishing: the reloaded model is
scored on the test set and the maximum absolute difference from the original
probabilities is asserted below 1e-9. In the run reported here it was exactly
**0.0**.

## 21. Inference pipeline

```
USER INPUT → VALIDATION → PREPROCESSING → MODEL → PROBABILITY
           → CLASSIFICATION → EXPLANATION
```

Implemented in `src/inference.py`. `validate_record` checks presence, type,
finiteness, numeric range, integrality where the schema demands it, categorical
membership, and rejects unknown fields — raising a `ValidationError` carrying one
message per offending field, phrased with the interface label rather than the
column code. The validated row is passed to the exported pipeline, which imputes,
scales and encodes it exactly as during training, and the estimator returns a
probability. The threshold turns that into a classification, and the explainer
produces the ranked contributions.

`get_predictor()` is a thread-safe process-wide singleton, so the artefacts are
deserialised once at application start-up, never per request. Measured response
time in the running app: **197 ms** on the first request (explainer warm-up),
**~35 ms** on subsequent ones.

The API layers Pydantic (shape, types, unknown fields) over the domain
validation above, and converts every failure into JSON with a message per field.
No Python traceback ever reaches the browser — asserted by a test.

## 22. UI / UX

### Product intent

The interface is presented as *an AI-powered medical risk classification
research tool*, not as a diagnostic service. That framing drives every decision
below.

### Structure

Six sections in one page: **Home · Prediction · How it works · Model ·
Performance · About**, with a sticky header, scroll-spy navigation
(`aria-current`), skip link, and a permanently visible medical disclaimer bar
directly beneath the header — not a footnote.

### Prediction flow

The form is **generated at run time from `GET /api/schema`**, which serves the
same schema object the training pipeline uses. The interface therefore cannot
drift from the model: adding or renaming a feature in `src/config.py` changes
both. Inputs are grouped into Patient Information, Symptoms, Vitals & Blood
Tests, ECG & Exercise Test, and Imaging, with human labels, units and help text
throughout — "Chest Pain Type", never `cp`.

Validation runs on blur and on submit. Failures produce a focusable error
summary (`role="alert"`, `tabindex="-1"`) listing every problem as a link to the
offending field, alongside an inline message bound to the input with
`aria-describedby` and `aria-invalid`. Submitting disables the button, shows
"Analysing patient data…" over a skeleton, and blocks duplicate submissions.

The result card reports classification, model-estimated probability, a
probability meter with the decision threshold marked in place, distance from the
threshold, the model name, response time, and the six most influential inputs
with direction and share of total influence. "Load a real record" pulls an
actual row from the held-out test split, so the demo does not require inventing
plausible clinical values. Reset clears inputs, errors and the previous result.

### Design decisions

- **No red-for-sick, green-for-healthy.** A classification is not a verdict.
  Positive uses a calm orange with a flag icon, negative a blue with a check,
  and colour never carries meaning alone — every state also has an icon and a
  text label.
- **Design skills were consulted, not name-dropped.** The `ui-ux-pro-max` skill
  was queried for a healthcare design system (returning the Swiss/minimal
  direction, a calm cyan palette and the Figtree/Noto Sans pairing used here),
  for accessible form-error patterns (the focusable error summary above comes
  directly from it), and for focus-visibility requirements. Its recommended
  landing pattern — hero, testimonials, CTA — was **rejected**: testimonials
  would mean fabricating social proof for a student project, which conflicts
  with the honesty constraint. The `dataviz` skill supplied the chart palette,
  the mark specifications (2px lines, hairline recessive grids, ≥8px markers),
  and the rule that text never wears a data colour.
- **The chart palette was validated, not eyeballed.** The blue/orange pair was
  run through the data-visualisation palette validator against this project's
  actual surfaces. All six checks pass in both modes — lightness band, chroma
  floor, colour-vision-deficiency separation (worst pair ΔE 9.2 light / 9.4
  dark), normal-vision floor (24.0 / 20.9), and contrast.
- **Contrast was computed for every token pair.** Two failures were found and
  fixed: the muted text token measured 4.31:1 on the secondary surface and was
  darkened to 5.22:1, and the confusion-matrix ramp originally put white text on
  a mid-blue fill at 2.50:1 — the ramp was re-stepped so a single dark ink
  colour clears 4.5:1 on every cell.
- **Accessibility.** Semantic landmarks and headings, keyboard navigable end to
  end, 2px focus rings offset 2px, `scroll-padding-top` so focus is never hidden
  behind the sticky header (WCAG 2.2 *Focus Not Obscured*), 44px+ touch targets
  (48px in the mobile menu), `prefers-reduced-motion` honoured, SVG icons with
  no emoji used as iconography, and every chart carrying a descriptive
  `role="img"` label with the underlying numbers also present as a real table.
- **Responsive.** Verified at 375, 390, 768, 1280 and 1440 px with no horizontal
  overflow at any width. The form collapses to one column, the result panel
  stops being sticky, tables scroll inside their own container, and the
  navigation becomes a disclosure menu below 860 px.
- **Themes.** Light and dark, with an explicit toggle that overrides the OS
  setting in both directions; charts re-render with the theme's own validated
  colour steps.
- **No dependencies.** No UI framework, no chart library. Every chart is
  hand-drawn inline SVG, so nothing can fail to load.

### Integration

```
Browser form
   → POST /api/predict (JSON)
      → Pydantic shape validation
      → src.inference.validate_record   (domain rules)
      → models/final_model.pkl          (preprocessing + estimator, loaded once)
      → probability
      → threshold from models/metadata.json
      → models/local_explainer.pkl      (per-feature attribution)
   → JSON response
→ result card
```

The Performance and Model sections are rendered from `GET /api/metrics` and
`GET /api/model`, which read `models/metrics.json` and `models/metadata.json`.
No metric is written into the front-end source. Deleting the artefacts causes
the interface to display an "artefacts unavailable" state with instructions,
which is also covered by a test.

## 23. Ethical considerations

- **Not a medical device.** The disclaimer appears in the header bar, in the
  About section, and in every API response body. The application never uses the
  phrase "probability of disease" — only "model-estimated probability".
- **Sex is a strong predictor here, and that is a hazard.** `sex` has the
  second-largest coefficient (+0.984, odds ratio 2.68). The cohort is roughly
  two-thirds male, so the model has seen far fewer female patients, and the
  female positive rate in the data (25.8% vs 55.3%) partly reflects referral
  patterns of a 1988 cardiology clinic rather than biology. A model trained on
  this data can reproduce that historical selection. The limitation is stated in
  the interface, not only in this report.
- **Explanation is not causation.** Every explanation surface says so
  explicitly, and the wording is enforced by a test that rejects causal or
  diagnostic language in the explanation note.
- **Threshold choices are value judgements.** Setting sensitivity ≥ 0.85 encodes
  a belief about the relative cost of the two errors. That belief is stated,
  its cost is quantified (specificity 0.697 on test), and the API allows the
  threshold to be overridden.
- **No personal data.** The application stores nothing; predictions are not
  persisted. The example records come from a public research dataset.
- **Provenance.** Dataset source, licence page, SHA-256, random seed and full
  environment are recorded in the exported metadata.

## 24. Limitations

1. **303 records from one hospital in 1988.** Small, old, and unrepresentative
   of contemporary or non-US populations.
2. **~68% male cohort**, so female performance rests on far fewer records — and
   sex is one of the two strongest coefficients.
3. **The label is angiographic narrowing above 50%**, not a clinical diagnosis
   and not an outcome such as myocardial infarction or death.
4. **61 test records.** Every test metric has a wide confidence interval. A
   difference of a few points between models is noise, and this report treats it
   as such.
5. **Five models are statistically tied.** The selected model is not
   demonstrably the most accurate; it is the one a pre-declared rule picks among
   equals.
6. **Tuning is not nested inside the evaluation.** Hyper-parameters were chosen
   by CV on the training split, and the repeated-CV figures reuse that split, so
   they are mildly optimistic. The held-out test split is the unbiased estimate.
7. **Several inputs require a hospital exercise test, fluoroscopy or a thallium
   scan.** This is not, and cannot be, a self-assessment tool.
8. **`restecg = 1` occurs in 4 records.** Its encoded column is fitted on almost
   nothing and should not be interpreted.
9. **No external validation cohort** was used. The Hungarian, Switzerland and
   Long Beach subsets in the same archive were deliberately not merged in, since
   they have far more missing data and different measurement protocols.
10. **Single train/test split.** With repeated CV on the training portion this is
    reasonable, but a repeated hold-out or bootstrap would quantify the test-set
    uncertainty properly.

## 25. Future work

- **External validation** on the Hungarian, Switzerland and Long Beach VA
  subsets — the strongest available test of whether this generalises beyond
  Cleveland, and the natural next experiment.
- **Bootstrap confidence intervals** on every test metric, so uncertainty is
  visible rather than implied.
- **Nested cross-validation**, putting tuning inside the evaluation loop to
  remove the optimism noted in limitation 6.
- **Subgroup reporting** by sex and age band, with sample sizes shown, to make
  the fairness question measurable rather than rhetorical.
- **Interactive threshold control** in the interface, letting a user watch
  sensitivity and specificity trade off live — the API already accepts a
  threshold override.
- **Decision-curve analysis**, which turns the threshold choice into an explicit
  statement about the cost ratio a user is willing to accept.
- **Prediction logging with a model version tag** to enable monitoring, and
  containerisation with a checksum-pinned dataset in CI so a silent upstream
  change is caught.

## 26. Conclusion

A complete classification system was built on the UCI Cleveland heart disease
dataset and delivered as a running application. Four algorithm families were
trained on an identical leakage-controlled pipeline, tuned on the training split
alone, and compared over 25 cross-validation fits each. **Logistic regression
was selected** by a rule fixed before the results were known, reaching **ROC-AUC
0.959, PR-AUC 0.942, sensitivity 0.964 and specificity 0.697** on 61 held-out
records at a deliberately chosen 0.38 threshold — or accuracy 0.869 with
sensitivity 0.929 and specificity 0.818 at the conventional 0.50.

The substantive finding is not the headline number. It is that **five very
different algorithms are statistically indistinguishable on this data**
(0.005 CV ROC-AUC spread against ±0.04 fold-to-fold variation), so the
interesting choices lie elsewhere: in preventing leakage, in choosing the
operating threshold to match the cost of a missed case, in checking that the
probabilities mean something, in explaining individual predictions without
implying causation, and in reporting the result honestly enough that a reader
can see exactly how much confidence 61 test records support.

The model is exported, verified to reload identically, served through a
validated API, and driven by an accessible, responsive interface in which every
displayed number originates from the training run — not from the front-end
source.

---

**Reproducibility.** The full pipeline was executed twice. Every exported value
was bit-identical between runs apart from wall-clock timings, so the seeds do
what they claim. Dataset: UCI ML Repository dataset 45,
`processed.cleveland.data`, SHA-256 recorded in `data/raw/SOURCE.json`. Seed 42.
Split: stratified 80/20, 242 / 61. CV: `RepeatedStratifiedKFold(5 × 5)`.
Environment: Python 3.14.7, scikit-learn 1.9.0, XGBoost 3.4.1, SHAP 0.52.0,
NumPy 2.5.2, pandas 3.0.5, Windows 11. Reproduce with `python -m src.train`.

**Medical disclaimer.** Educational use only. This project demonstrates machine
learning classification on a public dataset. Model predictions are not medical
diagnoses, medical advice, or a substitute for evaluation by a qualified
healthcare professional.
