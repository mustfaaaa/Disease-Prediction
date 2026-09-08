"""Single source of truth: paths, seeds, and the dataset feature schema.

Everything downstream (training, evaluation, inference, the API, and the UI
form) reads its notion of "what a patient record looks like" from FEATURES
below, so the UI can never drift away from what the model was trained on.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
PREDICTIONS = OUTPUTS / "predictions"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES, METRICS, PREDICTIONS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- experiment
RANDOM_SEED = 42
TEST_SIZE = 0.20          # held out, untouched until final evaluation
CV_FOLDS = 5
CV_REPEATS = 5            # repeated CV: n=303 makes a single 5-fold split noisy

# ------------------------------------------------------------------ dataset
DATASET_NAME = "UCI Heart Disease (Cleveland)"
DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
DATASET_FILE = "processed.cleveland.data"
RAW_CSV = DATA_RAW / "processed_cleveland.csv"
CLEAN_CSV = DATA_PROCESSED / "heart_clean.csv"

TARGET = "target"
TARGET_LABELS = {0: "Negative", 1: "Positive"}
TARGET_DESCRIPTION = (
    "Angiographic heart disease status. 0 = less than 50% diameter narrowing "
    "in all major vessels (Negative). 1 = more than 50% narrowing in at least "
    "one major vessel (Positive). Derived from the original 'num' column by "
    "binarising values 1-4 to 1."
)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# ------------------------------------------------------------------- schema
# role: "numeric"  -> median imputation + standard scaling
#       "binary"   -> mode imputation, passed through as 0/1
#       "nominal"  -> mode imputation + one-hot encoding
FEATURES: list[dict] = [
    {
        "name": "age", "role": "numeric", "group": "Patient Information",
        "label": "Age", "unit": "years", "min": 18, "max": 100, "step": 1,
        "description": "Patient age in whole years.",
        "help": "Range observed in this dataset: 29-77.",
    },
    {
        "name": "sex", "role": "binary", "group": "Patient Information",
        "label": "Sex", "unit": None,
        "choices": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}],
        "description": "Biological sex as recorded in the dataset (1 = male, 0 = female).",
        "help": "The Cleveland cohort is roughly two-thirds male - see limitations.",
    },
    {
        "name": "cp", "role": "nominal", "group": "Symptoms",
        "label": "Chest Pain Type", "unit": None,
        "choices": [
            {"value": 1, "label": "Typical angina"},
            {"value": 2, "label": "Atypical angina"},
            {"value": 3, "label": "Non-anginal pain"},
            {"value": 4, "label": "Asymptomatic"},
        ],
        "description": "Category of chest pain reported at admission.",
        "help": "'Asymptomatic' means no chest pain was reported.",
    },
    {
        "name": "exang", "role": "binary", "group": "Symptoms",
        "label": "Exercise-Induced Angina", "unit": None,
        "choices": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}],
        "description": "Whether chest pain was provoked by the exercise test.",
        "help": None,
    },
    {
        "name": "trestbps", "role": "numeric", "group": "Vitals & Blood Tests",
        "label": "Resting Blood Pressure", "unit": "mm Hg",
        "min": 80, "max": 220, "step": 1,
        "description": "Resting systolic blood pressure on admission to hospital.",
        "help": "Range observed in this dataset: 94-200.",
    },
    {
        "name": "chol", "role": "numeric", "group": "Vitals & Blood Tests",
        "label": "Serum Cholesterol", "unit": "mg/dl",
        "min": 100, "max": 600, "step": 1,
        "description": "Total serum cholesterol measured from a blood sample.",
        "help": "Range observed in this dataset: 126-564.",
    },
    {
        "name": "fbs", "role": "binary", "group": "Vitals & Blood Tests",
        "label": "Fasting Blood Sugar above 120 mg/dl", "unit": None,
        "choices": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}],
        "description": "Whether fasting blood sugar exceeded 120 mg/dl.",
        "help": None,
    },
    {
        "name": "restecg", "role": "nominal", "group": "ECG & Exercise Test",
        "label": "Resting ECG Result", "unit": None,
        "choices": [
            {"value": 0, "label": "Normal"},
            {"value": 1, "label": "ST-T wave abnormality"},
            {"value": 2, "label": "Left ventricular hypertrophy"},
        ],
        "description": "Resting electrocardiographic findings.",
        "help": None,
    },
    {
        "name": "thalach", "role": "numeric", "group": "ECG & Exercise Test",
        "label": "Maximum Heart Rate Achieved", "unit": "bpm",
        "min": 60, "max": 220, "step": 1,
        "description": "Highest heart rate reached during the exercise test.",
        "help": "Range observed in this dataset: 71-202.",
    },
    {
        "name": "oldpeak", "role": "numeric", "group": "ECG & Exercise Test",
        "label": "ST Depression (Exercise vs Rest)", "unit": "mm",
        "min": 0, "max": 7, "step": 0.1,
        "description": "ST-segment depression induced by exercise, relative to rest.",
        "help": "Recorded to one decimal place. Range observed: 0.0-6.2.",
    },
    {
        "name": "slope", "role": "nominal", "group": "ECG & Exercise Test",
        "label": "Peak Exercise ST Segment Slope", "unit": None,
        "choices": [
            {"value": 1, "label": "Upsloping"},
            {"value": 2, "label": "Flat"},
            {"value": 3, "label": "Downsloping"},
        ],
        "description": "Slope of the ST segment at peak exercise.",
        "help": None,
    },
    {
        "name": "ca", "role": "numeric", "group": "Imaging",
        "label": "Major Vessels Coloured by Fluoroscopy", "unit": "count",
        "min": 0, "max": 3, "step": 1,
        "description": "Number of major vessels (0-3) made visible by fluoroscopy.",
        "help": "Treated as an ordered count: 0 < 1 < 2 < 3.",
    },
    {
        "name": "thal", "role": "nominal", "group": "Imaging",
        "label": "Thallium Stress Test Result", "unit": None,
        "choices": [
            {"value": 3, "label": "Normal"},
            {"value": 6, "label": "Fixed defect"},
            {"value": 7, "label": "Reversible defect"},
        ],
        "description": "Result of the thallium myocardial perfusion scan.",
        "help": None,
    },
]

FEATURE_ORDER = [f["name"] for f in FEATURES]
NUMERIC_FEATURES = [f["name"] for f in FEATURES if f["role"] == "numeric"]
BINARY_FEATURES = [f["name"] for f in FEATURES if f["role"] == "binary"]
NOMINAL_FEATURES = [f["name"] for f in FEATURES if f["role"] == "nominal"]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}

GROUP_ORDER = [
    "Patient Information", "Symptoms", "Vitals & Blood Tests",
    "ECG & Exercise Test", "Imaging",
]


def friendly_name(raw: str) -> str:
    """Map an encoded model-matrix column back to a human-readable label.

    Examples: "num__age" -> "Age";
              "nom__cp_4.0" -> "Chest Pain Type: Asymptomatic".
    """
    token = raw.split("__")[-1]
    if token in FEATURE_BY_NAME:
        return FEATURE_BY_NAME[token]["label"]
    for feat in FEATURES:
        prefix = feat["name"] + "_"
        if token.startswith(prefix):
            value = token[len(prefix):]
            for choice in feat.get("choices", []):
                if value in (str(choice["value"]), str(float(choice["value"]))):
                    return f"{feat['label']}: {choice['label']}"
            return f"{feat['label']} = {value}"
    return raw
