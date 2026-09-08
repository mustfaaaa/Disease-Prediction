"""Phase 1 - evidence-based dataset selection.

Downloads/loads each candidate dataset and prints REAL statistics so the
choice of primary dataset can be justified with numbers, not assumptions.
"""
import io, zipfile, urllib.request, sys
import numpy as np
import pandas as pd

def survey(name, df, target, source):
    print("=" * 70)
    print(f"CANDIDATE: {name}")
    print(f"  source        : {source}")
    print(f"  shape         : {df.shape[0]} rows x {df.shape[1]} cols "
          f"({df.shape[1]-1} predictors)")
    y = df[target]
    vc = y.value_counts().sort_index()
    print(f"  target        : '{target}' -> {dict(vc)}")
    print(f"  class balance : {dict((vc/len(y)*100).round(1))} %  "
          f"| minority = {vc.min()/len(y)*100:.1f}%")
    print(f"  duplicates    : {int(df.duplicated().sum())}")
    miss = df.isna().sum()
    print(f"  missing cells : {int(miss.sum())} across {int((miss>0).sum())} columns")
    nuniq = df.drop(columns=[target]).nunique()
    cat_like = int((nuniq <= 10).sum())
    print(f"  feature mix   : {cat_like} low-cardinality (<=10 uniq, categorical-like), "
          f"{len(nuniq)-cat_like} continuous")
    print()

# ---------------------------------------------------------------- Heart (UCI 45)
z = urllib.request.urlopen(
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip", timeout=60).read()
zf = zipfile.ZipFile(io.BytesIO(z))
print("files in UCI heart+disease.zip:", zf.namelist())
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","num"]
heart = pd.read_csv(io.BytesIO(zf.read("processed.cleveland.data")),
                    names=cols, na_values="?")
heart["target"] = (heart["num"] > 0).astype(int)
heart = heart.drop(columns=["num"])
survey("UCI Heart Disease (Cleveland)", heart, "target",
       "archive.ics.uci.edu/dataset/45")
print("  per-column missing:", {k: int(v) for k, v in heart.isna().sum().items() if v})
print("  dtypes:", dict(heart.dtypes.astype(str)))
print()

# ------------------------------------------------------- Breast Cancer (sklearn)
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
bcd = bc.frame.rename(columns={"target": "target"})
survey("Breast Cancer Wisconsin (Diagnostic)", bcd, "target",
       "sklearn.datasets.load_breast_cancer / UCI 17")
print("  first 6 feature names:", list(bc.feature_names[:6]))
print()

# --------------------------------------------------------------- Pima Diabetes
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    raw = urllib.request.urlopen(url, timeout=60).read()
    pcols = ["pregnancies","glucose","blood_pressure","skin_thickness",
             "insulin","bmi","diabetes_pedigree","age","target"]
    pima = pd.read_csv(io.BytesIO(raw), names=pcols)
    survey("Pima Indians Diabetes", pima, "target", url)
    zeros = (pima[["glucose","blood_pressure","skin_thickness","insulin","bmi"]] == 0).sum()
    print("  biologically impossible zeros (hidden missing):", dict(zeros))
except Exception as e:
    print("Pima download failed:", type(e).__name__, e)
