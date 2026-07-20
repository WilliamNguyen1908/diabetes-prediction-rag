"""Standalone prediction: reproduces the notebook's preprocess_patient / predict_patient.

Loads the XGBoost model (diabetes_xgb.json) and the preprocessing bundle
(preprocess_bundle.joblib) so the FastAPI app uses the exact same transforms as
training. Keep this the single source of truth for turning a raw patient dict into
a prediction.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

_DIR = Path(__file__).parent
_BUNDLE = joblib.load(_DIR / "preprocess_bundle.joblib")

SCALER = _BUNDLE["scaler"]
NUM_COLS = _BUNDLE["num_cols"]
FEATURE_COLS = _BUNDLE["feature_cols"]
CATS = _BUNDLE["cats"]
STAGE_MAP = _BUNDLE["stage_map"]
NUM_DEFAULTS = _BUNDLE["num_defaults"]
BINARY = _BUNDLE["binary"]
CHOICES = _BUNDLE["choices"]
INV_MAP = {v: k for k, v in STAGE_MAP.items()}

_MODEL = xgb.XGBClassifier()
_MODEL.load_model(str(_DIR / "diabetes_xgb.json"))


def preprocess_patient(raw: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw])
    row["gender"] = (row["gender"] == "Male").astype(int)   # match training
    row = pd.get_dummies(row, columns=CATS)                  # expand categoricals
    row = row.reindex(columns=FEATURE_COLS, fill_value=0)    # add missing dummies, fix order
    row[NUM_COLS] = row[NUM_COLS].astype(float)
    row[NUM_COLS] = SCALER.transform(row[NUM_COLS])          # same fitted scaler
    return row


def predict_patient(raw: dict):
    row = preprocess_patient(raw)
    pred = int(_MODEL.predict(row)[0])
    proba = {
        INV_MAP[int(c)]: float(round(p, 3))
        for c, p in zip(_MODEL.classes_, _MODEL.predict_proba(row)[0])
    }
    return INV_MAP[pred], proba


if __name__ == "__main__":
    sample = dict(NUM_DEFAULTS)
    sample.update({b: 0 for b in BINARY})
    sample.update({k: opts[0] for k, opts in CHOICES.items()})
    label, proba = predict_patient(sample)
    print("raw fields:", len(sample))
    print("predicted:", label)
    print("proba:", proba)
