"""Use the trained model on one patient: validation, risk estimate and explanation."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config as C
from . import explain


def load_metadata() -> dict:
    """Training summary written by src.train (best model name, thresholds, metrics ...)."""
    with open(C.METRICS_PATH) as f:
        return json.load(f)


def load_model():
    """Load the saved best model; refit it quickly if the file cannot be read on this computer."""
    try:
        return joblib.load(C.BEST_MODEL_PATH)
    except Exception:  # e.g. different scikit-learn version, or the file is missing
        from .train import rebuild_best_model
        return rebuild_best_model(save=True)


@lru_cache(maxsize=1)
def _background() -> pd.DataFrame:
    """A fixed sample of training rows used as the reference for explanations."""
    train = pd.read_csv(C.PROCESSED_DIR / "train.csv")
    return train[C.FEATURES].sample(200, random_state=C.RANDOM_STATE).reset_index(drop=True)


def validate_input(patient: dict) -> dict:
    """Check that all six values are present, numeric and inside the training range."""
    problems, clean = [], {}
    for col in C.FEATURES:
        if col not in patient or patient[col] is None:
            problems.append(f"{col} is missing")
            continue
        try:
            value = float(patient[col])
        except (TypeError, ValueError):
            problems.append(f"{col} must be a number")
            continue
        lo, hi = C.VALID_RANGES[col]
        if not lo <= value <= hi:
            problems.append(f"{col} = {value:g} is outside the supported range {lo}-{hi}")
        clean[col] = value
    if "smoking" in clean and clean["smoking"] not in (0.0, 1.0):
        problems.append("smoking must be 0 or 1")
    if problems:
        raise ValueError("; ".join(problems))
    return clean


def risk_band(probability: float, screening_threshold: float) -> str:
    """Lower / Elevated / High, using cut-offs that come from the training data."""
    cut = min(screening_threshold, 0.5)
    if probability < cut:
        return "Lower risk"
    if probability < 0.5:
        return "Elevated risk"
    return "High risk"


def predict_patient(patient: dict, model=None, meta: dict | None = None) -> dict:
    """Return the predicted disease probability and a simple risk band for one patient."""
    clean = validate_input(patient)
    model = model or load_model()
    meta = meta or load_metadata()
    X = pd.DataFrame([clean])[C.FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    t = float(meta["screening_threshold"])
    return {
        "probability": prob,
        "predicted_label": int(prob >= 0.5),
        "screening_flag": bool(prob >= t),
        "screening_threshold": t,
        "risk_band": risk_band(prob, t),
        "model": meta["best_model"],
    }


def explain_patient(patient: dict, model=None) -> pd.DataFrame:
    """Per-feature contribution to this patient's risk, in percentage points.

    The table's `contribution` column plus the `base` value (the average predicted risk of the
    reference patients) adds up exactly to the predicted probability.
    """
    clean = validate_input(patient)
    model = model or load_model()
    X = pd.DataFrame([clean])[C.FEATURES]
    ex = explain.probability_explanation(model, X, _background())
    out = pd.DataFrame({
        "feature": [C.FEATURE_LABELS[f] for f in C.FEATURES],
        "value": [clean[f] for f in C.FEATURES],
        "contribution_pp": ex.values[0] * 100,
    })
    out.attrs["base_probability"] = float(ex.base_values[0])
    return out.reindex(out["contribution_pp"].abs().sort_values(ascending=False).index).reset_index(drop=True)
