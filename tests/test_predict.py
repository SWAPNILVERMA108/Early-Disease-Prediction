"""Checks on the saved model, the prediction function and the explanations."""
import pandas as pd
import pytest

from src import config as C
from src import data_utils as D
from src import evaluate as E
from src import predict
from src.train import rebuild_best_model

TYPICAL = {"age": 50, "bmi": 28.0, "blood_pressure": 100, "glucose": 134, "cholesterol": 215, "smoking": 0}


@pytest.fixture(scope="module")
def model():
    return predict.load_model()


@pytest.fixture(scope="module")
def meta():
    return predict.load_metadata()


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(C.PROCESSED_DIR / "test.csv")


def test_artifacts_exist():
    for path in [C.BEST_MODEL_PATH, C.METRICS_PATH, C.COMPARISON_PATH,
                 C.PROCESSED_DIR / "train.csv", C.PROCESSED_DIR / "test.csv"]:
        assert path.exists(), path


def test_saved_model_reproduces_the_reported_test_auc(model, meta, test_df):
    X, y = D.xy(test_df)
    auc = E.roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert auc == pytest.approx(meta["models"][meta["best_model"]]["test"]["roc_auc"], abs=1e-9)


def test_rebuilt_model_matches_the_saved_one(model, test_df):
    """If the .joblib file cannot be loaded, predict.load_model refits from the saved settings."""
    rebuilt = rebuild_best_model(save=False)
    X, _ = D.xy(test_df)
    diff = abs(rebuilt.predict_proba(X)[:, 1] - model.predict_proba(X)[:, 1]).max()
    assert diff < 1e-6


def test_prediction_output(model, meta):
    out = predict.predict_patient(TYPICAL, model=model, meta=meta)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["predicted_label"] == int(out["probability"] >= 0.5)
    assert out["screening_flag"] == (out["probability"] >= meta["screening_threshold"])
    assert out["risk_band"] in {"Lower risk", "Elevated risk", "High risk"}


def test_higher_glucose_gives_higher_risk(model, meta):
    low = predict.predict_patient({**TYPICAL, "glucose": 70}, model=model, meta=meta)["probability"]
    high = predict.predict_patient({**TYPICAL, "glucose": 200}, model=model, meta=meta)["probability"]
    assert high > low + 0.3


@pytest.mark.parametrize("bad", [
    {"glucose": 500},            # above the trained range
    {"age": 10},                 # below the trained range
    {"smoking": 2},              # not 0 / 1
    {"bmi": "abc"},              # not a number
    {"cholesterol": None},       # missing value
])
def test_bad_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        predict.validate_input({**TYPICAL, **bad})


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in TYPICAL.items() if k != "age"}
    with pytest.raises(ValueError, match="age is missing"):
        predict.validate_input(incomplete)


def test_risk_band_cut_offs():
    assert predict.risk_band(0.10, 0.39) == "Lower risk"
    assert predict.risk_band(0.45, 0.39) == "Elevated risk"
    assert predict.risk_band(0.50, 0.39) == "High risk"


def test_explanation_adds_up_to_the_prediction(model, meta):
    patient = {"age": 62, "bmi": 33.0, "blood_pressure": 128, "glucose": 171, "cholesterol": 244, "smoking": 1}
    prob = predict.predict_patient(patient, model=model, meta=meta)["probability"]
    contrib = predict.explain_patient(patient, model=model)
    total = contrib.attrs["base_probability"] + contrib["contribution_pp"].sum() / 100
    assert total == pytest.approx(prob, abs=1e-6)
    assert contrib.loc[0, "feature"] == C.FEATURE_LABELS["glucose"]    # glucose is the biggest driver here
    assert len(contrib) == len(C.FEATURES)
