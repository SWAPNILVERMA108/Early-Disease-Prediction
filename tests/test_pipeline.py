"""Checks on the model pipelines and the evaluation helpers."""
import numpy as np
import pytest

from src import config as C
from src import data_utils as D
from src import evaluate as E
from src.models import MODEL_NAMES, build_estimator


@pytest.fixture(scope="module")
def splits():
    df = D.load_data()
    train, test = D.split_data(df)
    return train, test


def test_scaler_learns_from_training_rows_only(splits):
    train, _ = splits
    X_train, y_train = D.xy(train)
    model = build_estimator("Logistic Regression").fit(X_train, y_train)
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy())     # no test / validation rows involved


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_gives_valid_probabilities(name, splits):
    train, test = splits
    X_train, y_train = D.xy(train.head(600))
    X_test, y_test = D.xy(test.head(200))
    model = build_estimator(name).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (200, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_models_beat_chance_on_the_full_training_data(splits):
    train, test = splits
    X_train, y_train = D.xy(train)
    X_test, y_test = D.xy(test)
    proba = build_estimator("Logistic Regression").fit(X_train, y_train).predict_proba(X_test)[:, 1]
    assert E.roc_auc_score(y_test, proba) > 0.75


def test_compute_metrics_on_perfect_and_worst_predictions():
    y = np.array([0, 0, 1, 1])
    perfect = E.compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["accuracy"] == perfect["roc_auc"] == perfect["recall"] == perfect["specificity"] == 1.0
    worst = E.compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert worst["accuracy"] == 0.0 and worst["roc_auc"] == 0.0
    assert (perfect["tn"], perfect["fp"], perfect["fn"], perfect["tp"]) == (2, 0, 0, 2)


def test_threshold_for_recall_really_reaches_the_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    t = E.choose_threshold_for_recall(y, proba, target_recall=0.85)
    recall = (proba[y == 1] >= t).mean()
    assert recall >= 0.85
    # and it is the HIGHEST such cut-off: a slightly higher one drops below the target
    assert (proba[y == 1] >= t + 0.01).mean() < 0.85 + 0.02


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.3 * y + rng.normal(0.35, 0.2, 500), 0, 1)
    lo, hi = E.bootstrap_ci(y, proba, n_boot=300)
    assert lo < E.roc_auc_score(y, proba) < hi
