"""SHAP explanations for the three model types.

Two views are offered:
* global_explanation(): the standard fast SHAP explainer for each model type
  (Linear for Logistic Regression, Tree for Random Forest / XGBoost). Used for the
  model-wide plots in the notebooks.
* probability_explanation(): an exact, model-agnostic SHAP calculation in PROBABILITY units,
  so a single patient's result reads as "+8 percentage points of risk". Used by the app.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from . import config as C

NAMES = [C.FEATURE_LABELS[f] for f in C.FEATURES]


def global_explanation(model, X: pd.DataFrame, background: pd.DataFrame | None = None):
    """SHAP values for every row of X, using the model's native explainer.

    Returns (shap.Explanation, output_space) where output_space is 'log-odds' or 'probability'.
    The Explanation keeps the REAL feature values in `.data`, so plots show age in years etc.
    """
    clf = model.named_steps["clf"]
    Xv = X[C.FEATURES]
    bg = (Xv if background is None else background)[C.FEATURES]
    bg = bg.sample(min(100, len(bg)), random_state=C.RANDOM_STATE)   # reference rows for SHAP

    if isinstance(clf, LogisticRegression):
        scaler = model.named_steps["scaler"]
        explainer = shap.LinearExplainer(clf, scaler.transform(bg))
        values = np.asarray(explainer.shap_values(scaler.transform(Xv)))
        base, space = float(np.ravel(explainer.expected_value)[0]), "log-odds"
    elif isinstance(clf, XGBClassifier):
        explainer = shap.TreeExplainer(clf)
        values = np.asarray(explainer.shap_values(Xv.values))
        base, space = float(np.ravel(explainer.expected_value)[0]), "log-odds"
    elif isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Xv.values)
        values = np.asarray(values[1] if isinstance(values, list) else values)
        if values.ndim == 3:                       # (rows, features, classes) in newer SHAP
            values = values[:, :, 1]
        ev = np.ravel(explainer.expected_value)
        base, space = float(ev[1] if len(ev) > 1 else ev[0]), "probability"
    else:
        raise TypeError(f"No SHAP explainer set up for {type(clf).__name__}")

    expl = shap.Explanation(values=values, base_values=np.full(len(Xv), base),
                            data=Xv.values, feature_names=NAMES)
    return expl, space


def probability_explanation(model, X: pd.DataFrame, background: pd.DataFrame,
                            n_background: int = 100, seed: int = C.RANDOM_STATE):
    """Exact SHAP values in probability units for any fitted model (6 features = fast).

    base_value + sum(values) equals the model's predicted probability for each row.
    """
    bg = background[C.FEATURES].sample(min(n_background, len(background)), random_state=seed)

    def predict_pos(data):
        return model.predict_proba(pd.DataFrame(data, columns=C.FEATURES))[:, 1]

    masker = shap.maskers.Independent(bg.values, max_samples=len(bg))
    explainer = shap.explainers.Exact(predict_pos, masker, feature_names=NAMES)
    return explainer(X[C.FEATURES].values.astype(float))


def mean_abs_shap(expl: shap.Explanation) -> pd.Series:
    """Average absolute SHAP value per feature (global importance), largest first."""
    return pd.Series(np.abs(expl.values).mean(axis=0), index=NAMES).sort_values(ascending=False)


# ---- plots -------------------------------------------------------------------
def plot_beeswarm(expl, title: str):
    plt.figure(figsize=(7, 4))
    shap.plots.beeswarm(expl, show=False, max_display=6)
    plt.title(title, fontweight="bold", fontsize=12)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_bar(expl, title: str):
    plt.figure(figsize=(6.5, 3.6))
    shap.plots.bar(expl, show=False, max_display=6)
    plt.title(title, fontweight="bold", fontsize=12)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_waterfall(single_expl, title: str):
    plt.figure(figsize=(7, 4))
    shap.plots.waterfall(single_expl, show=False, max_display=7)
    plt.title(title, fontweight="bold", fontsize=11)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_dependence(expl, feature_label: str, title: str):
    plt.figure(figsize=(5.8, 4))
    shap.plots.scatter(expl[:, feature_label], show=False, color=expl[:, "Age (years)"])
    plt.title(title, fontweight="bold", fontsize=12)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
