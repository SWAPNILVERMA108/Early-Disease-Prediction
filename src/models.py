"""Model definitions and hyper-parameter search spaces.

Every model is a scikit-learn Pipeline whose last step is called "clf".
Because scaling happens INSIDE the pipeline, it is learned from the training part of
each cross-validation fold only, so no information leaks from validation or test rows.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


def build_estimator(name: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Return an un-fitted pipeline for the given model name.

    Class imbalance is handled inside the models (class weights), never by copying rows,
    so the test set always keeps its natural class ratio.
    """
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                       random_state=C.RANDOM_STATE)),
        ])
    if name == "Random Forest":
        return Pipeline([
            ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                           random_state=C.RANDOM_STATE)),
        ])
    if name == "XGBoost":
        return Pipeline([
            ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  tree_method="hist", n_jobs=1, random_state=C.RANDOM_STATE,
                                  scale_pos_weight=scale_pos_weight)),
        ])
    raise ValueError(f"Unknown model: {name}")


SEARCH_SPACES = {
    "Logistic Regression": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
    "Random Forest": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [3, 4, 5, 6, 8, 12],
        "clf__min_samples_leaf": [3, 5, 10, 20, 40],
        "clf__max_features": ["sqrt", 0.5, 1.0],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [1, 5, 10],
    },
}

SCORING = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall",
           "precision": "precision", "accuracy": "accuracy"}


def make_search(name: str, estimator: Pipeline, cv):
    """Grid search for the small Logistic Regression space, random search for the others."""
    common = dict(scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1, return_train_score=False)
    if name == "Logistic Regression":
        return GridSearchCV(estimator, SEARCH_SPACES[name], **common)
    return RandomizedSearchCV(estimator, SEARCH_SPACES[name], n_iter=C.SEARCH_ITERATIONS,
                              random_state=C.RANDOM_STATE, **common)
