"""Train, compare and save the three models.

Run from the project folder:    python -m src.train
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

from . import config as C
from . import data_utils as D
from . import evaluate as E
from . import viz
from .models import MODEL_NAMES, build_estimator, make_search

SLUG = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest",
        "XGBoost": "xgboost"}


def _py(obj):
    """Make numpy values JSON-friendly."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


def _clean_params(params: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _class_weight_ratio(y) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def original_only_diagnostic() -> dict:
    """How well can models learn from the 500 ORIGINAL rows alone?

    Cross-validated AUC near 0.5 means the original labels carry no learnable signal.
    """
    df = D.clean_data(D.load_data())
    orig = df[D.source_of(df) == "original"]
    X, y = D.xy(orig)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=C.RANDOM_STATE)
    out = {}
    for name in ["Logistic Regression", "Random Forest"]:
        est = build_estimator(name)
        if name == "Random Forest":
            est.set_params(clf__n_estimators=300, clf__max_depth=5, clf__min_samples_leaf=10)
        scores = cross_val_score(est, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        out[name] = {"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std())}
    out["n_rows"] = int(len(orig))
    return out


def run_training(save: bool = True, verbose: bool = True) -> dict:
    """Full experiment: split, tune 3 models with 5-fold CV, test once, save everything."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    viz.set_style()

    df = D.clean_data(D.load_data())
    validation = D.validate_data(df)
    assert validation["passed"], validation["issues"]

    train_df, test_df = D.split_data(df)
    X_train, y_train = D.xy(train_df)
    X_test, y_test = D.xy(test_df)
    log(f"Train {X_train.shape}, test {X_test.shape}; "
        f"disease share train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    # ---- baseline: always predict the majority class ----------------------------
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline = E.compute_metrics(y_test, dummy.predict_proba(X_test)[:, 1], 0.5)

    # ---- tune and evaluate each model -------------------------------------------
    fitted, results, proba_test = {}, {}, {}
    spw = _class_weight_ratio(y_train)
    for name in MODEL_NAMES:
        search = make_search(name, build_estimator(name, scale_pos_weight=spw), cv)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        i = search.best_index_
        cvr = search.cv_results_
        proba = best.predict_proba(X_test)[:, 1]
        fitted[name], proba_test[name] = best, proba
        results[name] = {
            "best_params": _clean_params(search.best_params_),
            "cv_roc_auc_mean": float(cvr["mean_test_roc_auc"][i]),
            "cv_roc_auc_std": float(cvr["std_test_roc_auc"][i]),
            "cv_f1": float(cvr["mean_test_f1"][i]),
            "cv_recall": float(cvr["mean_test_recall"][i]),
            "cv_precision": float(cvr["mean_test_precision"][i]),
            "cv_accuracy": float(cvr["mean_test_accuracy"][i]),
            "test": E.compute_metrics(y_test, proba, 0.5),
        }
        log(f"{name:20s} CV AUC {results[name]['cv_roc_auc_mean']:.3f} "
            f"(+/- {results[name]['cv_roc_auc_std']:.3f})  test AUC {results[name]['test']['roc_auc']:.3f}")

    # ---- choose the final model using CROSS-VALIDATION only, not the test set ----
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best_model = fitted[best_name]
    log(f"Selected model: {best_name}")

    # ---- screening threshold from out-of-fold training predictions ---------------
    oof = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = E.choose_threshold_for_recall(y_train, oof, C.TARGET_RECALL)
    screening = E.compute_metrics(y_test, proba_test[best_name], threshold)
    ci_lo, ci_hi = E.bootstrap_ci(y_test, proba_test[best_name])

    # ---- diagnostics on the two data sources -------------------------------------
    src_test = D.source_of(test_df)
    by_source = {}
    for s in ["original", "synthetic"]:
        mask = (src_test == s).to_numpy()
        by_source[s] = {"n_test_rows": int(mask.sum()),
                        "roc_auc": float(E.roc_auc_score(y_test[mask], proba_test[best_name][mask]))}
    diagnostics = {"original_rows_only_cv": original_only_diagnostic(),
                   "best_model_test_auc_by_source": by_source}

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "shap": shap.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "settings": {"random_state": C.RANDOM_STATE, "test_size": C.TEST_SIZE,
                     "cv_folds": C.CV_FOLDS, "search_iterations": C.SEARCH_ITERATIONS,
                     "target_recall": C.TARGET_RECALL},
        "data": {"n_rows": len(df), "n_train": len(train_df), "n_test": len(test_df),
                 "disease_share_train": float(y_train.mean()),
                 "disease_share_test": float(y_test.mean()), "validation": validation},
        "baseline_majority_class": baseline,
        "models": results,
        "best_model": best_name,
        "best_params": results[best_name]["best_params"],
        "screening_threshold": threshold,
        "best_model_test_at_default_threshold": results[best_name]["test"],
        "best_model_test_at_screening_threshold": screening,
        "best_model_test_auc_ci95": [ci_lo, ci_hi],
        "diagnostics": diagnostics,
    }

    if save:
        C.MODELS_DIR.mkdir(exist_ok=True)
        C.REPORTS_DIR.mkdir(exist_ok=True)
        for name, m in fitted.items():
            joblib.dump(m, C.MODELS_DIR / f"{SLUG[name]}.joblib")
        joblib.dump(best_model, C.BEST_MODEL_PATH)
        D.save_processed_splits(train_df, test_df)
        with open(C.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2, default=_py)
        rows = []
        for name, r in results.items():
            rows.append({"model": name, "cv_roc_auc": r["cv_roc_auc_mean"], "cv_roc_auc_std": r["cv_roc_auc_std"],
                         "cv_f1": r["cv_f1"], "cv_recall": r["cv_recall"],
                         **{f"test_{k}": v for k, v in r["test"].items()
                            if k in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]}})
        pd.DataFrame(rows).round(4).to_csv(C.COMPARISON_PATH, index=False)

        # figures used by the notebooks, the app and the report
        import matplotlib.pyplot as plt
        viz.save_fig(E.plot_model_comparison(results), "model_comparison")
        viz.save_fig(E.plot_roc(y_test, proba_test), "roc_curves")
        viz.save_fig(E.plot_pr(y_test, proba_test), "pr_curves")
        viz.save_fig(E.plot_calibration(y_test, proba_test), "calibration")
        viz.save_fig(E.plot_confusion(results[best_name]["test"], f"{best_name}\nthreshold 0.50"), "confusion_default")
        viz.save_fig(E.plot_confusion(screening, f"{best_name}\nscreening threshold {threshold:.2f}"), "confusion_screening")
        viz.save_fig(E.plot_threshold_tradeoff(y_train, oof, threshold), "threshold_tradeoff")
        plt.close("all")
        log(f"Saved models, metrics and figures ({metrics['seconds']} s)")

    return {"models": fitted, "best_name": best_name, "train_df": train_df, "test_df": test_df,
            "results": results, "threshold": threshold, "proba_test": proba_test,
            "oof_best": oof, "metrics": metrics}


def rebuild_best_model(save: bool = True):
    """Refit the selected model with its saved settings (fast, a few seconds).

    Used automatically when the saved .joblib file cannot be loaded, for example because
    scikit-learn / XGBoost on another computer is a different version.
    """
    with open(C.METRICS_PATH) as f:
        meta = json.load(f)
    df = D.clean_data(D.load_data())
    train_df, _ = D.split_data(df)
    X_train, y_train = D.xy(train_df)
    model = build_estimator(meta["best_model"], scale_pos_weight=_class_weight_ratio(y_train))
    model.set_params(**meta["best_params"])
    model.fit(X_train, y_train)
    if save:
        C.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, C.BEST_MODEL_PATH)
    return model


if __name__ == "__main__":
    run_training()
