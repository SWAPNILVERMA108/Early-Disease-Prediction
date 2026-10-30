"""Metrics and evaluation plots."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config as C
from . import viz


# ---- numbers -----------------------------------------------------------------
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All headline metrics for one set of predicted probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),       # sensitivity
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold_for_recall(y_true, proba, target_recall: float = C.TARGET_RECALL) -> float:
    """Highest probability cut-off that still catches at least `target_recall` of the sick patients.

    Meant to be used with out-of-fold training predictions, never with the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds[0])
    return float(round(thresholds[ok[-1]], 4))


def bootstrap_ci(y_true, proba, metric_fn=roc_auc_score, n_boot: int = 1000,
                 seed: int = C.RANDOM_STATE, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for a metric (default: ROC-AUC)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true), np.asarray(proba)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---- plots -------------------------------------------------------------------
def plot_roc(y_true, proba_by_model: dict):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=2, color=viz.MODEL_COLORS.get(name, viz.GREY),
                label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=viz.GREY, lw=1, label="Chance (AUC 0.500)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curves on the test set", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    return fig


def plot_pr(y_true, proba_by_model: dict):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in proba_by_model.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, lw=2, color=viz.MODEL_COLORS.get(name, viz.GREY),
                label=f"{name} (AP {average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), ls="--", color=viz.GREY, lw=1, label="Share of disease cases")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left")
    return fig


def plot_confusion(metrics: dict, title: str):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    return fig


def plot_calibration(y_true, proba_by_model: dict, n_bins: int = 8):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in proba_by_model.items():
        frac, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", lw=2, color=viz.MODEL_COLORS.get(name, viz.GREY),
                label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=viz.GREY, lw=1, label="Perfectly calibrated")
    ax.set(xlabel="Predicted probability", ylabel="Observed share with disease",
           title="Calibration (test set)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    return fig


def plot_model_comparison(results: dict):
    """Cross-validated AUC (with spread) next to test AUC for each model."""
    names = list(results)
    cv_mean = [results[n]["cv_roc_auc_mean"] for n in names]
    cv_std = [results[n]["cv_roc_auc_std"] for n in names]
    test_auc = [results[n]["test"]["roc_auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - 0.2, cv_mean, 0.4, yerr=cv_std, capsize=4, color=viz.BLUE, label="Cross-validated AUC (train)")
    ax.bar(x + 0.2, test_auc, 0.4, color=viz.ORANGE, label="Test AUC")
    for xi, v, s in zip(x - 0.2, cv_mean, cv_std):
        ax.text(xi, v + s + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, test_auc):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set(ylim=(0.5, 0.95), ylabel="ROC-AUC", title="Model comparison")
    ax.legend(loc="upper right")
    return fig


def plot_threshold_tradeoff(y_true, proba, chosen: float):
    """Precision and recall for every cut-off, with the chosen screening threshold marked."""
    precision, recall, thr = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(thr, precision[:-1], color=viz.BLUE, lw=2, label="Precision")
    ax.plot(thr, recall[:-1], color=viz.RED, lw=2, label="Recall")
    ax.axvline(chosen, color=viz.GREY, ls="--", lw=1.2)
    ax.text(chosen + 0.01, 0.52, f"screening\nthreshold {chosen:.2f}", color=viz.GREY, fontsize=9)
    ax.axhline(C.TARGET_RECALL, color=viz.RED, ls=":", lw=1)
    ax.set(xlabel="Probability threshold", ylabel="Score", ylim=(0.45, 1.02),
           title="Choosing the screening threshold (training data, out-of-fold)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    return fig
