"""
Mindgard — Evaluation & Reporting Module
Generates all metrics, plots, and reports the jury wants to see.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, REPORTS_DIR


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute every metric the hackathon criteria mentions."""

    if class_names is None:
        class_names = CLASS_ORDER

    metrics: dict[str, Any] = {}

    # ─── Primary metric: Macro F1 ───
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))

    # ─── Secondary metrics ───
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ─── ROC-AUC (if probabilities available) ───
    if y_prob is not None:
        try:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            metrics["roc_auc_weighted"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception as e:
            metrics["roc_auc_macro"] = None
            metrics["roc_auc_weighted"] = None
            print(f"  ⚠ ROC-AUC error: {e}")

    # ─── Per-class metrics ───
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {}
    for cls in class_names:
        if cls in report:
            metrics["per_class"][cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

    return metrics


def print_metrics(metrics: dict[str, Any], class_names: list[str] | None = None) -> None:
    """Pretty-print all metrics."""
    if class_names is None:
        class_names = CLASS_ORDER

    print("\n" + "=" * 60)
    print("CYBERSENTINEL — EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  ★ MACRO F1-SCORE: {metrics['macro_f1']:.4f} ★")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    if metrics.get("roc_auc_macro"):
        print(f"  ROC-AUC (macro):  {metrics['roc_auc_macro']:.4f}")

    print(f"\n  {'─'*55}")
    print(f"  {'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'─'*55}")
    for cls in class_names:
        if cls in metrics.get("per_class", {}):
            m = metrics["per_class"][cls]
            print(f"  {cls:12s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")


def save_metrics(metrics: dict[str, Any]) -> Path:
    """Save metrics to JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  ✅ Metrics saved to: {path}")
    return path


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> Path:
    """Save text classification report."""
    if class_names is None:
        class_names = CLASS_ORDER
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    path = REPORTS_DIR / "classification_report.txt"
    path.write_text(report_text, encoding="utf-8")
    print(f"  ✅ Classification report saved to: {path}")
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    macro_f1: float | None = None,
) -> Path:
    """Generate and save a beautiful confusion matrix heatmap."""
    if class_names is None:
        class_names = CLASS_ORDER
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ─── Raw counts ───
    ax = axes[0]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix (Counts)", fontweight="bold")

    # ─── Normalized (recall per class) ───
    ax = axes[1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Normalized (Recall per Class)", fontweight="bold")

    title = "Mindgard — Confusion Matrix"
    if macro_f1 is not None:
        title += f" | Macro F1 = {macro_f1:.4f}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved to: {path}")
    return path


def plot_feature_importance(
    feature_importance: list[dict[str, Any]], top_n: int = 20
) -> Path:
    """Plot top-N feature importance horizontal bar chart."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = feature_importance[:top_n]
    names = [d["feature"] for d in data][::-1]
    values = [d["importance"] for d in data][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Mindgard — Top Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Feature importance plot saved to: {path}")
    return path


def plot_per_class_metrics(metrics: dict[str, Any]) -> Path:
    """Bar chart comparing precision, recall, F1 per class."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    classes = list(metrics.get("per_class", {}).keys())
    precision = [metrics["per_class"][c]["precision"] for c in classes]
    recall = [metrics["per_class"][c]["recall"] for c in classes]
    f1 = [metrics["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x, recall, width, label="Recall", color="#e74c3c", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1-Score", color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(
        f"Mindgard — Per-Class Metrics | Macro F1 = {metrics['macro_f1']:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    fig.tight_layout()
    path = REPORTS_DIR / "per_class_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Per-class metrics plot saved to: {path}")
    return path


def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    feature_importance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full evaluation pipeline: metrics + plots + reports."""

    print("\n" + "=" * 60)
    print("CYBERSENTINEL — EVALUATION PIPELINE")
    print("=" * 60)

    # ─── Compute metrics ───
    print("\n[1/5] Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)

    # ─── Save metrics JSON ───
    print("\n[2/5] Saving metrics...")
    save_metrics(metrics)
    save_classification_report(y_true, y_pred)

    # ─── Confusion matrix ───
    print("\n[3/5] Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, macro_f1=metrics["macro_f1"])

    # ─── Per-class metrics chart ───
    print("\n[4/5] Generating per-class metrics chart...")
    plot_per_class_metrics(metrics)

    # ─── Feature importance ───
    if feature_importance:
        print("\n[5/5] Generating feature importance plot...")
        plot_feature_importance(feature_importance)

    print(f"\n{'═'*60}")
    print(f"  ALL REPORTS SAVED TO: {REPORTS_DIR}")
    print(f"{'═'*60}")

    return metrics
