"""
Mindgard — SHAP Explainability Module
Provides global and local explanations for XGBoost predictions.

SHAP (SHapley Additive exPlanations) shows WHY the model flagged something —
this is what non-ML jury members remember from a demo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, MODELS_DIR, REPORTS_DIR


def create_explainer(
    model: Any,
    X_background: np.ndarray | None = None,
) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for XGBoost.
    
    TreeExplainer is exact for tree-based models (not approximate).
    Background data is optional for TreeExplainer but helps with output type.
    """
    print("  Creating SHAP TreeExplainer...")
    start = time.time()
    
    if X_background is not None and len(X_background) > 500:
        # Subsample background for efficiency
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_background), 500, replace=False)
        X_background = X_background[idx]
    
    explainer = shap.TreeExplainer(model)
    elapsed = time.time() - start
    print(f"  TreeExplainer created in {elapsed:.1f}s")
    
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    max_samples: int = 1000,
) -> tuple[np.ndarray | list, np.ndarray]:
    """
    Compute SHAP values for a set of samples.
    
    For multi-class, returns shape (n_samples, n_features, n_classes).
    Subsamples if X is too large for performance.
    """
    if len(X) > max_samples:
        print(f"  Subsampling {max_samples} from {len(X)} for SHAP computation...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    print(f"  Computing SHAP values for {len(X_sub)} samples...")
    start = time.time()
    shap_values = explainer.shap_values(X_sub)
    elapsed = time.time() - start
    print(f"  SHAP values computed in {elapsed:.1f}s")
    
    return shap_values, X_sub


def plot_global_summary(
    shap_values: np.ndarray | list,
    X: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
) -> list[Path]:
    """
    Generate global SHAP summary plots:
    1. Combined beeswarm for all classes
    2. Per-class summary for each attack type
    3. Mean absolute SHAP bar chart
    
    Returns list of saved plot paths.
    """
    if class_names is None:
        class_names = CLASS_ORDER
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    
    # Normalize shap_values to a list of (n_samples, n_features) arrays
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            shap_values_list = [shap_values]
    elif isinstance(shap_values, list):
        shap_values_list = shap_values
    else:
        # fallback
        shap_values_list = [shap_values]

    # ─── 1. Mean |SHAP| bar chart (all classes) ───
    print("  Generating SHAP global importance bar chart...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Average absolute SHAP across classes and samples
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_list], axis=0)
    
    # Top 20 features
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx][::-1]
    top_values = mean_abs[top_idx][::-1]
    
    colors = plt.cm.magma(np.linspace(0.25, 0.85, len(top_names)))
    ax.barh(range(len(top_names)), top_values, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Mindgard — SHAP Global Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    
    path = REPORTS_DIR / "shap_global_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)
    print(f"  ✅ Global SHAP importance: {path}")
    
    # ─── 2. Per-class SHAP summary ───
    if len(shap_values_list) == len(class_names):
        print("  Generating per-class SHAP summary plots...")
        
        fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 7))
        if len(class_names) == 1:
            axes = [axes]
        
        for cls_idx, (cls_name, ax) in enumerate(zip(class_names, axes)):
            sv = shap_values_list[cls_idx]
            mean_abs_cls = np.abs(sv).mean(axis=0)
            top_idx_cls = np.argsort(mean_abs_cls)[::-1][:10]
            names_cls = [feature_names[i] for i in top_idx_cls][::-1]
            values_cls = mean_abs_cls[top_idx_cls][::-1]
            
            cls_colors = {
                "DoS": "#e74c3c", "Probe": "#f39c12", "R2L": "#9b59b6",
                "U2R": "#e67e22", "normal": "#2ecc71"
            }
            color = cls_colors.get(cls_name, "#3498db")
            
            ax.barh(range(len(names_cls)), values_cls, color=color, alpha=0.85)
            ax.set_yticks(range(len(names_cls)))
            ax.set_yticklabels(names_cls, fontsize=8)
            ax.set_xlabel("Mean |SHAP|", fontsize=9)
            ax.set_title(f"{cls_name}", fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
        
        fig.suptitle("Mindgard — SHAP Feature Importance by Attack Class",
                      fontsize=14, fontweight="bold")
        fig.tight_layout()
        
        path = REPORTS_DIR / "shap_per_class.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        print(f"  ✅ Per-class SHAP: {path}")
    
    # ─── 3. SHAP beeswarm (top class) ───
    try:
        print("  Generating SHAP beeswarm plot...")
        if len(shap_values_list) == len(class_names):
            for cls_idx, cls_name in enumerate(class_names):
                fig = plt.figure(figsize=(12, 7))
                shap.summary_plot(
                    shap_values_list[cls_idx], X,
                    feature_names=feature_names,
                    show=False,
                    max_display=15,
                    plot_size=None,
                )
                plt.title(f"SHAP Beeswarm — {cls_name} Detection", fontsize=13, fontweight="bold")
                plt.tight_layout()
                path = REPORTS_DIR / f"shap_beeswarm_{cls_name}.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                saved_paths.append(path)
                print(f"  ✅ Beeswarm ({cls_name}): {path}")
    except Exception as e:
        print(f"  ⚠ Beeswarm plot failed: {e}")
    
    return saved_paths


def explain_single_prediction(
    explainer: shap.TreeExplainer,
    X_single: np.ndarray,
    feature_names: list[str],
    predicted_class: int,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """
    Explain a single prediction with SHAP waterfall plot.
    Returns the top contributing features for the predicted class.
    """
    if class_names is None:
        class_names = CLASS_ORDER
    
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    shap_values = explainer.shap_values(X_single)
    
    # Get SHAP values for predicted class
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[0, :, predicted_class]
    else:
        sv = shap_values[0]
    
    # Top contributing features
    top_idx = np.argsort(np.abs(sv))[::-1][:10]
    contributions = []
    for idx in top_idx:
        contributions.append({
            "feature": feature_names[idx],
            "shap_value": float(sv[idx]),
            "feature_value": float(X_single[0, idx]),
        })
    
    # Generate waterfall plot
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / "shap_waterfall_example.png"
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [contributions[i]["feature"] for i in range(len(contributions))][::-1]
        values = [contributions[i]["shap_value"] for i in range(len(contributions))][::-1]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
        
        ax.barh(range(len(names)), values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
        ax.set_title(
            f"Why classified as {class_names[predicted_class]}?",
            fontsize=13, fontweight="bold"
        )
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels
        for i, (v, name) in enumerate(zip(values, names)):
            ax.text(
                v + (0.001 if v >= 0 else -0.001),
                i,
                f"{v:+.3f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Waterfall plot: {save_path}")
    except Exception as e:
        print(f"  ⚠ Waterfall plot failed: {e}")
    
    return {
        "predicted_class": class_names[predicted_class],
        "top_contributions": contributions,
    }


def save_explainer(explainer: shap.TreeExplainer) -> Path:
    """Save SHAP explainer for later use."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "shap_explainer.pkl"
    joblib.dump(explainer, path)
    print(f"  ✅ SHAP explainer saved: {path}")
    return path


def run_shap_analysis(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Full SHAP analysis pipeline.
    Generates global + per-class + example waterfall plots.
    """
    print("\n" + "=" * 60)
    print("CYBERSENTINEL — SHAP EXPLAINABILITY")
    print("=" * 60)
    
    # ─── Create explainer ───
    print("\n[1/4] Creating SHAP TreeExplainer...")
    explainer = create_explainer(model, X_train)
    
    # ─── Compute SHAP values on test set ───
    print("\n[2/4] Computing SHAP values on test set...")
    shap_values, X_sub = compute_shap_values(explainer, X_test, max_samples=1000)
    
    # ─── Generate global plots ───
    print("\n[3/4] Generating global SHAP plots...")
    plot_paths = plot_global_summary(shap_values, X_sub, feature_names)
    
    # ─── Example single prediction explanation ───
    print("\n[4/4] Generating example prediction explanation...")
    # Find one sample of each class for explanation
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        mask = y_pred == cls_idx
        if mask.sum() > 0:
            sample_idx = np.where(mask)[0][0]
            explain_single_prediction(
                explainer, X_test[sample_idx], feature_names,
                predicted_class=cls_idx,
                save_path=REPORTS_DIR / f"shap_waterfall_{cls_name}.png",
            )
    
    # ─── Save explainer ───
    save_explainer(explainer)
    
    return {
        "explainer": explainer,
        "plot_paths": [str(p) for p in plot_paths],
    }
