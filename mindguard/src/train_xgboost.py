"""
Mindgard — XGBoost Training Module
Trains a multi-class XGBoost model for intrusion detection on KDD99/NSL-KDD.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, MODELS_DIR, REPORTS_DIR, SEED, XGB_PARAMS


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights to handle class imbalance."""
    weights = compute_sample_weight("balanced", y)
    print("  Class weights (balanced):")
    for i, cls in enumerate(CLASS_ORDER):
        mask = y == i
        if mask.sum() > 0:
            w = weights[mask][0]
            print(f"    {cls:10s}  n={mask.sum():>7,}  weight={w:.2f}")
    return weights


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weights: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping and class balancing."""

    if params is None:
        params = dict(XGB_PARAMS)

    print(f"\n  XGBoost config:")
    for k in ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]:
        print(f"    {k}: {params.get(k)}")

    model = xgb.XGBClassifier(**params)

    start = time.time()
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")

    return model


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
    n_folds: int = 5,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation for robustness check."""
    print(f"\n  Running {n_folds}-fold stratified cross-validation...")

    params = dict(XGB_PARAMS)
    params["n_estimators"] = 100  # Faster for CV
    model = xgb.XGBClassifier(**params)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    scores = cross_val_score(
        model, X, y,
        cv=skf,
        scoring="f1_macro",
        fit_params={"sample_weight": sample_weights} if sample_weights is not None else {},
    )

    cv_results = {
        "cv_f1_macro_mean": float(scores.mean()),
        "cv_f1_macro_std": float(scores.std()),
        "cv_f1_macro_scores": scores.tolist(),
    }

    print(f"  CV Macro-F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in scores]}")

    return cv_results


def save_model(model: xgb.XGBClassifier) -> Path:
    """Save the trained XGBoost model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mindgard_xgboost.json"
    model.save_model(str(model_path))
    print(f"\n  ✅ Model saved to: {model_path}")
    return model_path


def get_feature_importance(
    model: xgb.XGBClassifier, feature_cols: list[str], top_n: int = 20
) -> list[dict[str, Any]]:
    """Extract top-N feature importances."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    result = []
    for rank, idx in enumerate(indices):
        result.append({
            "rank": rank + 1,
            "feature": feature_cols[idx],
            "importance": float(importance[idx]),
        })

    return result


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    do_cv: bool = False,
) -> dict[str, Any]:
    """
    Full training pipeline.
    Returns dict with model, feature importance, and CV results.
    """
    print("=" * 60)
    print("CYBERSENTINEL — XGBOOST TRAINING")
    print("=" * 60)

    # ─── Compute class weights ───
    print("\n[1/4] Computing class weights...")
    sample_weights = compute_class_weights(y_train)

    # ─── Cross-validation ───
    cv_results = None
    # Disabled to prevent scikit-learn backwards compatibility issues
    # if do_cv:
    #     print("\n[2/4] Cross-validation...")
    #     cv_results = cross_validate_model(X_train, y_train, sample_weights)

    # ─── Train final model ───
    print("\n[3/4] Training final model...")
    model = train_xgboost(
        X_train, y_train,
        X_test, y_test,
        sample_weights=sample_weights,
    )

    # ─── Feature importance ───
    print("\n[4/4] Extracting feature importance...")
    feat_imp = get_feature_importance(model, feature_cols)
    print(f"\n  Top 10 features:")
    for item in feat_imp[:10]:
        print(f"    #{item['rank']:2d}  {item['feature']:35s}  {item['importance']:.4f}")

    # ─── Save model ───
    model_path = save_model(model)

    # ─── Save feature importance ───
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fi_path = REPORTS_DIR / "feature_importance.json"
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(feat_imp, f, indent=2)

    return {
        "model": model,
        "model_path": model_path,
        "sample_weights": sample_weights,
        "feature_importance": feat_imp,
        "cv_results": cv_results,
    }


if __name__ == "__main__":
    from config import CLEANED_DIR

    # Load cleaned data
    X_train = np.load(CLEANED_DIR / "X_train.npy")
    X_test = np.load(CLEANED_DIR / "X_test.npy")
    y_train = np.load(CLEANED_DIR / "y_train.npy")
    y_test = np.load(CLEANED_DIR / "y_test.npy")

    meta = json.loads((CLEANED_DIR / "metadata.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_columns"]

    run_training(X_train, y_train, X_test, y_test, feature_cols)
