"""
Mindgard — Per-Class Threshold Optimizer (Stable Version)
Uses Optuna to find optimal decision thresholds per class, maximizing Macro F1.

Handles edge cases:
  - Multiple classes above threshold → risk-aware tie-breaking
  - No class above threshold → fallback to argmax
  - Stable predictions guaranteed
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, MODELS_DIR, SEED

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Risk priority for tie-breaking: U2R > R2L > Probe > DoS > normal
# Higher value = higher priority when multiple classes exceed threshold
CLASS_PRIORITY = {
    "U2R": 5,
    "R2L": 4,
    "Probe": 3,
    "DoS": 2,
    "normal": 1,
}
PRIORITY_ARRAY = np.array([CLASS_PRIORITY[c] for c in CLASS_ORDER], dtype=np.float32)


def _predict_with_thresholds(
    y_prob: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """
    Apply per-class thresholds with stable edge-case handling.

    Logic:
    1. For each sample, check which classes exceed their threshold
    2. If multiple classes pass → pick highest priority (U2R > R2L > ...)
    3. If no class passes → fallback to argmax (highest probability)
    4. Among tied-priority classes → pick highest probability
    """
    n_samples, n_classes = y_prob.shape
    predictions = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        probs = y_prob[i]

        # Which classes exceed their threshold?
        passing = probs >= thresholds

        if passing.any():
            # Multiple classes pass -> risk-aware selection
            # STRICT_HIERARCHY_INDICES defines security priority:
            # U2R (index 3), R2L (index 2), Probe (index 1), DoS (index 0), normal (index 4)
            STRICT_HIERARCHY_INDICES = [
                CLASS_ORDER.index("U2R"),
                CLASS_ORDER.index("R2L"),
                CLASS_ORDER.index("Probe"),
                CLASS_ORDER.index("DoS"),
                CLASS_ORDER.index("normal")
            ]
            
            for cls_idx in STRICT_HIERARCHY_INDICES:
                if passing[cls_idx]:
                    predictions[i] = cls_idx
                    break
            else:
                predictions[i] = np.argmax(probs)
        else:
            # No class passes -> fallback to argmax
            predictions[i] = np.argmax(probs)

    return predictions


def optimize_thresholds(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    n_trials: int = 300,
    random_state: int = SEED,
) -> dict[str, Any]:
    """
    Use Optuna to find per-class thresholds that maximize Macro F1.

    The search space is [0.05, 0.95] per class — wider range for rare classes.
    """
    n_classes = y_prob_val.shape[1]

    # Baseline: default argmax
    y_pred_default = np.argmax(y_prob_val, axis=1)
    baseline_f1 = f1_score(y_val, y_pred_default, average="macro")
    print(f"\n  Baseline Macro F1 (argmax): {baseline_f1:.4f}")

    # Per-class baseline
    print(f"  Per-class F1 (baseline):")
    for i, cls in enumerate(CLASS_ORDER):
        mask = y_val == i
        if mask.sum() > 0:
            cls_f1 = f1_score(y_val == i, y_pred_default == i, average="binary")
            print(f"    {cls:10s}  F1={cls_f1:.4f}  support={mask.sum():,}")

    start = time.time()

    def objective(trial: optuna.Trial) -> float:
        thresholds = np.array([
            trial.suggest_float(f"threshold_{CLASS_ORDER[i]}", 0.05, 0.95)
            for i in range(n_classes)
        ])
        y_pred = _predict_with_thresholds(y_prob_val, thresholds)
        return f1_score(y_val, y_pred, average="macro")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - start

    # Extract best thresholds
    best_thresholds = np.array([
        study.best_params[f"threshold_{CLASS_ORDER[i]}"]
        for i in range(n_classes)
    ])

    optimized_f1 = study.best_value
    improvement = optimized_f1 - baseline_f1

    print(f"\n  Optimized Macro F1: {optimized_f1:.4f} ({'+' if improvement >= 0 else ''}{improvement:.4f})")
    print(f"  Optimization time: {elapsed:.1f}s ({n_trials} trials)")
    print(f"\n  Optimal thresholds per class:")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls:10s}  threshold={best_thresholds[i]:.4f}")

    # Verify stability: test on same data
    y_pred_opt = _predict_with_thresholds(y_prob_val, best_thresholds)
    verify_f1 = f1_score(y_val, y_pred_opt, average="macro")
    print(f"\n  Verification F1: {verify_f1:.4f} (should match optimized)")

    # Save thresholds
    result = {
        "thresholds": {cls: float(best_thresholds[i]) for i, cls in enumerate(CLASS_ORDER)},
        "thresholds_array": best_thresholds.tolist(),
        "baseline_macro_f1": float(baseline_f1),
        "optimized_macro_f1": float(optimized_f1),
        "improvement": float(improvement),
        "n_trials": n_trials,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    thresh_path = MODELS_DIR / "optimal_thresholds.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✅ Thresholds saved to: {thresh_path}")

    return result


def apply_optimized_thresholds(
    y_prob: np.ndarray, thresholds: np.ndarray | list[float]
) -> np.ndarray:
    """Apply previously optimized thresholds to new predictions."""
    if isinstance(thresholds, list):
        thresholds = np.array(thresholds)
    return _predict_with_thresholds(y_prob, thresholds)


def load_thresholds() -> np.ndarray | None:
    """Load saved thresholds from disk, or return None."""
    thresh_path = MODELS_DIR / "optimal_thresholds.json"
    if not thresh_path.exists():
        return None
    data = json.loads(thresh_path.read_text(encoding="utf-8"))
    return np.array(data["thresholds_array"])
