"""
Mindgard — Hybrid Intelligent IDS Pipeline (v4)
One command: python src/pipeline.py

Architecture:
  Raw Data → Clean → Features → PARALLEL INTELLIGENCE LAYERS →
  [XGBoost + Isolation Forest + Behavioral Engine] →
  Decision Fusion → Risk Scoring → SHAP → Report

This is a HYBRID system that detects:
  1. Known attacks (XGBoost classifier)
  2. Unknown/zero-day attacks (Isolation Forest anomaly detection)
  3. Behavioral anomalies (temporal pattern analysis)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_ORDER, CLEANED_DIR, MODELS_DIR, REPORTS_DIR, SEED,
    OPTUNA_N_TRIALS, VAL_SPLIT,
)

from clean_data import run_cleaning
from feature_engineering import engineer_features_array, FEATURE_DESCRIPTIONS
from train_xgboost import run_training
from threshold_optimizer import optimize_thresholds
from anomaly_model import run_anomaly_detection
from behavior_engine import run_behavioral_analysis, compute_behavior_score
from fusion_engine import run_fusion, EXTENDED_CLASSES, ALERT_LEVEL_NAMES, ALERT_LEVEL_EMOJI
from risk_scoring import (
    compute_batch_advanced_risk, compute_batch_rarity_scores,
    risk_summary, format_risk_alert,
)
from evaluate import run_evaluation
from explainer import run_shap_analysis


def main() -> None:
    pipeline_start = time.time()

    print("\n" + "█" * 65)
    print("█" + " " * 63 + "█")
    print("█" + "  MINDGARD — HYBRID INTELLIGENT INTRUSION DETECTION SYSTEM  ".center(63) + "█")
    print("█" + "  Pipeline v4 (XGBoost + Anomaly + Behavior + Fusion)      ".center(63) + "█")
    print("█" + " " * 63 + "█")
    print("█" * 65 + "\n")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: DATA CLEANING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 1: DATA CLEANING (RobustScaler)")
    print("▓" * 65)

    cleaned = run_cleaning()
    X_train = cleaned["X_train"]
    X_test = cleaned["X_test"]
    y_train = cleaned["y_train"]
    y_test = cleaned["y_test"]
    feature_cols = cleaned["feature_cols"]

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 2: FEATURE ENGINEERING")
    print("▓" * 65)

    print("\n  Engineering domain-driven features...")
    X_train, feature_cols = engineer_features_array(X_train, feature_cols)
    X_test, _ = engineer_features_array(X_test, feature_cols[:len(cleaned["feature_cols"])])
    if X_test.shape[1] != X_train.shape[1]:
        X_test_orig = cleaned["X_test"]
        X_test, _ = engineer_features_array(X_test_orig, cleaned["feature_cols"])

    print(f"\n  Final feature matrix:")
    print(f"    Train: {X_train.shape}")
    print(f"    Test:  {X_test.shape}")
    print(f"    Engineered features: {len(feature_cols) - len(cleaned['feature_cols'])}")

    for feat, desc in FEATURE_DESCRIPTIONS.items():
        if feat in feature_cols:
            print(f"    • {feat}: {desc}")

    np.save(CLEANED_DIR / "X_train_engineered.npy", X_train)
    np.save(CLEANED_DIR / "X_test_engineered.npy", X_test)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: PARALLEL INTELLIGENCE LAYERS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 3: PARALLEL INTELLIGENCE LAYERS")
    print("▓" * 65)

    # ─── Layer 1: XGBoost Classifier (threshold-aware) ───
    print("\n" + "─" * 65)
    print("  LAYER 1: XGBoost Classifier")
    print("─" * 65)

    # Split for threshold optimization
    print(f"\n  Splitting: {1-VAL_SPLIT:.0%} train / {VAL_SPLIT:.0%} validation...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    X_train_inner = X_train[train_idx]
    y_train_inner = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    print(f"    Train inner: {X_train_inner.shape[0]:,}")
    print(f"    Validation:  {X_val.shape[0]:,}")

    # Train inner model for threshold optimization
    print("\n  Training XGBoost (inner split)...")
    inner_results = run_training(
        X_train_inner, y_train_inner,
        X_val, y_val,
        feature_cols=feature_cols,
        do_cv=False,
    )

    # Optimize thresholds
    y_val_prob = inner_results["model"].predict_proba(X_val)
    threshold_results = optimize_thresholds(y_val, y_val_prob, n_trials=OPTUNA_N_TRIALS)
    optimal_thresholds = np.array(threshold_results["thresholds_array"])

    # Train final model on full data
    print("\n  Training FINAL XGBoost (full training set)...")
    train_results = run_training(
        X_train, y_train,
        X_test, y_test,
        feature_cols=feature_cols,
        do_cv=True,
    )
    model = train_results["model"]
    y_prob = model.predict_proba(X_test)

    # ─── Layer 2: Anomaly Detection (Isolation Forest) ───
    print("\n" + "─" * 65)
    print("  LAYER 2: Anomaly Detection (Isolation Forest)")
    print("─" * 65)

    anomaly_results = run_anomaly_detection(X_train, y_train, X_test, contamination=0.1)
    anomaly_scores_test = anomaly_results["test_scores"]

    # ─── Layer 3: Behavioral Analysis ───
    print("\n" + "─" * 65)
    print("  LAYER 3: Behavioral Analysis Engine")
    print("─" * 65)

    # Compute behavior on test (use argmax as initial prediction for logging)
    y_pred_initial = np.argmax(y_prob, axis=1)
    behavior_results = run_behavioral_analysis(X_test, y_pred_initial, feature_cols)
    behavior_scores_test = behavior_results["behavior_scores"]

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: DECISION FUSION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 4: DECISION FUSION")
    print("▓" * 65)

    fusion_results = run_fusion(
        y_prob=y_prob,
        anomaly_scores=anomaly_scores_test,
        behavior_scores=behavior_scores_test,
        optimized_thresholds=optimal_thresholds,
        y_true=y_test,
    )

    fused_predictions = fusion_results["predictions"]
    fused_confidences = fusion_results["confidences"]
    alert_levels = fusion_results["alert_levels"]

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: ADVANCED RISK SCORING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 5: ADVANCED RISK SCORING")
    print("▓" * 65)

    # Compute rarity scores
    print("\n  Computing rarity scores (distance from training distribution)...")
    median = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - median), axis=0) + 1e-6
    X_train_stats = {
        "median": median,
        "mad": mad,
    }
    rarity_scores = compute_batch_rarity_scores(X_test, X_train_stats)
    print(f"  Rarity: mean={rarity_scores.mean():.4f}, max={rarity_scores.max():.4f}")

    # Save train stats for inference
    import joblib
    joblib.dump(X_train_stats, MODELS_DIR / "train_stats.pkl")

    # Advanced risk scores
    print("\n  Computing advanced multi-signal risk scores...")
    risk_scores = compute_batch_advanced_risk(
        predictions=fused_predictions,
        confidences=fused_confidences,
        anomaly_scores=anomaly_scores_test,
        behavior_scores=behavior_scores_test,
        rarity_scores=rarity_scores,
        y_prob=y_prob,
    )
    risk_stats = risk_summary(risk_scores)

    print(f"\n  Risk Summary ({risk_stats['total']:,} connections):")
    print(f"    Mean risk score: {risk_stats['mean_risk_score']}/100")
    print(f"    Max risk score:  {risk_stats['max_risk_score']}/100")
    print(f"    Alert rate:      {risk_stats['alert_rate']}%")
    print(f"\n  Risk levels:")
    for level, count in risk_stats["risk_levels"].items():
        pct = 100 * count / risk_stats["total"]
        print(f"    {level:10s}  {count:>7,}  ({pct:.1f}%)")

    # Top alerts
    print(f"\n  Top 5 highest-risk connections:")
    sorted_risks = sorted(risk_scores, key=lambda r: r["risk_score"], reverse=True)
    for r in sorted_risks[:5]:
        print(format_risk_alert(r))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "risk_summary.json", "w", encoding="utf-8") as f:
        json.dump(risk_stats, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: EVALUATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 6: EVALUATION")
    print("▓" * 65)

    # Evaluate known-class predictions only (exclude UNKNOWN from F1 calc)
    known_mask = fused_predictions < len(CLASS_ORDER)
    if known_mask.sum() > 0 and known_mask.sum() < len(fused_predictions):
        print(f"\n  Evaluating {known_mask.sum():,} known-class predictions "
              f"({(~known_mask).sum():,} classified as UNKNOWN)")

    # Use known predictions for standard metrics
    y_pred_known = fused_predictions[known_mask]
    y_test_known = y_test[known_mask]
    y_prob_known = y_prob[known_mask]

    if len(y_pred_known) > 0:
        metrics = run_evaluation(
            y_true=y_test_known,
            y_pred=y_pred_known,
            y_prob=y_prob_known,
            feature_importance=train_results["feature_importance"],
        )
    else:
        print("  ⚠ No known-class predictions to evaluate")
        metrics = {"macro_f1": 0, "accuracy": 0, "macro_precision": 0, "macro_recall": 0}

    # Also compute overall accuracy including UNKNOWN
    print(f"\n  Overall stats (including UNKNOWN):")
    print(f"    Total predictions: {len(fused_predictions):,}")
    print(f"    Known class:       {known_mask.sum():,}")
    print(f"    Unknown (zero-day): {(~known_mask).sum():,}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: SHAP EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("PHASE 7: SHAP EXPLAINABILITY")
    print("▓" * 65)

    shap_results = run_shap_analysis(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_pred=y_pred_initial,
        feature_names=feature_cols,
    )

    # ═══════════════════════════════════════════════════════════
    # PHASE 8: FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - pipeline_start

    pipeline_summary = {
        "pipeline_version": "v4",
        "system": "Mindgard Hybrid Intelligent IDS",
        "pipeline_time_seconds": round(elapsed, 1),
        "architecture": {
            "layer_1": "XGBoost Classifier (known attacks)",
            "layer_2": "Isolation Forest (anomaly/zero-day detection)",
            "layer_3": "Behavioral Analysis Engine (temporal patterns)",
            "fusion": "Decision Fusion with open-set recognition",
            "risk": "Advanced multi-signal risk scoring",
            "explainability": "SHAP TreeExplainer",
        },
        "dataset": "KDD Cup 1999 (corrected 10%)",
        "n_features_original": len(cleaned["feature_cols"]),
        "n_features_engineered": len(feature_cols),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "classes": EXTENDED_CLASSES,
        "threshold_results": {
            "optimal_thresholds": threshold_results["thresholds"],
        },
        "fusion_stats": fusion_results["stats"],
        "risk_summary": risk_stats,
        "metrics": {
            "macro_f1": metrics.get("macro_f1", 0),
            "accuracy": metrics.get("accuracy", 0),
            "macro_precision": metrics.get("macro_precision", 0),
            "macro_recall": metrics.get("macro_recall", 0),
            "roc_auc_macro": metrics.get("roc_auc_macro"),
        },
        "cv_results": train_results.get("cv_results"),
        "top_5_features": train_results["feature_importance"][:5],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2, default=str)

    md_report = generate_markdown_report(pipeline_summary, metrics, train_results,
                                          threshold_results, risk_stats, fusion_results["stats"])
    md_path = REPORTS_DIR / "final_report.md"
    md_path.write_text(md_report, encoding="utf-8")

    print("\n" + "█" * 65)
    print("█" + " " * 63 + "█")
    print("█" + "  MINDGARD v4 — PIPELINE COMPLETE                           ".center(63) + "█")
    print("█" + " " * 63 + "█")
    print("█" * 65)
    print(f"\n  ⏱  Total time:          {elapsed:.1f} seconds")
    print(f"  ★  Macro F1-Score:      {metrics.get('macro_f1', 0):.4f}")
    print(f"  📊 Accuracy:            {metrics.get('accuracy', 0):.4f}")
    print(f"  🔍 Unknown detected:    {(~known_mask).sum():,}")
    print(f"  🚨 Alert rate:          {risk_stats['alert_rate']}%")
    print(f"  🧠 SHAP plots:          {len(shap_results.get('plot_paths', []))} generated")
    print(f"\n  📁 Reports:     {REPORTS_DIR}")
    print(f"  🤖 Model:       {MODELS_DIR / 'mindgard_xgboost.json'}")
    print(f"  📄 Full report: {md_path}")
    print(f"\n  🚀 Start API:  python src/api.py")
    print(f"  📡 API docs:   http://127.0.0.1:8000/docs")
    print()


def generate_markdown_report(
    summary: dict, metrics: dict, train_results: dict,
    threshold_results: dict, risk_stats: dict, fusion_stats: dict,
) -> str:
    """Generate technical report for v4 hybrid system."""

    per_class = metrics.get("per_class", {})
    class_rows = ""
    for cls in CLASS_ORDER:
        if cls in per_class:
            m = per_class[cls]
            class_rows += (
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['f1']:.4f} | {m['support']:,} |\n"
            )

    feat_rows = ""
    for item in train_results.get("feature_importance", [])[:15]:
        feat_rows += f"| {item['rank']} | {item['feature']} | {item['importance']:.4f} |\n"

    thresh = threshold_results or {}
    thresh_rows = ""
    for cls, val in thresh.get("thresholds", {}).items():
        thresh_rows += f"| {cls} | {val:.4f} |\n"

    risk_level_rows = ""
    for level, count in risk_stats.get("risk_levels", {}).items():
        pct = 100 * count / max(risk_stats.get("total", 1), 1)
        risk_level_rows += f"| {level} | {count:,} | {pct:.1f}% |\n"

    alert_rows = ""
    for level, count in fusion_stats.get("alert_levels", {}).items():
        total = max(sum(fusion_stats.get("alert_levels", {}).values()), 1)
        pct = 100 * count / total
        alert_rows += f"| {level} | {count:,} | {pct:.1f}% |\n"

    report = f"""# Mindgard — Technical Report (v4)
## Hybrid Intelligent Intrusion Detection System

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| **Macro F1-Score** | **{metrics.get('macro_f1', 0):.4f}** |
| Accuracy | {metrics.get('accuracy', 0):.4f} |
| Macro Precision | {metrics.get('macro_precision', 0):.4f} |
| Macro Recall | {metrics.get('macro_recall', 0):.4f} |
| ROC-AUC (macro) | {metrics.get('roc_auc_macro', 'N/A')} |
| Unknown threats detected | {fusion_stats.get('unknown_attacks', 0):,} |
| Pipeline time | {summary['pipeline_time_seconds']}s |

## 2. System Architecture

```
Raw Data → Cleaning (RobustScaler) → Feature Engineering
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
 XGBoost      Isolation Forest   Behavioral
 Classifier   (Anomaly)          Engine
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
           Decision Fusion
           (Open-set recognition)
                    ↓
           Risk Scoring Engine
           (Multi-signal fusion)
                    ↓
           SHAP Explainability
```

### Intelligence Layers

| Layer | Model | Purpose |
|---|---|---|
| Classifier | XGBoost (multi:softprob) | Detect known attack categories |
| Anomaly | Isolation Forest (200 trees) | Detect zero-day/unknown attacks |
| Behavioral | Rule-based feature analysis | Detect temporal attack patterns |

## 3. Methodology

### 3.1 Cost-Sensitive Learning
- Balanced sample weights (no synthetic data — preserves realism)
- Class weights scale inversely with frequency

### 3.2 Per-Class Threshold Optimization
| Class | Optimal Threshold |
|---|---|
{thresh_rows}

### 3.3 Decision Fusion Rules
1. **RULE 1** — High anomaly + low confidence → UNKNOWN_ATTACK
2. **RULE 2** — Very low max-prob + moderate anomaly → UNKNOWN_ATTACK
3. **RULE 3** — Optimized thresholds with risk-aware tie-breaking
4. **RULE 4** — Fallback to argmax

### 3.4 Advanced Risk Scoring
```
risk = (w1×confidence + w2×anomaly + w3×behavior + w4×rarity) × class_weight × uncertainty_penalty
```
- Entropy-based uncertainty: uncertain predictions get escalated
- Rarity score: statistical distance from training distribution

### 3.5 False Positive Reduction
Secondary validation: if classifier says "attack" but anomaly AND behavior are low → downgrade alert.

## 4. Results

### 4.1 Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
{class_rows}

### 4.2 Alert Distribution
| Alert Level | Count | % |
|---|---|---|
{alert_rows}

### 4.3 Risk Distribution
| Risk Level | Count | % |
|---|---|---|
{risk_level_rows}
Alert rate: {risk_stats.get('alert_rate', 0)}%

### 4.4 Top Features
| Rank | Feature | Importance |
|---|---|---|
{feat_rows}

## 5. API

```bash
python src/api.py
# POST /predict — single connection → prediction + risk + SHAP
# POST /predict/batch — CSV upload → batch analysis
# GET /model/info — system metadata
```

## 6. Artifacts
| Artifact | Path |
|---|---|
| XGBoost model | `models/mindgard_xgboost.json` |
| Anomaly detector | `models/anomaly_detector.pkl` |
| Preprocessing | `models/preprocessing.pkl` |
| Thresholds | `models/optimal_thresholds.json` |
| Train stats | `models/train_stats.pkl` |
| SHAP explainer | `models/shap_explainer.pkl` |
| SHAP plots | `reports/shap_*.png` |
| Fusion stats | `reports/fusion_stats.json` |
| Risk summary | `reports/risk_summary.json` |
"""
    return report


if __name__ == "__main__":
    main()
