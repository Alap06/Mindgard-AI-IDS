import os
import sys
import torch
import numpy as np
import joblib
import json
import time
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "lstm-smartgrid-IDS" / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(CURRENT_DIR)) # Add current dir to path for absolute imports

# Absolute imports (fix for 'relative import with no known parent package')
from models import IDSClassifier, AnomalyVAE
from dataset_v2 import load_and_prepare_data, get_loaders, CyberDataset
from engine_v2 import train_classifier, train_vae, get_predictions, get_vae_scores
from explainer_v2 import run_shap_analysis, generate_performance_plots

# Reusing modules from original 'src'
try:
    from behavior_engine import run_behavioral_analysis
    from fusion_engine import run_fusion, EXTENDED_CLASSES, ALERT_LEVEL_NAMES, ALERT_LEVEL_EMOJI
    from risk_scoring import (
        compute_batch_advanced_risk, compute_batch_rarity_scores,
        risk_summary, format_risk_alert
    )
    from config import CLASS_ORDER # For consistent labels
except ImportError:
    print("⚠ CRITICAL: Could not import core src modules. Ensure 'lstm-smartgrid-IDS/src' exists.")
    CLASS_ORDER = ["DoS", "Probe", "R2L", "U2R", "normal"]

def run_production_v2_pipeline(epochs=40, batch_size=2048):
    pipeline_start = time.time()
    
    # ─── Setup Environment ───
    DATA_DIR = PROJECT_ROOT / "dataset"
    MODELS_DIR = PROJECT_ROOT / "models_v2"
    REPORTS_DIR = PROJECT_ROOT / "reports_v2"
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("\n" + "█" * 70)
    print("█" + "  MINDGARD V2 — FULL DEEP HYBRID INTELLIGENT IDS          ".center(68) + "█")
    print("█" + f"  Hardware: RTX 3050 | Dataset: {DATA_DIR.name} ".center(68) + "█")
    print("█" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1 & 2: DATA CLEANING & FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1 & 2: DATA CLEANING & DOMAIN ENGINEERING")
    print("▓" * 70)
    X, y, feature_names, le, raw_scaler = load_and_prepare_data(str(DATA_DIR), use_full=False)
    
    # Class weights for imbalanced KDD data
    class_counts = np.bincount(y)
    total = len(y)
    weights = torch.tensor(total / (len(class_counts) * class_counts), dtype=torch.float32)
    
    train_loader, val_loader = get_loaders(X, y, batch_size=batch_size)
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: PARALLEL INTELLIGENCE LAYERS
    # ═══════════════════════════════════════════════════════════
    print("\nPHASE 3: INTELLIGENCE LAYERS TRAINING")
    print("▓" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Layer 1: Deep Classifier ---
    print(f"\n  [Layer 1] Training CNN-LSTM-Attention Classifier on {device}...")
    classifier = IDSClassifier(X.shape[1], len(le.classes_))
    classifier = train_classifier(classifier, train_loader, val_loader, device, epochs=epochs, weight=weights)
    
    # --- Layer 2: Anomaly Detection (VAE) ---
    print(f"\n  [Layer 2] Training Anomaly VAE (Normal Class Only)...")
    normal_cls_idx = list(le.classes_).index("normal")
    X_normal = X[y == normal_cls_idx]
    
    from torch.utils.data import DataLoader
    normal_ds = CyberDataset(X_normal, torch.zeros(len(X_normal)))
    t_n, v_n = torch.utils.data.random_split(normal_ds, [int(0.8*len(normal_ds)), len(normal_ds)-int(0.8*len(normal_ds))])
    vae_train_ldr = DataLoader(t_n, batch_size=batch_size, shuffle=True)
    vae_val_ldr = DataLoader(v_n, batch_size=batch_size, shuffle=False)
    
    vae = AnomalyVAE(X.shape[1])
    vae = train_vae(vae, vae_train_ldr, vae_val_ldr, device, epochs=max(10, epochs//2))

    # --- Layer 3: Behavioral Analysis ---
    print(f"\n  [Layer 3] Analyzing Behavioral Signals...")
    # Get initial predictions for behavioral engine
    y_prob, y_true = get_predictions(classifier, val_loader, device)
    y_pred_initial = np.argmax(y_prob, axis=1)
    
    # Reconstruct X_val for behavioral engine
    X_val_list = []
    for bx, _ in val_loader: X_val_list.append(bx.numpy())
    X_val = np.concatenate(X_val_list)
    
    behavior_results = run_behavioral_analysis(X_val, y_pred_initial, feature_names)
    behavior_scores = behavior_results["behavior_scores"]

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: DECISION FUSION
    # ═══════════════════════════════════════════════════════════
    print("\nPHASE 4: DECISION FUSION (Open-set Recognition)")
    print("▓" * 70)
    anomaly_scores = get_vae_scores(vae, val_loader, device)
    
    fusion_results = run_fusion(
        y_prob=y_prob,
        anomaly_scores=anomaly_scores,
        behavior_scores=behavior_scores,
        y_true=y_true
    )
    fused_predictions = fusion_results["predictions"]

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: ADVANCED RISK SCORING
    # ═══════════════════════════════════════════════════════════
    print("\nPHASE 5: ADVANCED RISK SCORING")
    print("▓" * 70)
    # Train stats for Rarity
    X_train_sub = X[:20000]
    median = np.median(X_train_sub, axis=0)
    mad = np.median(np.abs(X_train_sub - median), axis=0) + 1e-6
    rarity_scores = compute_batch_rarity_scores(X_val, {"median": median, "mad": mad})
    
    risk_scores = compute_batch_advanced_risk(
        predictions=fused_predictions,
        confidences=fusion_results["confidences"],
        anomaly_scores=anomaly_scores,
        behavior_scores=behavior_scores,
        rarity_scores=rarity_scores,
        y_prob=y_prob
    )
    risk_stats = risk_summary(risk_scores)

    # ═══════════════════════════════════════════════════════════
    # PHASE 6 & 7: EVALUATION & SHAP
    # ═══════════════════════════════════════════════════════════
    print("\nPHASE 6 & 7: EVALUATION & EXPLAINABILITY")
    print("▓" * 70)
    
    # Standard classification metrics
    report_dict = classification_report(y_true, fused_predictions[:len(y_true)], 
                                        target_names=le.classes_, output_dict=True)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    
    # Plots
    cm_path = generate_performance_plots(y_true, fused_predictions[:len(y_true)], le, REPORTS_DIR)
    shap_path = run_shap_analysis(classifier, X[:1000], X_val, feature_names, device, REPORTS_DIR)

    # ═══════════════════════════════════════════════════════════
    # PHASE 8: FINAL SUMMARY & MARKDOWN REPORT
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - pipeline_start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s. Macro F1: {macro_f1:.4f}")
    
    summary = {
        "pipeline_version": "v2_deep",
        "system": "Mindgard Deep Hybrid IDS",
        "time_seconds": round(elapsed, 1),
        "macro_f1": macro_f1,
        "n_samples": len(y),
        "device": str(device)
    }
    
    # Save Artifacts
    torch.save(classifier.state_dict(), MODELS_DIR / "classifier.pth")
    torch.save(vae.state_dict(), MODELS_DIR / "vae.pth")
    joblib.dump({"le": le, "scaler": raw_scaler, "names": feature_names}, MODELS_DIR / "prepro.pkl")
    
    # Generate Markdown Report (similar to src/pipeline.py)
    full_report = generate_v2_markdown_report(summary, report_dict, risk_stats, fusion_results["stats"])
    (REPORTS_DIR / "final_report_v2.md").write_text(full_report)
    
    print(f"  📄 Full report generated: {REPORTS_DIR / 'final_report_v2.md'}")

def generate_v2_markdown_report(summary, metrics, risk_stats, fusion_stats):
    class_rows = ""
    for cls in CLASS_ORDER:
        if cls in metrics:
            m = metrics[cls]
            class_rows += f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support']):,} |\n"

    alert_rows = ""
    for level, count in fusion_stats.get("alert_levels", {}).items():
        total = max(sum(fusion_stats.get("alert_levels", {}).values()), 1)
        alert_rows += f"| {level} | {count:,} | {100*count/total:.1f}% |\n"

    return f"""# Mindgard V2 — Technical Report
## Deep Hybrid Intelligent Intrusion Detection System (RTX 3050 Optimized)

| Metric | Value |
|---|---|
| **Macro F1-Score** | **{summary['macro_f1']:.4f}** |
| Total Time | {summary['time_seconds']}s |
| Device | {summary['device']} |
| Zero-day detected | {fusion_stats.get('unknown_attacks', 0):,} |

### Intelligence Layers
- **Classifier**: CNN + Bi-LSTM + Multi-Head Attention
- **Anomaly**: Variational Autoencoder (VAE)
- **Behavior**: Rule-based Temporal Analysis

### Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
{class_rows}

### Alert Distribution
| Alert Level | Count | % |
|---|---|---|
{alert_rows}
"""

if __name__ == "__main__":
    run_production_v2_pipeline()
