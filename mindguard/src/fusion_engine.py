"""
Mindgard — Decision Fusion Engine

WHY THIS EXISTS:
No single model catches everything. XGBoost misses zero-days. Isolation Forest
generates false positives on rare-but-known attacks. Behavioral analysis flags
normal power users as suspicious. The Fusion Engine combines all three intelligence
layers into a unified, stable decision.

Decision Logic (priority order):
  1. HIGH anomaly + LOW confidence → UNKNOWN_ATTACK (zero-day candidate)
  2. Optimized thresholds pass      → use threshold-based prediction
  3. Fallback                       → argmax of classifier probabilities

Secondary validation reduces false positives:
  - If classifier says "attack" but anomaly is LOW and behavior is LOW
    → downgrade alert level (likely false positive)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, MODELS_DIR, REPORTS_DIR

# Index for UNKNOWN class (appended after CLASS_ORDER)
UNKNOWN_CLASS = "UNKNOWN_ATTACK"
EXTENDED_CLASSES = CLASS_ORDER + [UNKNOWN_CLASS]

# ─── Fusion thresholds ───
# These are the decision boundaries for the fusion logic
ANOMALY_HIGH_THRESHOLD = 0.7      # Above this = "this connection is weird"
CONFIDENCE_LOW_THRESHOLD = 0.4    # Below this = "classifier isn't sure"
UNKNOWN_CONFIDENCE_THRESHOLD = 0.3  # Max prob below this = likely unknown


class FusionEngine:
    """
    Combines classifier, anomaly detector, and behavioral engine
    into a unified prediction with open-set recognition.
    """

    def __init__(
        self,
        optimized_thresholds: np.ndarray | None = None,
        anomaly_high: float = ANOMALY_HIGH_THRESHOLD,
        confidence_low: float = CONFIDENCE_LOW_THRESHOLD,
        unknown_threshold: float = UNKNOWN_CONFIDENCE_THRESHOLD,
    ):
        self.optimized_thresholds = optimized_thresholds
        self.anomaly_high = anomaly_high
        self.confidence_low = confidence_low
        self.unknown_threshold = unknown_threshold

        # Risk priority for tie-breaking (same as threshold_optimizer)
        self.class_priority = np.array([2, 3, 4, 5, 1], dtype=np.float32)  # DoS,Probe,R2L,U2R,normal

    def fuse(
        self,
        y_prob: np.ndarray,
        anomaly_scores: np.ndarray,
        behavior_scores: np.ndarray,
    ) -> dict[str, Any]:
        """
        Fused prediction for a batch of samples.

        Returns:
            predictions: int array (indices into EXTENDED_CLASSES)
            confidence: float array (fused confidence)
            is_unknown: bool array (True if classified as UNKNOWN_ATTACK)
            decision_path: list of strings explaining each decision
        """
        n_samples = len(y_prob)
        predictions = np.empty(n_samples, dtype=np.int64)
        confidences = np.empty(n_samples, dtype=np.float32)
        is_unknown = np.zeros(n_samples, dtype=bool)
        decision_paths = []

        for i in range(n_samples):
            probs = y_prob[i]
            anomaly = float(anomaly_scores[i])
            behavior = float(behavior_scores[i])
            max_prob = float(probs.max())
            max_class = int(probs.argmax())

            # ─── RULE 1: Zero-day detection (Extreme Anomaly Override) ───
            # High anomaly forcefully overrides model, regardless of confidence
            if anomaly > 0.85:
                predictions[i] = len(CLASS_ORDER)  # UNKNOWN_ATTACK index
                confidences[i] = anomaly
                is_unknown[i] = True
                decision_paths.append("RULE1_UNKNOWN: anomaly > 0.85 (strict override)")
                continue

            # ─── RULE 2: Open-set recognition ───
            if anomaly > 0.7 and max_prob < 0.8:
                predictions[i] = len(CLASS_ORDER)
                confidences[i] = anomaly
                is_unknown[i] = True
                decision_paths.append("RULE2_UNKNOWN: anomaly > 0.7 and confidence < 0.8")
                continue

            # ─── RULE 3: Optimized thresholds (strict hierarchy) ───
            if self.optimized_thresholds is not None:
                passing = probs >= self.optimized_thresholds
                if passing.any():
                    # Strict security hierarchy
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
                            confidences[i] = probs[cls_idx]
                            decision_paths.append(f"RULE3_THRESHOLD_HIERARCHY: {CLASS_ORDER[cls_idx]}")
                            break
                    continue

            # ─── RULE 4: Fallback to argmax ───
            predictions[i] = max_class
            confidences[i] = max_prob
            decision_paths.append(f"RULE4_ARGMAX: {CLASS_ORDER[max_class]}")

        return {
            "predictions": predictions,
            "confidences": confidences,
            "is_unknown": is_unknown,
            "decision_paths": decision_paths,
        }

    def validate_alerts(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        anomaly_scores: np.ndarray,
        behavior_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Secondary validation to reduce false positives.

        If the classifier predicts an attack but BOTH anomaly and behavior
        scores are low, the connection is probably normal → downgrade.

        WHY: XGBoost can misclassify normal traffic as Probe/R2L when
        features slightly resemble attack patterns. The anomaly detector
        acts as a sanity check.

        Returns adjusted alert levels: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL
        """
        n = len(predictions)
        alert_levels = np.ones(n, dtype=np.int32)  # Default: MEDIUM

        for i in range(n):
            pred = int(predictions[i])
            conf = float(confidences[i])
            anomaly = float(anomaly_scores[i])
            behavior = float(behavior_scores[i])

            # Unknown attacks always HIGH or CRITICAL
            if pred == len(CLASS_ORDER):  # UNKNOWN_ATTACK
                alert_levels[i] = 3 if anomaly > 0.8 else 2
                continue

            class_name = CLASS_ORDER[pred] if pred < len(CLASS_ORDER) else UNKNOWN_CLASS

            # Normal traffic
            if class_name == "normal":
                if anomaly > 0.7:
                    alert_levels[i] = 1  # MEDIUM — anomalous but classified normal
                else:
                    alert_levels[i] = 0  # LOW
                continue

            # Attack prediction validation
            supporting_evidence = 0
            if conf > 0.5:
                supporting_evidence += 1
            if anomaly > 0.4:
                supporting_evidence += 1
            if behavior > 0.3:
                supporting_evidence += 1

            if supporting_evidence >= 3:
                alert_levels[i] = 3  # CRITICAL — all signals agree
            elif supporting_evidence >= 2:
                alert_levels[i] = 2  # HIGH — strong evidence
            elif supporting_evidence >= 1:
                alert_levels[i] = 1  # MEDIUM — some evidence
            else:
                # Classifier says attack but no supporting evidence
                # → likely false positive, downgrade
                alert_levels[i] = 0  # LOW — probable false positive

        return alert_levels


ALERT_LEVEL_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_LEVEL_EMOJI = ["🟢", "🟡", "🟠", "🔴"]


def run_fusion(
    y_prob: np.ndarray,
    anomaly_scores: np.ndarray,
    behavior_scores: np.ndarray,
    optimized_thresholds: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Full decision fusion pipeline.
    Returns fused predictions, alert levels, and statistics.
    """
    print("\n" + "=" * 60)
    print("MINDGARD — DECISION FUSION ENGINE")
    print("=" * 60)

    engine = FusionEngine(optimized_thresholds=optimized_thresholds)

    # Fuse predictions
    print("\n  Fusing classifier + anomaly + behavior signals...")
    fusion_result = engine.fuse(y_prob, anomaly_scores, behavior_scores)
    predictions = fusion_result["predictions"]
    confidences = fusion_result["confidences"]
    is_unknown = fusion_result["is_unknown"]
    decision_paths = fusion_result["decision_paths"]

    # Validate and assign alert levels
    print("  Running secondary validation (false positive reduction)...")
    alert_levels = engine.validate_alerts(predictions, confidences, anomaly_scores, behavior_scores)

    # Statistics
    n_unknown = is_unknown.sum()
    n_total = len(predictions)

    print(f"\n  Fusion results:")
    print(f"    Total connections: {n_total:,}")
    print(f"    Unknown attacks:   {n_unknown:,} ({100*n_unknown/n_total:.1f}%)")

    # Decision path distribution
    path_counts = {}
    for p in decision_paths:
        rule = p.split(":")[0]
        path_counts[rule] = path_counts.get(rule, 0) + 1
    print(f"\n  Decision path distribution:")
    for rule, count in sorted(path_counts.items()):
        print(f"    {rule:30s}  {count:>7,}  ({100*count/n_total:.1f}%)")

    # Alert level distribution
    print(f"\n  Alert level distribution:")
    for level_idx, (name, emoji) in enumerate(zip(ALERT_LEVEL_NAMES, ALERT_LEVEL_EMOJI)):
        count = (alert_levels == level_idx).sum()
        print(f"    {emoji} {name:10s}  {count:>7,}  ({100*count/n_total:.1f}%)")

    # Per-class distribution (including UNKNOWN)
    print(f"\n  Prediction distribution:")
    for i, cls in enumerate(EXTENDED_CLASSES):
        count = (predictions == i).sum()
        if count > 0:
            print(f"    {cls:15s}  {count:>7,}  ({100*count/n_total:.1f}%)")

    # Save fusion stats
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fusion_stats = {
        "total": int(n_total),
        "unknown_attacks": int(n_unknown),
        "decision_paths": path_counts,
        "alert_levels": {name: int((alert_levels == i).sum()) for i, name in enumerate(ALERT_LEVEL_NAMES)},
    }
    stats_path = REPORTS_DIR / "fusion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(fusion_stats, f, indent=2)
    print(f"\n  ✅ Fusion stats saved: {stats_path}")

    return {
        "engine": engine,
        "predictions": predictions,
        "confidences": confidences,
        "is_unknown": is_unknown,
        "alert_levels": alert_levels,
        "decision_paths": decision_paths,
        "stats": fusion_stats,
    }
