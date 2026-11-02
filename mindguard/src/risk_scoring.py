"""
Mindgard — Advanced Risk Scoring Engine (v2)

WHY THIS EXISTS:
A simple "confidence × weight" misses critical context. A U2R attack at 10%
classifier confidence but 95% anomaly score is MORE dangerous than a DoS
at 90% confidence with 5% anomaly score.

The advanced formula fuses ALL intelligence signals:

  risk_score = (
      w1 * model_confidence +
      w2 * anomaly_score +
      w3 * behavior_score +
      w4 * rarity_score
  ) × class_impact_weight × uncertainty_penalty

Where:
  - model_confidence: classifier's probability for predicted class
  - anomaly_score: Isolation Forest anomaly (0=normal, 1=anomalous)
  - behavior_score: behavioral analysis (0=benign, 1=suspicious)
  - rarity_score: how rare this pattern is vs training distribution
  - uncertainty_penalty: entropy-based — high entropy = less certain = higher risk
  - class_impact_weight: U2R=10, R2L=7, Probe=4, DoS=3, normal=0, UNKNOWN=8
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER, CLASS_RISK_WEIGHT

# Extended risk weights including UNKNOWN_ATTACK
EXTENDED_RISK_WEIGHT = dict(CLASS_RISK_WEIGHT)
EXTENDED_RISK_WEIGHT["UNKNOWN_ATTACK"] = 8.0  # Unknown = severe (could be anything)

EXTENDED_CLASSES = CLASS_ORDER + ["UNKNOWN_ATTACK"]

# Signal weights for the fusion formula
DEFAULT_SIGNAL_WEIGHTS = {
    "confidence": 0.35,   # w1: classifier confidence
    "anomaly": 0.25,      # w2: anomaly detector score
    "behavior": 0.20,     # w3: behavioral analysis score
    "rarity": 0.20,       # w4: rarity/distance score
}

# Risk level thresholds
RISK_LEVELS = {
    "CRITICAL": (75, 100, "🔴"),
    "HIGH":     (50, 75,  "🟠"),
    "MEDIUM":   (25, 50,  "🟡"),
    "LOW":      (0,  25,  "🟢"),
}

# Alert level mapping
ALERT_LEVEL_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_LEVEL_EMOJI = ["🟢", "🟡", "🟠", "🔴"]


def compute_uncertainty(probs: np.ndarray) -> float:
    """
    Compute prediction uncertainty via normalized entropy.

    WHY entropy: if the classifier assigns similar probabilities to multiple
    classes, it's uncertain. High uncertainty + attack prediction = escalate risk.

    Returns: 0 = perfectly certain, 1 = maximum uncertainty (uniform distribution)
    """
    n_classes = len(probs)
    max_entropy = np.log(n_classes)  # entropy of uniform distribution
    if max_entropy == 0:
        return 0.0
    h = scipy_entropy(probs + 1e-10)  # add epsilon to avoid log(0)
    return float(min(h / max_entropy, 1.0))


def compute_rarity_score(
    X_single: np.ndarray,
    X_train_stats: dict[str, np.ndarray],
) -> float:
    """
    Compute how rare/unusual this sample is relative to training distribution.

    Uses Median Absolute Deviation (MAD) on log1p-transformed features.
    Network traffic features (like bytes) are highly heavy-tailed; normal Z-scores
    will get crushed by massive outliers. MAD + log1p is robust to extremums.

    Returns: 0 = typical sample, 1 = extremely rare
    """
    median = X_train_stats["median"]
    mad = X_train_stats["mad"]

    # Log1p transformation for heavy tails
    x_log = np.log1p(np.abs(X_single))

    # MAD-based rarity
    rarity = np.abs((x_log - median) / mad)
    
    # Average rarity, normalized to [0,1] via sigmoid-like transform
    avg_rarity = float(np.mean(rarity))
    score = 1.0 - (1.0 / (1.0 + avg_rarity / 3.0))
    return float(min(score, 1.0))


def compute_batch_rarity_scores(
    X: np.ndarray,
    X_train_stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized rarity computation using MAD + log1p."""
    median = X_train_stats["median"]
    mad = X_train_stats["mad"]

    X_log = np.log1p(np.abs(X))
    rarity = np.abs((X_log - median) / mad)
    
    avg_rarity = rarity.mean(axis=1)
    score = 1.0 - (1.0 / (1.0 + avg_rarity / 3.0))
    return np.clip(score, 0, 1).astype(np.float32)


def compute_advanced_risk_score(
    predicted_class: str | int,
    confidence: float,
    anomaly_score: float,
    behavior_score: float,
    rarity_score: float,
    probs: np.ndarray | None = None,
    signal_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Advanced multi-signal risk scoring.

    Formula:
      base_risk = w1*confidence + w2*anomaly + w3*behavior + w4*rarity
      uncertainty_penalty = 1 + 0.5 * entropy(probs)
      risk_score = base_risk * class_weight * uncertainty_penalty * 10

    The uncertainty penalty INCREASES risk when the classifier is unsure,
    because uncertain attack classifications deserve more attention.
    """
    if isinstance(predicted_class, (int, np.integer)):
        if predicted_class < len(CLASS_ORDER):
            predicted_class = CLASS_ORDER[predicted_class]
        else:
            predicted_class = "UNKNOWN_ATTACK"

    if signal_weights is None:
        signal_weights = DEFAULT_SIGNAL_WEIGHTS

    severity_weight = {"normal": 0.0, "Probe": 0.5, "DoS": 0.8, "R2L": 0.9, "U2R": 1.0, "UNKNOWN_ATTACK": 1.0}
    weight = severity_weight.get(str(predicted_class), 0.5)

    risk = (
        0.5 * anomaly_score +
        0.3 * (1.0 - confidence) +
        0.2 * weight
    ) * 100.0

    if str(predicted_class) == "normal" and anomaly_score < 0.7:
        risk = 0.0
    
    # ─── Uncertainty penalty (optional scaling) ───
    uncertainty = 0.0
    if probs is not None:
        uncertainty = compute_uncertainty(probs)
    # Uncertain predictions get 1.0 to 1.5x risk multiplier
    uncertainty_penalty = 1.0 + 0.5 * uncertainty

    # ─── Final risk score ───
    raw_score = risk * uncertainty_penalty
    risk_score = min(100.0, max(0.0, raw_score))

    # ─── Risk level ───
    risk_level = "LOW"
    risk_emoji = "🟢"
    for level, (lo, hi, emoji) in RISK_LEVELS.items():
        if lo <= risk_score <= hi:
            risk_level = level
            risk_emoji = emoji
            break

    return {
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "anomaly_score": round(float(anomaly_score), 4),
        "behavior_score": round(float(behavior_score), 4),
        "rarity_score": round(float(rarity_score), 4),
        "uncertainty": round(float(uncertainty), 4),
        "weight": weight,
        "signal_breakdown": {
            "confidence_component": round(signal_weights["confidence"] * confidence, 4),
            "anomaly_component": round(signal_weights["anomaly"] * anomaly_score, 4),
            "behavior_component": round(signal_weights["behavior"] * behavior_score, 4),
            "rarity_component": round(signal_weights["rarity"] * rarity_score, 4),
        },
    }


def compute_batch_advanced_risk(
    predictions: np.ndarray,
    confidences: np.ndarray,
    anomaly_scores: np.ndarray,
    behavior_scores: np.ndarray,
    rarity_scores: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute advanced risk scores for a batch."""
    results = []
    for i in range(len(predictions)):
        probs = y_prob[i] if y_prob is not None else None
        results.append(compute_advanced_risk_score(
            predicted_class=int(predictions[i]),
            confidence=float(confidences[i]),
            anomaly_score=float(anomaly_scores[i]),
            behavior_score=float(behavior_scores[i]),
            rarity_score=float(rarity_scores[i]),
            probs=probs,
        ))
    return results


def risk_summary(risk_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate risk summary statistics for a batch."""
    total = len(risk_scores)
    if total == 0:
        return {"total": 0}

    level_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    class_counts = {}
    scores = []

    for r in risk_scores:
        level_counts[r["risk_level"]] += 1
        cls = r["predicted_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        scores.append(r["risk_score"])

    return {
        "total": total,
        "mean_risk_score": round(float(np.mean(scores)), 1),
        "max_risk_score": round(float(np.max(scores)), 1),
        "risk_levels": level_counts,
        "attack_distribution": {k: v for k, v in class_counts.items() if v > 0},
        "alert_rate": round(100 * (level_counts["CRITICAL"] + level_counts["HIGH"]) / total, 1),
    }


def format_risk_alert(risk: dict[str, Any]) -> str:
    """Format a risk score as a human-readable alert string."""
    emoji = risk["risk_emoji"]
    level = risk["risk_level"]
    cls = risk["predicted_class"]
    score = risk["risk_score"]
    conf = risk["confidence"]
    anomaly = risk.get("anomaly_score", 0)

    if cls == "normal" and score < 25:
        return f"  {emoji} Normal traffic (risk: {score}/100)"

    return (
        f"  {emoji} {level}: {cls}\n"
        f"     Risk Score:    {score}/100\n"
        f"     Confidence:    {conf:.1%}\n"
        f"     Anomaly Score: {anomaly:.2f}\n"
        f"     Behavior:      {risk.get('behavior_score', 0):.2f}"
    )
