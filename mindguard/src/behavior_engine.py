"""
Mindgard — Behavioral Analysis Engine

WHY THIS EXISTS:
Individual connections look innocent in isolation. But ATTACK CAMPAIGNS
have temporal patterns: port scans hit many ports in sequence, brute force
attacks generate rapid login failures, DoS floods spike connection counts.

This engine computes behavioral signals from the feature vector itself
(since we don't have raw packet timestamps), extracting:
  1. Connection frequency patterns (count, srv_count)
  2. Error rate concentration (how persistent are errors?)
  3. Service diversity (scanning behavior)
  4. Entropy of categorical distributions (randomness = suspicious)

These behavioral scores feed into the risk scoring engine alongside
the classifier confidence and anomaly score.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_ORDER


class StatefulBehaviorEngine:
    """
    Streaming sliding-window engine for real-time behavior tracking.
    This tracks real temporal connections per IP over a fixed window (e.g. 10s).
    """
    def __init__(self, window_sec: int = 10):
        self.window_sec = window_sec
        # Maps src_ip -> deque of timestamps
        self.state = defaultdict(deque)

    def update(self, src_ip: str, timestamp: float) -> float:
        """
        Record a connection event and return recent connections / second.
        """
        now = timestamp
        q = self.state[src_ip]

        q.append(now)

        # Remove old events outside the window
        while q and now - q[0] > self.window_sec:
            q.popleft()

        return len(q) / float(self.window_sec)


def compute_connection_frequency_score(X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """
    Score based on connection frequency indicators.

    HIGH frequency connections are suspicious:
      - High `count` = many connections to same host in 2-second window
      - High `srv_count` = many connections to same service
      - Combination of both = flooding behavior (DoS/Probe)

    Score: 0 = low frequency (normal), 1 = extreme frequency (attack-like)
    """
    scores = np.zeros(len(X), dtype=np.float32)

    if "count" in feature_cols:
        idx = feature_cols.index("count")
        count_vals = X[:, idx]
        # Normalize: higher count = higher score
        count_max = max(np.percentile(np.abs(count_vals), 99), 1e-8)
        scores += np.clip(np.abs(count_vals) / count_max, 0, 1) * 0.5

    if "srv_count" in feature_cols:
        idx = feature_cols.index("srv_count")
        srv_vals = X[:, idx]
        srv_max = max(np.percentile(np.abs(srv_vals), 99), 1e-8)
        scores += np.clip(np.abs(srv_vals) / srv_max, 0, 1) * 0.5

    return np.clip(scores, 0, 1)


def compute_error_concentration_score(X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """
    Score based on error rate patterns.

    HIGH error rates indicate scanning/attack behavior:
      - serror_rate: SYN errors (port scanning signature)
      - rerror_rate: REJ errors (connection refused = probing)
      - Combined host-level errors = persistent attack

    Score: 0 = no errors (normal), 1 = extreme error concentration
    """
    error_features = [
        "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ]

    scores = np.zeros(len(X), dtype=np.float32)
    n_found = 0

    for feat in error_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            # Error rates are already 0-1 in raw data, but may be scaled
            vals = np.abs(X[:, idx])
            val_max = max(np.percentile(vals, 99), 1e-8)
            scores += np.clip(vals / val_max, 0, 1)
            n_found += 1

    if n_found > 0:
        scores /= n_found

    return np.clip(scores, 0, 1)


def compute_service_diversity_score(X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """
    Score based on service access diversity.

    Normal users access a few services repeatedly (same_srv_rate ≈ 1).
    Attackers (especially probes) access many different services
    (same_srv_rate ≈ 0, diff_srv_rate ≈ 1).

    Score: 0 = consistent service use, 1 = high diversity (suspicious)
    """
    scores = np.zeros(len(X), dtype=np.float32)

    if "same_srv_rate" in feature_cols:
        idx = feature_cols.index("same_srv_rate")
        # Low same_srv_rate = high diversity = suspicious
        vals = X[:, idx]
        # After scaling, negative values mean below-median same_srv_rate
        val_max = max(np.percentile(np.abs(vals), 99), 1e-8)
        normalized = np.clip(vals / val_max, -1, 1)
        # Invert: low same_srv_rate → high score
        scores += (1.0 - np.clip((normalized + 1) / 2, 0, 1)) * 0.5

    if "diff_srv_rate" in feature_cols:
        idx = feature_cols.index("diff_srv_rate")
        vals = X[:, idx]
        val_max = max(np.percentile(np.abs(vals), 99), 1e-8)
        scores += np.clip(np.abs(vals) / val_max, 0, 1) * 0.5

    return np.clip(scores, 0, 1)


def compute_privilege_escalation_score(X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """
    Score based on privilege escalation indicators.

    U2R attacks specifically target root access:
      - root_shell = got root shell
      - su_attempted = tried to su
      - num_root = root access operations
      - num_compromised = compromised conditions

    ANY non-zero value in these features is highly suspicious.

    Score: 0 = no privilege activity, 1 = active escalation
    """
    priv_features = ["root_shell", "su_attempted", "num_root", "num_compromised"]

    scores = np.zeros(len(X), dtype=np.float32)
    n_found = 0

    for feat in priv_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            vals = np.abs(X[:, idx])
            # Binary-ish: any non-zero scaled value is suspicious
            scores += (vals > 0.01).astype(np.float32)
            n_found += 1

    if n_found > 0:
        scores /= n_found

    return np.clip(scores, 0, 1)


def compute_behavior_score(
    X: np.ndarray,
    feature_cols: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Combined behavioral score from all sub-engines.

    Final score is a weighted combination of:
      - Connection frequency (flooding detection)
      - Error concentration (scanning detection)
      - Service diversity (probe detection)
      - Privilege escalation (U2R detection)

    Score: 0 = normal behavior, 1 = highly suspicious behavior
    """
    if weights is None:
        weights = {
            "frequency": 0.25,
            "errors": 0.30,
            "diversity": 0.20,
            "privilege": 0.25,
        }

    freq_score = compute_connection_frequency_score(X, feature_cols)
    error_score = compute_error_concentration_score(X, feature_cols)
    diversity_score = compute_service_diversity_score(X, feature_cols)
    priv_score = compute_privilege_escalation_score(X, feature_cols)

    combined = (
        weights["frequency"] * freq_score
        + weights["errors"] * error_score
        + weights["diversity"] * diversity_score
        + weights["privilege"] * priv_score
    )

    return np.clip(combined, 0, 1).astype(np.float32)


def run_behavioral_analysis(
    X_test: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Full behavioral analysis pipeline.
    Returns behavior scores + per-component breakdown.
    """
    print("\n" + "=" * 60)
    print("MINDGARD — BEHAVIORAL ANALYSIS ENGINE")
    print("=" * 60)

    print("\n  Computing behavioral sub-scores...")
    freq = compute_connection_frequency_score(X_test, feature_cols)
    errors = compute_error_concentration_score(X_test, feature_cols)
    diversity = compute_service_diversity_score(X_test, feature_cols)
    privilege = compute_privilege_escalation_score(X_test, feature_cols)
    combined = compute_behavior_score(X_test, feature_cols)

    print(f"\n  Behavioral score distribution (test):")
    print(f"    Combined: mean={combined.mean():.4f}, max={combined.max():.4f}")
    print(f"    Frequency:  mean={freq.mean():.4f}")
    print(f"    Errors:     mean={errors.mean():.4f}")
    print(f"    Diversity:  mean={diversity.mean():.4f}")
    print(f"    Privilege:  mean={privilege.mean():.4f}")

    # Per-class behavior scores
    print(f"\n  Mean behavior score by predicted class:")
    for i, cls in enumerate(CLASS_ORDER):
        mask = y_pred == i
        if mask.sum() > 0:
            print(f"    {cls:10s}  behavior={combined[mask].mean():.4f}  n={mask.sum():,}")

    return {
        "behavior_scores": combined,
        "sub_scores": {
            "frequency": freq,
            "errors": errors,
            "diversity": diversity,
            "privilege": privilege,
        },
    }
