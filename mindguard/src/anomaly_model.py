"""
Mindgard — Anomaly Detection Module (Isolation Forest)

WHY THIS EXISTS:
XGBoost only classifies into KNOWN attack categories. But real networks face
zero-day attacks that don't match any training pattern. Isolation Forest detects
statistical outliers — connections that are "weird" regardless of class label.

A high anomaly score + low classifier confidence = likely UNKNOWN ATTACK.
This is the zero-day detection capability that separates good IDS from great IDS.

Architecture:
  - Trained on X_train (normal operating conditions)
  - Scores every connection on how "anomalous" it is
  - Scores normalized to [0, 1] where 1 = most anomalous
  - Integrated into the Decision Fusion Layer
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, SEED, CLASS_ORDER


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for zero-day attack detection.

    Isolation Forest works by randomly partitioning data — anomalies
    are easier to isolate (fewer splits needed), so they get higher scores.
    This is ideal for network traffic because:
      1. It's unsupervised — no attack labels needed
      2. It's fast — O(n log n) training, O(log n) inference
      3. It handles high-dimensional data well (our 120+ features)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        random_state: int = SEED,
    ):
        """
        Args:
            contamination: Expected proportion of anomalies in training data.
                KDD has ~20% attacks in the 10% subset, but we want sensitivity.
            n_estimators: Number of isolation trees. More = stable but slower.
            max_samples: Samples per tree. 'auto' = min(256, n_samples).
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            warm_start=False,
        )
        self.is_fitted = False
        self._score_min = 0.0
        self._score_max = 1.0

    def fit(self, X_train: np.ndarray) -> "AnomalyDetector":
        """
        Fit Isolation Forest on training data.

        WHY train on ALL data (not just normal):
        We want the forest to learn what "typical" data looks like.
        Anomalies in training help calibrate the contamination boundary.
        """
        print(f"  Training Isolation Forest ({self.model.n_estimators} trees)...")
        start = time.time()

        self.model.fit(X_train)

        # Calibrate score range for normalization
        # decision_function: normal is positive, anomalies are negative
        # We invert so anomalies get high scores.
        raw_scores = -self.model.decision_function(X_train)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())

        elapsed = time.time() - start
        self.is_fitted = True

        # Stats
        labels = self.model.predict(X_train)
        n_anomalies = (labels == -1).sum()
        pct = 100 * n_anomalies / len(X_train)
        print(f"  Isolation Forest trained in {elapsed:.1f}s")
        print(f"  Detected {n_anomalies:,} anomalies in training ({pct:.1f}%)")

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1].

        Isolation Forest's decision_function returns negative scores
        for anomalies. We invert and normalize:
          - 0.0 = perfectly normal
          - 1.0 = extremely anomalous

        WHY normalize: so the fusion engine can weight anomaly_score
        alongside confidence and behavior_score on the same scale.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector not fitted. Call .fit() first.")

        raw = -self.model.decision_function(X)

        score_range = self._score_max - self._score_min + 1e-8
        normalized = (raw - self._score_min) / score_range
        
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction: 1 = normal, -1 = anomaly."""
        return self.model.predict(X)

    def save(self, path: Path | None = None) -> Path:
        """Save fitted model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "anomaly_detector.pkl"
        joblib.dump({
            "model": self.model,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "is_fitted": self.is_fitted,
        }, path)
        print(f"  ✅ Anomaly detector saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "AnomalyDetector":
        """Load a fitted model from disk."""
        if path is None:
            path = MODELS_DIR / "anomaly_detector.pkl"
        data = joblib.load(path)
        detector = cls()
        detector.model = data["model"]
        detector._score_min = data["score_min"]
        detector._score_max = data["score_max"]
        detector.is_fitted = data["is_fitted"]
        return detector


def run_anomaly_detection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    contamination: float = 0.1,
) -> dict[str, Any]:
    """
    Full anomaly detection pipeline.
    Returns detector + train/test anomaly scores.
    """
    print("\n" + "=" * 60)
    print("MINDGARD — ANOMALY DETECTION (Isolation Forest)")
    print("=" * 60)

    print("\n  Filtering X_train for normal traffic only (prevent IF poisoning)...")
    mask_normal = (y_train == CLASS_ORDER.index("normal"))
    X_train_normal = X_train[mask_normal]
    
    detector = AnomalyDetector(contamination=contamination)
    # Fit strictly on normal data
    detector.fit(X_train_normal)

    print("\n  Scoring test set...")
    test_scores = detector.score(X_test)
    train_scores = detector.score(X_train)

    # Distribution stats
    print(f"\n  Anomaly score distribution (test):")
    print(f"    Mean:   {test_scores.mean():.4f}")
    print(f"    Median: {np.median(test_scores):.4f}")
    print(f"    Std:    {test_scores.std():.4f}")
    print(f"    >0.5:   {(test_scores > 0.5).sum():,} ({100*(test_scores > 0.5).mean():.1f}%)")
    print(f"    >0.8:   {(test_scores > 0.8).sum():,} ({100*(test_scores > 0.8).mean():.1f}%)")

    # Save
    detector.save()

    return {
        "detector": detector,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }
