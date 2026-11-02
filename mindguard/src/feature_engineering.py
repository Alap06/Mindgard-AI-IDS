"""
Mindgard — Feature Engineering Module
Domain-driven features that capture network attack signatures.
These features are YOUR unique contribution — not Googleable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expert-crafted features to a DataFrame (before encoding/scaling).
    Call this BEFORE one-hot encoding and scaling.
    """
    df = df.copy()

    # ── RATIO FEATURES ──
    # DoS: huge src_bytes, tiny dst_bytes → high ratio
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    # Error rate aggregation — probes and DoS generate errors
    df["total_error_rate"] = df["serror_rate"] + df["rerror_rate"]
    df["total_srv_error_rate"] = df["srv_serror_rate"] + df["srv_rerror_rate"]

    # Host-level anomaly — R2L/U2R target specific hosts
    df["host_srv_ratio"] = df["dst_host_srv_count"] / (df["dst_host_count"] + 1)

    # Service diversity — probes hit many different services
    df["same_diff_gap"] = df["same_srv_rate"] - df["diff_srv_rate"]

    # ── LOG TRANSFORMS ──
    # Compress heavy-tailed distributions (src_bytes: 0 to 1.3B)
    df["log_src_bytes"] = np.log1p(df["src_bytes"])
    df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
    df["log_duration"] = np.log1p(df["duration"])

    # ── INTERACTION FEATURES ──
    # High count + high error = likely DoS flooding
    df["count_x_error"] = df["count"] * df["serror_rate"]

    # Connection behavior signature
    df["srv_count_ratio"] = df["srv_count"] / (df["count"] + 1)

    # Host connection concentration
    df["host_port_diff"] = (
        df["dst_host_same_src_port_rate"] - df["dst_host_srv_diff_host_rate"]
    )

    # ── HOST ERROR PATTERN ──
    # Host-level error pattern (persistent attack indicator)
    df["host_error_total"] = df["dst_host_serror_rate"] + df["dst_host_rerror_rate"]

    return df


def engineer_features_array(
    X: np.ndarray, feature_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    Add engineered features to a scaled numpy array.
    Call this AFTER scaling if you want to engineer on scaled data.
    
    Alternative approach — works on numpy arrays instead of DataFrames.
    Use this when you already have X_train/X_test as numpy arrays.
    """
    df = pd.DataFrame(X, columns=feature_cols)

    # Find original column indices (before one-hot encoding, some might not exist)
    # Only engineer features from columns that exist
    new_features = {}

    if "src_bytes" in feature_cols and "dst_bytes" in feature_cols:
        new_features["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"].abs() + 1)

    if "serror_rate" in feature_cols and "rerror_rate" in feature_cols:
        new_features["total_error_rate"] = df["serror_rate"] + df["rerror_rate"]

    if "srv_serror_rate" in feature_cols and "srv_rerror_rate" in feature_cols:
        new_features["total_srv_error_rate"] = df["srv_serror_rate"] + df["srv_rerror_rate"]

    if "dst_host_srv_count" in feature_cols and "dst_host_count" in feature_cols:
        new_features["host_srv_ratio"] = df["dst_host_srv_count"] / (df["dst_host_count"].abs() + 1)

    if "same_srv_rate" in feature_cols and "diff_srv_rate" in feature_cols:
        new_features["same_diff_gap"] = df["same_srv_rate"] - df["diff_srv_rate"]

    if "count" in feature_cols and "serror_rate" in feature_cols:
        new_features["count_x_error"] = df["count"] * df["serror_rate"]

    if "srv_count" in feature_cols and "count" in feature_cols:
        new_features["srv_count_ratio"] = df["srv_count"] / (df["count"].abs() + 1)

    if "dst_host_serror_rate" in feature_cols and "dst_host_rerror_rate" in feature_cols:
        new_features["host_error_total"] = df["dst_host_serror_rate"] + df["dst_host_rerror_rate"]

    if not new_features:
        print("  ⚠ No features could be engineered (column names may not match)")
        return X, feature_cols

    new_df = pd.DataFrame(new_features)
    new_cols = list(new_features.keys())

    # Replace inf/NaN
    new_df = new_df.replace([np.inf, -np.inf], 0).fillna(0)

    X_new = np.hstack([X, new_df.values.astype(np.float32)])
    all_cols = feature_cols + new_cols

    print(f"  Engineered {len(new_cols)} new features: {new_cols}")
    print(f"  Total features: {len(feature_cols)} → {len(all_cols)}")

    return X_new, all_cols


# ── Feature descriptions for the report ──
FEATURE_DESCRIPTIONS = {
    "bytes_ratio": "src_bytes / (dst_bytes + 1) — Asymmetric traffic indicator (DoS, exfiltration)",
    "total_error_rate": "serror_rate + rerror_rate — Connection error concentration (Probe, DoS)",
    "total_srv_error_rate": "srv_serror_rate + srv_rerror_rate — Service-level error pattern",
    "host_srv_ratio": "dst_host_srv_count / (dst_host_count + 1) — Host service concentration (R2L)",
    "same_diff_gap": "same_srv_rate - diff_srv_rate — Service access pattern anomaly (Probe)",
    "log_src_bytes": "log(1 + src_bytes) — Compressed source volume",
    "log_dst_bytes": "log(1 + dst_bytes) — Compressed destination volume",
    "log_duration": "log(1 + duration) — Compressed duration",
    "count_x_error": "count × serror_rate — Flooding with errors indicator (DoS)",
    "srv_count_ratio": "srv_count / (count + 1) — Service request concentration",
    "host_port_diff": "dst_host_same_src_port_rate - dst_host_srv_diff_host_rate — Port targeting pattern",
    "host_error_total": "dst_host_serror_rate + dst_host_rerror_rate — Host-level persistent error",
}
