"""
Mindgard — Data Cleaning Pipeline
Handles: loading, dropping useless columns, mapping attacks to 5 categories,
         encoding categoricals, handling inf/NaN, and scaling.

Supports KDD Cup 1999 format (labels end with period: "normal." → "normal").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ALL_COLUMNS, ATTACK_MAP, CAT_COLS, CLASS_ORDER, DROP_COLS,
    CLEANED_DIR, MODELS_DIR, TRAIN_PATH, TEST_PATH, SEED,
    DATASET_FORMAT,
)


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw KDD/NSL-KDD CSV with column names."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    print(f"  Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Strip trailing period from KDD Cup 99 labels (e.g., 'normal.' → 'normal')."""
    if DATASET_FORMAT == "kddcup99":
        df = df.copy()
        df["label"] = df["label"].str.rstrip(".")
        print("  Stripped trailing periods from labels (KDD Cup 99 format)")
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance and metadata columns."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped columns: {cols_to_drop}")
    return df


def map_attack_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map raw attack labels to 5 categories: normal, DoS, Probe, R2L, U2R."""
    df = df.copy()
    df["attack_cat"] = df["label"].map(ATTACK_MAP)

    # Handle unknown attacks (in test set — zero-day scenario)
    unknown_count = df["attack_cat"].isna().sum()
    if unknown_count > 0:
        unknown_labels = df[df["attack_cat"].isna()]["label"].unique()
        print(f"  ⚠ {unknown_count} rows with unmapped labels: {unknown_labels.tolist()}")
        print(f"    → Mapping to nearest known category based on attack nature")
        # Heuristic mapping for common KDD test-only attacks
        fallback_map = {
            "snmpgetattack": "R2L", "named": "R2L", "xlock": "R2L",
            "xsnoop": "R2L", "sendmail": "R2L", "httptunnel": "R2L",
            "worm": "R2L", "snmpguess": "R2L",
            "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
            "mailbomb": "DoS",
            "mscan": "Probe", "saint": "Probe",
            "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
        }
        for label in unknown_labels:
            cat = fallback_map.get(label, "R2L")  # default R2L for truly unknown
            df.loc[df["label"] == label, "attack_cat"] = cat
            print(f"      {label} → {cat}")

    remaining_na = df["attack_cat"].isna().sum()
    if remaining_na > 0:
        df["attack_cat"] = df["attack_cat"].fillna("R2L")
        print(f"  Filled {remaining_na} remaining NaN with 'R2L'")

    df = df.drop(columns=["label"])
    return df, unknown_count


def encode_categoricals(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode categorical columns, aligning train and test."""
    df_train["_split"] = "train"
    df_test["_split"] = "test"
    combined = pd.concat([df_train, df_test], ignore_index=True)

    combined = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)

    df_train_enc = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    df_test_enc = combined[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

    onehot_cols = [c for c in df_train_enc.columns if any(c.startswith(f"{cat}_") for cat in CAT_COLS)]
    print(f"  One-hot encoded: {len(onehot_cols)} new columns from {CAT_COLS}")

    return df_train_enc, df_test_enc, onehot_cols


def handle_inf_nan(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Replace inf with NaN, then fill NaN with 0."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols]).sum().sum()
    nan_count = df[num_cols].isna().sum().sum()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    print(f"  {name}: fixed {inf_count} inf + {nan_count} NaN values")
    return df


def encode_labels(
    y_train_raw: pd.Series, y_test_raw: pd.Series
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode string labels to integers with consistent class ordering."""
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    print(f"  Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return y_train, y_test, le


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Fit RobustScaler on train, transform both. Uses median/IQR — immune to outliers."""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    print(f"  Scaled features: RobustScaler (median/IQR) — fitted on train only")
    return X_train_scaled, X_test_scaled, scaler


def run_cleaning() -> dict[str, Any]:
    """
    Execute the full cleaning pipeline.
    Returns a dict with all artifacts for downstream use.
    
    For KDD Cup 1999: uses 10% corrected subset, splits into train/test.
    For NSL-KDD: uses separate KDDTrain+.txt / KDDTest+.txt.
    """
    print("=" * 60)
    print("CYBERSENTINEL — DATA CLEANING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Load ───
    print("\n[1/8] Loading raw data...")
    
    if DATASET_FORMAT == "kddcup99":
        # KDD Cup 99: load 10% corrected, then split into train/test
        print("  Dataset format: KDD Cup 1999 (corrected)")
        df_full = load_raw(TRAIN_PATH)  # 10% corrected = ~494K rows
        
        # Clean labels (strip trailing period)
        print("\n[2/8] Cleaning labels...")
        df_full = clean_labels(df_full)
        
        # Drop useless columns
        print("\n[3/8] Dropping useless columns...")
        df_full = drop_useless_columns(df_full)
        
        # Map attack categories BEFORE splitting
        print("\n[4/8] Mapping attack labels to 5 categories...")
        df_full, unknown_count = map_attack_categories(df_full)
        
        # Handle inf/NaN
        print("\n[5/8] Handling inf/NaN values...")
        df_full = handle_inf_nan(df_full, "Full")
        
        # Split into train/test (stratified by attack_cat)
        print("\n[6/8] Splitting into train/test (80/20, stratified)...")
        df_train, df_test = train_test_split(
            df_full, test_size=0.2, random_state=SEED,
            stratify=df_full["attack_cat"],
        )
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        print(f"  Train: {len(df_train):,} rows")
        print(f"  Test:  {len(df_test):,} rows")
        
    else:
        # NSL-KDD: separate train/test files
        print("  Dataset format: NSL-KDD")
        df_train = load_raw(TRAIN_PATH)
        df_test = load_raw(TEST_PATH)
        
        print("\n[2/8] Cleaning labels...")
        df_train = clean_labels(df_train)
        df_test = clean_labels(df_test)
        
        print("\n[3/8] Dropping useless columns...")
        df_train = drop_useless_columns(df_train)
        df_test = drop_useless_columns(df_test)
        
        print("\n[4/8] Mapping attack labels to 5 categories...")
        df_train, _ = map_attack_categories(df_train)
        df_test, unknown_count = map_attack_categories(df_test)
        
        print("\n[5/8] Handling inf/NaN values...")
        df_train = handle_inf_nan(df_train, "Train")
        df_test = handle_inf_nan(df_test, "Test")

    # ─── Encode categoricals ───
    print("\n[7/8] Encoding categorical variables...")
    df_train, df_test, onehot_cols = encode_categoricals(df_train, df_test)

    # ─── Separate features & labels, encode labels ───
    print("\n[7.5/8] Encoding labels...")
    y_train_raw = df_train["attack_cat"]
    y_test_raw = df_test["attack_cat"]
    feature_cols = [c for c in df_train.columns if c != "attack_cat"]
    X_train_df = df_train[feature_cols].astype(np.float32)
    X_test_df = df_test[feature_cols].astype(np.float32)
    y_train, y_test, le = encode_labels(y_train_raw, y_test_raw)

    # ─── Scale ───
    print("\n[8/8] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_df.values, X_test_df.values)

    # ─── Print summary ───
    print(f"\n{'─'*60}")
    print("CLEANING SUMMARY")
    print(f"{'─'*60}")
    print(f"  Features:       {X_train_scaled.shape[1]}")
    print(f"  Train samples:  {X_train_scaled.shape[0]:,}")
    print(f"  Test samples:   {X_test_scaled.shape[0]:,}")
    print(f"  Classes:        {le.classes_.tolist()}")
    for i, cls in enumerate(le.classes_):
        n_train = (y_train == i).sum()
        n_test = (y_test == i).sum()
        print(f"    {cls:10s}  train={n_train:>7,}  test={n_test:>7,}")

    # ─── Save cleaned data ───
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CLEANED_DIR / "X_train.npy", X_train_scaled)
    np.save(CLEANED_DIR / "X_test.npy", X_test_scaled)
    np.save(CLEANED_DIR / "y_train.npy", y_train)
    np.save(CLEANED_DIR / "y_test.npy", y_test)

    # Save preprocessing artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        "feature_columns": feature_cols,
        "onehot_columns": onehot_cols,
        "scaler": scaler,
        "label_encoder_classes": le.classes_.tolist(),
        "attack_map": ATTACK_MAP,
        "class_order": CLASS_ORDER,
        "n_features": X_train_scaled.shape[1],
        "dataset_format": DATASET_FORMAT,
    }
    joblib.dump(preprocessing, MODELS_DIR / "preprocessing.pkl")

    metadata = {
        "train_shape": list(X_train_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "n_features": int(X_train_scaled.shape[1]),
        "classes": le.classes_.tolist(),
        "feature_columns": feature_cols,
        "unknown_test_attacks": int(unknown_count),
        "dataset_format": DATASET_FORMAT,
    }
    (CLEANED_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n✅ Cleaned data saved to: {CLEANED_DIR}")
    print(f"✅ Preprocessing artifacts saved to: {MODELS_DIR / 'preprocessing.pkl'}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "label_encoder": le,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_cleaning()
