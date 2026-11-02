"""
Mindgard — Data Audit & Exploratory Data Analysis
Run standalone: python src/data_audit.py
Generates: reports/data_audit.txt + reports/class_distribution.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ALL_COLUMNS, ATTACK_MAP, CAT_COLS, CLASS_ORDER,
    REPORTS_DIR, TRAIN_PATH, TEST_PATH, SEED, DATASET_FORMAT,
)


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw KDD/NSL-KDD data with proper column names."""
    return pd.read_csv(path, header=None, names=ALL_COLUMNS)


def run_audit() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    log("=" * 70)
    log("CYBERSENTINEL — DATA AUDIT REPORT")
    log("=" * 70)

    # ─── Load ───
    df_train = load_raw(TRAIN_PATH)
    df_test = load_raw(TEST_PATH)
    log(f"\nTrain shape: {df_train.shape}")
    log(f"Test shape:  {df_test.shape}")

    # ─── Missing values ───
    log(f"\n{'─'*40}")
    log("MISSING VALUES")
    log(f"{'─'*40}")
    train_missing = df_train.isnull().sum()
    missing_cols = train_missing[train_missing > 0]
    if len(missing_cols) == 0:
        log("  No missing values in train set ✓")
    else:
        log(f"  Missing columns:\n{missing_cols.to_string()}")

    test_missing = df_test.isnull().sum()
    missing_test = test_missing[test_missing > 0]
    if len(missing_test) == 0:
        log("  No missing values in test set ✓")
    else:
        log(f"  Missing columns (test):\n{missing_test.to_string()}")

    # ─── Data types ───
    log(f"\n{'─'*40}")
    log("DATA TYPES")
    log(f"{'─'*40}")
    log(f"  Numeric columns: {df_train.select_dtypes(include=[np.number]).shape[1]}")
    log(f"  Object columns:  {df_train.select_dtypes(include=['object']).shape[1]}")

    # ─── Categorical columns ───
    log(f"\n{'─'*40}")
    log("CATEGORICAL COLUMNS")
    log(f"{'─'*40}")
    for col in CAT_COLS:
        train_vals = df_train[col].unique()
        test_vals = df_test[col].unique()
        test_only = set(test_vals) - set(train_vals)
        log(f"  {col}:")
        log(f"    Train unique: {len(train_vals)}  |  Test unique: {len(test_vals)}")
        if test_only:
            log(f"    ⚠ Values in test but NOT in train: {test_only}")

    # ─── Zero-variance features ───
    log(f"\n{'─'*40}")
    log("ZERO-VARIANCE FEATURES")
    log(f"{'─'*40}")
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    zero_var = [c for c in num_cols if df_train[c].std() == 0]
    if zero_var:
        for c in zero_var:
            log(f"  ✗ {c} (constant = {df_train[c].iloc[0]})")
    else:
        log("  None found")

    # ─── Near-zero variance ───
    near_zero = [c for c in num_cols if 0 < df_train[c].std() < 0.01 and c not in zero_var]
    if near_zero:
        log(f"\n  Near-zero variance:")
        for c in near_zero:
            log(f"    {c}: std={df_train[c].std():.6f}, unique={df_train[c].nunique()}")

    # ─── Raw label distribution (train) ───
    log(f"\n{'─'*40}")
    log("RAW LABEL DISTRIBUTION (TRAIN)")
    log(f"{'─'*40}")
    train_labels = df_train["label"].value_counts()
    for label, count in train_labels.items():
        pct = 100.0 * count / len(df_train)
        log(f"  {label:30s}  {count:>7d}  ({pct:5.2f}%)")

    # ─── Category distribution ───
    log(f"\n{'─'*40}")
    log("ATTACK CATEGORY DISTRIBUTION")
    log(f"{'─'*40}")
    # Strip trailing period from KDD Cup 99 labels (e.g., "normal." → "normal")
    if DATASET_FORMAT == "kddcup99":
        df_train["label"] = df_train["label"].str.rstrip(".")
        df_test["label"] = df_test["label"].str.rstrip(".")
    df_train["attack_cat"] = df_train["label"].map(ATTACK_MAP)
    df_test["attack_cat"] = df_test["label"].map(ATTACK_MAP)

    unknown_train = df_train["attack_cat"].isna().sum()
    unknown_test = df_test["attack_cat"].isna().sum()
    log(f"  Unmapped labels — Train: {unknown_train}, Test: {unknown_test}")

    if unknown_test > 0:
        unmapped = df_test[df_test["attack_cat"].isna()]["label"].unique()
        log(f"  Unmapped test labels: {unmapped.tolist()}")

    for split_name, df in [("Train", df_train), ("Test", df_test)]:
        cat_counts = df["attack_cat"].value_counts()
        log(f"\n  {split_name}:")
        for cat in CLASS_ORDER:
            count = cat_counts.get(cat, 0)
            pct = 100.0 * count / len(df)
            bar = "█" * int(pct / 2)
            log(f"    {cat:10s}  {count:>7d}  ({pct:5.2f}%)  {bar}")

    # ─── Numerical feature statistics ───
    log(f"\n{'─'*40}")
    log("NUMERICAL FEATURE RANGES (top outlier candidates)")
    log(f"{'─'*40}")
    desc = df_train[num_cols].describe().T
    extreme = desc.nlargest(10, "max")[["min", "mean", "std", "max"]]
    log(extreme.to_string())

    # ─── Inf values check ───
    log(f"\n{'─'*40}")
    log("INF VALUES CHECK")
    log(f"{'─'*40}")
    inf_train = np.isinf(df_train.select_dtypes(include=[np.number])).sum().sum()
    inf_test = np.isinf(df_test.select_dtypes(include=[np.number])).sum().sum()
    log(f"  Train inf count: {inf_train}")
    log(f"  Test inf count:  {inf_test}")

    # ─── Duplicate rows ───
    log(f"\n{'─'*40}")
    log("DUPLICATE ROWS")
    log(f"{'─'*40}")
    dup_train = df_train.duplicated().sum()
    dup_test = df_test.duplicated().sum()
    log(f"  Train duplicates: {dup_train} ({100*dup_train/len(df_train):.2f}%)")
    log(f"  Test duplicates:  {dup_test} ({100*dup_test/len(df_test):.2f}%)")

    # ─── Save text report ───
    report_path = REPORTS_DIR / "data_audit.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"\n✅ Audit report saved to: {report_path}")

    # ─── Generate class distribution plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (split_name, df) in zip(axes, [("Train", df_train), ("Test", df_test)]):
        cat_counts = df["attack_cat"].fillna("Unknown").value_counts()
        ordered = [c for c in CLASS_ORDER if c in cat_counts.index]
        if "Unknown" in cat_counts.index:
            ordered.append("Unknown")
        colors = ["#e74c3c", "#f39c12", "#9b59b6", "#e67e22", "#2ecc71", "#95a5a6"]
        bars = ax.bar(
            range(len(ordered)),
            [cat_counts[c] for c in ordered],
            color=colors[: len(ordered)],
        )
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=0, fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} Set — Class Distribution", fontweight="bold")
        ax.set_yscale("log")
        # Add count labels
        for bar, cat in zip(bars, ordered):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1,
                f"{cat_counts[cat]:,}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    fig.suptitle("Mindgard — KDD Dataset Class Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plot_path = REPORTS_DIR / "class_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"✅ Class distribution plot saved to: {plot_path}")


if __name__ == "__main__":
    run_audit()
