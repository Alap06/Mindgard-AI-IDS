import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Absolute imports from original src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "lstm-smartgrid-IDS" / "src"
sys.path.append(str(SRC_DIR))

# Mandatory imports for the pipeline
from clean_data import clean_labels, drop_useless_columns, map_attack_categories, handle_inf_nan
from feature_engineering import engineer_features_df
from config import ALL_COLUMNS, CAT_COLS, CLASS_ORDER, DROP_COLS

class CyberDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def load_and_prepare_data(data_dir: str, use_full: bool = False):
    """
    Complete data preparation following the DATASET_CLEANING_GUIDE.md protocol.
    """
    dataset_path = Path(data_dir)
    train_file = "kddcup.data_10_percent_corrected"
    if use_full:
        train_file = "kddcup.data.corrected"
    
    file_path = dataset_path / train_file
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    print(f"  Reading dataset: {train_file}...")
    df = pd.read_csv(file_path, header=None, names=ALL_COLUMNS)
    
    # --- PHASE 1: Cleaning ---
    print("  [Phase 1] Cleaning according to guide...")
    if "label" in df.columns:
        df["label"] = df["label"].str.rstrip(".") # KDD format fix
    
    # Drop zero-variance
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Map categories
    # Re-implementing mapping to be safe if src import fails
    ATTACK_MAP = {
        "normal": "normal",
        "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
        "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
        "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L", "multihop": "R2L",
        "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
        "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R"
    }
    df["attack_cat"] = df["label"].map(ATTACK_MAP).fillna("R2L") # R2L fallback for unknowns
    
    # --- PHASE 2: Feature Engineering ---
    print("  [Phase 2] Domain-driven feature engineering...")
    try:
        from feature_engineering import engineer_features_df
        df = engineer_features_df(df)
    except ImportError:
        # Mini-fallback FE logic
        df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        df["total_error_rate"] = df["serror_rate"] + df["rerror_rate"]

    # Handle numeric health
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # One-Hot encoding
    df = pd.get_dummies(df, columns=CAT_COLS)
    
    # Separate and Encode
    y_raw = df["attack_cat"]
    X_df = df.drop(columns=["attack_cat", "label"], errors='ignore')
    
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)
    y = le.transform(y_raw)
    
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    # --- PHASE 4: Robust Scaling ---
    print("  [Phase 4] Robust Scaling (Median/IQR protocol)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Secondary scaler for VAE (deep models love [0,1])
    minmax = MinMaxScaler()
    X_minmax = minmax.fit_transform(X_scaled)

    return X_minmax, y, feature_names, le, scaler

def get_loaders(X, y, batch_size=2048, val_ratio=0.2):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )
    
    train_ds = CyberDataset(X_train, y_train)
    val_ds = CyberDataset(X_val, y_val)
    
    # Use pin_memory and num_workers for RTX 3050 optimization
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        pin_memory=True, num_workers=2
    )
    
    return train_loader, val_loader
