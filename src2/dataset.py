import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, RobustScaler
import sys
from pathlib import Path

# Add the original src to path to reuse cleaning logic
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

try:
    from clean_data import clean_labels, drop_useless_columns, map_attack_categories, handle_inf_nan
    from config import ALL_COLUMNS, CAT_COLS, DROP_COLS, CLASS_ORDER
except ImportError:
    # Fallback if src is not accessible or needs direct definitions
    from .src_fallback import ALL_COLUMNS, CAT_COLS, DROP_COLS, CLASS_ORDER

class KDDDataset(Dataset):
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

def prepare_kaggle_data(dataset_path: str = None):
    """
    Cleans and prepares KDD data from a Kaggle path or local directory.
    """
    if dataset_path is None:
        # Try to find locally first in the project structure
        local_data = ROOT_DIR.parent / "dataset"
        if local_data.exists():
            dataset_path = str(local_data)
        else:
            print("Local dataset not found. Downloading via kagglehub...")
            dataset_path = kagglehub.dataset_download("galaxyh/kdd-cup-1999-data")

    # Find the data file
    possible_files = [
        'kddcup.data_10_percent_corrected',
        'kddcup.data_10_percent',
        'kddcup.data.corrected'
    ]
    
    file_path = None
    for f_name in possible_files:
        test_path = os.path.join(dataset_path, f_name)
        if os.path.exists(test_path):
            file_path = test_path
            break
            
    if not file_path:
        # Fallback: find any large file that isn't .gz or .names
        all_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        all_files = [f for f in all_files if not f.endswith(('.gz', '.names', '.txt'))]
        if all_files:
            file_path = os.path.join(dataset_path, all_files[0])
    
    if not file_path:
        raise FileNotFoundError(f"Could not find a valid data file in {dataset_path}")

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, header=None, names=ALL_COLUMNS)
    
    # 1. Cleaning
    df = clean_labels(df)
    df = drop_useless_columns(df)
    df, _ = map_attack_categories(df)
    
    # 2. FEATURE ENGINEERING (Domain-driven)
    print("\n[Phase 2] Engineering expert features...")
    # Import engineer_features_df from src if available
    try:
        from feature_engineering import engineer_features_df
        df = engineer_features_df(df)
    except ImportError:
        pass # Fallback to original features if eng script is missing
        
    df = handle_inf_nan(df, "Dataset")
    
    # 3. Encoding
    # Separate labels
    y_raw = df["attack_cat"]
    X_raw = df.drop(columns=["attack_cat"])
    
    # One-hot encoding
    X_df = pd.get_dummies(X_raw, columns=CAT_COLS)
    
    # Label encoding
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)
    y = le.transform(y_raw)
    
    feature_cols = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)
    
    # 4. Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, le, scaler

def get_dataloaders(X, y, batch_size=512, split_ratio=0.2):
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=split_ratio, random_state=42, stratify=y
    )
    
    train_ds = KDDDataset(X_train, y_train)
    val_ds = KDDDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
