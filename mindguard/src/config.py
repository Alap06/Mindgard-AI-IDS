"""
Mindgard — Central Configuration
All paths, constants, and mappings in one place.
"""
from __future__ import annotations
from pathlib import Path

# ─── PATHS ───
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# KDD Cup 1999 dataset paths (from archive)
ARCHIVE_DIR = ROOT_DIR.parent / "archive (1)"
TRAIN_PATH = ARCHIVE_DIR / "kddcup.data_10_percent_corrected"
TEST_PATH = ARCHIVE_DIR / "kddcup.data.corrected"

CLEANED_DIR = DATA_DIR / "cleaned"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ─── DATASET FORMAT ───
# KDD Cup 1999: 41 features + label (no difficulty column)
# Labels have trailing period: "normal." → "normal"
DATASET_FORMAT = "kddcup99"  # "kddcup99" or "nsl-kdd"

# ─── RANDOM SEED ───
SEED = 42

# ─── COLUMN NAMES (41 features + label) ───
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
ALL_COLUMNS = FEATURE_NAMES + ["label"]

# ─── CATEGORICAL COLUMNS ───
CAT_COLS = ["protocol_type", "service", "flag"]

# ─── COLUMNS TO DROP ───
DROP_COLS = ["num_outbound_cmds"]

# ─── ATTACK TYPE → CATEGORY MAPPING ───
ATTACK_MAP = {
    # Normal
    "normal": "normal",
    # DoS — Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS",
    # Probe — Network reconnaissance
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L — Remote to Local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "worm": "R2L",
    # U2R — User to Root  
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

# ─── CLASS ORDER (for consistent label encoding) ───
CLASS_ORDER = ["DoS", "Probe", "R2L", "U2R", "normal"]

# ─── XGBOOST HYPERPARAMETERS ───
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(CLASS_ORDER),
    "eval_metric": "mlogloss",
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 1,
}

# ─── THRESHOLD OPTIMIZATION ───
OPTUNA_N_TRIALS = 300
VAL_SPLIT = 0.2

# ─── SHAP ───
SHAP_MAX_SAMPLES = 1000  # Max test samples for SHAP computation

# ─── RISK SCORING ───
# Weights reflect real-world impact: root compromise >> service disruption
CLASS_RISK_WEIGHT = {
    "U2R": 10.0,    # Root compromise = catastrophic
    "R2L": 7.0,     # Remote access = severe
    "Probe": 4.0,   # Reconnaissance = moderate
    "DoS": 3.0,     # Service disruption = moderate
    "normal": 0.0,  # No risk
}

# ─── API ───
API_HOST = "127.0.0.1"
API_PORT = 8000
