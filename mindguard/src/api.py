"""
Mindgard — FastAPI Real-Time Hybrid IDS Backend (v2)
Run: python src/api.py

Every prediction returns:
  - predicted_class (including UNKNOWN_ATTACK)
  - confidence
  - anomaly_score
  - behavior_score
  - risk_score (0-100)
  - explanation (top SHAP features)
  - alert_level (LOW / MEDIUM / HIGH / CRITICAL)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_ORDER, MODELS_DIR, REPORTS_DIR, FEATURE_NAMES, CAT_COLS,
    API_HOST, API_PORT,
)
from feature_engineering import engineer_features_df
from risk_scoring import compute_advanced_risk_score, compute_rarity_score
from threshold_optimizer import load_thresholds
from anomaly_model import AnomalyDetector
from behavior_engine import compute_behavior_score, StatefulBehaviorEngine
from fusion_engine import FusionEngine, EXTENDED_CLASSES, ALERT_LEVEL_NAMES, ALERT_LEVEL_EMOJI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts at startup."""
    global model, preprocessing, thresholds, anomaly_detector
    global train_stats, shap_explainer, fusion_engine, pipeline_summary
    global stateful_behavior_engine

    print("\n  Loading Mindgard v4 artifacts...")

    # Model
    model_path = MODELS_DIR / "mindgard_xgboost.json"
    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        print(f"  ✅ XGBoost model loaded")
    else:
        print(f"  ⚠ Model not found — run 'python src/pipeline.py' first!")

    # Preprocessing
    prep_path = MODELS_DIR / "preprocessing.pkl"
    if prep_path.exists():
        preprocessing = joblib.load(prep_path)
        print(f"  ✅ Preprocessing loaded")

    # Thresholds
    thresholds = load_thresholds()
    if thresholds is not None:
        print(f"  ✅ Thresholds loaded")

    # Anomaly detector
    ad_path = MODELS_DIR / "anomaly_detector.pkl"
    if ad_path.exists():
        anomaly_detector = AnomalyDetector.load(ad_path)
        print(f"  ✅ Anomaly detector loaded")

    # Train stats
    stats_path = MODELS_DIR / "train_stats.pkl"
    if stats_path.exists():
        train_stats = joblib.load(stats_path)
        print(f"  ✅ Train stats loaded")

    # Fusion engine
    fusion_engine = FusionEngine(optimized_thresholds=thresholds)
    print(f"  ✅ Fusion engine initialized")

    # Stateful behavioral engine
    stateful_behavior_engine = StatefulBehaviorEngine(window_sec=10)
    print(f"  ✅ Stateful behavioral engine initialized")

    # SHAP explainer
    shap_path = MODELS_DIR / "shap_explainer.pkl"
    if shap_path.exists():
        shap_explainer = joblib.load(shap_path)
        print(f"  ✅ SHAP explainer loaded")

    # Pipeline summary
    summary_path = REPORTS_DIR / "pipeline_summary.json"
    if summary_path.exists():
        pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"\n  🚀 Mindgard Hybrid IDS API ready!")
    yield  # App runs here
    # Shutdown logic (if needed) goes after yield

# ─── App Setup ───
app = FastAPI(
    title="Mindgard Hybrid IDS API",
    description="Hybrid Intelligent Intrusion Detection — XGBoost + Anomaly + Behavior + Fusion",
    version="4.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Global state ───
model: xgb.XGBClassifier | None = None
preprocessing: dict | None = None
thresholds: np.ndarray | None = None
anomaly_detector: AnomalyDetector | None = None
train_stats: dict | None = None
shap_explainer: Any = None
fusion_engine: FusionEngine | None = None
pipeline_summary: dict | None = None
stateful_behavior_engine: StatefulBehaviorEngine | None = None


class ConnectionInput(BaseModel):
    """Single network connection for prediction."""
    src_ip: str = "192.168.1.100"
    timestamp: float | None = None
    duration: float = 0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = 0
    dst_bytes: float = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    srv_diff_host_rate: float = 0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate: float = 0
    dst_host_srv_rerror_rate: float = 0


def _preprocess_input(df: pd.DataFrame) -> np.ndarray:
    """Run raw DataFrame through the same preprocessing pipeline as training."""
    global preprocessing
    df = df.copy()  # Avoid mutating the caller's DataFrame
    # Auto-fill missing required columns from default ConnectionInput schema
    default_vals = ConnectionInput().model_dump()
    for col, val in default_vals.items():
        if col not in df.columns and col not in ["src_ip", "timestamp"]:
            df[col] = val

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feature_cols = preprocessing["feature_columns"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    scaler = preprocessing["scaler"]
    scaled_array = scaler.transform(df.values.astype(np.float32))
    
    # Engineer extra features after scaling to match Phase 2 of pipeline
    try:
        from src.feature_engineering import engineer_features_array
    except ImportError:
        from feature_engineering import engineer_features_array
        
    X_engineered, _ = engineer_features_array(scaled_array, feature_cols)
    return X_engineered


def _full_predict(X: np.ndarray, src_ip: str, timestamp: float | None = None) -> dict[str, Any]:
    """Full hybrid prediction: classifier + anomaly + behavior + fusion + risk."""
    global model, preprocessing, anomaly_detector, stateful_behavior_engine, train_stats, fusion_engine, shap_explainer
    feature_cols = preprocessing["feature_columns"]

    # 1. Classifier probabilities
    probs = model.predict_proba(X)[0]

    # 2. Anomaly score
    anomaly_score = float(anomaly_detector.score(X)[0]) if anomaly_detector else 0.0

    # 3. Behavior score
    if timestamp is None:
        timestamp = time.time()
    
    conn_rate = stateful_behavior_engine.update(src_ip, timestamp) if stateful_behavior_engine else 0.0
    behavior_score = float(compute_behavior_score(X, feature_cols)[0])
    # Augment behavior score with real-time connection rate
    rate_score = min(conn_rate / 100.0, 1.0)
    behavior_score = max(behavior_score, rate_score)

    # 4. Rarity score
    rarity_score = 0.0
    if train_stats:
        rarity_score = compute_rarity_score(X[0], train_stats)

    # 5. Decision fusion
    anomaly_arr = np.array([anomaly_score], dtype=np.float32)
    behavior_arr = np.array([behavior_score], dtype=np.float32)
    if fusion_engine:
        fusion_result = fusion_engine.fuse(probs.reshape(1, -1), anomaly_arr, behavior_arr)
        pred_idx = int(fusion_result["predictions"][0])
        confidence = float(fusion_result["confidences"][0])
        decision = fusion_result["decision_paths"][0]

        alert_levels = fusion_engine.validate_alerts(
            fusion_result["predictions"], fusion_result["confidences"],
            anomaly_arr, behavior_arr,
        )
        alert_level_idx = int(alert_levels[0])
    else:
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        decision = "ARGMAX_FALLBACK"
        alert_level_idx = 1

    pred_class = EXTENDED_CLASSES[pred_idx] if pred_idx < len(EXTENDED_CLASSES) else "UNKNOWN_ATTACK"

    # 6. Risk scoring
    risk = compute_advanced_risk_score(
        predicted_class=pred_class,
        confidence=confidence,
        anomaly_score=anomaly_score,
        behavior_score=behavior_score,
        rarity_score=rarity_score,
        probs=probs,
    )

    # 7. SHAP explanation
    top_reasons = []
    if shap_explainer is not None:
        try:
            shap_values = shap_explainer.shap_values(X)
            target_idx = min(pred_idx, len(CLASS_ORDER) - 1)
            sv = shap_values[target_idx][0] if isinstance(shap_values, list) else shap_values[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            for idx in top_idx:
                direction = "increases" if sv[idx] > 0 else "decreases"
                top_reasons.append({
                    "feature": feature_cols[idx],
                    "impact": f"{sv[idx]:+.4f}",
                    "direction": f"{direction} {pred_class} probability",
                })
        except Exception:
            pass

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "anomaly_score": round(anomaly_score, 4),
        "behavior_score": round(behavior_score, 4),
        "rarity_score": round(rarity_score, 4),
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "alert_level": ALERT_LEVEL_NAMES[alert_level_idx],
        "alert_emoji": ALERT_LEVEL_EMOJI[alert_level_idx],
        "uncertainty": risk.get("uncertainty", 0),
        "decision_path": decision,
        "all_probabilities": {cls: round(float(probs[i]), 4) for i, cls in enumerate(CLASS_ORDER)},
        "signal_breakdown": risk.get("signal_breakdown", {}),
        "explanation": top_reasons,
    }





@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "Mindgard Hybrid IDS v4",
        "model_loaded": model is not None,
        "anomaly_detector_loaded": anomaly_detector is not None,
        "thresholds_loaded": thresholds is not None,
        "shap_loaded": shap_explainer is not None,
        "fusion_active": fusion_engine is not None,
    }


@app.get("/model/info")
async def model_info():
    if pipeline_summary is None:
        return {"error": "No pipeline summary. Run pipeline first."}
    return pipeline_summary


@app.post("/predict")
async def predict(connection: ConnectionInput):
    """Full hybrid prediction for a single network connection."""
    if model is None or preprocessing is None:
        raise HTTPException(500, "Model not loaded. Run 'python src/pipeline.py' first.")

    data = connection.model_dump()
    src_ip = data.pop("src_ip", "unknown")
    timestamp = data.pop("timestamp", time.time())
    
    df = pd.DataFrame([data])
    try:
        X = _preprocess_input(df)
        return _full_predict(X, src_ip, timestamp)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV upload."""
    if model is None or preprocessing is None:
        raise HTTPException(500, "Model not loaded.")
    try:
        content = await file.read()
        # Drop strict header requirements so it accepts generic CSVs
        df = pd.read_csv(__import__("io").BytesIO(content))
        X = _preprocess_input(df)
        feature_cols = preprocessing["feature_columns"]

        probs = model.predict_proba(X)
        anomaly_scores = anomaly_detector.score(X) if anomaly_detector else np.zeros(len(X))
        behavior_scores = compute_behavior_score(X, feature_cols)

        results = []
        for i in range(len(X)):
            rarity = compute_rarity_score(X[i], train_stats) if train_stats else 0.0
            fusion_r = fusion_engine.fuse(probs[i:i+1], anomaly_scores[i:i+1], behavior_scores[i:i+1])
            pred_idx = int(fusion_r["predictions"][0])
            pred_class = EXTENDED_CLASSES[pred_idx] if pred_idx < len(EXTENDED_CLASSES) else "UNKNOWN_ATTACK"

            risk = compute_advanced_risk_score(
                pred_class, float(fusion_r["confidences"][0]),
                float(anomaly_scores[i]), float(behavior_scores[i]),
                rarity, probs[i],
            )
            results.append({
                "index": i,
                "predicted_class": pred_class,
                "confidence": round(float(fusion_r["confidences"][0]), 4),
                "anomaly_score": round(float(anomaly_scores[i]), 4),
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "alert_level": fusion_r["decision_paths"][0].split(":")[0],
            })

        return {"total": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(400, f"Batch failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Mindgard — Hybrid Intelligent IDS API")
    print("=" * 55)
    print(f"\n  Docs:    http://{API_HOST}:{API_PORT}/docs")
    print(f"  Health:  http://{API_HOST}:{API_PORT}/health")
    print(f"  Predict: POST http://{API_HOST}:{API_PORT}/predict")
    print()
    uvicorn.run(app, host=API_HOST, port=API_PORT)
