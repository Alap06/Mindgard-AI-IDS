# Mindgard — Technical Report (v4)
## Hybrid Intelligent Intrusion Detection System

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| **Macro F1-Score** | **0.9784** |
| Accuracy | 0.9999 |
| Macro Precision | 0.9643 |
| Macro Recall | 0.9956 |
| ROC-AUC (macro) | 0.999992700924962 |
| Unknown threats detected | 21,814 |
| Pipeline time | 635.3s |

## 2. System Architecture

```
Raw Data → Cleaning (RobustScaler) → Feature Engineering
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
 XGBoost      Isolation Forest   Behavioral
 Classifier   (Anomaly)          Engine
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
           Decision Fusion
           (Open-set recognition)
                    ↓
           Risk Scoring Engine
           (Multi-signal fusion)
                    ↓
           SHAP Explainability
```

### Intelligence Layers

| Layer | Model | Purpose |
|---|---|---|
| Classifier | XGBoost (multi:softprob) | Detect known attack categories |
| Anomaly | Isolation Forest (200 trees) | Detect zero-day/unknown attacks |
| Behavioral | Rule-based feature analysis | Detect temporal attack patterns |

## 3. Methodology

### 3.1 Cost-Sensitive Learning
- Balanced sample weights (no synthetic data — preserves realism)
- Class weights scale inversely with frequency

### 3.2 Per-Class Threshold Optimization
| Class | Optimal Threshold |
|---|---|
| DoS | 0.0847 |
| Probe | 0.8770 |
| R2L | 0.2590 |
| U2R | 0.9407 |
| normal | 0.0508 |


### 3.3 Decision Fusion Rules
1. **RULE 1** — High anomaly + low confidence → UNKNOWN_ATTACK
2. **RULE 2** — Very low max-prob + moderate anomaly → UNKNOWN_ATTACK
3. **RULE 3** — Optimized thresholds with risk-aware tie-breaking
4. **RULE 4** — Fallback to argmax

### 3.4 Advanced Risk Scoring
```
risk = (w1×confidence + w2×anomaly + w3×behavior + w4×rarity) × class_weight × uncertainty_penalty
```
- Entropy-based uncertainty: uncertain predictions get escalated
- Rarity score: statistical distance from training distribution

### 3.5 False Positive Reduction
Secondary validation: if classifier says "attack" but anomaly AND behavior are low → downgrade alert.

## 4. Results

### 4.1 Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| DoS | 1.0000 | 1.0000 | 1.0000 | 56,916 |
| Probe | 0.9975 | 0.9874 | 0.9924 | 397 |
| R2L | 0.9909 | 0.9909 | 0.9909 | 220 |
| U2R | 0.8333 | 1.0000 | 0.9091 | 10 |
| normal | 0.9997 | 0.9998 | 0.9997 | 19,448 |


### 4.2 Alert Distribution
| Alert Level | Count | % |
|---|---|---|
| LOW | 19,273 | 19.5% |
| MEDIUM | 321 | 0.3% |
| HIGH | 57,014 | 57.7% |
| CRITICAL | 22,197 | 22.5% |


### 4.3 Risk Distribution
| Risk Level | Count | % |
|---|---|---|
| CRITICAL | 14 | 0.0% |
| HIGH | 21,870 | 22.1% |
| MEDIUM | 556 | 0.6% |
| LOW | 76,365 | 77.3% |

Alert rate: 22.1%

### 4.4 Top Features
| Rank | Feature | Importance |
|---|---|---|
| 1 | service_telnet | 0.1283 |
| 2 | service_http | 0.1200 |
| 3 | is_guest_login | 0.0868 |
| 4 | count | 0.0612 |
| 5 | service_ftp_data | 0.0561 |
| 6 | service_smtp | 0.0466 |
| 7 | diff_srv_rate | 0.0466 |
| 8 | srv_count | 0.0381 |
| 9 | num_failed_logins | 0.0312 |
| 10 | service_eco_i | 0.0310 |
| 11 | flag_RSTO | 0.0292 |
| 12 | dst_host_serror_rate | 0.0242 |
| 13 | protocol_type_udp | 0.0241 |
| 14 | service_ftp | 0.0236 |
| 15 | protocol_type_icmp | 0.0211 |


## 5. API

```bash
python src/api.py
# POST /predict — single connection → prediction + risk + SHAP
# POST /predict/batch — CSV upload → batch analysis
# GET /model/info — system metadata
```

## 6. Artifacts
| Artifact | Path |
|---|---|
| XGBoost model | `models/mindgard_xgboost.json` |
| Anomaly detector | `models/anomaly_detector.pkl` |
| Preprocessing | `models/preprocessing.pkl` |
| Thresholds | `models/optimal_thresholds.json` |
| Train stats | `models/train_stats.pkl` |
| SHAP explainer | `models/shap_explainer.pkl` |
| SHAP plots | `reports/shap_*.png` |
| Fusion stats | `reports/fusion_stats.json` |
| Risk summary | `reports/risk_summary.json` |
