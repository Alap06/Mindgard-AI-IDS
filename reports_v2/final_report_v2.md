# Mindgard V2 — Technical Report
## Deep Hybrid Intelligent Intrusion Detection System (RTX 3050 Optimized)

| Metric | Value |
|---|---|
| **Macro F1-Score** | **0.7281** |
| Total Time | 13299.6s |
| Device | cuda |
| Zero-day detected | 0 |

### Intelligence Layers
- **Classifier**: CNN + Bi-LSTM + Multi-Head Attention
- **Anomaly**: Variational Autoencoder (VAE)
- **Behavior**: Rule-based Temporal Analysis

### Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| DoS | 0.9999 | 0.9948 | 0.9973 | 78,292 |
| Probe | 0.8543 | 0.9915 | 0.9178 | 822 |
| R2L | 0.5000 | 0.9822 | 0.6627 | 225 |
| U2R | 0.0408 | 1.0000 | 0.0784 | 10 |
| normal | 0.9888 | 0.9794 | 0.9841 | 19,456 |


### Alert Distribution
| Alert Level | Count | % |
|---|---|---|
| LOW | 19,623 | 19.9% |
| MEDIUM | 76,099 | 77.0% |
| HIGH | 3,083 | 3.1% |
| CRITICAL | 0 | 0.0% |

