# Mindgard — Advanced AI-Powered Network Intrusion Detection System
**ISIMa Hackathon 2026 — Comprehensive Technical Report & Architecture Guide**

---

## 📌 1. Executive Summary & Problem Statement
Modern networks, particularly IoT and Smart Grid infrastructures, generate millions of connection events daily. Traditional Intrusion Detection Systems (IDS) rely on static signature-matching (e.g., Snort, Suricata), which renders them blind to zero-day attacks and polymorphic malware. 

**Mindgard** is a state-of-the-art, hybrid IDS that bridges the gap between Advanced Machine Learning and Generative AI. It completely overhauls traditional tabular ML approaches by implementing a **Multi-Stage Fusion Engine** consisting of an XGBoost Classifier (for known threats), an Isolation Forest Anomaly Detector (for Zero-Days), and a Stateful Behavior Engine (for temporal tracking). Finally, it augments human Security Operations Center (SOC) teams with a **Tier-3 Generative AI Analyst (Gemini 2.5 Pro)** capable of reading packet telemetry and autonomously writing tactical defensive playbooks mapped to the MITRE ATT&CK framework.

### Project Goals
1. **Zero-Day Detection:** Identify unknown threats using an Unsupervised Isolation Forest, overriding the main classifier when necessary.
2. **Explainability (XAI):** Prove *why* an attack was flagged using SHAP (SHapley Additive exPlanations) directly in the UI, avoiding the "black-box" ML trap.
3. **Resilience to Extreme Class Imbalance:** Accurately detect extremely rare cyber-attacks using strictly mathematical algorithms (balanced sample weights) instead of injecting fake synthesized packets, preserving the realistic purity of the dataset.
4. **Automated Incident Response:** Use cutting-edge LLMs (Gemini 2.5 Pro) to transform raw network alarms into immediate, actionable firewall mitigation commands.
5. **Real-time Pipeline:** Seamlessly process live network interfaces (Pcap/Scapy) to microservice inference in milliseconds.

---

## 📊 2. Dataset Architecture & Threat Taxonomy
The system is originally trained on the highly regarded **KDD Cup 99 / NSL-KDD benchmark dataset**. 
We map over 22 nuanced cyber-attack sub-types into 4 primary malicious classes, plus 1 normal class:

*   **Normal (0):** Standard, benign TCP/UDP/ICMP traffic.
*   **DoS (Denial of Service):** Exhaustion of target resources (e.g., SYN Floods, Smurf, Neptune).
*   **Probe (Surveillance):** Information gathering and port scanning (e.g., Nmap, Portsweep, Satan).
*   **R2L (Remote to Local):** Unauthorized access from a remote machine (e.g., password guessing, FTP write).
*   **U2R (User to Root):** Unauthorized access to local superuser privileges (e.g., Buffer overflows, Rootkits).
*   **UNKNOWN_ATTACK (Fallback):** Triggered dynamically by the Fusion Engine when behavior deviates strongly from the known 5 classes.

---

## 🧠 3. The Machine Learning Pipeline (`src/pipeline.py`)
The core intelligence of Mindgard is generated through a rigorous automated pipeline, designed to extract maximum signal from network noise while preventing synthetic data hallucination.

### Phase 1: Data Preprocessing & Dimensionality Expansion
*   **One-Hot Encoding:** The raw dataset contains baseline features including categorical variables (`protocol_type`, `service`, `flag`). Our pipeline employs one-hot encoding to accurately represent categorical space, expanding the dataset from basic fields to **125 model-ready dimensions**.
*   **Robust Scaling (Median/IQR):** Network traffic is prone to massive outliers (e.g., a DoS attack sending 1.5 billion bytes). We abandoned standard Mean/Variance scaling (`StandardScaler`) in favor of `RobustScaler`, ensuring that extreme attack outliers do not skew the mathematical distribution of normal traffic profiles.

### Phase 2: Domain-Driven Feature Engineering
We engineered multiple novel mathematical features that capture the *behavior* of an attack rather than just its volume:
*   **`bytes_ratio` (src/dst):** Detects asymmetric exfiltration or heavy payload drops.
*   **`total_error_rate` & `total_srv_error_rate`:** Identifies SYN scanners that leave half-open connections.
*   **`host_srv_ratio`:** Flags anomalous focus on specific target services.
*   **`same_diff_gap`:** Highlights unusual deviations in accessing repetitive versus different internal services.

### Phase 3: Pure-Mathematical Imbalance Handling (No SMOTE)
In enterprise network telemetry, `R2L` and `U2R` attacks account for less than 0.05% of traffic. Standard ML models ignore them to artificially boost global accuracy. 
Instead of generating fake synthetic packets (like SMOTE) which destroys the pristine nature of network logs, **Mindgard uses Native Scikit-Learn Class Weight Balancing (`compute_sample_weight("balanced")`)**. This surgically forces the XGBoost model to pay extreme mathematical penalty for missing a rare `U2R` attack, successfully solving the imbalance while retaining 100% realistic training data.

### Phase 4: XGBoost Classifier (Main AI)
*   **Why XGBoost?** Deep Learning (LSTMs) are computationally heavy. XGBoost (Gradient Boosted Decision Trees) offers superior tabular data performance, hyper-fast inference speeds critical for line-rate packet inspection, and native handling of non-linear network behaviors. It outputs a `multi:softprob` across the 5 known classes.

### Phase 5: Isolation Forest (Zero-Day Detector)
While XGBoost catches known malware, Mindgard simultaneously trains an **Isolation Forest anomaly detector**. This unsupervised model learns what "Normal" traffic looks like. If a piece of traffic falls outside the known structure of normal traffic *and* the XGBoost attack profiles, it triggers a catastrophic Anomaly Score.

### Phase 6: Explainable AI (SHAP TreeExplainer)
Cybersecurity analysts cannot block traffic based on a "black box" guess. Mindgard employs **SHAP values** based on cooperative game theory:
*   Allows the system to calculate the exact percentage of contribution each packet attribute (like `duration`, `protocol_tcp`, `src_bytes`) had on the final decision.
*   Locally generates dynamic waterfall plots inside the dashboard to explain individual packet anomalies.

---

## ⚙️ 4. The Multi-Stage Architecture & Microservices
Mindgard uses a highly decoupled architecture designed for enterprise deployment.

### A. The Fusion Engine (`src/fusion_engine.py`)
This is the brain of the IDS. It does not blindly trust XGBoost. It takes three inputs (XGBoost probabilities, Isolation Forest Anomaly Score, and Stateful Behavior Score) and routes them through a strict cybersecurity hierarchy:
1.  **Rule 1: Zero-Day Extreme Override:** If the anomaly score is > 0.85, the prediction is forcefully converted to `UNKNOWN_ATTACK` (Zero-Day) regardless of XGBoost's confidence.
2.  **Rule 2: Open-Set Recognition:** If anomaly > 0.70 and AI confidence is weak (< 0.80), the threat is flagged as a Zero-Day.
3.  **Rule 3: Strict Security Hierarchy:** If multiple classes cross the threshold, the Fusion Engine picks the most dangerous one (`U2R` > `R2L` > `Probe` > `DoS`).
4.  **Rule 4: Argmax Fallback:** If traffic behaves relatively normally, standard AI confidence applies.

### B. Stateful Behavior Engine (`src/behavior_engine.py`)
Attacks don't happen in a vacuum. The Behavior Engine tracks IP addresses over time (temporal data). If a specific IP address suddenly spikes in connection rate (e.g., standard ping sweeps), the Behavior Engine dynamically augments the live risk score in real-time.

### C. FastAPI Microservice (`src/api.py`)
The `_full_predict` endpoint serves as the strict gateway. It dynamically pads missing network columns on the fly, computes rarity, anomaly, and behavior scores, pushes them through the Fusion Engine, and calculates an overall `Risk Score` (0-100) before delivering the final payload.

### D. Enterprise UI Dashboard (`src/dashboard.py`)
A comprehensive Graphical User Interface built on Streamlit featuring a customized **Dark-Mode / Hacker Aesthetic**.
*   Shows live Threat Metrics (Total Logs, Threats Blocked, Zero-Days Intercepted).
*   Live DataFrame mapping attacks to explicit color codes (`Probe` = Orange, `CRITICAL` = Red).
*   In-browser dynamic SHAP generation for visual forensics.

### E. Elite Generative AI SOC Agent (`src/soc_agent.py`)
Integrated natively with the **Google Gemini 2.5 Pro API** (`google.generativeai`). 
When an anomaly hits, an analyst simply clicks "Execute Neural Audit." Mindgard dynamically transforms the XGBoost probabilities, SHAP vectors, Risk Score, and raw network data into an intensely detailed engineering prompt. 
Gemini 2.5 Pro immediately returns a **Targeted Mitigation Directive**, outlining the MITRE ATT&CK phase (e.g., T1190), interpreting the machine learning logic, and generating pure Linux/Bash firewall rules to quarantine the attacker. (Includes an automatic rule-based fail-safe playbook if API limits are hit).

---

## 🚀 5. Deployment and Setup Instructions

### Pre-requisites
Ensure Python 3.9+ is installed.
```bash
pip install -r requirements.txt
```

### 1. Run the ML Pipeline Setup from Scratch
If you wish to retrain the XGBoost system and Anomaly detect on your dataset, run the pipeline. Note: this generates all local models and artifacts.
```bash
python src/pipeline.py
```

### 2. Activate the Generative AI (LLM) Feature
Set the Gemini Native API key in your terminal environment.
```powershell
$env:GEMINI_API_KEY="AIzaSy...YourKeyHere"
```

### 3. Launch the Server (Optional for Microservice setup)
Start the FastAPI inference engine to listen for traffic logs globally.
```bash
uvicorn src.api:app --reload --port 8000
```

### 4. Boot the Visual Enterprise SOC Dashboard
Launch the Streamlit frontend. This connects the trained models, runs the SHAP visualizations, and loads the Gemini Hacker Playbooks.
```bash
streamlit run mindguard/src/dashboard.py
```

---

## 🔬 6. Mindgard V2: Deep Learning & Variational Autoencoder (`src2/`)
In addition to the high-throughput XGBoost fusion engine, this repository includes an advanced Deep Learning architecture located in `src2/`:
*   **Hybrid CNN-LSTM with Self-Attention:** Extracts localized spatial features and captures long-term temporal dependencies in connection flows.
*   **Variational Autoencoder (VAE):** Models the latent probability distribution of normal traffic for ultra-fine-grained Zero-Day anomaly detection.
*   **Google Colab GPU Notebook (`IDS_Optimized_Training_Colab.ipynb`):** Ready-to-run GPU pipeline on T4 for fast distributed training on the KDD Cup 99 dataset.
*   **Pretrained Weights & Metrics:** Available in `models_v2/` (`classifier.pth`, `vae.pth`, `prepro.pkl`) and `reports_v2/`.

---

## 📚 7. Technical Documentation & Hackathon Brief
Comprehensive academic and technical reports are available in the repository:
*   📄 [`Hackathon_ISIMA.pdf`](Hackathon_ISIMA.pdf) — ISIMa CyberSentinel AI Challenge specification & guidelines.
*   📄 [`docs/AI-Based Intrusion Detection Systems Report.pdf`](docs/AI-Based%20Intrusion%20Detection%20Systems%20Report.pdf) — Complete benchmark and comparative analysis of modern AI-based IDS frameworks.
*   📄 [`docs/Rapport_Technique_Avance_Arabe.pdf`](docs/Rapport_Technique_Avance_Arabe.pdf) — Comprehensive study in Arabic covering algorithmic architectures, operational frameworks, and empirical evaluation.

---
**Crafted for the ISIMa Hackathon 2026**
*Merging High-Performance Mathematical Fusion Engines with Elite Generative AI for Next-Generation Network Security.*
