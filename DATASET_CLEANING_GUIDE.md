# Mindgard IDS — Dataset Cleaning & Preprocessing Guide

This document outlines the strict cleaning protocol applied to raw network traffic datasets (e.g., KDD Cup 1999, NSL-KDD) before they are fed into the Mindgard Hybrid IDS pipeline. 

Poorly cleaned network data leads to heavy overfitting, zero-variance exceptions, and inflated F1 scores that collapse in production. Below is the step-by-step strategy implemented in `src/clean_data.py`.

---

## 1. Zero-Variance Feature Removal
Network datasets often contain columns designed for features that never successfully triggered during data collection.
- **Analysis:** We identify columns where every single value is identical (e.g., all `0`).
- **Action:** Features like `num_outbound_cmds` and `is_host_login` (frequently empty in KDD) are hard-dropped. Leaving them in wastes memory and creates singular matrix issues in distance algorithms (like our Rarity Score).

## 2. Attack Taxonomy Mapping
The raw KDD dataset contains exactly 22 distinct attack signatures (e.g., `smurf`, `neptune`, `back`, `guess_passwd`). Training a model to output 23 tiny distinct classes prevents generalizability.
- **Action:** We map raw strings into **5 high-level categories**:
   1. **DoS** (e.g., `neptune`, `smurf`, `pod`, `teardrop`) — Denial of Service
   2. **Probe** (e.g., `satan`, `ipsweep`, `nmap`) — Surveillance and scanning
   3. **R2L** (e.g., `guess_passwd`, `ftp_write`) — Remote to Local (Unauthorized access from a remote machine)
   4. **U2R** (e.g., `buffer_overflow`, `rootkit`) — User to Root (Privilege escalation)
   5. **normal** — Benign background traffic
- **Formatting Fix:** KDD labels natively end with trailing periods (e.g., `neptune.`). These are systematically stripped using `.str.rstrip('.')`.

## 3. Dealing with Difficulty/Metadata Columns
Different dataset versions vary slightly in their column count.
- **Action:** The NSL-KDD version appends an extra "difficulty level" column at the end (forming 43 columns). The original KDD'99 has 42. Our cleaning script dynamically checks column dimensionality and drops purely metadata columns that would otherwise leak target difficulty to the models.

## 4. Addressing Heavy-Tailed Distributions (Robust Scaling)
Variables like `src_bytes` and `duration` follow violent Power Law distributions. A DoS attack might dump 1.3 billion bytes, while a normal ping sends 64.
- **Why Standard scaling fails:** `StandardScaler` sets the mean based on that 1.3 billion outlier, effectively shrinking all normal traffic variations to zero.
- **Action:** We deploy `RobustScaler`. It relies purely on the Median and Interquartile Range (IQR), making it immune to extreme DoS volumetric outliers. 

## 5. One-Hot Encoding (Categorical Protocol Safety)
Categorical features (`protocol_type`, `service`, `flag`) contain critical semantic value but cannot be mapped to continuous integers safely (because HTTP is not mathematically "greater" than FTP).
- **Action:** We map these using standard One-Hot Encoding (`pd.get_dummies`).
- **Safety check:** The pipeline strictly saves the exact column alignment during training. During live API inference, if a completely unknown service string appears, it safely aligns it with standard `0`s across known categories without crashing the backend matrices.

## Summary Checklist
Before the `pipeline.py` starts training the models, it guarantees that:
- [x] All data is cleanly separated into 5 standard classes.
- [x] Zero-variance columns are destroyed.
- [x] Extreme outliers are mitigated via IQR scaling (`RobustScaler`).
- [x] Categorical data is cleanly binarized.
