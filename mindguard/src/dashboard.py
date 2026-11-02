import sys
from pathlib import Path
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

# Configure paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, CLASS_ORDER, REPORTS_DIR
from api import _preprocess_input, _full_predict
from soc_agent import SOCAgent

import joblib

st.set_page_config(
    page_title="Mindgard IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ─── Custom CSS for Professional SOC Look ───
st.markdown("""
<style>
    /* Global background and text */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1 {
        font-weight: 700;
        color: #58a6ff !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #30363d;
    }
    h2, h3 {
        color: #79c0ff !important;
    }
    
    /* Metrics panel styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: bold;
        color: #ffffff;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: rgba(240, 246, 252, 0.2);
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.5);
    }
    
    /* Dataframes and tables */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #30363d;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Hide Streamlit default marks */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Mindgard Enterprise SOC")
st.markdown("**Real-time intrusion detection intercepting zero-days and APTs** utilizing advanced XGBoost ensembles, Isolation Forests, and autonomous Tier-3 LLM responders.")

# ─── Load Artifacts ───
@st.cache_resource
def load_all_artifacts():
    try:
        model = xgb.XGBClassifier()
        model_path = MODELS_DIR / "mindgard_xgboost.json"
        
        # Load the latest model
        if not model_path.exists():
            return None, None
            
        model.load_model(model_path)
        
        prep_path = MODELS_DIR / "preprocessing.pkl"
        if prep_path.exists():
            preprocessing = joblib.load(prep_path)
        else:
            preprocessing = None
            
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, preprocessing = load_all_artifacts()

if model is None or preprocessing is None:
    st.error("🚨 Models not found. You must run the Training Pipeline first!")
    st.stop()

# ─── Dashboard State ───
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# ─── Sidebar controls ───
with st.sidebar:
    st.markdown("<h1 style='font-size: 3rem;'>🛡️</h1>", unsafe_allow_html=True)
    st.markdown("## Global Command Center")
    st.caption("v2.5.0 • Mindgard Neural Engine")
    st.markdown("---")
    st.write("Deploy a raw packet capture (CSV) for deep inline assessment.")

    uploaded_file = st.file_uploader("📥 Ingest Traffic Data", type="csv", help="Limit: First 100 rows analyzed for demo.")
    scan_btn = st.button("🚀 INITIATE SCAN", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Engine Load:** <span style='color:#00ffcc'>Optimal</span>", unsafe_allow_html=True)
    st.markdown("**LLM Node:** <span style='color:#00ffcc'>Online</span>", unsafe_allow_html=True)

raw_df = None
if uploaded_file is not None:
    # Just take top 100 rows for fast demonstration
    raw_df = pd.read_csv(uploaded_file).head(100)
    st.success(f"✅ Securely loaded {len(raw_df)} network events.", icon="🔐")
else:
    st.info("Awaiting telemetry payload... Please upload a network traffic CSV from the sidebar.", icon="📡")

# ─── Execute Scan ───
if scan_btn and raw_df is not None:
    with st.spinner("Initializing Deep Packet Inspection..."):
        time.sleep(0.5) # UI effect

        # Ensure API globals are loaded for the full predict function and preprocessing
        import api
        api.model = model
        api.preprocessing = preprocessing
        
        stats_path = MODELS_DIR / "train_stats.pkl"
        if stats_path.exists():
            api.train_stats = joblib.load(stats_path)
        else:
            api.train_stats = None
            
        from anomaly_model import AnomalyDetector

        anom_path = MODELS_DIR / "anomaly_detector.pkl"
        if anom_path.exists():
            api.anomaly_detector = AnomalyDetector.load(anom_path)

        from threshold_optimizer import load_thresholds
        api.thresholds = load_thresholds()

        from behavior_engine import StatefulBehaviorEngine
        api.stateful_behavior_engine = StatefulBehaviorEngine(window_sec=10)

        from fusion_engine import FusionEngine
        api.fusion_engine = FusionEngine(optimized_thresholds=api.thresholds)
        
        shap_path = MODELS_DIR / "shap_explainer.pkl"
        if shap_path.exists():
            api.shap_explainer = joblib.load(shap_path)

        results = []
        # Preprocess everything first
        try:
            X_batch = api._preprocess_input(raw_df)
        except Exception as e:
            st.error(f"Preprocessing failed. Ensure CSV matches expected columns.\n{e}")
            st.stop()

        # Run prediction directly via API logic (simulating stream)
        progress_bar = st.progress(0)

        for idx in range(len(raw_df)):
            x_single = X_batch[idx].reshape(1, -1)

            # Map default fields for behavior mapping
            src_ip = raw_df.iloc[idx].get("src_ip", f"192.168.1.{np.random.randint(2, 250)}")
            timestamp = time.time() + (idx * 0.1)  # Simulate time separation
            
            # Use out-of-the-box _full_predict directly from the unified api
            out = getattr(api, '_full_predict')(x_single, src_ip, timestamp)
            
            res_row = {
                "Connection": f"CONN-{idx:03d}",
                "IP": src_ip,
                "Prediction": out.get("predicted_class", "UNKNOWN_ATTACK"),
                "Alert Level": out.get("alert_level", "LOW"),
                "Risk Score": round(out.get("risk_score", 0), 3),
                "Confidence": round(out.get("confidence", 0), 3),
                "Anomaly": round(out.get("anomaly_score", 0), 3),
                "Behavior": round(out.get("behavior_score", 0), 3),
                "Fusion Rule": out.get("decision_path", "Auto"),
                "X_index": idx # For SHAP
            }
            results.append(res_row)
            progress_bar.progress((idx + 1) / len(raw_df))
            
        st.session_state.results_df = pd.DataFrame(results)
        st.session_state.predicted = True
        st.session_state.X_batch = X_batch
        st.session_state.raw_df = raw_df

# ─── Results Dashboard ───
if st.session_state.predicted and st.session_state.results_df is not None:
    res_df = st.session_state.results_df
    
    st.markdown("### 📊 Enterprise Threat Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total_scanned = len(res_df)
    total_threats = len(res_df[res_df['Prediction'] != "normal"])
    total_zeros = len(res_df[res_df['Prediction'] == "UNKNOWN_ATTACK"])
    avg_risk = res_df['Risk Score'].mean()

    with col1:
        st.metric(label="🌐 Logs Ingested", value=f"{total_scanned:,}")
    with col2:
        st.metric(label="🎯 Threats Blocked", value=f"{total_threats:,}", 
                  delta=f"{total_threats/total_scanned*100:.1f}%" if total_scanned > 0 else "0%", delta_color="inverse")
    with col3:
        st.metric(label="👻 Zero-Days Intercepted", value=f"{total_zeros:,}", delta_color="inverse")
    with col4:
        st.metric(label="🔥 Network Risk Index", value=f"{avg_risk:.1f}/100")
    
    st.markdown("---")
    
    st.markdown("### 📋 Active Intercept Registry")
    st.caption("Live feed of AI-scored connections, confidence intervals, and anomaly heuristics.")
    def style_dataframe(val):
        val_str = str(val)
        if val_str in ['normal', 'Benign', 'LOW']:
            return 'color: #3fb950; font-weight: bold;'
        elif val_str in ['Warning', 'MEDIUM']:
            return 'color: #d29922; font-weight: bold;'
        elif val_str in ['HIGH']:
            return 'color: #ff7b72; font-weight: bold;'
        elif val_str in ['CRITICAL']:
            return 'color: #f85149; font-weight: bold;'
        elif val_str in ['UNKNOWN_ATTACK']:
            return 'color: #bc8cff; font-weight: bold;'
        elif val_str in ['Probe', 'DoS', 'R2L', 'U2R']:
            return 'color: #ffa657; font-weight: bold;'
        return ''

    st.dataframe(res_df.style.map(style_dataframe, subset=['Prediction', 'Alert Level']),use_container_width=True)
    
    # ─── SHAP Explainability & SOC Agent ───
    st.markdown("---")
    st.markdown("### 🧠 Autonomous Threat Intelligence Engine")
    st.caption("AI-powered Forensic Audit & MITRE ATT&CK Playbook Generation")
    st.write("Target a specific compromised node index to unpack the neural network's decision boundary and synthesize mitigation steps.")
    
    col_l, col_r = st.columns([1, 1], gap="large")
    selected_idx = col_l.number_input("🔍 Target Connection Index", min_value=0, max_value=len(res_df)-1, value=0, help="Matches the leftmost index column in the registry above.")
    
    if col_l.button("⚡ EXECUTE NEURAL AUDIT & PLAYBOOK", type="primary", use_container_width=True):
        import api
        import shap
        
        if not hasattr(api, 'shap_explainer') or api.shap_explainer is None:
            shap_path = MODELS_DIR / "shap_explainer.pkl"
            if shap_path.exists():
                api.shap_explainer = joblib.load(shap_path)
                
        if hasattr(api, 'shap_explainer') and api.shap_explainer is not None:
            explainer = api.shap_explainer
            X_batch = st.session_state.X_batch
            
            x_single = X_batch[selected_idx].reshape(1, -1)
            raw_shap_values = explainer.shap_values(x_single)
            
            pred_class = res_df.iloc[selected_idx]['Prediction']
            row_data = res_df.iloc[selected_idx].to_dict()
            
            # --- 🔍 MULTI-LAYER ENGINE TELEMETRY ---
            with col_l:
                with st.expander("🛠️ View Inner Multi-Layer Engine Telemetry", expanded=True):
                    st.markdown(f"**L1: XGBoost Classifier** &nbsp;&nbsp;→&nbsp;&nbsp; `Confidence: {row_data.get('Confidence', 0)*100:.1f}%`")
                    st.markdown(f"**L2: Isolation Forest** &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp; `Anomaly Score: {row_data.get('Anomaly', 0)}`")
                    st.markdown(f"**L3: Behavior Engine** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp; `Temporal Risk: {row_data.get('Behavior', 0)}`")
                    st.markdown("---")
                    st.markdown(f"**L4: Fusion Decision** &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp; <span style='color:#58a6ff; font-family: monospace; font-weight: bold; background-color: rgba(88, 166, 255, 0.1); padding: 4px 6px; border-radius: 4px;'>{row_data.get('Fusion Rule', 'N/A')}</span>", unsafe_allow_html=True)
            
            # --- SHAP Figure ---
            if "UNKNOWN_ATTACK" in pred_class:
               col_l.warning("This was classified as a Zero-Day through the Isolation Forest override. SHAP explanations primarily apply to XGBoost known classifications.")
               target_idx = 0 # Default to DoS for illustration
            else:
               target_idx = CLASS_ORDER.index(pred_class) if pred_class in CLASS_ORDER else 0
               
            try:
                # Handle 3D vs List output from XGBoost SHAP natively
                if isinstance(raw_shap_values, list):
                    sv = raw_shap_values[target_idx][0]
                elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
                    sv = raw_shap_values[0, :, target_idx]
                else:
                    sv = raw_shap_values[0]
                
                # Extract top 5 features for the LLM
                feature_names = preprocessing["feature_columns"]
                
                # Check for engineered features mismatch (e.g. 125 vs 117)
                if len(feature_names) < x_single.shape[1]:
                    import feature_engineering
                    _, feature_names = feature_engineering.engineer_features_array(
                        np.zeros((1, len(feature_names))), feature_names
                    )
                
                top_features_idx = np.argsort(np.abs(sv))[::-1][:5]
                top_features = [{"feature": feature_names[i], "shap_value": float(sv[i]), "feature_value": float(x_single[0, i])} for i in top_features_idx]
                
                with col_l:
                    fig = plt.figure(figsize=(9, 5))
                    shap.plots.waterfall(shap.Explanation(values=sv, base_values=0, data=x_single[0], feature_names=feature_names), max_display=10, show=False)
                    # Styling the plot to be dark-mode friendly
                    ax = plt.gca()
                    ax.set_facecolor('#0e1117')
                    fig.patch.set_facecolor('#0e1117')
                    ax.spines['bottom'].set_color('#8b949e')
                    ax.spines['left'].set_color('#8b949e')
                    ax.tick_params(axis='x', colors='#c9d1d9')
                    ax.tick_params(axis='y', colors='#c9d1d9')
                    ax.yaxis.label.set_color('#c9d1d9')
                    ax.xaxis.label.set_color('#c9d1d9')
                    ax.title.set_color('#ffffff')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # --- LLM SOC Agent ---
                with col_r:
                    with st.spinner("🤖 Assembling Cyber Threat Playbook via Gemini..."):
                        agent = SOCAgent()
                        playbook = agent.generate_playbook(row_data, top_features)
                        
                        st.markdown("""
                        <div style="background-color: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #ff7b72; 
                        box-shadow: 0 0 15px rgba(255, 123, 114, 0.15);">
                        <h3 style="color: #ff7b72; margin-top: 0;">🚨 Targeted Mitigation Directive</h3>
                        <p style="color: #8b949e; font-size: 0.9em; margin-bottom: 20px;">AUTHORIZED BY: MINDGARD TIER-3 AI<br>
                        FRAMEWORK: MITRE ATT&CK</p>
                        """, unsafe_allow_html=True)
                        st.markdown(playbook)
                        st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as eval_e:
                col_l.error(f"Could not render SHAP diagram. Explainer dimensions might differ. Error: {eval_e}")
                
        else:
            st.error("SHAP explainer model missing from artifacts.")
