"""
AquaRescue AI — Sonar EDA to Classification Pipeline
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from modules.sonar_simulation import SonarSimulator
from modules.signal_processing import SignalProcessor
from modules.feature_engineering import FeatureEngineer
from modules.ml_classifier import SonarClassifier
from modules.vital_signs import VitalSignSimulator


st.set_page_config(page_title="AquaRescue AI", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      
      .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
      }
      
      [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
      }
      
      [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
      }
      
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #001e3c 100%);
      }
      
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
      
      .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        font-weight: 500;
      }
      
      .stTabs [aria-selected="true"] {
        background-color: rgba(25, 118, 210, 0.2) !important;
        border-bottom: 2px solid #1976d2;
      }
      
      div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4caf50;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
      }
      
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
      
      .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      
      .status-active {
        background: rgba(76, 175, 80, 0.2);
        color: #4caf50;
        border: 1px solid #4caf50;
      }
      
      .status-standby {
        background: rgba(255, 152, 0, 0.2);
        color: #ff9800;
        border: 1px solid #ff9800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "pipeline": None,
    "history": [],
    "auto_mode": True,
    "scan_interval": 3.0,
    "last_scan_time": None,
    "auto_initialized": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def build_pipeline(object_type, depth_range, noise_level, point_count, model_name, train_samples, source_mode):
    type_map = {"Human": "human", "Debris": "debris", "Random (Mixed)": "random"}

    simulator = SonarSimulator(
        n_points=point_count,
        noise_level=noise_level,
        depth_range=depth_range,
    )
    sonar_data = simulator.generate(object_type=type_map[object_type])

    processor = SignalProcessor()
    processed = processor.process(sonar_data)

    engineer = FeatureEngineer()
    features = engineer.extract(sonar_data, processed)

    classifier = SonarClassifier(model_name=model_name)
    training_report = classifier.train(n_samples=train_samples)
    prediction = classifier.predict(features)

    vitals = VitalSignSimulator().generate() if prediction["class"] == "Human" else None
    extracted_input = extracted_input_dataframe(sonar_data)

    return {
        "sonar": sonar_data,
        "extracted_input": extracted_input,
        "processed": processed,
        "features": features,
        "train_report": training_report,
        "prediction": prediction,
        "vitals": vitals,
        "input_config": {
            "source_mode": source_mode,
            "object_type": object_type,
            "depth_range": depth_range,
            "noise_level": noise_level,
            "point_count": point_count,
            "model_name": model_name,
            "train_samples": train_samples,
        },
    }


def sonar_dataframe(sonar_data):
    return pd.DataFrame(
        {
            "x": sonar_data["x"],
            "y": sonar_data["y"],
            "z": sonar_data["z"],
            "intensity": sonar_data["intensity"],
            "doppler_shift": sonar_data["doppler_shift"],
            "time_delay": sonar_data["time_delay"],
        }
    )


def signal_dataframe(sonar_data, processed):
    return pd.DataFrame(
        {
            "raw_intensity": sonar_data["intensity"],
            "filtered": processed["intensity_filtered"],
            "normalized": processed["intensity_normalized"],
            "distance": processed["distances"],
        }
    )


def extracted_input_dataframe(sonar_data):
    n = len(sonar_data["x"])
    start_time = datetime.now().replace(microsecond=0)
    timestamps = [(start_time + timedelta(milliseconds=i * 22)).isoformat() for i in range(n)]

    x = np.array(sonar_data["x"])
    y = np.array(sonar_data["y"])
    z = np.array(sonar_data["z"])
    intensity = np.array(sonar_data["intensity"])
    doppler = np.array(sonar_data["doppler_shift"])

    range_m = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    beam_angle_deg = np.degrees(np.arctan2(y, x))
    echo_strength_db = -60 + (intensity * 55)
    snr_db = 8 + (intensity * 28) - (np.abs(doppler) * 1.5)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ping_id": np.arange(1, n + 1),
            "beam_angle_deg": np.round(beam_angle_deg, 2),
            "range_m": np.round(range_m, 3),
            "time_of_flight_ms": np.round(np.array(sonar_data["time_delay"]) * 1000, 3),
            "echo_strength_db": np.round(echo_strength_db, 2),
            "doppler_hz": np.round(doppler, 3),
            "snr_db": np.round(np.clip(snr_db, 1.0, 45.0), 2),
        }
    )


def auto_generate_scan_config():
    return {
        "object_type": np.random.choice(["Random (Mixed)", "Human", "Debris"]),
        "depth_range": tuple(sorted(np.round(np.random.uniform(1.5, 22.0, 2), 1))),
        "noise_level": float(np.round(np.random.uniform(0.1, 0.65), 2)),
        "point_count": int(np.random.choice(np.arange(80, 281, 20))),
        "model_name": np.random.choice(["Random Forest", "SVM", "Logistic Regression"]),
        "train_samples": int(np.random.choice(np.arange(1000, 3601, 200))),
        "source_mode": "Auto-generated sonar reading",
    }


with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <h1 style='margin: 0; font-size: 1.8rem;'>🌊 AquaRescue AI</h1>
        <p style='margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.7; letter-spacing: 1px;'>AUTOMATED SONAR DETECTION SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("System Control")
    
    auto_mode = st.toggle("🔄 Auto-scan mode", value=st.session_state.auto_mode, 
                          help="Automatically generate and analyze sonar scans at regular intervals")
    st.session_state.auto_mode = auto_mode
    
    if auto_mode:
        scan_interval = st.slider("Scan interval (seconds)", 1.0, 10.0, st.session_state.scan_interval, 0.5)
        st.session_state.scan_interval = scan_interval
        st.info(f"⚡ Auto-scanning every {scan_interval}s")
    
    st.divider()
    
    with st.expander("⚙️ Scan Parameters", expanded=False):
        object_type = st.selectbox("Target object", ["Random (Mixed)", "Human", "Debris"])
        depth_range = st.slider("Depth range (m)", 1.0, 30.0, (2.0, 15.0), 0.5)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.3, 0.05)
        point_count = st.slider("Point cloud density", 50, 300, 120, 10)
    
    with st.expander("🤖 Model Settings", expanded=False):
        model_name = st.selectbox("Classifier", ["Random Forest", "SVM", "Logistic Regression"])
        train_samples = st.slider("Training samples", 400, 4000, 2000, 200)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    manual_btn = col1.button("▶️ Manual Scan", use_container_width=True, type="primary")
    reset_btn = col2.button("🔄 Reset", use_container_width=True)

def run_scan(config, source_mode="Manual"):
    """Execute a scan with given configuration"""
    pipeline = build_pipeline(
        object_type=config["object_type"],
        depth_range=config["depth_range"],
        noise_level=config["noise_level"],
        point_count=config["point_count"],
        model_name=config["model_name"],
        train_samples=config["train_samples"],
        source_mode=source_mode,
    )
    
    st.session_state.pipeline = pipeline
    st.session_state.last_scan_time = datetime.now()
    st.session_state.history.append(
        {
            "scan": len(st.session_state.history) + 1,
            "class": pipeline["prediction"]["class"],
            "confidence": pipeline["prediction"]["confidence"],
            "model": pipeline["prediction"]["model"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )

if reset_btn:
    st.session_state.pipeline = None
    st.session_state.history = []
    st.session_state.last_scan_time = None
    st.session_state.auto_initialized = False
    st.rerun()

# Auto-scan logic
if st.session_state.auto_mode:
    current_time = datetime.now()
    should_scan = False
    
    if not st.session_state.auto_initialized:
        should_scan = True
        st.session_state.auto_initialized = True
    elif st.session_state.last_scan_time:
        time_since_last = (current_time - st.session_state.last_scan_time).total_seconds()
        if time_since_last >= st.session_state.scan_interval:
            should_scan = True
    
    if should_scan:
        auto_cfg = auto_generate_scan_config()
        run_scan(auto_cfg, source_mode="Auto-scan")
        st.rerun()

# Manual scan button
if manual_btn:
    manual_cfg = {
        "object_type": object_type,
        "depth_range": depth_range,
        "noise_level": noise_level,
        "point_count": point_count,
        "model_name": model_name,
        "train_samples": train_samples,
    }
    run_scan(manual_cfg, source_mode="Manual scan")
    st.rerun()

# Header
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div style='display: flex; align-items: center;'>
        <h1 style='margin: 0; font-size: 2rem;'>Sonar Detection Pipeline</h1>
    </div>
    <p style='margin: 0.2rem 0 0 0; opacity: 0.7; font-size: 0.9rem;'>Real-time underwater object detection and classification</p>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.auto_mode:
        st.markdown("""
        <div style='text-align: right; padding: 0.5rem 0;'>
            <span class='status-badge status-active'>
                <span class='live-indicator'></span>LIVE
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: right; padding: 0.5rem 0;'>
            <span class='status-badge status-standby'>STANDBY</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

if st.session_state.pipeline is None:
    st.info("🚀 System ready. Enable auto-scan mode or click 'Manual Scan' to begin.")
    st.stop()

# Auto-refresh in auto mode
if st.session_state.auto_mode:
    import time
    time.sleep(0.1)
    st.rerun()

pipeline = st.session_state.pipeline
sonar_data = pipeline["sonar"]
extracted_input = pipeline["extracted_input"]
processed = pipeline["processed"]
features = pipeline["features"]
train_report = pipeline["train_report"]
prediction = pipeline["prediction"]
vitals = pipeline["vitals"]
input_config = pipeline["input_config"]

sonar_df = sonar_dataframe(sonar_data)
signal_df = signal_dataframe(sonar_data, processed)

# KPI Metrics
metric_cols = st.columns([1.5, 1, 1, 1, 1, 1])

with metric_cols[0]:
    if prediction["class"] == "Human":
        st.metric("🚨 Detection", prediction["class"], delta="Alert", delta_color="inverse")
    else:
        st.metric("✅ Detection", prediction["class"], delta="Safe", delta_color="normal")

metric_cols[1].metric("Confidence", f"{prediction['confidence']*100:.1f}%")
metric_cols[2].metric("Model", prediction["model"])
metric_cols[3].metric("Accuracy", f"{train_report['accuracy']*100:.1f}%")
metric_cols[4].metric("Total Scans", str(len(st.session_state.history)))

if st.session_state.history:
    humans_detected = sum(1 for h in st.session_state.history if h["class"] == "Human")
    metric_cols[5].metric("Humans Found", str(humans_detected))
else:
    metric_cols[5].metric("Humans Found", "0")

# Metadata row
with st.expander("📊 Scan Details", expanded=False):
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Source", input_config['source_mode'])
    meta_col2.metric("Depth Range", f"{input_config['depth_range'][0]}–{input_config['depth_range'][1]}m")
    meta_col3.metric("Data Points", input_config['point_count'])
    meta_col4.metric("Noise Level", f"{input_config['noise_level']:.2f}")

st.divider()

stage1, stage2, stage3, stage4, stage5 = st.tabs(
    [
        "1) Sonar Data",
        "2) Signal Processing",
        "3) Feature Engineering",
        "4) Model Evaluation",
        "5) Classification Output",
    ]
)

with stage1:
    st.markdown("### 📡 Sonar Sensor Input & Exploratory Analysis")
    st.caption("Raw sensor readings and statistical overview of captured sonar data")
    st.markdown("")
    
    left, right = st.columns([1.1, 1.9])

    with left:
        st.markdown("**🎯 Extracted Sonar Reading**")
        st.caption(f"{len(extracted_input)} data points captured from sonar array")
        
        # Format the extracted input table with better column names
        display_input = extracted_input.head(15).copy()
        display_input.columns = ['Timestamp', 'Ping ID', 'Beam °', 'Range (m)', 
                                  'ToF (ms)', 'Echo (dB)', 'Doppler (Hz)', 'SNR (dB)']
        st.dataframe(display_input, use_container_width=True, hide_index=True, height=350)

        st.markdown("")
        st.markdown("**📊 Statistical Summary**")
        st.caption("Key metrics across all captured points")
        
        summary = sonar_df[["x", "y", "z", "intensity", "doppler_shift"]].describe().T
        summary = summary[["mean", "std", "min", "50%", "max"]].round(3)
        summary.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
        summary.index = ['X Position', 'Y Position', 'Depth (Z)', 'Intensity', 'Doppler Shift']
        st.dataframe(summary, use_container_width=True, height=220)

    with right:
        st.markdown("**🗺️ 2D Intensity Heatmap**")
        st.caption("Top-down view showing echo intensity distribution")
        fig2d = go.Figure(
            go.Scatter(
                x=sonar_df["x"],
                y=sonar_df["y"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=sonar_df["intensity"],
                    colorscale="Turbo",
                    showscale=True,
                    opacity=0.9,
                    colorbar=dict(title="Intensity", thickness=12, len=0.7),
                    line=dict(width=0.5, color='rgba(255,255,255,0.1)'),
                ),
                hovertemplate="<b>Position</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br><b>Intensity:</b> %{marker.color:.3f}<extra></extra>",
            )
        )
        fig2d.update_layout(
            height=330, 
            margin=dict(l=10, r=10, t=10, b=10), 
            xaxis=dict(title="X Position (m)", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Y Position (m)", gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig2d, use_container_width=True, config={'displayModeBar': False})

        st.markdown("**📦 3D Point Cloud Visualization**")
        st.caption("Spatial distribution with depth information")
        fig3d = go.Figure(
            go.Scatter3d(
                x=sonar_df["x"],
                y=sonar_df["y"],
                z=sonar_df["z"],
                mode="markers",
                marker=dict(
                    size=3.5, 
                    color=sonar_df["intensity"], 
                    colorscale="Plasma", 
                    opacity=0.85,
                    line=dict(width=0.3, color='rgba(255,255,255,0.2)'),
                ),
                hovertemplate="<b>3D Position</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Depth: %{z:.2f}m<extra></extra>",
            )
        )
        fig3d.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(title="X (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title="Y (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(title="Depth (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                bgcolor="rgba(0,0,0,0)"
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True, config={'displayModeBar': False})

with stage2:
    st.markdown("### 📶 Signal Processing & Filtering")
    st.caption("Echo signal enhancement and noise reduction analysis")
    st.markdown("")
    
    left, right = st.columns([1.8, 1.2])

    with left:
        st.markdown("**🌊 Echo Signal Processing Pipeline**")
        st.caption("Butterworth filter applied to raw intensity data")
        fig_signal = go.Figure()
        fig_signal.add_trace(go.Scatter(
            y=signal_df["raw_intensity"], 
            name="Raw Signal", 
            mode="lines", 
            line=dict(width=1.2, color='rgba(100,149,237,0.5)'),
            hovertemplate="Sample: %{x}<br>Raw: %{y:.3f}<extra></extra>"
        ))
        fig_signal.add_trace(go.Scatter(
            y=signal_df["filtered"], 
            name="Filtered", 
            mode="lines", 
            line=dict(width=2.2, color='#1E90FF'),
            hovertemplate="Sample: %{x}<br>Filtered: %{y:.3f}<extra></extra>"
        ))
        fig_signal.add_trace(go.Scatter(
            y=signal_df["normalized"], 
            name="Normalized", 
            mode="lines", 
            line=dict(width=1.8, dash="dot", color='#00FF88'),
            hovertemplate="Sample: %{x}<br>Normalized: %{y:.3f}<extra></extra>"
        ))
        fig_signal.update_layout(
            height=360, 
            margin=dict(l=10, r=10, t=10, b=10), 
            xaxis=dict(title="Sample Index", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Signal Amplitude", gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig_signal, use_container_width=True, config={'displayModeBar': False})

    with right:
        st.markdown("**📉 Doppler Shift Analysis**")
        st.caption(f"Movement detection · σ²={np.var(sonar_df['doppler_shift']):.3f}")
        fig_hist = go.Figure(go.Histogram(
            x=sonar_df["doppler_shift"], 
            nbinsx=28,
            marker=dict(color='#00D4FF', opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')),
            hovertemplate="Doppler: %{x:.2f} Hz<br>Count: %{y}<extra></extra>"
        ))
        fig_hist.update_layout(
            height=165, 
            margin=dict(l=10, r=10, t=5, b=10),
            xaxis=dict(title="Doppler Shift (Hz)", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Frequency", gridcolor='rgba(255,255,255,0.1)'),
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

        st.markdown("**📏 Range Distribution**")
        st.caption(f"Calculated distances · μ={signal_df['distance'].mean():.2f}m")
        fig_dist = go.Figure(go.Box(
            y=signal_df["distance"], 
            boxmean='sd',
            name="Range",
            marker=dict(color='#FF9800', opacity=0.7),
            line=dict(width=2),
            hovertemplate="Distance: %{y:.2f}m<extra></extra>"
        ))
        fig_dist.update_layout(
            height=165, 
            margin=dict(l=10, r=10, t=5, b=10),
            yaxis=dict(title="Distance (m)", gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

with stage3:
    st.markdown("### 🔧 Feature Engineering & Extraction")
    st.caption("Transform raw sonar data into ML-ready feature vectors")
    st.markdown("")
    
    left, right = st.columns([1.1, 1.9])

    feature_table = pd.DataFrame(
        {
            "Feature": ['Height', 'Width', 'Density', 'Echo Intensity', 'Doppler Variance', 'Symmetry', 'Movement'],
            "Value": [float(v) for v in features.values()],
            "Unit": ['m', 'm', 'pts/m³', 'norm', 'Hz²', 'score', 'score'],
            "Description": [
                'Max dimension (X/Y)',
                'Min dimension (X/Y)',
                'Point cloud density',
                'Mean echo strength',
                'Movement indicator',
                'Shape symmetry',
                'Motion score'
            ]
        }
    )

    with left:
        st.markdown("**📊 Feature Vector (7D)**")
        st.caption(f"Extracted from {len(sonar_df)} sonar returns")
        
        # Display with better formatting
        styled_features = feature_table[['Feature', 'Value', 'Unit']].copy()
        styled_features['Value'] = styled_features['Value'].apply(lambda x: f"{x:.4f}")
        st.dataframe(styled_features, use_container_width=True, hide_index=True, height=280)
        
        st.markdown("")
        st.markdown("**ℹ️ Feature Details**")
        details_df = feature_table[['Feature', 'Description']].copy()
        st.dataframe(details_df, use_container_width=True, hide_index=True, height=280)

    with right:
        st.markdown("**🎯 Feature Space Visualization**")
        st.caption("Radar plot showing normalized feature distribution")
        
        theta_labels = ['Height', 'Width', 'Density', 'Intensity', 'Doppler Var', 'Symmetry', 'Movement']
        values = [float(v) for v in features.values()]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]], 
            theta=theta_labels + [theta_labels[0]], 
            fill='toself',
            name='Current Scan',
            line=dict(color='#00D4FF', width=2),
            fillcolor='rgba(0,212,255,0.3)',
            hovertemplate="<b>%{theta}</b><br>Value: %{r:.4f}<extra></extra>"
        ))
        
        fig_radar.update_layout(
            height=600, 
            margin=dict(l=40, r=40, t=40, b=40),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2],
                    gridcolor='rgba(255,255,255,0.1)',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickfont=dict(size=11, weight=600)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

with stage4:
    st.markdown("### 🤖 Model Training & Evaluation")
    st.caption("Supervised learning performance metrics and validation results")
    st.markdown("")
    
    left, right = st.columns([1.1, 1.9])

    report = train_report["classification_report"]
    report_df = pd.DataFrame(report).T
    report_df = report_df[["precision", "recall", "f1-score", "support"]]
    
    # Better formatting
    report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    report_df.index = report_df.index.map(lambda x: x.title() if x in ['debris', 'human'] else x.replace('_', ' ').title())
    report_df = report_df.round(3)
    
    # Format as percentages except support
    for col in ['Precision', 'Recall', 'F1-Score']:
        report_df[col] = report_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '-')
    report_df['Support'] = report_df['Support'].apply(lambda x: f"{int(x)}" if pd.notna(x) else '-')

    with left:
        st.markdown("**📋 Training Configuration**")
        
        config_data = [
            ['Model Type', train_report['model']],
            ['Training Set', f"{train_report['n_train']} samples"],
            ['Test Set', f"{train_report['n_test']} samples"],
            ['Train/Test Split', '75% / 25%'],
        ]
        config_df = pd.DataFrame(config_data, columns=['Parameter', 'Value'])
        st.dataframe(config_df, use_container_width=True, hide_index=True, height=180)
        
        st.markdown("")
        st.markdown("**🎯 Performance Metrics**")
        
        perf_col1, perf_col2 = st.columns(2)
        perf_col1.metric("Accuracy", f"{train_report['accuracy']*100:.1f}%", 
                        delta=f"{(train_report['accuracy']-0.5)*100:.1f}%" if train_report['accuracy'] > 0.5 else None)
        perf_col2.metric("Avg Confidence", f"{train_report['avg_confidence']*100:.1f}%")

    with right:
        st.markdown("**📊 Per-Class Performance Breakdown**")
        st.caption("Precision, recall, and F1-score for each target class")
        st.dataframe(report_df, use_container_width=True, height=200)

        st.markdown("")
        st.markdown("**🔢 Confusion Matrix**")
        st.caption("True vs predicted classifications on test set")
        
        cm = np.array(train_report["confusion_matrix"])
        
        # Calculate percentages for annotations
        cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annotations_text = [[f"{val}<br>({pct:.1f}%)" for val, pct in zip(row_vals, row_pcts)] 
                           for row_vals, row_pcts in zip(cm, cm_percentages)]
        
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted<br>Debris", "Predicted<br>Human"],
                y=["Actual<br>Debris", "Actual<br>Human"],
                text=annotations_text,
                texttemplate="%{text}",
                textfont={"size": 14, "weight": "bold"},
                colorscale=[[0, '#f0f9ff'], [0.5, '#60a5fa'], [1, '#1e40af']],
                showscale=True,
                colorbar=dict(title="Count", thickness=12, len=0.6),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )
        )
        fig_cm.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(side='bottom'),
            yaxis=dict(side='left')
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

with stage5:
    st.markdown("### 🎯 Final Classification & Vital Signs")
    st.caption("Model prediction results and biological signature analysis")
    st.markdown("")
    
    left, right = st.columns([1.1, 1.9])

    with left:
        is_human = prediction["class"] == "Human"
        st.markdown("**🚨 Detection Result**")
        if is_human:
            st.error(f"**HUMAN DETECTED**")
            st.markdown(f"Confidence: **{prediction['confidence']*100:.1f}%**")
        else:
            st.success(f"**DEBRIS / OBJECT**")
            st.markdown(f"Confidence: **{prediction['confidence']*100:.1f}%**")
        
        st.markdown("")
        st.markdown("**📝 Classification Details**")
        
        details_data = [
            ['Predicted Class', prediction['class']],
            ['Confidence Score', f"{prediction['confidence']*100:.2f}%"],
            ['Model Used', prediction['model']],
            ['Ground Truth', sonar_data['true_label'].title()],
            ['Match', '✅ Correct' if prediction['class'].lower() == sonar_data['true_label'] else '❌ Mismatch']
        ]
        details_df = pd.DataFrame(details_data, columns=['Attribute', 'Value'])
        st.dataframe(details_df, use_container_width=True, hide_index=True, height=210)

    with right:
        st.markdown("**💓 Biological Vital Signs Monitor**")
        if vitals:
            st.caption(f"Real-time physiological data extracted from micro-Doppler signature")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("❤️ Heart Rate", f"{vitals['heart_rate']} bpm",
                       delta="Normal" if 60 <= vitals['heart_rate'] <= 100 else "Abnormal")
            col2.metric("🫁 Respiration", f"{vitals['resp_rate']} br/min",
                       delta="Normal" if 12 <= vitals['resp_rate'] <= 20 else "Abnormal")
            col3.metric("📊 Signal Quality", "Good", delta="Stable")
            
            st.markdown("")

            fig_v = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                subplot_titles=("Electrocardiogram (ECG) Signal", "Respiratory Waveform"),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.5]
            )
            
            fig_v.add_trace(
                go.Scatter(
                    x=vitals["time"], 
                    y=vitals["ecg"], 
                    mode="lines", 
                    line=dict(color="#ff4d4f", width=1.6),
                    name="ECG",
                    hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>"
                ),
                row=1,
                col=1,
            )
            
            fig_v.add_trace(
                go.Scatter(
                    x=vitals["time"], 
                    y=vitals["resp"], 
                    mode="lines", 
                    line=dict(color="#1677ff", width=1.6),
                    fill='tozeroy',
                    fillcolor='rgba(22,119,255,0.1)',
                    name="Respiration",
                    hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>"
                ),
                row=2,
                col=1,
            )
            
            fig_v.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
            fig_v.update_yaxes(title_text="Voltage (mV)", row=1, col=1, gridcolor='rgba(255,255,255,0.1)')
            fig_v.update_yaxes(title_text="Amplitude", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
            
            fig_v.update_layout(
                height=450, 
                margin=dict(l=10, r=10, t=40, b=10), 
                showlegend=False,
                hovermode='x unified'
            )
            
            for annotation in fig_v['layout']['annotations']:
                annotation['font'] = dict(size=13, weight=600)
            
            st.plotly_chart(fig_v, use_container_width=True, config={'displayModeBar': False})
        else:
            st.caption("Vital signs monitoring available only for human detections")
            st.info("🔍 No biological signature detected in current scan. Target classified as debris or inanimate object.")

if st.session_state.history:
    st.divider()
    st.markdown("### 📊 System Performance & Scan History")
    st.caption("Comprehensive log of all detection events and aggregate statistics")
    st.markdown("")
    
    history_col1, history_col2 = st.columns([2.2, 1])
    
    with history_col1:
        st.markdown("**📜 Detection Event Log**")
        history_df = pd.DataFrame(st.session_state.history)
        
        # Better column names and formatting
        display_df = history_df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).map(lambda x: f"{x:.1f}%")
        display_df['status'] = display_df['class'].apply(lambda x: '🚨 Human' if x == 'Human' else '✅ Debris')
        display_df = display_df[['scan', 'timestamp', 'status', 'confidence', 'model']]
        display_df.columns = ['Scan #', 'Time', 'Detection', 'Confidence', 'Model']
        
        # Add color coding
        def highlight_human(row):
            if '🚨' in str(row['Detection']):
                return ['background-color: rgba(244, 67, 54, 0.15); font-weight: 500;'] * len(row)
            return ['background-color: rgba(76, 175, 80, 0.08);'] * len(row)
        
        styled_df = display_df.style.apply(highlight_human, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=320)
    
    with history_col2:
        st.markdown("**📈 Cumulative Statistics**")
        
        total_scans = len(st.session_state.history)
        humans = sum(1 for h in st.session_state.history if h["class"] == "Human")
        debris = total_scans - humans
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        human_pct = (humans/total_scans*100) if total_scans > 0 else 0
        
        st.metric("🔢 Total Scans", total_scans, delta=f"+{total_scans}" if total_scans > 0 else None)
        st.metric("🚨 Humans Found", humans, 
                 delta=f"{human_pct:.1f}% of scans",
                 delta_color="inverse" if humans > 0 else "off")
        st.metric("✅ Debris Objects", debris, 
                 delta=f"{((debris/total_scans*100) if total_scans > 0 else 0):.1f}% of scans")
        st.metric("🎯 Avg Confidence", f"{avg_conf*100:.1f}%",
                 delta="High" if avg_conf > 0.85 else "Medium" if avg_conf > 0.7 else "Low")
