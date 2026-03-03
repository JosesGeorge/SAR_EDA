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
    "scanning": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def create_human_wireframe():
    """Generate 3D wireframe human model for visualization"""
    # Head
    theta = np.linspace(0, 2*np.pi, 20)
    head_x = 0.15 * np.cos(theta)
    head_y = 0.15 * np.sin(theta)
    head_z = np.ones(20) * 1.7
    
    # Torso
    torso_x = [0, 0, 0, 0]
    torso_y = [0, 0, 0, 0]
    torso_z = [1.5, 1.0, 0.5, 0]
    
    # Arms
    left_arm_x = [0, -0.15, -0.3, -0.4]
    left_arm_y = [0, 0, 0, 0]
    left_arm_z = [1.4, 1.2, 1.0, 0.8]
    
    right_arm_x = [0, 0.15, 0.3, 0.4]
    right_arm_y = [0, 0, 0, 0]
    right_arm_z = [1.4, 1.2, 1.0, 0.8]
    
    # Legs
    left_leg_x = [-0.1, -0.1, -0.12, -0.12]
    left_leg_y = [0, 0, 0, 0]
    left_leg_z = [0, -0.4, -0.8, -1.2]
    
    right_leg_x = [0.1, 0.1, 0.12, 0.12]
    right_leg_y = [0, 0, 0, 0]
    right_leg_z = [0, -0.4, -0.8, -1.2]
    
    return {
        'head': (head_x, head_y, head_z),
        'torso': (torso_x, torso_y, torso_z),
        'left_arm': (left_arm_x, left_arm_y, left_arm_z),
        'right_arm': (right_arm_x, right_arm_y, right_arm_z),
        'left_leg': (left_leg_x, left_leg_y, left_leg_z),
        'right_leg': (right_leg_x, right_leg_y, right_leg_z),
    }


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


with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <h1 style='margin: 0; font-size: 1.8rem;'>🌊 AquaRescue AI</h1>
        <p style='margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.7; letter-spacing: 1px;'>REAL-TIME SONAR RESCUE SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🎯 Sonar Control")
    
    # Primary scan button
    scan_btn = st.button("🔍 START SONAR SCAN", use_container_width=True, type="primary")
    
    st.markdown("")
    
    with st.expander("⚙️ Scan Configuration", expanded=False):
        object_type = st.selectbox("Simulation Target", ["Random (Mixed)", "Human", "Debris"],
                                   help="For testing: select object type to simulate")
        depth_range = st.slider("Depth Range (m)", 1.0, 30.0, (2.0, 15.0), 0.5)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.3, 0.05)
        point_count = st.slider("Point Cloud Density", 50, 300, 120, 10)
    
    with st.expander("🤖 Classifier Settings", expanded=False):
        model_name = st.selectbox("ML Model", ["Random Forest", "SVM", "Logistic Regression"])
        train_samples = st.slider("Training Samples", 400, 4000, 2000, 200)
    
    st.divider()
    
    # Status indicator
    if st.session_state.pipeline is not None:
        st.markdown("""
        <div style='text-align: center; padding: 0.5rem; background: rgba(76, 175, 80, 0.15); border-radius: 8px; border: 1px solid #4caf50;'>
            <span style='color: #4caf50; font-weight: 600; font-size: 0.85rem;'>✓ SCAN COMPLETE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 0.5rem; background: rgba(255, 152, 0, 0.15); border-radius: 8px; border: 1px solid #ff9800;'>
            <span style='color: #ff9800; font-weight: 600; font-size: 0.85rem;'>⏸ STANDBY</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    reset_btn = st.button("🔄 Reset System", use_container_width=True)

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
    st.session_state.scanning = False
    st.rerun()

# Execute scan when button clicked
if scan_btn:
    scan_config = {
        "object_type": object_type,
        "depth_range": depth_range,
        "noise_level": noise_level,
        "point_count": point_count,
        "model_name": model_name,
        "train_samples": train_samples,
    }
    run_scan(scan_config, source_mode="Sonar Scan")
    st.rerun()

# Header
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div style='display: flex; align-items: center;'>
        <h1 style='margin: 0; font-size: 2rem;'>Real-Time Sonar Rescue System</h1>
    </div>
    <p style='margin: 0.2rem 0 0 0; opacity: 0.7; font-size: 0.9rem;'>Underwater object detection and human rescue classification</p>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.pipeline is not None:
        pred_class = st.session_state.pipeline["prediction"]["class"]
        if pred_class == "Human":
            st.markdown("""
            <div style='text-align: right; padding: 0.5rem 0;'>
                <span class='status-badge' style='background: rgba(244, 67, 54, 0.2); color: #f44336; border: 1px solid #f44336;'>
                    <span class='live-indicator' style='background: #f44336;'></span>HUMAN DETECTED
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: right; padding: 0.5rem 0;'>
                <span class='status-badge status-active'>
                    <span class='live-indicator'></span>DEBRIS
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: right; padding: 0.5rem 0;'>
            <span class='status-badge status-standby'>READY</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

if st.session_state.pipeline is None:
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem;'>
        <h2 style='color: rgba(255,255,255,0.9); margin-bottom: 1rem;'>🚨 System Ready for Operation</h2>
        <p style='font-size: 1.1rem; opacity: 0.7; margin-bottom: 2rem;'>
            Click <strong>"START SONAR SCAN"</strong> in the sidebar to begin underwater detection
        </p>
        <div style='background: rgba(33, 150, 243, 0.1); border: 1px solid rgba(33, 150, 243, 0.3); border-radius: 8px; padding: 1.5rem; max-width: 600px; margin: 0 auto;'>
            <h3 style='margin-top: 0; color: #2196f3;'>System Capabilities</h3>
            <ul style='text-align: left; line-height: 2;'>
                <li>🌊 Real-time sonar data simulation and processing</li>
                <li>🎯 ML-powered Human vs Debris classification</li>
                <li>📊 Advanced feature extraction and analysis</li>
                <li>💓 Vital signs monitoring for human detection</li>
                <li>📈 3D visualization and point cloud analysis</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

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

# KPI Metrics - Rescue Dashboard
metric_cols = st.columns([1.5, 1, 1, 1, 1, 1])

with metric_cols[0]:
    if prediction["class"] == "Human":
        st.metric("🚨 CLASSIFICATION", prediction["class"], delta="⚠️ RESCUE ALERT", delta_color="inverse")
    else:
        st.metric("✅ CLASSIFICATION", prediction["class"], delta="Safe", delta_color="normal")

metric_cols[1].metric("🎯 Confidence", f"{prediction['confidence']*100:.1f}%")
metric_cols[2].metric("🤖 ML Model", prediction["model"])
metric_cols[3].metric("📊 Model Acc.", f"{train_report['accuracy']*100:.1f}%")
metric_cols[4].metric("🔢 Total Scans", str(len(st.session_state.history)))

if st.session_state.history:
    humans_detected = sum(1 for h in st.session_state.history if h["class"] == "Human")
    if humans_detected > 0:
        metric_cols[5].metric("🚨 Humans Found", str(humans_detected), delta="Action Required", delta_color="inverse")
    else:
        metric_cols[5].metric("🚨 Humans Found", "0", delta="All Clear")
else:
    metric_cols[5].metric("🚨 Humans Found", "0")

# Scan metadata
with st.expander("📋 Current Scan Configuration", expanded=False):
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("🎯 Source", input_config['source_mode'])
    meta_col2.metric("📏 Depth Range", f"{input_config['depth_range'][0]}–{input_config['depth_range'][1]}m")
    meta_col3.metric("💠 Data Points", input_config['point_count'])
    meta_col4.metric("📡 Noise Level", f"{input_config['noise_level']:.2f}")

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
    st.markdown("### 🎯 Classification Result & Rescue Analysis")
    st.caption("Final detection outcome with confidence metrics and biological monitoring")
    st.markdown("")
    
    is_human = prediction["class"] == "Human"
    
    # Top section: Classification result
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        st.markdown("**🚨 Detection Classification**")
        if is_human:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(211, 47, 47, 0.15)); 
                        border: 2px solid #f44336; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <h1 style='color: #f44336; margin: 0; font-size: 2.5rem; font-weight: 700;'>⚠️ HUMAN</h1>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 600;'>Rescue Required</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(56, 142, 60, 0.15)); 
                        border: 2px solid #4caf50; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <h1 style='color: #4caf50; margin: 0; font-size: 2.5rem; font-weight: 700;'>✓ DEBRIS</h1>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 600;'>No Action Needed</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("**📊 Classification Details**")
        
        details_data = [
            ['Predicted Class', prediction['class']],
            ['Confidence Score', f"{prediction['confidence']*100:.2f}%"],
            ['Model Used', prediction['model']],
            ['Ground Truth', sonar_data['true_label'].title()],
            ['Accuracy', '✅ Correct' if prediction['class'].lower() == sonar_data['true_label'] else '❌ Mismatch']
        ]
        details_df = pd.DataFrame(details_data, columns=['Metric', 'Value'])
        st.dataframe(details_df, use_container_width=True, hide_index=True, height=210)

        st.markdown("")
        st.markdown("**🎯 Feature Summary**")
        feature_summary = pd.DataFrame({
            'Feature': ['Height', 'Density', 'Symmetry', 'Movement'],
            'Value': [
                f"{features['height']:.2f} m",
                f"{features['density']:.1f} pts/m³",
                f"{features['symmetry_score']:.3f}",
                f"{features['movement_score']:.3f}"
            ]
        })
        st.dataframe(feature_summary, use_container_width=True, hide_index=True, height=180)

    with result_col2:
        st.markdown("**📦 3D Sonar Point Cloud with Detection Overlay**")
        st.caption(f"Spatial visualization of detected object · {len(sonar_df)} data points")
        
        fig3d_result = go.Figure()
        
        # Add point cloud
        fig3d_result.add_trace(
            go.Scatter3d(
                x=sonar_df["x"],
                y=sonar_df["y"],
                z=sonar_df["z"],
                mode="markers",
                name="Sonar Returns",
                marker=dict(
                    size=4, 
                    color=sonar_df["intensity"], 
                    colorscale="Viridis", 
                    opacity=0.7,
                    colorbar=dict(title="Intensity", x=1.1, thickness=15, len=0.7),
                    line=dict(width=0.3, color='rgba(255,255,255,0.3)'),
                ),
                hovertemplate="<b>Sonar Point</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>",
            )
        )
        
        # Add human wireframe if detected
        if is_human:
            human_model = create_human_wireframe()
            
            # Center the model at the centroid of the point cloud
            cx, cy, cz = sonar_df["x"].mean(), sonar_df["y"].mean(), sonar_df["z"].mean()
            
            # Add head circle
            hx, hy, hz = human_model['head']
            fig3d_result.add_trace(go.Scatter3d(
                x=hx + cx, y=hy + cy, z=hz + cz - 0.5,
                mode='lines',
                name='Head',
                line=dict(color='#ff4d4f', width=6),
                hoverinfo='skip'
            ))
            
            # Add body parts
            for part_name, part_data in human_model.items():
                if part_name == 'head':
                    continue
                px, py, pz = part_data
                fig3d_result.add_trace(go.Scatter3d(
                    x=np.array(px) + cx,
                    y=np.array(py) + cy,
                    z=np.array(pz) + cz - 0.5,
                    mode='lines+markers',
                    name=part_name.replace('_', ' ').title(),
                    line=dict(color='#ff4d4f', width=5),
                    marker=dict(size=4, color='#ff1744'),
                    hoverinfo='skip'
                ))
        
        fig3d_result.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(title="X (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title="Y (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(title="Depth (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=is_human,
            legend=dict(x=0.7, y=0.95, bgcolor='rgba(0,0,0,0.5)')
        )
        st.plotly_chart(fig3d_result, use_container_width=True, config={'displayModeBar': True})

    # Vital signs section - only show for humans
    if is_human and vitals:
        st.markdown("---")
        st.markdown("### 💓 Human Vital Signs Monitoring")
        st.caption("Real-time physiological data extracted from micro-Doppler signature · Emergency response required")
        st.markdown("")
        
        vital_left, vital_right = st.columns([1, 2])
        
        with vital_left:
            st.markdown("**📊 Vital Statistics**")
            
            # Heart rate status
            hr = vitals['heart_rate']
            hr_status = "Normal" if 60 <= hr <= 100 else "⚠️ Abnormal"
            hr_delta = "Stable" if 60 <= hr <= 100 else "Alert"
            
            st.metric("❤️ Heart Rate", f"{hr} bpm", 
                     delta=hr_delta,
                     delta_color="normal" if 60 <= hr <= 100 else "inverse")
            
            # Respiration rate status
            rr = vitals['resp_rate']
            rr_status = "Normal" if 12 <= rr <= 20 else "⚠️ Abnormal"
            rr_delta = "Stable" if 12 <= rr <= 20 else "Alert"
            
            st.metric("🫁 Respiration Rate", f"{rr} br/min",
                     delta=rr_delta,
                     delta_color="normal" if 12 <= rr <= 20 else "inverse")
            
            st.metric("📡 Signal Quality", "Excellent", delta="Strong SNR")
            st.metric("⏱️ Monitoring Duration", "6.0 sec", delta="Real-time")
            
            st.markdown("")
            st.markdown("**🚑 Rescue Priority**")
            st.error("**HIGH PRIORITY** - Immediate rescue response recommended")
            
            st.markdown("")
            st.markdown("**📋 Assessment**")
            assessment_data = [
                ['Consciousness', 'Detected'],
                ['Movement', f"{features['movement_score']:.2f}"],
                ['Position', f"Depth {sonar_df['z'].mean():.1f}m"],
                ['Status', 'Alive']
            ]
            assessment_df = pd.DataFrame(assessment_data, columns=['Parameter', 'Value'])
            st.dataframe(assessment_df, use_container_width=True, hide_index=True, height=180)
        
        with vital_right:
            st.markdown("**🫀 Cardiac & Respiratory Waveforms**")
            st.caption("Live ECG and respiration patterns extracted from sonar micro-Doppler analysis")
            
            fig_vitals = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Electrocardiogram (ECG) · Cardiac Activity", "Respiratory Signal · Breathing Pattern"),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.5]
            )
            
            # ECG trace
            fig_vitals.add_trace(
                go.Scatter(
                    x=vitals["time"],
                    y=vitals["ecg"],
                    mode="lines",
                    line=dict(color="#ff4d4f", width=2),
                    name=f"ECG ({hr} bpm)",
                    fill='tozeroy',
                    fillcolor='rgba(255, 77, 79, 0.1)',
                    hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.2f} mV<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Respiration trace
            fig_vitals.add_trace(
                go.Scatter(
                    x=vitals["time"],
                    y=vitals["resp"],
                    mode="lines",
                    line=dict(color="#1677ff", width=2),
                    name=f"Respiration ({rr} br/min)",
                    fill='tozeroy',
                    fillcolor='rgba(22, 119, 255, 0.1)',
                    hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>"
                ),
                row=2, col=1
            )
            
            fig_vitals.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
            fig_vitals.update_yaxes(title_text="ECG (mV)", row=1, col=1, gridcolor='rgba(255,255,255,0.1)')
            fig_vitals.update_yaxes(title_text="Resp. Amplitude", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
            
            fig_vitals.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                hovermode='x unified'
            )
            
            for annotation in fig_vitals['layout']['annotations']:
                annotation['font'] = dict(size=12, weight=600)
            
            st.plotly_chart(fig_vitals, use_container_width=True, config={'displayModeBar': True})
    
    elif not is_human:
        st.markdown("---")
        st.markdown("### ℹ️ Vital Signs Monitoring")
        st.info("💡 **No biological signature detected** · Vital signs monitoring is only available for human detection. Current classification: Debris / Inanimate object.")

if st.session_state.history:
    st.divider()
    st.markdown("### 📊 Rescue Mission Log & Performance Statistics")
    st.caption("Complete detection history with aggregate metrics and mission outcomes")
    st.markdown("")
    
    history_col1, history_col2 = st.columns([2.2, 1])
    
    with history_col1:
        st.markdown("**📜 Scan History & Detection Log**")
        history_df = pd.DataFrame(st.session_state.history)
        
        # Enhanced formatting with rescue focus
        display_df = history_df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).map(lambda x: f"{x:.1f}%")
        display_df['status'] = display_df['class'].apply(lambda x: '🚨 HUMAN' if x == 'Human' else '✅ DEBRIS')
        display_df['action'] = display_df['class'].apply(lambda x: 'RESCUE' if x == 'Human' else 'IGNORE')
        display_df = display_df[['scan', 'timestamp', 'status', 'action', 'confidence', 'model']]
        display_df.columns = ['Scan #', 'Time', 'Detection', 'Action', 'Confidence', 'Model']
        
        # Color coding for rescue priority
        def highlight_rows(row):
            if 'HUMAN' in str(row['Detection']):
                return ['background: linear-gradient(90deg, rgba(244, 67, 54, 0.25), rgba(244, 67, 54, 0.05)); font-weight: 600; color: #ff5252;'] * len(row)
            return ['background: rgba(76, 175, 80, 0.08);'] * len(row)
        
        styled_df = display_df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=320)
    
    with history_col2:
        st.markdown("**📈 Mission Statistics**")
        
        total_scans = len(st.session_state.history)
        humans = sum(1 for h in st.session_state.history if h["class"] == "Human")
        debris = total_scans - humans
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        human_pct = (humans/total_scans*100) if total_scans > 0 else 0
        
        st.metric("🔢 Total Scans", total_scans, 
                 delta=f"Last scan: {st.session_state.history[-1]['timestamp']}")
        
        if humans > 0:
            st.metric("🚨 Humans Detected", humans, 
                     delta=f"{human_pct:.1f}% detection rate",
                     delta_color="inverse")
        else:
            st.metric("🚨 Humans Detected", humans, delta="No rescues needed")
        
        st.metric("✅ Debris Detected", debris, 
                 delta=f"{((debris/total_scans*100) if total_scans > 0 else 0):.1f}% of scans")
        
        st.metric("🎯 Avg Confidence", f"{avg_conf*100:.1f}%",
                 delta="Excellent" if avg_conf > 0.9 else "Good" if avg_conf > 0.8 else "Fair" if avg_conf > 0.7 else "Low")
        
        st.markdown("")
        
        # Rescue priority indicator
        if humans > 0:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(211, 47, 47, 0.1)); 
                        border: 2px solid #f44336; border-radius: 8px; padding: 1rem; text-align: center;'>
                <p style='margin: 0; font-size: 1.1rem; font-weight: 700; color: #f44336;'>
                    ⚠️ ACTIVE RESCUES<br/>
                    <span style='font-size: 2rem;'>{}</span>
                </p>
            </div>
            """.format(humans), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(56, 142, 60, 0.1)); 
                        border: 2px solid #4caf50; border-radius: 8px; padding: 1rem; text-align: center;'>
                <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #4caf50;'>
                    ✓ ALL CLEAR<br/>
                    <span style='font-size: 1rem;'>No rescue operations required</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

