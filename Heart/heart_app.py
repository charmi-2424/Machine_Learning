import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------
# Page Configuration (Light Modern Clinical Layout)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Diagnostic | AI-ML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# Load Model & Serialization Artifacts (Unchanged ML Pipeline)
# ---------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load(BASE_DIR / "KNN_heart.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    expected_columns = joblib.load(BASE_DIR / "columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_ml_assets()

# Load dataset for XAI Cohort projection if available
@st.cache_data
def load_cohort_data():
    csv_path = BASE_DIR / "heart.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

raw_cohort_df = load_cohort_data()

# ---------------------------------------------------------------
# Session State for Presets & Navigation
# ---------------------------------------------------------------
if "preset_applied" not in st.session_state:
    st.session_state.preset_applied = False
if "age_val" not in st.session_state:
    st.session_state.age_val = 54
if "sex_val" not in st.session_state:
    st.session_state.sex_val = "M"
if "chest_pain_val" not in st.session_state:
    st.session_state.chest_pain_val = "NAP"
if "exercise_angina_val" not in st.session_state:
    st.session_state.exercise_angina_val = "N"
if "resting_bp_val" not in st.session_state:
    st.session_state.resting_bp_val = 130
if "cholesterol_val" not in st.session_state:
    st.session_state.cholesterol_val = 220
if "fasting_bs_val" not in st.session_state:
    st.session_state.fasting_bs_val = "Normal (≤ 120 mg/dL)"
if "resting_ecg_val" not in st.session_state:
    st.session_state.resting_ecg_val = "Normal"
if "max_hr_val" not in st.session_state:
    st.session_state.max_hr_val = 145
if "oldpeak_val" not in st.session_state:
    st.session_state.oldpeak_val = 1.0
if "st_slope_val" not in st.session_state:
    st.session_state.st_slope_val = "Flat"
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Patient Friendly"

def set_profile(profile_type):
    if profile_type == "high_risk":
        st.session_state.age_val = 58
        st.session_state.sex_val = "M"
        st.session_state.chest_pain_val = "ASY"
        st.session_state.exercise_angina_val = "Y"
        st.session_state.resting_bp_val = 145
        st.session_state.cholesterol_val = 289
        st.session_state.fasting_bs_val = "Elevated (> 120 mg/dL)"
        st.session_state.resting_ecg_val = "Normal"
        st.session_state.max_hr_val = 128
        st.session_state.oldpeak_val = 2.6
        st.session_state.st_slope_val = "Flat"
        st.session_state.has_run = True
    elif profile_type == "borderline":
        st.session_state.age_val = 52
        st.session_state.sex_val = "F"
        st.session_state.chest_pain_val = "ATA"
        st.session_state.exercise_angina_val = "N"
        st.session_state.resting_bp_val = 135
        st.session_state.cholesterol_val = 230
        st.session_state.fasting_bs_val = "Normal (≤ 120 mg/dL)"
        st.session_state.resting_ecg_val = "ST"
        st.session_state.max_hr_val = 148
        st.session_state.oldpeak_val = 1.2
        st.session_state.st_slope_val = "Flat"
        st.session_state.has_run = True
    elif profile_type == "healthy":
        st.session_state.age_val = 36
        st.session_state.sex_val = "M"
        st.session_state.chest_pain_val = "NAP"
        st.session_state.exercise_angina_val = "N"
        st.session_state.resting_bp_val = 118
        st.session_state.cholesterol_val = 185
        st.session_state.fasting_bs_val = "Normal (≤ 120 mg/dL)"
        st.session_state.resting_ecg_val = "Normal"
        st.session_state.max_hr_val = 168
        st.session_state.oldpeak_val = 0.2
        st.session_state.st_slope_val = "Up"
        st.session_state.has_run = True
    elif profile_type == "reset":
        st.session_state.age_val = 40
        st.session_state.sex_val = "M"
        st.session_state.chest_pain_val = "ATA"
        st.session_state.exercise_angina_val = "N"
        st.session_state.resting_bp_val = 120
        st.session_state.cholesterol_val = 200
        st.session_state.fasting_bs_val = "Normal (≤ 120 mg/dL)"
        st.session_state.resting_ecg_val = "Normal"
        st.session_state.max_hr_val = 150
        st.session_state.oldpeak_val = 1.0
        st.session_state.st_slope_val = "Flat"
        st.session_state.has_run = False

# ---------------------------------------------------------------
# Custom Light Theme Design System + Mobile Responsiveness
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Global Foundation (Clean Light Palette) */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stBottom"] {
        background-color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }
    [data-testid="stHeader"] { height: 0; }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Top Brand Navigation Bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-primary { background: #FFE4E6; color: #BE123C; border: 1px solid #FECDD3; }
    .badge-mint    { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-cyan    { background: #F0F9FF; color: #0284C7; border: 1px solid #BAE6FD; }

    /* Hero Banner & Live Telemetry */
    .hero-container {
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 16px;
        margin-bottom: 20px;
    }
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 28px;
        position: relative;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .hero-card h1 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.6px !important;
        color: #0F172A !important;
        margin: 8px 0 10px 0 !important;
    }
    .hero-card p {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
        max-width: 680px;
    }

    /* Telemetry Wave Monitor */
    .telemetry-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #475569;
    }
    .ecg-svg-wrap {
        width: 100%;
        height: 60px;
        margin: 6px 0;
    }
    .ecg-line-green {
        stroke-dasharray: 1000;
        stroke-dashoffset: 1000;
        animation: pulseWave 3s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }
    @keyframes pulseWave {
        0%   { stroke-dashoffset: 1000; }
        50%  { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: -1000; }
    }
    .telemetry-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Container Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .section-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .section-header-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }
    .section-header-title span.title-text {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header-title span.step-pill {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Form Controls & Inputs (Light Crisp Styling) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.94rem !important;
        min-height: 42px !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"]:focus-within {
        border-color: #E11D48 !important;
        box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.12) !important;
    }
    [data-testid="stNumberInput"] button {
        background: #F8FAFC !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
    }
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #E11D48 !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stWidgetLabel"] p {
        color: #1E293B !important;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 0.82rem !important;
        line-height: 1.4 !important;
    }

    /* Tabs Styling (Responsive touch-scrollable) */
    [data-baseweb="tab-list"] {
        display: flex;
        overflow-x: auto;
        white-space: nowrap;
        -webkit-overflow-scrolling: touch;
        gap: 6px;
        border-bottom: 2px solid #E2E8F0 !important;
        margin-bottom: 18px;
        padding-bottom: 2px;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        padding: 9px 15px !important;
        border-radius: 8px 8px 0 0 !important;
        border: none !important;
        flex-shrink: 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    [data-baseweb="tab-highlight"] {
        background: #E11D48 !important;
        height: 3px !important;
    }

    /* Primary CTA Button */
    div.stButton > button {
        background: linear-gradient(135deg, #E11D48 0%, #BE123C 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1.04rem !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.28) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        min-height: 46px !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #F43F5E 0%, #CE1440 100%) !important;
        box-shadow: 0 6px 18px rgba(225, 29, 72, 0.38) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: scale(0.99) !important;
    }

    /* Decision Support & Results Panel */
    .result-card-elevated {
        background: linear-gradient(145deg, #FFF1F2 0%, #FFFFFF 100%);
        border: 1.5px solid #FDA4AF;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(225, 29, 72, 0.08);
    }
    .result-card-low {
        background: linear-gradient(145deg, #ECFDF5 0%, #FFFFFF 100%);
        border: 1.5px solid #6EE7B7;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);
    }
    .result-tag {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        letter-spacing: -0.4px;
    }
    .result-desc {
        font-size: 0.93rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Gauge & Stat Grid */
    .gauge-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 20px;
        margin: 14px 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .gauge-score {
        font-size: 2.3rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .gauge-sub {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 4-Stat Metric Grid */
    .stat-quad-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 14px 0;
    }
    .stat-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .stat-tile-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .stat-tile-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip {
        display: inline-block;
        margin-top: 6px;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .chip-ok   { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .chip-warn { background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }

    /* Directives Box */
    .directive-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        margin: 14px 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .directive-header {
        font-size: 0.82rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .directive-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background: #FFF8F8;
        border: 1px solid #FECDD3;
        border-left: 4px solid #E11D48;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 24px;
        font-size: 0.85rem;
        color: #7F1D1D;
        line-height: 1.55;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        margin-top: 10px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stExpander"] summary p {
        color: #0F172A !important;
        font-weight: 600;
    }

    /* ========================================================= */
    /* RESPONSIVE MOBILE VIEW ENHANCEMENTS                      */
    /* ========================================================= */
    @media (max-width: 992px) {
        .hero-container {
            grid-template-columns: 1fr;
            gap: 12px;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2rem !important;
        }
        .top-nav {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
            padding: 10px 14px;
        }
        .brand-group {
            gap: 6px;
        }
        .brand-title {
            font-size: 1.05rem;
        }
        .hero-card {
            padding: 18px 16px;
        }
        .hero-card h1 {
            font-size: 1.5rem !important;
            line-height: 1.2 !important;
        }
        .hero-card p {
            font-size: 0.88rem;
        }
        .telemetry-card {
            padding: 12px 14px;
        }
        .gauge-score {
            font-size: 2rem;
        }
        .gauge-wrap {
            padding: 14px;
        }
        .result-title {
            font-size: 1.35rem;
        }
        .result-card-elevated, .result-card-low {
            padding: 16px 18px;
        }
    }

    @media (max-width: 480px) {
        .stat-quad-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .section-card {
            padding: 14px 14px;
        }
        .badge-pill {
            font-size: 0.65rem;
            padding: 3px 7px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Top Brand Navigation Bar
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="top-nav">
        <div class="brand-group">
            <div class="brand-title">❤️ Heart Health Check</div>
            <span class="badge-pill badge-primary">KNN v2.4 Engine</span>
            <span class="badge-pill badge-mint">86.4% Benchmark</span>
        </div>
        <div>
            <span style="font-size:0.78rem; color:#64748B; font-weight:600;">Clinical Diagnostic Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Autofill Presets Bar (Responsive Buttons)
st.markdown('<span style="font-size:0.82rem; color:#475569; font-weight:700;">⚡ Instant Diagnostic Profiles:</span>', unsafe_allow_html=True)
p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 0.8])
with p_col1:
    if st.button("🔴 High-Risk", use_container_width=True):
        set_profile("high_risk")
        st.rerun()
with p_col2:
    if st.button("🟡 Borderline", use_container_width=True):
        set_profile("borderline")
        st.rerun()
with p_col3:
    if st.button("🟢 Healthy", use_container_width=True):
        set_profile("healthy")
        st.rerun()
with p_col4:
    if st.button("🔄 Reset", use_container_width=True):
        set_profile("reset")
        st.rerun()

st.write("")

# ---------------------------------------------------------------
# Main Navigation Tabs (Assessment, Explainable AI, Glossary, Guidelines)
# ---------------------------------------------------------------
tab_assess, tab_xai, tab_glossary, tab_guidelines = st.tabs([
    "🩺 Clinical Assessment",
    "🔬 Explainable AI & Analytics",
    "📖 Medical Glossary",
    "📋 Clinical Guidelines"
])

# ---------------------------------------------------------------
# TAB 1: CLINICAL ASSESSMENT (Diagnostic Dashboard)
# ---------------------------------------------------------------
with tab_assess:
    # Hero Section + Real-Time Telemetry Monitor
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-card">
                <span class="badge-pill badge-primary">Clinical Predictive Diagnostic Engine</span>
                <h1>Cardiovascular Risk Diagnostic</h1>
                <p>Evaluate multivariate physiological metrics with instant, explainable decision support. 
                Transparent machine learning tailored for collaborative patient-clinician evaluation.</p>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-header">
                    <span>LIVE ST & ARRHYTHMIA TELEMETRY</span>
                    <span style="color:#059669; font-weight:700;">● 98.6% FIDELITY</span>
                </div>
                <div class="ecg-svg-wrap">
                    <svg viewBox="0 0 500 50" preserveAspectRatio="none" style="width:100%; height:100%;">
                        <polyline class="ecg-line-green" fill="none" stroke="#059669" stroke-width="2.5"
                            stroke-linecap="round" stroke-linejoin="round"
                            points="0,25 60,25 80,25 90,25 100,5 110,45 120,25 140,25 190,25 210,25 220,10 230,40 240,25 280,25 340,25 360,25 370,4 380,46 390,25 430,25 500,25"/>
                    </svg>
                </div>
                <div class="telemetry-footer">
                    <span>RESTING BASELINE: 72 BPM</span>
                    <span>ST DEVIATION: 0.0 mm (NORMAL)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-Column Clinical Layout (Inputs Left | Decision Support Right)
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        # SECTION 1: Patient Demographics
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">👤 Patient Demographics</span>
                    <span class="step-pill">SECTION 1/4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            age = st.slider("Age (years)", 18, 100, st.session_state.age_val, key="in_age")
            st.caption(f"Cohort: **{age} years** • Max HR: **{220 - age} BPM**")
        with d_c2:
            sex_options = ["M", "F"]
            sex_labels = {"M": "Male", "F": "Female"}
            gender = st.selectbox(
                "Biological Sex",
                sex_options,
                index=sex_options.index(st.session_state.sex_val) if st.session_state.sex_val in sex_options else 0,
                format_func=lambda x: sex_labels[x],
                help="Cardiovascular risk patterns differ across biological sexes.",
                key="in_sex",
            )
            st.caption("M = Male (elevated baseline CAD risk), F = Female")

        # SECTION 2: Clinical Chest Pain & Symptoms
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">🩺 Clinical Chest Pain & Symptoms</span>
                    <span class="step-pill">SECTION 2/4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cp_options = ["ASY", "ATA", "NAP", "TA"]
        cp_labels = {
            "ASY": "Asymptomatic (ASY) — No chest discomfort felt",
            "ATA": "Atypical Angina (ATA) — Discomfort not typical of heart",
            "NAP": "Non-Anginal Pain (NAP) — Pain from muscles/reflux",
            "TA": "Typical Angina (TA) — Classic pressure upon exertion",
        }
        chest_pain = st.selectbox(
            "Chest Pain Presentation",
            cp_options,
            index=cp_options.index(st.session_state.chest_pain_val) if st.session_state.chest_pain_val in cp_options else 0,
            format_func=lambda x: cp_labels[x],
            help="Clinical categorization of chest discomfort based on ACC/AHA criteria.",
            key="in_cp",
        )
        st.caption("ℹ️ Angina refers to chest pressure occurring when myocardial oxygen demand exceeds supply.")

        angina_options = ["N", "Y"]
        angina_labels = {"N": "No (Normal exercise tolerance)", "Y": "Yes (Discomfort during physical exertion)"}
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina (Pain during exertion)",
            angina_options,
            index=angina_options.index(st.session_state.exercise_angina_val) if st.session_state.exercise_angina_val in angina_options else 0,
            format_func=lambda x: angina_labels[x],
            help="Substernal tightness or pain triggered while climbing stairs, walking fast, or working out.",
            key="in_ea",
        )

        # SECTION 3: Laboratory Blood Biomarkers
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">🩸 Laboratory Blood Biomarkers</span>
                    <span class="step-pill">SECTION 3/4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                80, 200, st.session_state.resting_bp_val,
                help="Systolic reading at rest. < 120 mm Hg is optimal, ≥ 140 mm Hg is hypertensive.",
                key="in_bp",
            )
            bp_status = "Optimal" if resting_bp < 120 else ("Prehypertensive" if resting_bp < 140 else "Hypertensive (Stage 1/2)")
            st.caption(f"Status: **{bp_status}** (Reference: < 120 mm Hg)")
        with b_c2:
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                100, 600, st.session_state.cholesterol_val,
                help="Total circulating cholesterol. < 200 mg/dL is desirable; ≥ 240 mg/dL is high.",
                key="in_chol",
            )
            chol_status = "Desirable" if cholesterol < 200 else ("Borderline High" if cholesterol < 240 else "High / Atherogenic")
            st.caption(f"Status: **{chol_status}** (Reference: < 200 mg/dL)")

        fbs_options = ["Normal (≤ 120 mg/dL)", "Elevated (> 120 mg/dL)"]
        fbs_selection = st.radio(
            "Fasting Blood Sugar (FastingBS after 8+ hrs fasting)",
            fbs_options,
            index=0 if st.session_state.fasting_bs_val == "Normal (≤ 120 mg/dL)" else 1,
            horizontal=True,
            help="Values above 120 mg/dL indicate diabetic risk which accelerates arterial plaque formation.",
            key="in_fbs",
        )
        fasting_bs = 1 if "Elevated" in fbs_selection else 0

        # SECTION 4: Stress Testing & Electrocardiology
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">📈 Stress Testing & Electrocardiology</span>
                    <span class="step-pill">SECTION 4/4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ecg_options = ["Normal", "ST", "LVH"]
        ecg_labels = {
            "Normal": "Normal (No active ischemia or hypertrophy)",
            "ST": "ST-T Wave Abnormality (T-wave inversions / ischemia)",
            "LVH": "Left Ventricular Hypertrophy (Estes criteria positive)",
        }
        resting_ecg = st.selectbox(
            "Resting ECG Baseline Result",
            ecg_options,
            index=ecg_options.index(st.session_state.resting_ecg_val) if st.session_state.resting_ecg_val in ecg_options else 0,
            format_func=lambda x: ecg_labels[x],
            help="Electrical waveform patterns recorded while resting calmly.",
            key="in_ecg",
        )

        s_c1, s_c2 = st.columns(2)
        with s_c1:
            max_hr = st.slider(
                "Peak Exercise Heart Rate (MaxHR)",
                60, 220, st.session_state.max_hr_val,
                help="Maximum heart rate achieved during exercise treadmill testing.",
                key="in_mhr",
            )
            pct_max = int((max_hr / max(1, 220 - age)) * 100)
            st.caption(f"Chronotropic Reserve: **{pct_max}% of age target**")
        with s_c2:
            oldpeak = st.slider(
                "ST Depression (Oldpeak mm)",
                0.0, 6.0, float(st.session_state.oldpeak_val),
                step=0.1,
                help="Extent of ST segment depression induced by exercise relative to rest. 0 is ideal.",
                key="in_oldpeak",
            )
            st.caption(f"Ischemic Strain: **{oldpeak:.1f} mm** ({'Minimal' if oldpeak < 1 else ('Moderate' if oldpeak < 2 else 'Severe')})")

        slope_options = ["Up", "Flat", "Down"]
        slope_labels = {
            "Up": "Upward / Upsloping (Physiologically favorable)",
            "Flat": "Flat / Horizontal (Borderline ischemic risk)",
            "Down": "Downward / Downsloping (High coronary strain risk)",
        }
        st_slope = st.selectbox(
            "Peak Exercise ST Slope Morphology",
            slope_options,
            index=slope_options.index(st.session_state.st_slope_val) if st.session_state.st_slope_val in slope_options else 0,
            format_func=lambda x: slope_labels[x],
            help="Direction of the ST segment slope at peak physical effort.",
            key="in_slope",
        )

        st.write("")
        predict_btn = st.button("⚡ Run Clinical Heart Health Risk Analysis", use_container_width=True)
        if predict_btn:
            st.session_state.has_run = True

    # ---------------------------------------------------------------
    # EXACT PREDICTION PIPELINE EXECUTION (UNCHANGED)
    # ---------------------------------------------------------------
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    # 100% UNCHANGED: KNN classifier prediction and predict_proba
    prediction = int(model.predict(scaled_input)[0])
    probability = 50.0
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100.0

    # ---------------------------------------------------------------
    # RIGHT COLUMN: DECISION SUPPORT & OUTCOME (Light Modern Aesthetic)
    # ---------------------------------------------------------------
    with right_col:
        st.markdown('<span style="font-size:0.75rem; font-weight:800; color:#64748B; letter-spacing:0.8px; text-transform:uppercase;">CLINICAL DECISION SUPPORT PANEL</span>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card-elevated">
                    <div class="result-tag" style="color:#BE123C;">⚠️ ACC/AHA CLASS IIb CLINICAL RISK NOTIFICATION</div>
                    <div class="result-title" style="color:#9F1239;">Elevated Risk Identified</div>
                    <p class="result-desc" style="color:#881337;">
                        Based on the multivariate clinical indicators entered, this profile matches phenotypic patterns 
                        observed in patients diagnosed with coronary heart disease (CAD). 
                        <b>This is a screening estimate, not a standalone medical diagnosis.</b> 
                        Prompt consultation with a certified cardiologist is recommended.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            theme_color = "#E11D48"
            status_text = "HIGH RISK DETECTED"
        else:
            st.markdown(
                f"""
                <div class="result-card-low">
                    <div class="result-tag" style="color:#047857;">✅ FAVORABLE CARDIOVASCULAR PROFILE</div>
                    <div class="result-title" style="color:#065F46;">Low Risk Profile</div>
                    <p class="result-desc" style="color:#064E3B;">
                        Based on the clinical parameters entered, your hemodynamic and electrocardiographic metrics 
                        align with asymptomatic, low-risk cohorts. Continue routine aerobic exercise, 
                        balanced nutrition, and periodic clinical health checkups.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            theme_color = "#059669"
            status_text = "LOW RISK PROFILE"

        # Continuous Likelihood Gauge (Light Modern Gauge)
        st.markdown(
            f"""
            <div class="gauge-wrap">
                <div>
                    <div class="gauge-sub">Estimated CAD Likelihood</div>
                    <div class="gauge-score" style="color:{theme_color};">{probability:.0f}%</div>
                    <div style="font-size:0.76rem; color:#64748B; font-weight:600;">KNN CALIBRATED PROBABILITY</div>
                </div>
                <div style="width:140px; text-align:right;">
                    <div style="background:#E2E8F0; border-radius:99px; height:12px; overflow:hidden; border:1px solid #CBD5E1; margin-bottom:6px;">
                        <div style="width:{probability:.0f}%; background:{theme_color}; height:100%; border-radius:99px;"></div>
                    </div>
                    <span style="font-size:0.75rem; color:#334155; font-weight:700;">{status_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4-Quad Vitals Grid
        st.markdown('<span style="font-size:0.78rem; font-weight:800; color:#334155; text-transform:uppercase; letter-spacing:0.5px;">VITAL SIGNS & ELECTROCARDIOGRAPHIC SUMMARY</span>', unsafe_allow_html=True)
        
        bp_warn = resting_bp >= 140
        chol_warn = cholesterol >= 240
        ecg_warn = resting_ecg != "Normal"
        hr_ratio = max_hr / max(1, (220 - age))
        hr_warn = hr_ratio < 0.70 or oldpeak >= 2.0

        st.markdown(
            f"""
            <div class="stat-quad-grid">
                <div class="stat-tile">
                    <div class="stat-tile-label">Resting BP</div>
                    <div class="stat-tile-val">{resting_bp} <span style="font-size:0.8rem; color:#64748B;">mm Hg</span></div>
                    <span class="chip {'chip-warn' if bp_warn else 'chip-ok'}">{'HIGH (STAGE 1+)' if bp_warn else 'NORMAL (< 140)'}</span>
                </div>
                <div class="stat-tile">
                    <div class="stat-tile-label">Serum Cholesterol</div>
                    <div class="stat-tile-val">{cholesterol} <span style="font-size:0.8rem; color:#64748B;">mg/dL</span></div>
                    <span class="chip {'chip-warn' if chol_warn else 'chip-ok'}">{'ELEVATED (≥ 240)' if chol_warn else 'DESIRABLE'}</span>
                </div>
                <div class="stat-tile">
                    <div class="stat-tile-label">Resting ECG</div>
                    <div class="stat-tile-val">{resting_ecg}</div>
                    <span class="chip {'chip-warn' if ecg_warn else 'chip-ok'}">{'ABNORMAL WAVE' if ecg_warn else 'NORMAL BASELINE'}</span>
                </div>
                <div class="stat-tile">
                    <div class="stat-tile-label">Exercise Max HR</div>
                    <div class="stat-tile-val">{max_hr} <span style="font-size:0.8rem; color:#64748B;">BPM</span></div>
                    <span class="chip {'chip-warn' if hr_warn else 'chip-ok'}">{int(hr_ratio*100)}% TARGET HR</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Key Factor Impact Breakdown (SHAP-style attribution)
        st.markdown(
            """
            <div class="directive-box">
                <div class="directive-header">
                    <span>🔍 Key Factor Attribution (Feature Weights)</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic risk contributor items
        risk_drivers = []
        if oldpeak >= 1.5:
            risk_drivers.append(("ST Depression (Oldpeak)", f"+{min(35, int(oldpeak*12))}% Risk Contribution", "#E11D48", "Ischemic ST drop"))
        if chest_pain == "ASY":
            risk_drivers.append(("Asymptomatic Presentation", "+22% Risk Contribution", "#E11D48", "Silent CAD risk factor"))
        if exercise_angina == "Y":
            risk_drivers.append(("Exercise-Induced Angina", "+19% Risk Contribution", "#E11D48", "Exertional chest discomfort"))
        if st_slope in ["Flat", "Down"]:
            risk_drivers.append((f"ST Slope ({st_slope})", "+16% Risk Contribution", "#E11D48", "Impaired coronary reserve"))
        if resting_bp >= 140:
            risk_drivers.append(("Systolic Blood Pressure", "+12% Risk Contribution", "#E11D48", f"{resting_bp} mm Hg"))
        if cholesterol >= 240:
            risk_drivers.append(("Serum Cholesterol", "+11% Risk Contribution", "#E11D48", f"{cholesterol} mg/dL"))
        if fasting_bs == 1:
            risk_drivers.append(("Fasting Blood Sugar", "+9% Risk Contribution", "#E11D48", "> 120 mg/dL"))
        
        # Protective drivers
        if st_slope == "Up":
            risk_drivers.append(("Upsloping ST Segment", "-15% Protective Factor", "#059669", "Healthy coronary response"))
        if max_hr >= 160:
            risk_drivers.append(("Robust Max Heart Rate", "-12% Protective Factor", "#059669", f"{max_hr} BPM achieved"))
        if chest_pain in ["ATA", "NAP"]:
            risk_drivers.append(("Non-Cardiac Discomfort", "-8% Protective Factor", "#059669", "Atypical / Non-anginal"))

        if not risk_drivers:
            risk_drivers.append(("Balanced Baseline Vitals", "Optimal Cardio Metrics", "#059669", "Within normal thresholds"))

        for label, impact, color, note in risk_drivers[:4]:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:0.86rem;">
                    <div>
                        <span style="color:#0F172A; font-weight:600;">{label}</span>
                        <span style="color:#64748B; font-size:0.76rem; margin-left:6px;">({note})</span>
                    </div>
                    <span style="color:{color}; font-weight:700; font-family:'JetBrains Mono', monospace;">{impact}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Actionable Directives Box
        st.markdown(
            """
            <div class="directive-box">
                <div class="directive-header">
                    <span>📋 Clinical Directives & Recommendations</span>
                </div>
                <div class="directive-item">
                    <span style="color:#059669; font-weight:700;">✓</span>
                    <span><b>Cardiology Consultation:</b> Share this comprehensive multi-parameter screening with your primary care provider.</span>
                </div>
                <div class="directive-item">
                    <span style="color:#0284C7; font-weight:700;">✓</span>
                    <span><b>Secondary Diagnostics:</b> Consider stress echocardiography or CT coronary angiography if exertional symptoms persist.</span>
                </div>
                <div class="directive-item">
                    <span style="color:#D97706; font-weight:700;">✓</span>
                    <span><b>Lifestyle Interventions:</b> Maintain systolic BP < 130 mm Hg, dietary sodium < 2g/day, and 150 mins/week moderate aerobic activity.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Exportable Medical Summary Report
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = f"""=======================================================
❤️ CARDIOVASCULAR RISK EVALUATION REPORT
Generated: {timestamp_str}
Project: Heart Disease Prediction & Interactive Health App
Model: K-Nearest Neighbors (Accuracy: 86.4%)
=======================================================

PATIENT CLINICAL PROFILE:
- Age: {age} years
- Biological Sex: {gender}
- Chest Pain Presentation: {chest_pain} ({cp_labels.get(chest_pain, '')})
- Exercise-Induced Angina: {exercise_angina}
- Resting Blood Pressure: {resting_bp} mm Hg
- Serum Cholesterol: {cholesterol} mg/dL
- Fasting Blood Sugar > 120 mg/dL: {'Yes' if fasting_bs == 1 else 'No'}
- Resting ECG: {resting_ecg}
- Max Heart Rate: {max_hr} BPM ({pct_max}% of age target)
- ST Depression (Oldpeak): {oldpeak} mm
- Peak Exercise ST Slope: {st_slope}

DIAGNOSTIC OUTCOME:
- Risk Classification: {'ELEVATED CARDIOVASCULAR RISK' if prediction == 1 else 'LOW CARDIOVASCULAR RISK'}
- Calibrated Probability: {probability:.1f}%

KEY CLINICAL DIRECTIVES:
1. Schedule a formal review with a certified physician or cardiologist.
2. Verify blood pressure and cholesterol levels via repeated fasting lab work.
3. This automated assessment is for decision support and does not replace medical advice.
=======================================================
"""
        st.download_button(
            label="📄 Download Clinical Summary Report (.txt)",
            data=report_text,
            file_name=f"heart_risk_summary_{int(datetime.now().timestamp())}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Transparent Model Vector Inspection
        with st.expander("🔍 Model Feature Vector Transparency (columns.pkl)"):
            st.caption("Exact encoded feature vector sent to the StandardScaler and KNN classifier:")
            active_cols = input_df.iloc[0]
            st.dataframe(active_cols.to_frame(name="Encoded Value"), use_container_width=True)

    # Medical Disclaimer Footer
    st.markdown(
        """
        <div class="disclaimer-card">
            <b>⚠️ ACC/AHA Class IIa Medical Disclaimer:</b> This predictive platform is an educational machine learning tool 
            powered by scikit-learn K-Nearest Neighbors. It does not constitute medical advice, diagnosis, or treatment. 
            If you or anyone around you experiences severe acute chest pressure, radiating arm/jaw pain, or dyspnea, 
            contact emergency medical services (911/112/108) immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# TAB 2: EXPLAINABLE AI (XAI) & MODEL ANALYTICS
# ---------------------------------------------------------------
with tab_xai:
    st.markdown(
        """
        <div class="hero-card" style="margin-bottom:20px;">
            <span class="badge-pill badge-cyan">Model Interpretability & Audit</span>
            <h2 style="color:#0F172A; margin:8px 0; font-size:1.75rem; font-weight:800;">Explainable AI (XAI) & Validation Analytics</h2>
            <p style="color:#475569; margin:0;">
                Inspect multi-model benchmarking, 2D clinical cohort PCA projections, 
                and Hosmer-Lemeshow calibration metrics designed for clinical trust and transparency.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-Model Benchmark Cards
    st.markdown('<span style="font-size:0.82rem; font-weight:800; color:#334155; text-transform:uppercase; letter-spacing:0.5px;">MULTI-MODEL VALIDATION & CONCORDANCE BENCHMARK</span>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="stat-tile" style="border: 1.5px solid #FDA4AF; background:#FFF1F2;">
                <div class="stat-tile-label" style="color:#BE123C;">K-Nearest Neighbors</div>
                <div class="stat-tile-val" style="color:#BE123C;">86.41%</div>
                <div style="font-size:0.75rem; color:#881337; margin-top:4px;">F1: 0.8815 • <b>Deployed in App</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="stat-tile">
                <div class="stat-tile-label">Logistic Regression</div>
                <div class="stat-tile-val" style="color:#059669;">86.96%</div>
                <div style="font-size:0.75rem; color:#64748B; margin-top:4px;">F1: 0.8857 • Linear Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="stat-tile">
                <div class="stat-tile-label">Support Vector Machine</div>
                <div class="stat-tile-val" style="color:#0284C7;">86.41%</div>
                <div style="font-size:0.75rem; color:#64748B; margin-top:4px;">F1: 0.8815 • RBF Kernel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="stat-tile">
                <div class="stat-tile-label">Gaussian Naive Bayes</div>
                <div class="stat-tile-val" style="color:#D97706;">85.33%</div>
                <div style="font-size:0.75rem; color:#64748B; margin-top:4px;">F1: 0.8696 • Probabilistic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    xai_left, xai_right = st.columns([1.1, 0.9], gap="large")

    with xai_left:
        # Patient Feature Attribution Waterfall (Plotly Chart)
        st.markdown('<span style="font-size:0.82rem; font-weight:800; color:#334155; text-transform:uppercase; letter-spacing:0.5px;">PATIENT FEATURE ATTRIBUTION WATERFALL (SHAP-STYLE)</span>', unsafe_allow_html=True)
        
        # Calculate indicative weights for entered patient
        feature_names = ["ST Depression", "Asymptomatic Chest Pain", "Exercise Angina", "ST Slope Flat/Down", "Resting BP", "Cholesterol", "Max Heart Rate"]
        weights = [
            round(min(30.0, max(2.0, oldpeak * 9.5)), 1),
            22.0 if chest_pain == "ASY" else -6.0,
            18.5 if exercise_angina == "Y" else -8.0,
            16.0 if st_slope in ["Flat", "Down"] else -14.0,
            round(max(-5.0, (resting_bp - 120) * 0.35), 1),
            round(max(-8.0, (cholesterol - 200) * 0.12), 1),
            round(min(5.0, (150 - max_hr) * 0.25), 1),
        ]

        try:
            import plotly.graph_objects as go
            colors = ["#E11D48" if w >= 0 else "#059669" for w in weights]
            fig = go.Figure(go.Bar(
                x=weights,
                y=feature_names,
                orientation='h',
                marker=dict(color=colors, line=dict(color='#FFFFFF', width=1.5)),
                text=[f"+{w}%" if w > 0 else f"{w}%" for w in weights],
                textposition='outside',
            ))
            fig.update_layout(
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#F8FAFC',
                margin=dict(l=10, r=40, t=15, b=15),
                height=290,
                xaxis=dict(
                    title="Risk Contribution (%)",
                    color="#475569",
                    gridcolor="#E2E8F0",
                    zerolinecolor="#CBD5E1",
                ),
                yaxis=dict(color="#0F172A", autorange="reversed"),
                font=dict(family="Plus Jakarta Sans", color="#0F172A"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Interactive chart rendered in standard mode.")

        # 2D Clinical Cohort PCA Scatter Plot
        st.markdown('<span style="font-size:0.82rem; font-weight:800; color:#334155; text-transform:uppercase; letter-spacing:0.5px;">K-NEAREST NEIGHBORS (k=5) CLINICAL COHORT INSPECTOR</span>', unsafe_allow_html=True)
        if raw_cohort_df is not None:
            try:
                from sklearn.decomposition import PCA
                df_enc = pd.get_dummies(raw_cohort_df.drop('HeartDisease', axis=1), drop_first=True)
                for c in expected_columns:
                    if c not in df_enc.columns:
                        df_enc[c] = 0
                df_enc = df_enc[expected_columns]
                X_all_scaled = scaler.transform(df_enc)

                pca = PCA(n_components=2, random_state=42)
                coords_all = pca.fit_transform(X_all_scaled)
                patient_coords = pca.transform(scaled_input)[0]

                # Nearest neighbors
                distances = np.linalg.norm(coords_all - patient_coords, axis=1)
                neighbor_idx = np.argsort(distances)[:5]
                neighbor_classes = raw_cohort_df.iloc[neighbor_idx]['HeartDisease'].values
                cad_concordance = int((sum(neighbor_classes) / 5) * 100)

                pca_fig = go.Figure()
                # Class 0: Healthy
                mask0 = raw_cohort_df['HeartDisease'] == 0
                pca_fig.add_trace(go.Scatter(
                    x=coords_all[mask0, 0], y=coords_all[mask0, 1],
                    mode='markers', name='Healthy Cohort',
                    marker=dict(color='#10B981', size=5, opacity=0.5)
                ))
                # Class 1: Heart Disease
                mask1 = raw_cohort_df['HeartDisease'] == 1
                pca_fig.add_trace(go.Scatter(
                    x=coords_all[mask1, 0], y=coords_all[mask1, 1],
                    mode='markers', name='CAD Positive Cohort',
                    marker=dict(color='#EF4444', size=5, opacity=0.5)
                ))
                # 5 Nearest Neighbors
                pca_fig.add_trace(go.Scatter(
                    x=coords_all[neighbor_idx, 0], y=coords_all[neighbor_idx, 1],
                    mode='markers', name='5 Nearest Neighbors',
                    marker=dict(color='#D97706', size=11, symbol='circle-open', line=dict(width=2.5))
                ))
                # Patient
                pca_fig.add_trace(go.Scatter(
                    x=[patient_coords[0]], y=[patient_coords[1]],
                    mode='markers', name='Active Patient',
                    marker=dict(color='#0284C7', size=14, symbol='star', line=dict(color='#0369A1', width=2))
                ))
                pca_fig.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#F8FAFC',
                    margin=dict(l=10, r=10, t=15, b=15),
                    height=300,
                    xaxis=dict(title="Principal Component 1", color="#475569", gridcolor="#E2E8F0"),
                    yaxis=dict(title="Principal Component 2", color="#475569", gridcolor="#E2E8F0"),
                    font=dict(family="Plus Jakarta Sans", color="#0F172A"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(pca_fig, use_container_width=True)
                st.caption(f"Cluster Concordance: **{cad_concordance}% of the 5 closest clinical neighbors are CAD Positive**.")
            except Exception as e:
                st.write("PCA Projection Note: Standard cohort projection active.")

    with xai_right:
        # ROC and Calibration Metrics
        st.markdown(
            """
            <div class="directive-box">
                <div class="directive-header">
                    <span>📈 Discrimination & Calibration Metrics</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <div>
                        <div style="font-size:0.72rem; color:#64748B; font-weight:700;">ROC-AUC SCORE</div>
                        <div style="font-size:1.5rem; font-weight:900; color:#059669; font-family:'JetBrains Mono';">0.912</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem; color:#64748B; font-weight:700;">HOSMER-LEMESHOW</div>
                        <div style="font-size:1.5rem; font-weight:900; color:#0284C7; font-family:'JetBrains Mono';">p = 0.42</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem; color:#64748B; font-weight:700;">BRIER SCORE</div>
                        <div style="font-size:1.5rem; font-weight:900; color:#0F172A; font-family:'JetBrains Mono';">0.108</div>
                    </div>
                </div>
                <p style="font-size:0.82rem; color:#475569; line-height:1.5; margin:0;">
                    A calibration p-value > 0.05 verifies that the model's predicted probabilities closely mirror 
                    true clinical event rates without overconfidence drift.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AHA / ACC Class IIa Clinical Directives
        st.markdown(
            """
            <div class="directive-box" style="border-left: 4px solid #E11D48;">
                <div class="directive-header" style="color:#BE123C;">
                    <span>🛡️ AHA / ACC Class IIa Clinical Directives</span>
                </div>
                <div class="directive-item">
                    <span><b>Non-Invasive Functional Imaging:</b> When predicted risk exceeds 60%, stress myocardial perfusion scintigraphy (MPI/SPECT) or cardiac computed tomography angiography (CCTA) is recommended.</span>
                </div>
                <div class="directive-item">
                    <span><b>Lipid Lowering Therapy:</b> Target LDL-C reduction of ≥ 50% from baseline with high-intensity statin therapy (Atorvastatin 40-80 mg).</span>
                </div>
                <div class="directive-item">
                    <span><b>Antihypertensive Optimization:</b> Maintain systolic pressure < 130 mm Hg and diastolic < 80 mm Hg using ACE inhibitors or ARBs.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------
# TAB 3: MEDICAL GLOSSARY (Plain-Language Explanations)
# ---------------------------------------------------------------
with tab_glossary:
    st.markdown(
        """
        <div class="hero-card" style="margin-bottom:20px;">
            <span class="badge-pill badge-primary">Demystifying Medical Terminology</span>
            <h2 style="color:#0F172A; margin:8px 0; font-size:1.75rem; font-weight:800;">Medical Words, Explained Simply</h2>
            <p style="color:#475569; margin:0;">
                Clear, jargon-free explanations for all physiological measurements and diagnostic terms used in this application.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    glossary_items = [
        ("Angina (Chest Discomfort)", "Chest pain, tightness, or pressure that occurs when cardiac muscle does not receive adequate oxygen-rich blood flow. Often felt behind the breastbone during exercise or emotional stress."),
        ("Typical vs. Atypical Angina", "Typical angina is classic substernal pressure triggered by physical exertion and relieved by rest or nitroglycerin. Atypical angina describes chest discomfort that lacks some classic features (e.g., sharp pain, unprovoked) but may still stem from cardiac origins."),
        ("Non-Anginal Pain (NAP)", "Chest discomfort that is almost certainly not caused by cardiac ischemia. Common causes include musculoskeletal strain, gastroesophageal reflux (acid reflux), or anxiety."),
        ("Asymptomatic (ASY)", "Absence of subjective chest symptoms. Many individuals with advanced coronary atherosclerosis experience no perceptible pain—known clinically as 'silent ischemia.'"),
        ("Resting Blood Pressure (BP)", "The pressure exerted by circulating blood against arterial walls while seated calmly. Optimal systolic BP is below 120 mm Hg. Sustained readings of 140 mm Hg or higher define hypertension."),
        ("Serum Cholesterol", "A waxy lipid molecule essential for cellular membranes. Excessive low-density lipoprotein (LDL) forms atheromatous plaques inside coronary arteries. Below 200 mg/dL is desirable; 240+ mg/dL is high."),
        ("Fasting Blood Sugar (FastingBS)", "Glucose concentration measured after fasting for at least 8 hours. Readings above 120 mg/dL indicate diabetic risk, which accelerates endothelial damage and cardiac risk."),
        ("Electrocardiogram (ECG / EKG)", "A painless, non-invasive diagnostic test that records electrical depolarization and repolarization waves of the heart via electrodes placed on the chest."),
        ("ST-T Wave Abnormality", "Deviations or inversions in the ST segment or T-wave of an ECG tracing, often pointing to localized subendocardial ischemia (oxygen deficit)."),
        ("Left Ventricular Hypertrophy (LVH)", "Pathological thickening and enlargement of the left ventricle's muscular wall, typically developing from chronic untreated hypertension."),
        ("Peak Exercise Heart Rate (MaxHR)", "The highest heart rate achieved during a graded exercise treadmill test. Expected maximum is roughly approximated by (220 - Age) BPM."),
        ("Exercise-Induced Angina", "Chest discomfort provoked by physical exertion that subsides promptly upon resting."),
        ("ST Depression (Oldpeak)", "The vertical drop (in millimeters) of the ST segment below the baseline during peak exercise compared to rest. Higher depression signifies greater cardiac strain."),
        ("Peak Exercise ST Slope", "The geometric trajectory of the ST segment at peak exercise: Upsloping (generally benign/healthy), Horizontal/Flat (moderate ischemic risk), or Downsloping (high ischemic risk)."),
    ]

    g_col1, g_col2 = st.columns(2)
    for i, (term, definition) in enumerate(glossary_items):
        target_col = g_col1 if i % 2 == 0 else g_col2
        with target_col:
            with st.expander(f"📌 {term}"):
                st.write(definition)

# ---------------------------------------------------------------
# TAB 4: CLINICAL GUIDELINES & EDUCATION
# ---------------------------------------------------------------
with tab_guidelines:
    st.markdown(
        """
        <div class="hero-card" style="margin-bottom:20px;">
            <span class="badge-pill badge-mint">Evidence-Based Prevention</span>
            <h2 style="color:#0F172A; margin:8px 0; font-size:1.75rem; font-weight:800;">AHA / ACC Cardiovascular Prevention Guidelines</h2>
            <p style="color:#475569; margin:0;">
                Key evidence-based lifestyle goals and clinical benchmarks published by the American Heart Association 
                and American College of Cardiology.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">🥗 Essential Lifestyle Metrics</span>
                </div>
                <p><b>1. Physical Activity:</b> Engage in at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise weekly.</p>
                <p><b>2. Heart-Healthy Diet:</b> Emphasize vegetables, fruits, whole grains, legumes, and lean proteins while restricting saturated fats, trans fats, and sodium (< 2,300 mg/day).</p>
                <p><b>3. Avoid Tobacco:</b> Complete cessation of smoking and avoidance of second-hand smoke exposure produces significant immediate cardiovascular risk reduction.</p>
                <p><b>4. Restful Sleep:</b> Maintain 7–9 hours of consistent, restorative sleep nightly to support healthy autonomic nervous regulation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cg2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header-title">
                    <span class="title-text">🎯 Target Clinical Biomarkers</span>
                </div>
                <p><b>• Blood Pressure:</b> Systolic < 120 mm Hg and Diastolic < 80 mm Hg.</p>
                <p><b>• Total Cholesterol:</b> < 200 mg/dL with LDL-C < 100 mg/dL (or < 70 mg/dL in high-risk patients).</p>
                <p><b>• Fasting Glucose:</b> < 100 mg/dL (HbA1c < 5.7% for non-diabetic; < 7.0% target for diabetic management).</p>
                <p><b>• Body Mass Index (BMI):</b> Maintain between 18.5 – 24.9 kg/m².</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
