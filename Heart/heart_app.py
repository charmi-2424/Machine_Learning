import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Heart Health Check by Charmi",
    page_icon="❤️",
    layout="centered",
)

# ---------------------------------------------------------------
# Load model files
# ---------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "KNN_heart.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
expected_columns = joblib.load(BASE_DIR / "columns.pkl")

# ---------------------------------------------------------------
# Styling: pure black background
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Base: black everywhere ---------- */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stBottom"] { background: #000000 !important; }
    [data-testid="stHeader"] { height: 0; }
    .block-container { padding-top: 1.6rem; max-width: 800px; }

    html, body, p, li, span, div { color: #E9E9E9; }
    [data-testid="stWidgetLabel"] p {
        color: #F4F4F4 !important; font-weight: 600; font-size: 0.98rem;
    }
    h1, h2, h3, h4 { color: #FFFFFF !important; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: #0B0B0B; border: 1px solid #242424;
        border-radius: 20px; padding: 30px 32px 22px 32px; margin-bottom: 20px;
    }
    .hero h1 {
        font-size: 2.1rem; line-height: 1.15; margin: 0 0 10px 0; padding: 0;
        letter-spacing: -0.5px;
    }
    .hero p { color: #A8A8A8; margin: 0; font-size: 1.02rem; line-height: 1.6; max-width: 600px; }
    .hero svg { display: block; width: 100%; height: 56px; margin-top: 18px; }
    .ecg-line {
        stroke-dasharray: 900; stroke-dashoffset: 900;
        animation: draw 2.6s ease-out forwards;
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @media (prefers-reduced-motion: reduce) {
        .ecg-line { animation: none; stroke-dashoffset: 0; }
    }

    /* ---------- Form sections ---------- */
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #FFFFFF;
        margin: 30px 0 4px 0; padding-left: 12px; border-left: 4px solid #FF4D5E;
    }
    .section-sub { color: #8F8F8F; font-size: 0.92rem; margin: 0 0 10px 16px; }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    [data-testid="stNumberInput"] input {
        background: #0E0E0E !important; color: #FFFFFF !important;
        border-color: #2A2A2A !important; border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"]:focus-within { border-color: #FF4D5E !important; }
    [data-testid="stNumberInput"] button { background: #161616 !important; color: #FFFFFF !important; }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li { background: #0E0E0E !important; color: #FFFFFF !important; }
    div[data-baseweb="popover"] li:hover { background: #1F1F1F !important; }
    [data-testid="stSlider"] [data-testid="stThumbValue"] { color: #FF4D5E !important; font-weight: 700; }
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #777777 !important; }

    /* ---------- Tabs ---------- */
    [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #242424; }
    button[data-baseweb="tab"] { background: transparent !important; color: #9A9A9A !important; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #FFFFFF !important; }
    [data-baseweb="tab-highlight"] { background: #FF4D5E !important; }

    /* ---------- Button ---------- */
    div.stButton > button {
        background: #FF4D5E; color: #000000; border: none; border-radius: 12px;
        padding: 0.85rem 1.4rem; font-weight: 700; font-size: 1.08rem; width: 100%;
        transition: background 0.15s ease, transform 0.1s ease;
    }
    div.stButton > button p { color: #000000 !important; font-weight: 700; }
    div.stButton > button:hover { background: #FF7482; color: #000000; }
    div.stButton > button:active { transform: scale(0.99); }
    div.stButton > button:focus-visible { outline: 3px solid #FFFFFF; outline-offset: 2px; }

    /* ---------- Result ---------- */
    .result-card {
        border-radius: 18px; padding: 24px 26px; margin-top: 18px; border: 1px solid;
    }
    .result-high { background: #1A0709; border-color: #FF4D5E; }
    .result-low  { background: #05160F; border-color: #3DD6A0; }
    .result-card h2 { margin: 0 0 8px 0; font-size: 1.5rem; padding: 0; }
    .result-high h2 { color: #FF6B79 !important; }
    .result-low h2  { color: #5CE6B6 !important; }
    .result-card p  { margin: 0; line-height: 1.6; color: #D8D8D8; }

    .meter-wrap { background: #1E1E1E; border-radius: 99px; height: 16px; margin: 18px 0 6px 0; overflow: hidden; }
    .meter-fill { height: 100%; border-radius: 99px; }
    .meter-label { font-size: 0.9rem; color: #9A9A9A; }

    /* ---------- Numbers at a glance ---------- */
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 8px; }
    .stat {
        background: #0B0B0B; border: 1px solid #242424; border-radius: 14px; padding: 14px 16px;
    }
    .stat .name { color: #8F8F8F; font-size: 0.85rem; margin-bottom: 4px; }
    .stat .val  { color: #FFFFFF; font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
    .stat .unit { color: #8F8F8F; font-size: 0.8rem; font-weight: 400; }
    .chip {
        display: inline-block; margin-top: 8px; padding: 3px 10px; border-radius: 99px;
        font-size: 0.78rem; font-weight: 600;
    }
    .chip-ok   { background: #0C2A1F; color: #5CE6B6; }
    .chip-warn { background: #2E0F13; color: #FF6B79; }

    .note-box {
        background: #0B0B0B; border: 1px solid #242424; border-left: 4px solid #FF4D5E;
        border-radius: 10px; padding: 12px 16px; margin: 8px 0; font-size: 0.96rem; color: #E0E0E0;
    }

    .disclaimer {
        font-size: 0.86rem; color: #8F8F8F; margin-top: 26px;
        border-top: 1px solid #242424; padding-top: 14px; line-height: 1.55;
    }

    /* ---------- Glossary ---------- */
    [data-testid="stExpander"] {
        background: #0B0B0B !important; border: 1px solid #242424 !important; border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary p { color: #FFFFFF !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Health Check</h1>
        <p>Answer a few questions about your body and test results. The tool estimates
        whether your heart may be at risk. Every question comes with a simple explanation,
        so you don't need any medical knowledge.</p>
        <svg viewBox="0 0 800 56" preserveAspectRatio="none" aria-hidden="true">
            <polyline class="ecg-line" fill="none" stroke="#FF4D5E" stroke-width="2.5"
                stroke-linejoin="round" stroke-linecap="round"
                points="0,32 120,32 150,32 165,32 180,8 198,52 214,32 250,32 330,32 360,32 375,32 390,14 406,48 420,32 470,32 560,32 590,32 605,32 620,6 640,52 656,32 700,32 800,32"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_check, tab_terms = st.tabs(["🩺 Check my risk", "📖 What do these words mean?"])

# ---------------------------------------------------------------
# Plain-language labels (values sent to the model stay the same)
# ---------------------------------------------------------------
SEX_LABELS = {"M": "Male", "F": "Female"}

CHEST_PAIN_LABELS = {
    "TA": "Typical angina: classic heart-related chest pain or pressure",
    "ATA": "Atypical angina: chest discomfort that is not the classic kind",
    "NAP": "Non-anginal pain: chest pain that is probably not from the heart",
    "ASY": "No chest pain (asymptomatic)",
}

ECG_LABELS = {
    "Normal": "Normal: nothing unusual",
    "ST": "ST-T abnormality: small changes in the heart's electrical wave",
    "LVH": "Thickened heart wall (LVH): the main pumping muscle is enlarged",
}

SLOPE_LABELS = {
    "Up": "Upward (rising): usually a healthier sign",
    "Flat": "Flat: a middle or less healthy sign",
    "Down": "Downward (falling): often a less healthy sign",
}

YES_NO_SUGAR = {0: "No, 120 or below (normal)", 1: "Yes, above 120 (high)"}
YES_NO_ANGINA = {"N": "No", "Y": "Yes"}


def stat_card(name, value, unit, is_warn, ok_text, warn_text):
    chip_class = "chip-warn" if is_warn else "chip-ok"
    chip_text = warn_text if is_warn else ok_text
    return (
        f'<div class="stat">'
        f'<div class="name">{name}</div>'
        f'<div class="val">{value} <span class="unit">{unit}</span></div>'
        f'<span class="chip {chip_class}">{chip_text}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------
# TAB 1: Inputs + prediction
# ---------------------------------------------------------------
with tab_check:

    # --- About you ---
    st.markdown('<div class="section-title">About you</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age (years)", 18, 100, 40)
    with c2:
        gender = st.selectbox(
            "Sex",
            ["M", "F"],
            help="Biological sex. Heart disease risk patterns differ between men and women.",
        )
        st.caption("M = Male, F = Female")

    # --- How you feel ---
    st.markdown('<div class="section-title">How you feel</div>', unsafe_allow_html=True)
    chest_pain = st.selectbox(
        "What kind of chest pain do you get?",
        ["ATA", "NAP", "TA", "ASY"],
        help="Angina means chest pain or tightness that happens when the heart does not get enough blood.",
    )
    st.caption("ℹ️ " + CHEST_PAIN_LABELS[chest_pain])
    exercise_angina = st.selectbox(
        "Do you get chest pain when you exercise?",
        ["Y", "N"],
        help="Pain, pressure or tightness in the chest during physical effort such as climbing stairs or walking fast.",
    )
    st.caption("Y = Yes, N = No")

    # --- Blood pressure and blood tests ---
    st.markdown('<div class="section-title">Blood pressure and blood tests</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">You can find these numbers in your recent lab report or doctor\'s notes.</div>',
        unsafe_allow_html=True,
    )
    c3, c4 = st.columns(2)
    with c3:
        resting_bp = st.number_input(
            "Blood pressure at rest (mm Hg)",
            80, 200, 120,
            help="The top number of your blood pressure reading, measured while sitting calmly. Around 120 is normal. 140 or more is high.",
        )
    with c4:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            100, 600, 200,
            help="Cholesterol is a fatty substance in your blood. Too much can clog the arteries. Under 200 is desirable. 240 or more is high.",
        )
    fasting_bs = st.selectbox(
        "Is your blood sugar above 120 mg/dL after not eating for 8+ hours?",
        [0, 1],
        help="This is called fasting blood sugar. A high value can point towards diabetes, which raises heart risk.",
    )
    st.caption("ℹ️ 0 = " + YES_NO_SUGAR[0] + "  |  1 = " + YES_NO_SUGAR[1])

    # --- Heart tests ---
    st.markdown('<div class="section-title">Heart tests</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">These come from an ECG (heart rhythm recording) or a treadmill stress test. If you are unsure, check your report or ask your doctor.</div>',
        unsafe_allow_html=True,
    )
    resting_ecg = st.selectbox(
        "Resting ECG result",
        ["Normal", "ST", "LVH"],
        help="An ECG records the electrical signals of your heart while you lie still.",
    )
    st.caption("ℹ️ " + ECG_LABELS[resting_ecg])
    c5, c6 = st.columns(2)
    with c5:
        max_hr = st.slider(
            "Highest heart rate reached (beats per minute)",
            60, 220, 150,
            help="The fastest your heart beat during exercise or a stress test.",
        )
    with c6:
        oldpeak = st.slider(
            "ST depression (Oldpeak)",
            0.0, 6.0, 1.0,
            help="A measure of how much the ECG line dips during exercise compared with rest. A bigger number means the heart may be under more strain. 0 is best.",
        )
    st_slope = st.selectbox(
        "Direction of the ECG line at peak exercise (ST slope)",
        ["Up", "Flat", "Down"],
        help="Shows which way the ECG line moves at the hardest point of exercise.",
    )
    st.caption("ℹ️ " + SLOPE_LABELS[st_slope])

    st.write("")
    predict_clicked = st.button("Check my heart risk")

    # --- Prediction (same logic as original) ---
    if predict_clicked:
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

        prediction = model.predict(scaled_input)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(scaled_input)[0][1]) * 100

        # --- Result card ---
        if prediction == 1:
            st.markdown(
                """
                <div class="result-card result-high">
                    <h2>⚠️ Higher risk of heart disease</h2>
                    <p>Based on the details you entered, your profile looks similar to people who
                    were diagnosed with heart disease. This is <b>not a diagnosis</b>.
                    Please book a visit with a doctor or cardiologist for proper tests.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bar_color = "#FF4D5E"
        else:
            st.markdown(
                """
                <div class="result-card result-low">
                    <h2>✅ Lower risk of heart disease</h2>
                    <p>Based on the details you entered, your profile looks similar to people who
                    did not have heart disease. Keep up healthy habits and have regular
                    check-ups.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bar_color = "#3DD6A0"

        if probability is not None:
            st.markdown(
                f"""
                <div class="meter-wrap">
                    <div class="meter-fill" style="width:{probability:.0f}%; background:{bar_color};"></div>
                </div>
                <div class="meter-label">Model's estimated chance of heart disease: <b style="color:#FFFFFF;">{probability:.0f}%</b></div>
                """,
                unsafe_allow_html=True,
            )

        # --- Numbers at a glance ---
        st.markdown('<div class="section-title">Your numbers at a glance</div>', unsafe_allow_html=True)
        cards = (
            stat_card("Blood pressure", resting_bp, "mm Hg", resting_bp >= 140, "Normal range", "High")
            + stat_card("Cholesterol", cholesterol, "mg/dL", cholesterol >= 240, "Okay", "High")
            + stat_card("Fasting sugar", "Above 120" if fasting_bs == 1 else "120 or below", "", fasting_bs == 1, "Normal", "High")
            + stat_card("ST depression", oldpeak, "", oldpeak >= 2, "Low", "Raised")
        )
        st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)

        # --- Plain-language pointers (general guidelines only) ---
        notes = []
        if resting_bp >= 140:
            notes.append("🩸 Your <b>blood pressure</b> is on the high side (140 or above).")
        if cholesterol >= 240:
            notes.append("🧈 Your <b>cholesterol</b> is high (240 or above).")
        if fasting_bs == 1:
            notes.append("🍬 Your <b>fasting blood sugar</b> is high, which can raise heart risk.")
        if exercise_angina == "Y":
            notes.append("🏃 <b>Chest pain during exercise</b> is worth discussing with a doctor.")
        if oldpeak >= 2:
            notes.append("📉 Your <b>ST depression</b> is fairly high, which can mean extra strain on the heart.")
        if st_slope == "Down":
            notes.append("📈 A <b>downward ST slope</b> is often linked with heart strain.")

        if notes:
            st.markdown('<div class="section-title">Things worth noticing</div>', unsafe_allow_html=True)
            for n in notes:
                st.markdown(f'<div class="note-box">{n}</div>', unsafe_allow_html=True)

        # --- Transparency: show exactly what was sent to the model ---
        with st.expander("🔍 See exactly what the model received"):
            received = input_df.iloc[0]
            st.caption("Only the columns that are not zero are shown. Check that your choices appear here.")
            st.dataframe(received[received != 0].rename("value"))

    st.markdown(
        """
        <div class="disclaimer">
        This tool is a student project that uses a machine learning model. It is for learning and
        awareness only and cannot replace a doctor. If you have chest pain, difficulty breathing,
        or feel faint, seek medical help immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# TAB 2: Glossary in everyday language
# ---------------------------------------------------------------
with tab_terms:
    st.markdown("### Medical words, made simple")

    glossary = [
        ("Angina",
         "Chest pain or tightness that happens when the heart muscle doesn't get enough blood. It often feels like pressure or squeezing."),
        ("Typical vs. atypical angina",
         "Typical angina is the classic heart-related chest pain. Atypical angina is chest discomfort that doesn't fit the classic pattern but may still be linked to the heart."),
        ("Non-anginal pain",
         "Chest pain that is most likely caused by something else, such as muscles, acidity or stress."),
        ("Asymptomatic",
         "Having no symptoms. Some people with heart disease feel nothing at all, which is why check-ups matter."),
        ("Resting blood pressure",
         "The pressure of blood against your artery walls while you are calm. Normal is around 120. 140 or higher is considered high."),
        ("Cholesterol",
         "A waxy, fat-like substance in your blood. Too much can build up inside arteries and block blood flow. Below 200 mg/dL is desirable."),
        ("Fasting blood sugar",
         "Your blood sugar level after not eating for at least 8 hours. Above 120 mg/dL can point towards diabetes."),
        ("ECG (Electrocardiogram)",
         "A quick, painless test that records the electrical signals of your heart using small stickers on your chest."),
        ("ST-T abnormality",
         "A small change in one part of the ECG wave. It can be a sign that the heart is not getting enough blood or oxygen."),
        ("LVH (Left ventricular hypertrophy)",
         "The heart's main pumping chamber has become thicker than normal, often because of long-term high blood pressure."),
        ("Maximum heart rate",
         "The fastest your heart beats during hard exercise or a stress test. It usually gets lower as you get older."),
        ("Exercise-induced angina",
         "Chest pain that starts when you exercise and eases when you rest."),
        ("Oldpeak (ST depression)",
         "How much the ECG line drops during exercise compared with rest. A higher number means the heart may be under more stress."),
        ("ST slope",
         "The direction the ECG line takes at the hardest point of exercise: rising (up), flat, or falling (down). Rising is usually the healthiest."),
    ]

    for term, meaning in glossary:
        with st.expander(term):
            st.write(meaning)