import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from fpdf import FPDF
import os
import google.generativeai as genai

from utils import preprocess_ecg, calculate_final_risk

# -----------------------------------
# CONFIG & THEME
# -----------------------------------
st.set_page_config(page_title="Heart Risk AI", page_icon="❤️", layout="wide")

page_bg_color = """
<style>
.stApp {
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.8);
}
h1, h2, h3 {
    color: #1c2833 !important;
    font-weight: 700 !important;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# SESSION STATE INITIALIZATION  (Fix 1)
# -----------------------------------
if "final_risk" not in st.session_state:
    st.session_state.final_risk = None
if "risk_label" not in st.session_state:
    st.session_state.risk_label = None
if "report_text" not in st.session_state:
    st.session_state.report_text = "AI report unavailable."
if "patient_name_display" not in st.session_state:
    st.session_state.patient_name_display = ""

# -----------------------------------
# LOAD MODELS
# -----------------------------------
@st.cache_resource
def load_models():
    clinical_path = os.path.join(BASE_DIR, "models", "clinical_model.pkl")
    ecg_path = os.path.join(BASE_DIR, "models", "hybrid_ecg_model.h5")

    clinical_model = None
    ecg_model = None

    if os.path.exists(clinical_path):
        clinical_model = joblib.load(clinical_path)
        if not hasattr(clinical_model, "predict_proba"):
            st.error("❌ Wrong file loaded! You loaded feature_names.pkl instead of clinical_model.pkl")
            st.stop()
    else:
        st.error("❌ clinical_model.pkl missing")
        st.stop()

    if os.path.exists(ecg_path):
        try:
            ecg_model = tf.keras.models.load_model(ecg_path, compile=False)
        except Exception:
            st.warning("⚠️ ECG model invalid or not trained")

    return clinical_model, ecg_model

clinical_model, ecg_model = load_models()

# -----------------------------------
# GOOGLE GEMINI SETUP
# -----------------------------------
# NOTE:
# The Google Gemini API key has been removed for security reasons.
# Please add your own API key to run the AI report functionality.
API_KEY = "Remove this and add your own Google API key to enable AI report generation."

llm = None
if not API_KEY:
    st.warning("⚠️ Add your Google API key to enable AI report")
else:
    try:
        genai.configure(api_key=API_KEY)
        llm = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.warning(f"Gemini error: {e}")

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=100)
    st.header("👤 Patient Demographics")
    patient_name = st.text_input("Patient Name", "Mia Khalifa")
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    st.info("Fill out the clinical data on the main screen to run the assessment.")

# -----------------------------------
# MAIN UI
# -----------------------------------
st.title("❤️ Heart Risk AI System")
st.write("Enter the patient's medical metrics below.")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🩸 Vitals & Blood Work")
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 50, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate (thalach)", 60, 220, 150)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    with st.container(border=True):
        st.subheader("🩺 Clinical Findings")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4])
        exang = st.selectbox("Exercise Angina (exang)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ECG Upload
with st.container(border=True):
    st.subheader("📈 ECG Analysis (Optional)")
    ecg_files = st.file_uploader(
        "Upload ECG (.dat + .hea OR image)",
        type=["png", "jpg", "jpeg", "dat", "hea"],
        accept_multiple_files=True
    )

st.write("")

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Generate Diagnostic Report", type="primary", use_container_width=True):

    if clinical_model is None:
        st.error("Clinical model not loaded")
        st.stop()

    with st.spinner("Analyzing patient data..."):

        # -------- Feature Engineering --------
        chol_age_ratio = chol / age if age > 0 else 0
        bp_high = 1 if trestbps > 140 else 0
        stress_index = oldpeak * thalach

        # -------- Clinical Prediction --------
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
            "ca": ca, "chol_age_ratio": chol_age_ratio, "bp_high": bp_high,
            "stress_index": stress_index
        }])

        try:
            clinical_prob = clinical_model.predict_proba(input_data)[0][1]
            clinical_risk = clinical_prob * 100
        except Exception as e:
            st.error(f"Clinical model error: {e}")
            st.stop()

        # -------- ECG Prediction --------
        ecg_risk = None
        ecg_image_to_show = None

        if ecg_files:
            try:
                import tempfile
                import wfdb

                file_names = [f.name.lower() for f in ecg_files]

                # IMAGE CASE
                if any(name.endswith(("png", "jpg", "jpeg")) for name in file_names):
                    image_file = next(
                        f for f in ecg_files
                        if f.name.lower().endswith(("png", "jpg", "jpeg"))
                    )
                    image = Image.open(image_file).convert("RGB")
                    ecg_image_to_show = image
                    image_input = preprocess_ecg(image)
                    signal_input = np.zeros((1, 1000, 12))

                    if ecg_model is not None:
                        ecg_prob = ecg_model.predict([signal_input, image_input])[0][0]
                        ecg_risk = ecg_prob * 100
                    else:
                        st.warning("⚠️ ECG model not loaded — skipping ECG analysis.")

                # SIGNAL CASE (.dat + .hea)
                else:
                    dat_file = next((f for f in ecg_files if f.name.endswith(".dat")), None)
                    hea_file = next((f for f in ecg_files if f.name.endswith(".hea")), None)

                    if dat_file and hea_file:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            dat_path = os.path.join(tmpdir, dat_file.name)
                            hea_path = os.path.join(tmpdir, hea_file.name)

                            with open(dat_path, "wb") as f:
                                f.write(dat_file.getbuffer())
                            with open(hea_path, "wb") as f:
                                f.write(hea_file.getbuffer())

                            record_name = dat_path.replace(".dat", "")
                            signal, _ = wfdb.rdsamp(record_name)
                            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                            signal_input = np.expand_dims(signal, axis=0)

                            import matplotlib.pyplot as plt
                            fig, axes = plt.subplots(6, 2, figsize=(12, 8))
                            axes = axes.flatten()
                            for lead in range(12):
                                axes[lead].plot(signal[:, lead], linewidth=1.2)
                                axes[lead].grid(True)
                                axes[lead].set_xticks([])
                                axes[lead].set_yticks([])
                            plt.tight_layout()

                            img_temp = os.path.join(tmpdir, "temp.png")
                            fig.savefig(img_temp)
                            plt.close(fig)

                            image = Image.open(img_temp).convert("RGB")
                            ecg_image_to_show = image
                            image_input = preprocess_ecg(image)

                            if ecg_model is not None:
                                ecg_prob = ecg_model.predict([signal_input, image_input])[0][0]
                                ecg_risk = ecg_prob * 100
                            else:
                                st.warning("⚠️ ECG model not loaded — skipping ECG analysis.")
                    else:
                        st.warning("⚠️ Please upload BOTH .dat and .hea files.")

            except Exception as e:
                st.warning(f"ECG processing failed: {e}")

        # -------- FINAL RISK (Fix 2: outside try/except, correct indentation) --------
        if ecg_risk is not None:
            final_risk = calculate_final_risk(clinical_risk, ecg_risk)
        else:
            final_risk = clinical_risk

        risk_label = "High" if final_risk > 70 else "Moderate" if final_risk > 40 else "Low"

        # Persist to session state
        st.session_state.final_risk = final_risk
        st.session_state.risk_label = risk_label
        st.session_state.patient_name_display = patient_name
        st.session_state.ecg_image_to_show = ecg_image_to_show if ecg_files else None

        # Generate AI report immediately while we have context
        report_text = "AI report unavailable."
        if llm:
            try:
                prompt = f"""
                You are a cardiologist.
                Patient: {patient_name}
                Risk Score: {final_risk:.2f}%
                Provide 3 clinical recommendations based on this score.
                """
                response = llm.generate_content(prompt)
                if hasattr(response, "text"):
                    report_text = response.text
            except Exception as e:
                report_text = f"AI error: {e}"

        st.session_state.report_text = report_text

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------
if st.session_state.final_risk is not None:

    final_risk = st.session_state.final_risk
    risk_label = st.session_state.risk_label
    report_text = st.session_state.report_text
    display_name = st.session_state.patient_name_display
    ecg_image_to_show = st.session_state.get("ecg_image_to_show", None)

    st.markdown("---")
    st.header(f"📊 Assessment for {display_name}")

    tab1, tab2, tab3 = st.tabs(["🎯 Risk Score", "🧠 AI Insights", "📄 Export"])

    with tab1:
        colA, colB = st.columns(2)

        with colA:
            st.metric("Final Risk Score", f"{final_risk:.2f}%")

            if risk_label == "High":
                st.error(f"Risk Category: **{risk_label}**")
            elif risk_label == "Moderate":
                st.warning(f"Risk Category: **{risk_label}**")
            else:
                st.success(f"Risk Category: **{risk_label}**")

            # Fix 3: use ecg_image_to_show instead of undefined ecg_file
            if ecg_image_to_show is not None:
                st.image(ecg_image_to_show, caption="Uploaded ECG Analyzed", width=300)

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_risk,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 40], "color": "#00cc96"},
                        {"range": [40, 70], "color": "#ffa15a"},
                        {"range": [70, 100], "color": "#ef553b"},
                    ],
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#1c2833"})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info(report_text)

    with tab3:
        st.write("Generate a formal PDF report of this assessment.")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Heart Risk AI Report", ln=True, align="C")

        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Patient: {display_name}", ln=True)
        pdf.cell(0, 10, f"Risk Score: {final_risk:.2f}%", ln=True)
        # Fix 4: use session state risk_label — no scope issues
        pdf.cell(0, 10, f"Risk Category: {risk_label}", ln=True)

        pdf.ln(5)
        safe_text = report_text.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe_text)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{display_name.replace(' ', '_')}_report.pdf",
        )

else:
    st.warning("⚠️ Please generate the diagnostic report first.")