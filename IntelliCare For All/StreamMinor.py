import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# PAGE CONFIG
st.set_page_config(
    page_title="💠 IntelliCare For All ",
    page_icon="💉",
    layout="wide"
)

# CUSTOM CSS 
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
    font-family: 'Poppins', sans-serif;
}
.main {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 2.5rem 3rem;
    box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    max-width: 1000px;
    margin: 40px auto;
    position: relative;
    overflow: hidden;
}
h1 {
    text-align: center;
    color: #007bff;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    color: #6c757d;
    font-size: 15px;
    margin-bottom: 2rem;
}
.section {
    background: linear-gradient(145deg, #f9fbff, #ecf3ff);
    padding: 1.6rem;
    border-radius: 15px;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}
.section h4 {
    color: #007bff;
    margin-bottom: 1rem;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #007bff, #00b4d8);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    height: 3rem;
    border: none;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 6px 14px rgba(0,123,255,0.3);
}
.stButton>button:hover {
    transform: translateY(-2px) scale(1.03);
}
.result-card {
    background: #f8faff;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-top: 1rem;
    transition: transform 0.3s ease;
}
.result-card:hover {
    transform: scale(1.02);
}
.positive {
    border-left: 6px solid #e63946;
    color: #e63946;
}
.negative {
    border-left: 6px solid #06d6a0;
    color: #06d6a0;
}
.ai-bubble {
    background: linear-gradient(90deg, #007bff, #00b4d8);
    color: white;
    padding: 12px 20px;
    border-radius: 15px;
    display: inline-block;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    margin: 10px 0;
    animation: floatIn 0.8s ease-in-out;
}
@keyframes floatIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.floating-bg {
    position: absolute;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(0,123,255,0.2), transparent);
    border-radius: 50%;
    animation: move 6s ease-in-out infinite alternate;
}
@keyframes move {
    from { transform: translate(-20px, -10px); }
    to { transform: translate(30px, 20px); }
}
.hospital-btn {
    background: linear-gradient(90deg, #007bff, #00b4d8);
    padding: 12px 25px;
    border-radius: 12px;
    color: white !important;
    text-decoration: none;
    font-weight: 600;
    transition: 0.3s ease-in-out;
}
.hospital-btn:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# FLOATING BACKGROUND 
st.markdown("""
<div class="floating-bg" style="top:-40px; left:-40px;"></div>
<div class="floating-bg" style="bottom:-60px; right:-40px; animation-delay:2s;"></div>
""", unsafe_allow_html=True)

# HEADER 
st.markdown("<h1>💠IntelliCare For All</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>India’s AI-Powered Smart Disease Diagnosis Assistant 🇮🇳</p>", unsafe_allow_html=True)
st.markdown("<div class='ai-bubble'>👋 Hello! I’m your AI health companion — let’s check your health today.</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# LOAD MODEL 
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "disease_model.pkl"))
    disease_encoder = joblib.load(os.path.join(base_dir, "disease_encoder.pkl"))
except Exception as e:
    st.error(f"⚠️ Model loading error: {e}")
    st.stop()

# PATIENT INFO 
st.markdown("<div class='section'><h4>👤 Patient Information</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Full Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
with col2:
    bp = st.selectbox("Blood Pressure", ["Low", "Medium", "High"])
    chol = st.selectbox("Cholesterol Level", ["Low", "Medium", "High"])
    date = st.date_input("Check-up Date", datetime.now())
st.markdown("</div>", unsafe_allow_html=True)

# INTERACTIVE SYMPTOMS 
st.markdown("<div class='section'><h4>🤒 Symptoms (Select Multiple)</h4>", unsafe_allow_html=True)
symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
selected_symptoms = st.multiselect("Choose symptoms:", symptoms, help="Select all that apply")
st.markdown("</div>", unsafe_allow_html=True)

# PREDICT BUTTON 
st.markdown("<h4 style='text-align:center;'>🧠 Diagnosis Report</h4>", unsafe_allow_html=True)
if st.button("🔍 Run Diagnosis"):
    if not name.strip():
        st.warning("⚠️ Please enter your name first.")
        st.stop()
    
    with st.spinner("🤖 Analyzing symptoms..."):
        time.sleep(2.5)
    
    # Prepare input in exact format used during training
    patient_data = {
        "Age": age,
        "Gender": 1 if gender == "Female" else 0,
        "Blood Pressure": {"Low": 0, "Medium": 1, "High": 2}[bp],
        "Cholesterol Level": {"Low": 0, "Medium": 1, "High": 2}[chol],
        "Fever": 1 if "Fever" in selected_symptoms else 0,
        "Cough": 1 if "Cough" in selected_symptoms else 0,
        "Fatigue": 1 if "Fatigue" in selected_symptoms else 0,
        "Difficulty Breathing": 1 if "Difficulty Breathing" in selected_symptoms else 0
    }

    df_input = pd.DataFrame([patient_data])

    # Align columns with model features
    for col in model.feature_names_in_:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model.feature_names_in_]

    try:
        pred_code = model.predict(df_input)[0]
        disease = disease_encoder.inverse_transform([pred_code])[0]
        
        # Save to patient_records.csv
        record = {
            "Name": name, "Gender": gender, "Age": age, "Blood Pressure": bp,
            "Cholesterol Level": chol, "Symptoms": ", ".join(selected_symptoms),
            "Predicted Disease": disease, "Date": date.strftime("%Y-%m-%d"),
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
        file_path = os.path.join(base_dir, "patient_records.csv")
        pd.DataFrame([record]).to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

        st.markdown("<hr>", unsafe_allow_html=True)
        if disease.lower() != "no disease":
            st.markdown(f"<div class='result-card positive'>🦠 {name}, signs indicate <b>{disease}</b>.<br>Consult a doctor immediately.</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; margin-top:20px;'><a href='https://www.google.com/maps/search/hospitals+near+me+India' target='_blank' class='hospital-btn'>🏥 Find Nearest Hospital</a></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card negative'>✅ {name}, no disease detected.<br>Maintain a healthy lifestyle 🌿</div>", unsafe_allow_html=True)
            st.balloons()
    except Exception as e:
        st.warning(f"⚠️ Prediction error: {e}")

# ------------------ FOOTER ------------------
st.markdown("<div class='ai-bubble'>✨ Thanks for using AI HealthMate! Stay healthy 💧</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color: gray; margin-top: 15px;'>Made with ❤️ in India | © 2025 AI HealthMate | Developed by <b>Navneet Kaur</b></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

