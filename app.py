import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page setup
st.set_page_config(page_title="Heart Stroke Predictor", layout="centered")

# Custom styling with blue background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #cce7ff;
        color: #000000;
    }
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: black !important;
    }
    .stButton>button {
        color: white !important;
        background-color: #007acc !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("❤️ Heart Stroke Prediction")
st.markdown("Please provide the following health information:")

# Input fields
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction button
if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(scaled_input)[0][1]  # Probability of class 1 (high risk)
    percentage = round(prob * 100, 2)

    if percentage >= 50:
        st.error(f"⚠️ High Risk of Heart Disease: {percentage}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease: {percentage}%")