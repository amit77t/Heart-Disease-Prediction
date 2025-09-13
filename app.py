import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page configuration
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# Title and description
st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Enter your health details to check your heart disease risk using a Machine Learning model.</p>",
    unsafe_allow_html=True
)

# Sidebar for additional info
st.sidebar.header("About This App")
st.sidebar.info(
    """
    This app predicts the risk of heart disease based on your health parameters.
    - Uses a trained KNN model
    - Provides personalized predictions
    - For educational purposes only
    """
)

st.header("Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40, help="Your age in years")
    sex = st.radio("Sex", ["Male", "Female"])
with col2:
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 22.0, help="Your BMI calculated as weight(kg)/height(m)^2")
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

st.header("Medical Details")
col3, col4 = st.columns(2)
with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="Type of chest pain you experience")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
with col4:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    family_history = st.radio("Family History of Heart Disease", ["Yes", "No"])

# Predict button
if st.button("Predict"):
    # Convert categorical inputs to model-ready format
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'BMI': bmi,
        'Sex_' + ("M" if sex=="Male" else "F"): 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + ("Y" if exercise_angina=="Yes" else "N"): 1,
        'ST_Slope_' + st_slope: 1,
        'SmokingStatus_' + smoking_status: 1,
        'FamilyHistory_' + ("Y" if family_history=="Yes" else "N"): 1
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]
    risk_proba = model.predict_proba(scaled_input)[0][1]  # probability of high risk

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! ({risk_proba*100:.1f}% probability)")
        st.info(
            """
            Recommendations:
            - Consult a cardiologist
            - Maintain a healthy diet
            - Regular exercise
            - Avoid smoking
            - Monitor blood pressure and cholesterol
            """
        )
    else:
        st.success(f"✅ Low Risk of Heart Disease ({(1-risk_proba)*100:.1f}% probability)")
        st.info(
            """
            Tips to maintain heart health:
            - Balanced diet
            - Regular physical activity
            - Avoid smoking
            - Routine checkups
            """
        )
