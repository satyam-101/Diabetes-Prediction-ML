import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and training medians
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
medians = joblib.load("diabetes_medians.pkl")

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient data to predict the **risk of diabetes**.")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, step=1)

if st.button("Predict"):
    # Prepare input dataframe
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Handle missing (zero) values with training medians
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        input_data[col] = input_data[col].replace(0, np.nan).fillna(medians[col])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Display result
    st.subheader("📊 Prediction Result")
    st.write("Probability of being diabetic: **{:.2f}%**".format(probability))

    if prediction == 1:
        st.warning("⚠️ This person is **diabetic**.")
    else:
        st.success("✅ This person is **not diabetic**.")
