import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, and medians
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
medians = joblib.load("diabetes_medians.pkl")  # Load saved medians used during training

# Define column names (same order as training)
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Patient input (raw values)
new_patient = pd.DataFrame([[8, 150, 64, 45, 0, 23.3, 0.672, 50]], columns=columns)

# Replace zeros in selected columns with NaN and fill with training medians
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    new_patient[col] = new_patient[col].replace(0, np.nan).fillna(medians[col])

# Scale the input
new_patient_scaled = scaler.transform(new_patient)

# Predict class and probability
prediction = model.predict(new_patient_scaled)[0]
probability = model.predict_proba(new_patient_scaled)[0][1]

# Output

print(f"\n📊 Probability of being diabetic: {probability:.2f}")
print("⚠️ Diabetic" if prediction == 1 else "✅ Not Diabetic")
