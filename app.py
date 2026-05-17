import streamlit as st
import pandas as pd
import joblib 


model = joblib.load('logistic_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("provide the following details:")
age = st.slider("age",18,100,40)
sex = st.selectbox("sex", ["male", "female"])
chest_pain = st.selectbox("chest pain type", ["ATA","NAP", "ASY", "TA"])
resting_bp = st.number_input("resting blood pressure", 80, 200, 120)
cholesterol = st.number_input("cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("fasting blood sugar > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("resting electrocardiographic results", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
max_hr = st.number_input("maximum heart rate achieved", 60, 220, 150)
exercise_angina = st.selectbox("exercise induced angina", ["yes", "no"])
st_slope = st.selectbox("ST slope", ["upsloping", "flat", "downsloping"])
oldpeak = st.number_input("oldpeak", 0.0, 6.0, 1.0)

if st.button("predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "chest_pain": [chest_pain],
        "resting_bp": [resting_bp],
        "cholesterol": [cholesterol],
        "fasting_bs": [fasting_bs],
        "resting_ecg": [resting_ecg],
        "max_hr": [max_hr],
        "exercise_angina": [exercise_angina],
        "st_slope": [st_slope],
        "oldpeak": [oldpeak]
    })
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_columns]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("high risk of heart disease")
    else:
        st.success("low risk of heart disease")