import streamlit as st
import numpy as np
import joblib

# Load model and scaler
# model = joblib.load(r"D:\Internship\Retrein_final\model\extra_trees_malaria_modelre.pkl")
# scaler = joblib.load(r"D:\Internship\Retrein_final\model\scaler2.pkl")
model = joblib.load("model/extra_trees_malaria_modelFolder.pkl")
scaler = joblib.load("model/scalerfolder.pkl")

# App title and instructions
st.title("🦟 Smart Malaria Predictor")
st.markdown("Enter symptoms and weather details to predict the risk of Malaria.")

# Input features
temperature = st.number_input("🌡️ Temperature (°C)", value=30.0)
humidity = st.number_input("💧 Humidity (%)", value=60.0)
rainfall = st.number_input("🌧️ Rainfall (mm)", value=5.0)
fever_days = st.number_input("🤒 Fever Duration (days)", value=2)
chills_days = st.number_input("🥶 Chills Duration (days)", value=1)
headache_days = st.number_input("🤕 Headache Duration (days)", value=1)
bodyache_days = st.number_input("💥 Bodyache Duration (days)", value=1)
rigors = st.selectbox("🌀 Rigors Present?", ["No", "Yes"])
rigors_value = 1 if rigors == "Yes" else 0

# Prepare input array
raw_input = np.array([[temperature, humidity, rainfall, fever_days, chills_days,
                       headache_days, bodyache_days, rigors_value]])
scaled_input = scaler.transform(raw_input)

# Display input data
st.markdown("📥 **Raw Input Data:**")
st.write(raw_input)
st.markdown("📊 **Scaled Input Data:**")
st.write(scaled_input)

# Prediction button
if st.button("🔍 Predict Malaria"):
    prob = model.predict_proba(scaled_input)[0]
    threshold = 0.7
    prediction = 1 if prob[1] > threshold else 0

    st.markdown("---")
    st.markdown("🧪 **Prediction Result**")
    if prediction == 1:
        st.error("⚠️ Malaria Present")
    else:
        st.success("✅ Malaria Absent")
