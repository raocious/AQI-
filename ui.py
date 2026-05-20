import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Set up a clean, minimalist layout
st.set_page_config(page_title="Bangladesh AQI Prediction", page_icon="🌍", layout="centered")

# --- LOAD TARGETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('bangladesh_aqi_ann_model.keras')
    scaler = joblib.load('aqi_scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Could not load model or scaler. Make sure 'bangladesh_aqi_ann_model.keras' and 'aqi_scaler.joblib' are in the same folder.")

# --- UI HEADER ---
st.title("Bangladesh Air Quality Index (AQI) Predictor")
st.markdown("Enter the coordinates and pollutant levels below to estimate the real-time Air Quality Index using our deep learning model.")
st.write("---")

# --- INPUT FIELDS ---
st.subheader("Input Features")

# Create a clean two-column input layout
col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Latitude", value=23.7298, format="%.4f")
    pm10 = st.number_input("PM10 Level (µg/m³)", value=25.0, min_value=0.0)
    carbon_monoxide = st.number_input("Carbon Monoxide (CO)", value=252.0, min_value=0.0)
    sulphur_dioxide = st.number_input("Sulphur Dioxide (SO2)", value=5.2, min_value=0.0)

with col2:
    lon = st.number_input("Longitude", value=90.3854, format="%.4f")
    pm2_5 = st.number_input("PM2.5 Level (µg/m³)", value=16.7, min_value=0.0)
    nitrogen_dioxide = st.number_input("Nitrogen Dioxide (NO2)", value=18.6, min_value=0.0)
    ozone = st.number_input("Ozone (O3)", value=13.0, min_value=0.0)

st.write("---")

# --- PREDICTION LOGIC ---
if st.button("Predict AQI", type="primary", use_container_width=True):
    # 1. Apply the exact same log1p transformation you did in preprocessing for skewed columns
    # Order must match your feature list exactly: ['lat', 'lon', 'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
    pm10_log = np.log1p(pm10)
    pm2_5_log = np.log1p(pm2_5)
    co_log = np.log1p(carbon_monoxide)
    no2_log = np.log1p(nitrogen_dioxide)
    so2_log = np.log1p(sulphur_dioxide)
    
    # Assemble row array
    raw_features = np.array([[lat, lon, pm10_log, pm2_5_log, co_log, no2_log, so2_log, ozone]])
    
    # 2. Scale using the saved scaler
    scaled_features = scaler.transform(raw_features)
    
    # 3. Generate prediction
    prediction = model.predict(scaled_features)
    predicted_aqi = float(prediction[0][0])
    
    # --- DISPLAY METRICS & CATEGORIES ---
    st.subheader("Prediction Result")
    
    # Determine look & category color matching standard global AQI health guidelines
    if predicted_aqi <= 50:
        color = "green"
        status = "Good 😄"
    elif predicted_aqi <= 100:
        color = "orange"
        status = "Moderate 😐"
    elif predicted_aqi <= 150:
        color = "orange"
        status = "Unhealthy for Sensitive Groups 😷"
    elif predicted_aqi <= 200:
        color = "red"
        status = "Unhealthy 🚨"
    else:
        color = "purple"
        status = "Hazardous ☠️"
        
    # Styled Display Card
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 8px solid {color};">
            <h4 style="margin: 0; color: #31333F;">Predicted AQI Score:</h4>
            <h1 style="margin: 5px 0; color: {color};">{predicted_aqi:.1f}</h1>
            <p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: #31333F;">Air Quality Status: {status}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )