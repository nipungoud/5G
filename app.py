import streamlit as st
import torch
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from src.model import LSTMModel

# Load trained model
@st.cache_resource
def load_model():
    model = LSTMModel(input_size=7)
    model.load_state_dict(torch.load("models/lstm_model.pth"))
    model.eval()
    return model

model = load_model()
geolocator = Nominatim(user_agent="5g-predictor")

st.title("📡 India 5G Bandwidth Predictor")

# Choice for input type
option = st.radio("Choose Input Mode", ("📍 Enter Location Name", "🧭 Enter Latitude and Longitude"))

lat, lon = None, None
place_name = ""

if option == "📍 Enter Location Name":
    place_name = st.text_input("Enter a place (e.g., Hyderabad, Mumbai)")

    if place_name:
        try:
            location = geolocator.geocode(place_name + ", India", timeout=10)
            if location:
                lat = location.latitude
                lon = location.longitude
                st.success(f"📍 Location: {location.address}")
                st.write(f"Latitude: **{lat:.4f}**, Longitude: **{lon:.4f}**")
            else:
                st.error("❗ Could not find location.")
        except GeocoderTimedOut:
            st.error("⏱️ Geocoding timeout. Try again.")
            
elif option == "🧭 Enter Latitude and Longitude":
    lat = st.number_input("Latitude", value=23.0, format="%.6f")
    lon = st.number_input("Longitude", value=80.0, format="%.6f")

    if lat and lon:
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
            if location:
                place_name = location.address.split(",")[0]
                st.success(f"📍 Approximate Location: {place_name}")
        except GeocoderTimedOut:
            st.warning("⏱️ Location lookup timed out.")

# If lat/lon resolved, do prediction
if lat and lon:
    # Simulate signal values
    mobility = np.random.choice([0, 30, 60, 90])
    rssi = np.random.normal(-85, 5)
    sinr = np.random.normal(15, 3)
    rsrp = np.random.normal(-95, 4)
    rsrq = np.random.normal(-10, 2)

    # Clip values to avoid out-of-distribution input
    rssi = np.clip(rssi, -120, -70)
    sinr = np.clip(sinr, 0, 30)
    rsrp = np.clip(rsrp, -120, -75)
    rsrq = np.clip(rsrq, -20, 0)

    # Build input tensor
    input_sequence = np.array([[lat, lon, mobility, rssi, sinr, rsrp, rsrq]] * 10)
    input_tensor = torch.tensor(input_sequence).unsqueeze(0).float()

    try:
        with torch.no_grad():
            prediction = model(input_tensor).item()
            st.markdown("### 📶 Predicted Bandwidth:")
            st.success(f"**{prediction:.2f} Mbps**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
