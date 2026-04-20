import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Room Occupancy Detection",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.big-title {
    font-size:42px;
    font-weight:bold;
    color:#00ffaa;
}
.sub-text {
    color:#bbbbbb;
    font-size:18px;
}
.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-title">🏠 Smart Room Occupancy Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Predict whether room is Occupied or Empty using sensor data</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙ Input Sensor Values")
temp = st.sidebar.number_input("🌡 Temperature", value=23.0)
hum = st.sidebar.number_input("💧 Humidity", value=27.0)
light = st.sidebar.number_input("💡 Light", value=400.0)
co2 = st.sidebar.number_input("🌫 CO2", value=700.0)
ratio = st.sidebar.number_input("📊 Humidity Ratio", value=0.0047, format="%.6f")

# Main Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Temperature", temp)

with col2:
    st.metric("Humidity", hum)

with col3:
    st.metric("CO2", co2)

st.write("")
st.write("")

# Predict Button
if st.button("🔍 Predict Occupancy", use_container_width=True):
    data = np.array([[temp, hum, light, co2, ratio]])
    pred = model.predict(data)[0]

    if pred == 1:
        st.success("✅ Room is Occupied")
        st.balloons()
    else:
        st.error("❌ Room is Not Occupied")

# Footer
st.markdown("---")
st.caption("Developed using Python + Machine Learning + Streamlit")