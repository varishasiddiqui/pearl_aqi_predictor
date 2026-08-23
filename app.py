
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
st.title("🌍 Pearls AQI Predictor - Karachi")

@st.cache_resource
def load_model():
    model = joblib.load("best_model_ridge.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    return model, scaler, features

model, scaler, feature_cols = load_model()

@st.cache_data(ttl=3600)
def fetch_current_data():
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
    LAT, LON = 24.8607, 67.0011
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution?lat={}&lon={}&appid={}"
    resp = requests.get(url.format(LAT, LON, API_KEY))
    pollution = resp.json()["list"][0]["components"]
    
    wurl = "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    wresp = requests.get(wurl.format(LAT, LON))
    weather = wresp.json()["current"]
    
    return pollution, weather

BREAKPOINTS = {
    "pm2_5": [(0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10": [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "no2": [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],
    "so2": [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200)],
    "o3": [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)],
    "co": [(0,4400,0,50),(4401,9400,51,100),(9401,12400,101,150),(12401,15400,151,200)]
}

def calc_aqi(pollutant, conc):
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS.get(pollutant, []):
        if conc <= c_hi:
            return round(((i_hi-i_lo)/(c_hi-c_lo))*(conc-c_lo)+i_lo, 1)
    return 500

def get_aqi(components):
    indices = {}
    for p in ["pm2_5","pm10","no2","so2","o3","co"]:
        indices[p] = calc_aqi(p, components.get(p, 0))
    return max(indices.values()), max(indices, key=indices.get)

def aqi_category(val):
    if val <= 50: return "Good", "🟢"
    elif val <= 100: return "Moderate", "🟡"
    elif val <= 150: return "Unhealthy for Sensitive", "🟠"
    elif val <= 200: return "Unhealthy", "🔴"
    elif val <= 300: return "Very Unhealthy", "🟣"
    else: return "Hazardous", "🟤"

try:
    pollution, weather = fetch_current_data()
    current_aqi, dominant = get_aqi(pollution)
    cat, emoji = aqi_category(current_aqi)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current AQI", f"{current_aqi} {emoji}")
    with col2:
        st.metric("Category", cat)
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M %d %b"))
    
    st.subheader("📊 Pollutant Levels")
    pcols = st.columns(6)
    for i, (p, val) in enumerate(list(pollution.items())[:6]):
        with pcols[i]:
            st.metric(p.upper(), f"{val:.1f}")
    
    st.subheader("📅 3-Day AQI Forecast")
    
    try:
        latest = {
            "pm2_5": pollution.get("pm2_5", 20),
            "so2": pollution.get("so2", 0.5),
            "aqi_lag_24": current_aqi,
            "pressure": weather.get("surface_pressure", 1000),
            "month": datetime.now().month,
            "co": pollution.get("co", 70),
            "wind_speed": weather.get("wind_speed_10m", 10),
            "no2": pollution.get("no2", 0.1),
            "o3": pollution.get("o3", 40)
        }
        
        forecast_times = []
        forecast_aqi = []
        
        for h in range(1, 73):
            if h <= 24:
                latest["aqi_lag_24"] = current_aqi
            else:
                latest["aqi_lag_24"] = forecast_aqi[h-25]
            
            X = pd.DataFrame([latest])[feature_cols]
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled).flatten()[0])
            pred = max(0, pred)
            forecast_times.append(datetime.now() + timedelta(hours=h))
            forecast_aqi.append(pred)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(forecast_times, forecast_aqi, color="steelblue", linewidth=1.5)
        ax.fill_between(forecast_times, [a-3 for a in forecast_aqi], [a+3 for a in forecast_aqi], alpha=0.2)
        ax.axhspan(0, 50, alpha=0.1, color="green")
        ax.axhspan(50, 100, alpha=0.1, color="yellow")
        ax.axhspan(100, 150, alpha=0.1, color="orange")
        ax.axhspan(150, 200, alpha=0.1, color="red")
        ax.set_ylabel("AQI")
        ax.set_title("3-Day AQI Forecast")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("📋 Day-wise Summary")
        for d in range(3):
            day_preds = forecast_aqi[d*24:(d+1)*24]
            if len(day_preds) > 0:
                day_avg = np.mean(day_preds)
                day_cat, day_emoji = aqi_category(day_avg)
                day_date = (datetime.now() + timedelta(days=d+1)).strftime("%d %b %Y")
                st.write(f"{day_emoji} **{day_date}**: Avg AQI **{day_avg:.1f}** ({day_cat})")
        
        if any(a > 150 for a in forecast_aqi):
            st.error("🚨 ALERT: Hazardous AQI expected!")
        elif any(a > 100 for a in forecast_aqi):
            st.warning("⚠️ WARNING: Unhealthy AQI expected")
        else:
            st.success("✅ AQI levels within safe range")
            
    except Exception as e:
        st.error(f"Forecast error: {e}")
    
except Exception as e:
    st.error(f"Error: {e}")

st.divider()
st.caption("Pearls AQI Predictor | MLOps Project")
