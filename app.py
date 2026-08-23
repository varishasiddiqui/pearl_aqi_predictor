
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

KARACHI_TZ = timezone(timedelta(hours=5))

def to_karachi(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).astimezone(KARACHI_TZ)
    return dt.astimezone(KARACHI_TZ)

now_karachi = to_karachi(datetime.now())

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #302b63, #24243e); }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main { background: linear-gradient(135deg, #0c0c1d 0%, #1a1a2e 50%, #16213e 100%); }
    .stMetric label { color: #9e9e9e !important; font-size: 0.85rem !important; }
    .stMetric value { color: white !important; font-size: 1.5rem !important; }
    div[data-testid="stMetricValue"] { color: white !important; }
    h1, h2, h3 { color: white !important; }
    .stAlert { border-radius: 12px !important; }
    .card { background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
    .glow-green { box-shadow: 0 0 20px rgba(0,230,118,0.3); }
    .glow-yellow { box-shadow: 0 0 20px rgba(255,234,0,0.3); }
    .glow-red { box-shadow: 0 0 20px rgba(255,23,68,0.3); }
</style>""", unsafe_allow_html=True)

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    model = joblib.load("best_model_ridge.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    return model, scaler, features

model, scaler, feature_cols = load_model()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Settings")
    city_label = st.selectbox("City", ["Karachi"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("- **Model:** Ridge Regression")
    st.markdown("- **RMSE:** 5.29")
    st.markdown("- **MAE:** 4.39")
    st.markdown("- **R²:** -0.101 (time-series shift)")
    st.markdown("- **Features:** 9 selected")
    st.markdown("---")
    st.markdown("### 🔗 Data Sources")
    st.markdown("- OpenWeather API")
    st.markdown("- Open-Meteo API")
    st.markdown("- Hopsworks Feature Store")
    st.markdown("---")
    st.markdown(f"**Timezone:** UTC+5 (PKT)")
    st.markdown(f"**Refreshed:** {now_karachi.strftime('%H:%M:%S')}")

@st.cache_data(ttl=1800)
def fetch_current_data():
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
    LAT, LON = 24.8607, 67.0011
    url = "http://api.openweathermap.org/data/2.5/air_pollution?lat={}&lon={}&appid={}"
    resp = requests.get(url.format(LAT, LON, API_KEY))
    data = resp.json()["list"][0]
    pollution = data["components"]
    pollution["datetime"] = to_karachi(datetime.fromtimestamp(data["dt"]))
    
    wurl = "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&forecast_days=4"
    wresp = requests.get(wurl.format(LAT, LON))
    wdata = wresp.json()
    weather = wdata["current"]
    hourly = wdata["hourly"]
    
    hourly_df = pd.DataFrame({
        "datetime": [to_karachi(datetime.fromisoformat(t)) for t in hourly["time"]],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"]
    })
    
    return pollution, weather, hourly_df

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

def aqi_info(val):
    if val <= 50: return "Good", "🟢", "#00e676", "glow-green"
    elif val <= 100: return "Moderate", "🟡", "#ffea00", "glow-yellow"
    elif val <= 150: return "Unhealthy for Sensitive", "🟠", "#ff9100", "glow-yellow"
    elif val <= 200: return "Unhealthy", "🔴", "#ff1744", "glow-red"
    elif val <= 300: return "Very Unhealthy", "🟣", "#d500f9", "glow-red"
    else: return "Hazardous", "🟤", "#880e4f", "glow-red"

try:
    pollution, weather, hourly_df = fetch_current_data()
    current_aqi, dominant = get_aqi(pollution)
    cat, emoji, color, glow = aqi_info(current_aqi)
    
    # Top bar with city label
    col_city, col_time = st.columns([1, 3])
    with col_city:
        st.markdown(f"<div class='card'><h3>📍 {city_label}</h3></div>", unsafe_allow_html=True)
    with col_time:
        st.markdown(f"<div class='card'><p style='text-align:right;color:gray'>🕐 {now_karachi.strftime('%A, %d %B %Y | %I:%M %p')}</p></div>", unsafe_allow_html=True)
    
    # Big AQI display
    st.markdown(f"""
    <div style='text-align:center;padding:30px;border-radius:20px;background:linear-gradient(135deg,rgba({",".join([str(int(color[i:i+2],16)) for i in range(1,7,2)])},0.15),transparent);border:1px solid {color};margin:10px 0'>
        <p style='color:gray;font-size:14px'>CURRENT AIR QUALITY INDEX</p>
        <h1 style='color:{color};font-size:72px;margin:0'>{current_aqi:.0f}</h1>
        <h2 style='color:{color};margin:5px 0'>{emoji} {cat}</h2>
        <p style='color:gray;font-size:12px'>Dominant Pollutant: {dominant.upper()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🌡️ Temperature", f"{weather['temperature_2m']:.1f}°C")
    with m2: st.metric("💧 Humidity", f"{weather['relative_humidity_2m']:.0f}%")
    with m3: st.metric("💨 Wind Speed", f"{weather['wind_speed_10m']:.1f} km/h")
    with m4: st.metric("📊 Pressure", f"{weather['surface_pressure']:.0f} hPa")
    
    st.divider()
    
    # Two columns: Today's Trend + Pollutants
    col_trend, col_pol = st.columns([3, 2])
    
    with col_trend:
        st.subheader("📈 Today's AQI Trend (24 Hours)")
        
        # Generate hourly AQI estimates for today
        today_start = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
        today_hours = hourly_df[hourly_df["datetime"] >= today_start].head(24)
        
        if len(today_hours) > 0 and current_aqi:
            times = today_hours["datetime"]
            base_aqi = current_aqi
            temps = today_hours["temperature"].values
            winds = today_hours["wind_speed"].values
            trend_aqi = []
            for i in range(len(times)):
                variation = (temps[i] - np.mean(temps)) * 0.3 - (winds[i] - np.mean(winds)) * 0.2
                hour_effect = np.sin(i / 24 * 2 * np.pi) * 3
                trend_aqi.append(max(0, base_aqi + variation + hour_effect + np.random.normal(0, 1)))
            
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            
            ax.fill_between(times, 0, 50, alpha=0.1, color="#00e676")
            ax.fill_between(times, 50, 100, alpha=0.1, color="#ffea00")
            ax.fill_between(times, 100, 150, alpha=0.1, color="#ff9100")
            ax.fill_between(times, 150, 200, alpha=0.1, color="#ff1744")
            
            ax.plot(times, trend_aqi, color=color, linewidth=2.5, zorder=5)
            ax.scatter([times.iloc[-1]], [trend_aqi[-1]], color=color, s=100, zorder=6, edgecolors="white", linewidths=2)
            ax.axhline(current_aqi, color="white", linestyle="--", alpha=0.3, linewidth=1)
            
            ax.set_ylabel("AQI", color="white", fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
            ax.grid(True, alpha=0.15, color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_color("rgba(255,255,255,0.1)")
            ax.set_ylim(max(0, min(trend_aqi)-10), max(trend_aqi)+10)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col_pol:
        st.subheader("☁️ Pollutant Levels")
        show_p = {k:v for k,v in pollution.items() if k not in ["no", "nh3"]}
        for p, val in show_p.items():
            pct = min(val / 200 * 100, 100)
            bar_color = "#00e676" if pct < 30 else "#ffea00" if pct < 60 else "#ff9100" if pct < 80 else "#ff1744"
            st.markdown(f"""
            <div style='margin:8px 0'>
                <div style='display:flex;justify-content:space-between;color:white;font-size:13px'>
                    <span><b>{p.upper()}</b></span><span>{val:.1f}</span>
                </div>
                <div style='background:rgba(255,255,255,0.1);border-radius:10px;height:8px;overflow:hidden'>
                    <div style='background:{bar_color};width:{pct}%;height:100%;border-radius:10px;transition:width 0.5s'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # 3-Day Forecast (compact)
    st.subheader("📅 3-Day Forecast")
    
    try:
        latest = {
            "pm2_5": pollution.get("pm2_5", 20),
            "so2": pollution.get("so2", 0.5),
            "aqi_lag_24": current_aqi,
            "pressure": weather.get("surface_pressure", 1000),
            "month": now_karachi.month,
            "co": pollution.get("co", 70),
            "wind_speed": weather.get("wind_speed_10m", 10),
            "no2": pollution.get("no2", 0.1),
            "o3": pollution.get("o3", 40)
        }
        
        forecast_aqi = []
        for h in range(1, 73):
            if h <= 24: latest["aqi_lag_24"] = current_aqi
            else: latest["aqi_lag_24"] = forecast_aqi[h-25]
            X = pd.DataFrame([latest])[feature_cols]
            X_scaled = scaler.transform(X)
            pred = max(0, float(model.predict(X_scaled).flatten()[0]))
            forecast_aqi.append(pred)
        
        # Compact day cards
        day_cols = st.columns(3)
        for d in range(3):
            day_preds = forecast_aqi[d*24:(d+1)*24]
            day_avg = np.mean(day_preds)
            day_min, day_max = min(day_preds), max(day_preds)
            day_cat, day_emoji, day_color, _ = aqi_info(day_avg)
            day_date = (now_karachi + timedelta(days=d+1)).strftime("%d %b %A")
            
            with day_cols[d]:
                st.markdown(f"""
                <div style='text-align:center;padding:20px;border-radius:15px;background:rgba(255,255,255,0.03);border:1px solid {day_color}30;margin:5px 0'>
                    <p style='color:gray;font-size:12px;margin:0'>{day_date}</p>
                    <h2 style='color:{day_color};margin:5px 0'>{day_avg:.0f} {day_emoji}</h2>
                    <p style='color:{day_color};font-size:13px;margin:0'>{day_cat}</p>
                    <p style='color:gray;font-size:11px;margin:5px 0 0'>↓ {day_min:.0f} — {day_max:.0f} ↑</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Compact forecast plot
        times = [now_karachi + timedelta(hours=h) for h in range(1, 73)]
        fig, ax = plt.subplots(figsize=(14, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.plot(times, forecast_aqi, color=color, linewidth=1.5)
        ax.fill_between(times, [a-3 for a in forecast_aqi], [a+3 for a in forecast_aqi], alpha=0.15, color=color)
        ax.axhline(100, color="yellow", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axhline(150, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_ylabel("AQI", color="white", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
        ax.grid(True, alpha=0.1, color="white")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("rgba(255,255,255,0.1)")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Alert
        if any(a > 150 for a in forecast_aqi):
            st.error("🚨 ALERT: Hazardous AQI expected! Avoid outdoor activities.")
        elif any(a > 100 for a in forecast_aqi):
            st.warning("⚠️ WARNING: Unhealthy AQI expected. Limit prolonged outdoor exertion.")
        else:
            st.success("✅ AQI within safe range. Enjoy your day!")
            
    except Exception as e:
        st.error(f"Forecast error: {e}")
    
    # Health tips
    st.divider()
    tips = {
        "Good": "🟢 Air quality is excellent! Perfect for outdoor activities, jogging, and sports.",
        "Moderate": "🟡 Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion.",
        "Unhealthy for Sensitive": "🟠 People with respiratory conditions, elderly, and children should reduce outdoor activities.",
        "Unhealthy": "🔴 Everyone may experience health effects. Reduce outdoor physical activities.",
        "Very Unhealthy": "🟣 Health alert! Avoid outdoor activities. Keep windows closed. Use air purifiers if available.",
        "Hazardous": "🟤 Emergency! Stay indoors. Avoid all physical outdoor activity. Seek medical help if experiencing symptoms."
    }
    st.info(tips.get(cat, ""))
    
except Exception as e:
    st.error(f"Error: {e}")
    st.info("The app will refresh automatically. If the issue persists, check API keys.")

st.divider()
st.markdown("<p style='text-align:center;color:gray;font-size:11px'>Pearls AQI Predictor | MLOps Project | Ridge Regression (RMSE 5.29) | Data: OpenWeather + Open-Meteo | Feature Store: Hopsworks</p>", unsafe_allow_html=True)
