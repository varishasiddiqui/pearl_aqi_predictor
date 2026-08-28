import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

KARACHI_TZ = timezone(timedelta(hours=5))
LAT, LON = 24.8607, 67.0011
FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_predictor_karachi"

now_karachi = pd.Timestamp.now(tz="UTC").tz_convert(KARACHI_TZ)

st.set_page_config(page_title="Pearl · AQI Station Karachi", layout="wide", page_icon="🌆")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg:#F4F6F9; --white:#FFFFFF; --white-2:#F8F9FB; --white-3:#EEF1F5;
        --border:#E2E6ED; --border-2:#D1D5DE;
        --ink:#0F172A; --ink-2:#334155; --ink-3:#64748B; --ink-4:#94A3B8;
        --good:#059669; --moderate:#D97706; --uhfs:#EA580C;
        --unhealthy:#DC2626; --very:#7C3AED; --hazard:#B91C1C;
        --shadow:0 1px 2px rgba(0,0,0,0.05);
        --shadow-sm:0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md:0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.03);
        --radius:12px; --radius-lg:16px;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp { background: var(--bg) !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.025em; }
    p, span, label, div { color: var(--ink-2); }

    /* ===== Kill sidebar completely ===== */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"], button[kind="header"],
    [data-testid="stToolbar"], [data-testid="stHeader"],
    .stApp header, .stApp > header,
    .stApp > div > div > header {
        display: none !important;
        width: 0 !important; height: 0 !important;
        min-width: 0 !important; min-height: 0 !important;
        padding: 0 !important; margin: 0 !important;
        overflow: hidden !important; border: none !important;
    }

    /* ===== Top info bar ===== */
    .top-bar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 0 8px; flex-wrap: wrap; gap: 6px;
    }
    .top-bar-left { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .top-bar-right { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .brand { font-weight: 800; font-size: 17px; color: var(--ink); letter-spacing: -0.03em; white-space: nowrap; }
    .tag {
        font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
        color: var(--ink-3); background: var(--white); border: 1px solid var(--border);
        padding: 3px 8px; border-radius: 6px; white-space: nowrap;
    }
    .tag-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
    .tag-dot-on { background: var(--good); }
    .tag-dot-off { background: var(--ink-4); }
    .clock {
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: var(--ink-3); text-align: right; white-space: nowrap;
    }

    /* ===== Hero row: AQI + weather side by side ===== */
    .hero-row { display: flex; gap: 16px; margin: 8px 0 12px; align-items: stretch; }
    .hero-card {
        background: var(--white); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 24px 28px;
        box-shadow: var(--shadow-sm); flex-shrink: 0;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-width: 200px; position: relative; overflow: hidden;
    }
    .hero-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent);
    }
    .hero-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; font-weight: 500; }
    .hero-value { font-weight: 800; font-size: 52px; line-height: 1; color: var(--accent); }
    .hero-cat { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; font-weight: 500; }
    .hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-4); margin-top: 8px; }

    /* Weather grid in hero row */
    .weather-grid {
        flex: 1; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    }
    @media (max-width: 700px) { .weather-grid { grid-template-columns: repeat(2, 1fr); } }
    .w-card {
        background: var(--white); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 16px 18px;
        box-shadow: var(--shadow-sm); display: flex; flex-direction: column; justify-content: center;
    }
    .w-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; font-weight: 500; }
    .w-val { font-weight: 700; font-size: 22px; color: var(--ink); line-height: 1.1; }

    /* ===== Breathing halo (mini, inside hero card) ===== */
    .halo-mini {
        width: 120px; height: 120px; border-radius: 50%; position: relative;
        display: flex; align-items: center; justify-content: center; margin-bottom: 8px;
    }
    .halo-mini-ring {
        position: absolute; inset: 0; border-radius: 50%;
        border: 2px solid var(--accent); opacity: 0.25;
        animation: breathe var(--breathe-speed) ease-in-out infinite;
    }
    .halo-mini-ring.r2 { inset: 8px; animation-delay: calc(var(--breathe-speed) / -2); opacity: 0.15; }
    .halo-mini-core {
        width: 96px; height: 96px; border-radius: 50%;
        background: var(--white); border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center; z-index: 2;
        box-shadow: var(--shadow-md);
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.25; }
        50% { transform: scale(1.1); opacity: 0.06; }
    }
    @media (prefers-reduced-motion: reduce) { .halo-mini-ring { animation: none !important; } }

    /* ===== Panel ===== */
    .panel {
        background: var(--white); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 20px 24px;
        margin: 12px 0; box-shadow: var(--shadow-sm);
    }
    .panel-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 14px; }
    .panel-title { font-weight: 700; font-size: 14px; color: var(--ink); margin: 0; }
    .panel-sub { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }

    /* ===== Pollutant Gauges ===== */
    .gauge-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
    @media (max-width: 900px) { .gauge-grid { grid-template-columns: repeat(3, 1fr); } }
    .gauge-cell { display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .gauge-name { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-4); letter-spacing: 0.03em; font-weight: 500; }
    .gauge-val { font-weight: 700; font-size: 12px; }
    .gauge-status { font-family: 'JetBrains Mono', monospace; font-size: 9px; text-transform: uppercase; font-weight: 500; }

    /* ===== Forecast tiles ===== */
    .day-tile {
        text-align: center; padding: 16px 12px; border-radius: var(--radius);
        background: var(--white); border: 1px solid var(--border); box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .day-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
    .day-tile .d-label { font-family: 'JetBrains Mono', monospace; color: var(--ink-4); font-size: 10px; margin: 0; letter-spacing: 0.03em; font-weight: 500; }
    .day-tile .d-val { font-weight: 800; font-size: 28px; margin: 6px 0 2px; }
    .day-tile .d-cat { font-family: 'JetBrains Mono', monospace; font-size: 10px; margin: 0; text-transform: uppercase; letter-spacing: 0.03em; font-weight: 500; }
    .day-tile .d-range { font-family: 'JetBrains Mono', monospace; color: var(--ink-4); font-size: 9.5px; margin-top: 8px; }

    /* ===== Guidance ===== */
    .guidance {
        display: flex; gap: 12px; align-items: flex-start; padding: 14px 18px;
        border-radius: var(--radius); border: 1px solid var(--gl-color);
        background: color-mix(in srgb, var(--gl-color) 5%, var(--white));
        box-shadow: var(--shadow-sm);
    }
    .guidance .g-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--gl-color); margin-top: 5px; flex-shrink: 0; }
    .guidance .g-title { font-weight: 600; font-size: 13px; color: var(--ink); margin: 0 0 2px; }
    .guidance .g-body { font-size: 12px; color: var(--ink-3); margin: 0; line-height: 1.45; }

    .stAlert { border-radius: var(--radius) !important; background: var(--white) !important; border: 1px solid var(--border) !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; }
    hr { border-color: var(--border) !important; }

    .footer-note {
        text-align: center; color: var(--ink-4);
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
        letter-spacing: 0.05em; margin-top: 8px; font-weight: 500;
    }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    hopsworks_key = st.secrets.get("HOPSWORKS_API_KEY", "")
    if hopsworks_key:
        try:
            import hopsworks
            project = hopsworks.login(api_key_value=hopsworks_key)
            mr = project.get_model_registry()
            all_versions = mr.get_models(MODEL_NAME)
            if not all_versions:
                raise RuntimeError(f"No versions registered yet for model '{MODEL_NAME}'.")
            all_versions = sorted(all_versions, key=lambda m: m.version, reverse=True)
            last_error = None
            for registry_model in all_versions:
                try:
                    import os
                    model_dir = registry_model.download()
                    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
                    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
                    features = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
                    st.session_state["model_source"] = f"Hopsworks v{registry_model.version}"
                    return model, scaler, features
                except Exception as version_err:
                    last_error = version_err
                    continue
            raise last_error
        except Exception as e:
            st.session_state["model_source"] = "local fallback"
            st.warning(f"Using local model. ({e})")

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    st.session_state["model_source"] = "local fallback"
    return model, scaler, features


model, scaler, feature_cols = load_model()

_has_ow = bool(st.secrets.get("OPENWEATHER_API_KEY", ""))
_has_hw = bool(st.secrets.get("HOPSWORKS_API_KEY", ""))

# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_recent_actuals_from_feature_store(lookback_hours=72):
    hopsworks_key = st.secrets.get("HOPSWORKS_API_KEY", "")
    if not hopsworks_key:
        return pd.DataFrame()
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=hopsworks_key)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        if df.empty:
            return df
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(KARACHI_TZ)
        window_start = now_karachi - pd.Timedelta(hours=lookback_hours)
        df = df[(df["datetime"] >= window_start) & (df["datetime"] <= now_karachi)]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_current_data():
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    curr_url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
                f"?lat={LAT}&lon={LON}&appid={API_KEY}")
    curr_resp = requests.get(curr_url, timeout=15)
    curr_resp.raise_for_status()
    pollution = curr_resp.json()["list"][0]["components"]

    fc_url = (f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
              f"?lat={LAT}&lon={LON}&appid={API_KEY}")
    fc_resp = requests.get(fc_url, timeout=15)
    fc_resp.raise_for_status()
    fc_list = fc_resp.json().get("list", [])
    poll_forecast_df = pd.DataFrame([
        {"datetime": pd.to_datetime(item["dt"], unit="s", utc=True).tz_convert(KARACHI_TZ),
         **item["components"]}
        for item in fc_list
    ])
    if not poll_forecast_df.empty:
        poll_forecast_df["datetime"] = poll_forecast_df["datetime"].dt.as_unit("ns")

    w_url = ("https://api.open-meteo.com/v1/forecast"
             f"?latitude={LAT}&longitude={LON}"
             "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
             "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
             "&forecast_days=4")
    w_resp = requests.get(w_url, timeout=15)
    w_resp.raise_for_status()
    w_data = w_resp.json()
    weather = w_data["current"]
    hourly = w_data["hourly"]

    hourly_times = pd.to_datetime(hourly["time"]).tz_localize("UTC").tz_convert(KARACHI_TZ).as_unit("ns")
    hourly_df = pd.DataFrame({
        "datetime": hourly_times,
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"],
    })

    combined_df = pd.DataFrame()
    if not poll_forecast_df.empty:
        combined_df = pd.merge_asof(
            poll_forecast_df.sort_values("datetime"),
            hourly_df.sort_values("datetime"),
            on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"),
        ).dropna(subset=["temperature"])

    return pollution, weather, combined_df


BREAKPOINTS = {
    "pm2_5": [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300)],
    "co": [(0, 4400, 0, 50), (4401, 9400, 51, 100), (9401, 12400, 101, 150), (12401, 15400, 151, 200)],
}


def calc_aqi(pollutant, conc):
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS.get(pollutant, []):
        if conc <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo, 1)
    return 500


def get_aqi(components):
    indices = {}
    for p in ["pm2_5", "pm10", "no2", "so2", "o3", "co"]:
        indices[p] = calc_aqi(p, components.get(p, 0))
    return max(indices.values()), max(indices, key=indices.get)


def aqi_info(val):
    if val <= 50: return "Good", "4.2s", "#059669", "Clear air"
    elif val <= 100: return "Moderate", "3.6s", "#D97706", "Acceptable"
    elif val <= 150: return "Sensitive Groups", "3.0s", "#EA580C", "Limit outdoors"
    elif val <= 200: return "Unhealthy", "2.4s", "#DC2626", "Reduce activity"
    elif val <= 300: return "Very Unhealthy", "1.8s", "#7C3AED", "Health alert"
    else: return "Hazardous", "1.3s", "#B91C1C", "Stay indoors"


def build_forecast(feature_df, hist_lookback_df, current_aqi, current_row, feature_cols, model, scaler, hours=72):
    df = feature_df.reset_index(drop=True)
    n = min(hours, len(df))
    aqi_history = list(hist_lookback_df["aqi"]) if hist_lookback_df is not None and not hist_lookback_df.empty else []
    pm25_history = list(hist_lookback_df["pm2_5"]) if hist_lookback_df is not None and not hist_lookback_df.empty else []
    aqi_history.append(current_aqi)
    pm25_history.append(current_row.get("pm2_5", np.nan))

    def lag(hist, k):
        return hist[-k] if len(hist) >= k else (hist[0] if hist else np.nan)
    def rolling(hist, k):
        w = hist[-k:] if len(hist) >= k else hist
        return float(np.mean(w)) if w else np.nan

    preds, times = [], []
    for h in range(n):
        row = df.iloc[h]
        dt = row["datetime"]
        feat = {
            "pm2_5": row.get("pm2_5", np.nan), "pm10": row.get("pm10", np.nan),
            "so2": row.get("so2", np.nan), "co": row.get("co", np.nan),
            "no2": row.get("no2", np.nan), "o3": row.get("o3", np.nan),
            "pressure": row.get("pressure", np.nan), "wind_speed": row.get("wind_speed", np.nan),
            "humidity": row.get("humidity", np.nan), "temperature": row.get("temperature", np.nan),
            "month": dt.month, "hour": dt.hour, "day_of_week": dt.dayofweek,
            "is_weekend": int(dt.dayofweek in (5, 6)),
            "aqi_lag_1": lag(aqi_history, 1), "aqi_lag_3": lag(aqi_history, 3),
            "aqi_lag_24": lag(aqi_history, 24), "pm25_lag_1": lag(pm25_history, 1),
            "pm25_lag_24": lag(pm25_history, 24), "aqi_rolling_3": rolling(aqi_history, 3),
            "aqi_rolling_6": rolling(aqi_history, 6), "aqi_rolling_24": rolling(aqi_history, 24),
            "pm25_rolling_24": rolling(pm25_history, 24),
        }
        X = pd.DataFrame([feat])[feature_cols]
        X_scaled = scaler.transform(X)
        pred = max(0, float(model.predict(X_scaled).flatten()[0]))
        preds.append(pred)
        times.append(dt)
        aqi_history.append(pred)
        pm25_history.append(row.get("pm2_5", np.nan))
    return times, preds


try:
    pollution, weather, combined_df = fetch_current_data()
    hist_lookback_df = fetch_recent_actuals_from_feature_store(lookback_hours=72)
    hist_df = hist_lookback_df[hist_lookback_df["datetime"] >= now_karachi.normalize()] if not hist_lookback_df.empty else hist_lookback_df
    current_aqi, dominant = get_aqi(pollution)
    cat, breathe_speed, color, cat_desc = aqi_info(current_aqi)

    # ===== TOP BAR: brand + all info in one line =====
    st.markdown(f"""
    <div class='top-bar'>
        <div class='top-bar-left'>
            <span class='brand'>Pearl AQI</span>
            <span class='tag'>Karachi 24.86°N</span>
            <span class='tag'><span class='tag-dot {"tag-dot-on" if _has_ow else "tag-dot-off"}'></span>OpenWeather</span>
            <span class='tag'><span class='tag-dot {"tag-dot-on" if _has_hw else "tag-dot-off"}'></span>Hopsworks</span>
            <span class='tag'>Model: {st.session_state.get("model_source", "—")}</span>
            <span class='tag'>{len(feature_cols)} features</span>
        </div>
        <div class='top-bar-right'>
            <span class='clock'>{now_karachi.strftime('%a %d %b')}<br>{now_karachi.strftime('%I:%M %p')} PKT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== HERO ROW: AQI card + 4 weather cards side by side =====
    st.markdown(f"""
    <div class='hero-row'>
        <div class='hero-card' style='--accent:{color}; --breathe-speed:{breathe_speed}'>
            <div class='halo-mini'>
                <div class='halo-mini-ring'></div>
                <div class='halo-mini-ring r2'></div>
                <div class='halo-mini-core'>
                    <span style='font-weight:800;font-size:36px;color:{color};line-height:1'>{current_aqi:.0f}</span>
                </div>
            </div>
            <div class='hero-label'>Air Quality Index</div>
            <div class='hero-cat' style='color:{color}'>{cat}</div>
            <div class='hero-sub'>{cat_desc} · {dominant.upper()}</div>
        </div>
        <div class='weather-grid'>
            <div class='w-card'>
                <div class='w-label'>Temperature</div>
                <div class='w-val'>{weather['temperature_2m']:.1f}°</div>
            </div>
            <div class='w-card'>
                <div class='w-label'>Humidity</div>
                <div class='w-val'>{weather['relative_humidity_2m']:.0f}%</div>
            </div>
            <div class='w-card'>
                <div class='w-label'>Wind</div>
                <div class='w-val'>{weather['wind_speed_10m']:.1f}<span style='font-size:12px;font-weight:500;color:var(--ink-3)'> km/h</span></div>
            </div>
            <div class='w-card'>
                <div class='w-label'>Pressure</div>
                <div class='w-val'>{weather['surface_pressure']:.0f}<span style='font-size:11px;font-weight:500;color:var(--ink-3)'> hPa</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== POLLUTANT GAUGES =====
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Pollutant Levels</p>
        <p class='panel-sub'>% of threshold</p>
    </div>""", unsafe_allow_html=True)
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    gauges = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#059669" if pct < 40 else "#D97706" if pct < 70 else "#DC2626"
        gauges += f"""
        <div class='gauge-cell'>
            <div style='width:52px;height:52px;border-radius:50%;background:conic-gradient({gcolor} {pct:.0f}%, var(--white-3) {pct:.0f}% 100%);display:flex;align-items:center;justify-content:center;'>
                <div style='width:40px;height:40px;border-radius:50%;background:var(--white);display:flex;align-items:center;justify-content:center;'>
                    <span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span>
                </div>
            </div>
            <span class='gauge-name'>{p.upper()}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    gauges += "</div>"
    st.markdown(gauges, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== TODAY'S TREND =====
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Today's AQI Trend</p>
        <p class='panel-sub'>Measured · Predicted</p>
    </div>""", unsafe_allow_html=True)
    try:
        if not hist_df.empty:
            hist_df = hist_df.copy().sort_values("datetime")
        today_end = now_karachi.replace(hour=23, minute=59, second=59, microsecond=0)
        future_times, future_preds = [], []
        if not combined_df.empty:
            ft_df = combined_df[(combined_df["datetime"] > now_karachi) & (combined_df["datetime"] <= today_end)].sort_values("datetime")
            if not ft_df.empty:
                future_times, future_preds = build_forecast(ft_df, hist_lookback_df, current_aqi, pollution, feature_cols, model, scaler, hours=len(ft_df))
        if hist_df.empty and not future_times:
            st.warning("No trend data available yet.")
        else:
            fig, ax = plt.subplots(figsize=(12, 3.8))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FAFBFC")
            ds = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            de = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([ds, de], 0, 50, alpha=0.06, color="#059669")
            ax.fill_between([ds, de], 50, 100, alpha=0.06, color="#D97706")
            ax.fill_between([ds, de], 100, 150, alpha=0.06, color="#EA580C")
            ax.fill_between([ds, de], 150, 200, alpha=0.06, color="#DC2626")
            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=2, label="Measured", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=1.8, linestyle="--", alpha=0.65, label="Predicted", zorder=5)
                all_vals += future_preds
            ax.scatter([now_karachi], [current_aqi], color=color, s=80, zorder=6, edgecolors="#FFF", linewidths=2, label="Now")
            ax.axhline(current_aqi, color="#CBD5E1", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_ylabel("AQI", color="#64748B", fontsize=10, fontweight=500)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, alpha=0.12, color="#E2E8F0", linestyle="-")
            ax.tick_params(colors="#64748B", labelsize=8)
            for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
            for s in ax.spines.values(): s.set_color("#E2E8F0")
            ax.legend(facecolor="#FFF", edgecolor="#E2E8F0", labelcolor="#334155", fontsize=8, loc="upper right", framealpha=0.95)
            if all_vals: ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== 3-DAY FORECAST =====
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>3-Day Forecast</p>
        <p class='panel-sub'>Hourly model prediction</p>
    </div>""", unsafe_allow_html=True)
    try:
        if combined_df.empty:
            st.warning("Forecast unavailable.")
        else:
            future_df = combined_df[combined_df["datetime"] > now_karachi].sort_values("datetime")
            times, forecast_aqi = build_forecast(future_df, hist_lookback_df, current_aqi, pollution, feature_cols, model, scaler, hours=72)
            if not times:
                st.warning("Not enough forecast data.")
            else:
                fdf = pd.DataFrame({"datetime": times, "aqi": forecast_aqi})
                fdf["date"] = fdf["datetime"].apply(lambda d: d.date())
                unique_dates = sorted(fdf["date"].unique())[:3]
                day_cols = st.columns(len(unique_dates))
                for d, day in enumerate(unique_dates):
                    dv = fdf.loc[fdf["date"] == day, "aqi"]
                    da, dmi, dmx = dv.mean(), dv.min(), dv.max()
                    dc, _, dcol, _ = aqi_info(da)
                    dl = pd.Timestamp(day).strftime("%d %b · %a")
                    with day_cols[d]:
                        st.markdown(f"""
                        <div class='day-tile' style='border-color:{dcol}25'>
                            <p class='d-label'>{dl}</p>
                            <p class='d-val' style='color:{dcol}'>{da:.0f}</p>
                            <p class='d-cat' style='color:{dcol}'>{dc}</p>
                            <p class='d-range'>↓ {dmi:.0f} — {dmx:.0f} ↑</p>
                        </div>""", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 2.4))
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#FAFBFC")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.5)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.08, color=color)
                ax.axhline(100, color="#D97706", linestyle="--", alpha=0.4, linewidth=0.7)
                ax.axhline(150, color="#EA580C", linestyle="--", alpha=0.4, linewidth=0.7)
                ax.set_ylabel("AQI", color="#64748B", fontsize=9, fontweight=500)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, alpha=0.12, color="#E2E8F0", linestyle="-")
                ax.tick_params(colors="#64748B", labelsize=7)
                for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
                for s in ax.spines.values(): s.set_color("#E2E8F0")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                if any(a > 150 for a in forecast_aqi):
                    gc, gt, gb = "#DC2626", "Hazardous AQI expected", "Avoid outdoor activity."
                elif any(a > 100 for a in forecast_aqi):
                    gc, gt, gb = "#D97706", "Elevated AQI expected", "Sensitive groups should limit outdoor time."
                else:
                    gc, gt, gb = "#059669", "Within safe range", "No elevated AQI expected."
                st.markdown(f"""<div class='guidance' style='--gl-color:{gc}'><span class='g-dot'></span><div><p class='g-title'>{gt}</p><p class='g-body'>{gb}</p></div></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Forecast: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== GUIDANCE =====
    tips = {
        "Good": ("Excellent air quality — perfect for outdoor activity.", "#059669"),
        "Moderate": ("Acceptable. Sensitive people should limit prolonged exertion.", "#D97706"),
        "Sensitive Groups": ("Sensitive groups should reduce outdoor activity.", "#EA580C"),
        "Unhealthy": ("Reduce outdoor physical activity for everyone.", "#DC2626"),
        "Very Unhealthy": ("Avoid outdoors — use air purifiers indoors.", "#7C3AED"),
        "Hazardous": ("Emergency. Stay indoors, seek medical help if needed.", "#B91C1C"),
    }
    tb, tc = tips.get(cat, ("", color))
    st.markdown(f"""<div class='guidance' style='--gl-color:{tc}'><span class='g-dot'></span><div><p class='g-title'>Right now: {cat}</p><p class='g-body'>{tb}</p></div></div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error: {e}")

st.markdown("<p class='footer-note'>PEARL AQI STATION · KARACHI · Hopsworks + OpenWeather + Open-Meteo</p>", unsafe_allow_html=True)
