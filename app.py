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

st.set_page_config(page_title="Pearl · AQI Station Karachi", layout="wide", initial_sidebar_state="expanded", page_icon="🫁")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg:#F4F6F9; --white:#FFFFFF; --white-2:#F8F9FB; --white-3:#EEF1F5;
        --border:#E2E6ED; --border-2:#D1D5DE;
        --ink:#111827; --ink-2:#374151; --ink-3:#6B7280; --ink-4:#9CA3AF;
        --good:#10B981; --moderate:#F59E0B; --uhfs:#F97316;
        --unhealthy:#EF4444; --very:#8B5CF6; --hazard:#DC2626;
        --shadow:0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md:0 4px 12px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
        --shadow-lg:0 10px 30px rgba(0,0,0,0.08), 0 4px 8px rgba(0,0,0,0.04);
        --radius:16px;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp { background: var(--bg) !important; }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1180px; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.02em; }
    p, span, label, div { color: var(--ink-2); }
    .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* ===== FIX: Hide "keyboard_double" sidebar toggle text ===== */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] div,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] p,
    button[kind="header"] span,
    button[kind="header"] div {
        font-size: 0 !important;
        color: transparent !important;
        line-height: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* ===== FIX: Hide Streamlit's default top toolbar menu text ===== */
    .stApp header,
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ===== FIX: Remove top gap so header is fully visible ===== */
    .stApp > div > div > div {
        padding-top: 0 !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: var(--white) !important;
        border-right: 1px solid var(--border);
        box-shadow: 2px 0 8px rgba(0,0,0,0.03);
        margin-top: 0 !important;
        padding-top: 2rem !important;
    }
    section[data-testid="stSidebar"] * { color: var(--ink-2) !important; }
    section[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

    .diag-label {
        font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-4);
        letter-spacing:0.1em; text-transform:uppercase; margin:0 0 8px; font-weight:500;
    }
    .diag-row {
        display:flex; align-items:center; gap:8px;
        font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--ink-2); padding:3px 0;
    }
    .dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
    .dot-on { background:var(--good); box-shadow:0 0 6px rgba(16,185,129,0.4); }
    .dot-off { background:var(--ink-4); }

    /* ---- Panel / Card ---- */
    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: var(--shadow);
    }
    .panel-head { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; }
    .panel-title { font-family:'Inter',sans-serif; font-weight:600; font-size:15px; color:var(--ink); margin:0; }
    .panel-sub { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-4); text-transform:uppercase; letter-spacing:0.08em; font-weight:500; }

    /* ---- Header Strip ---- */
    .station-header {
        display:flex; align-items:center; justify-content:space-between;
        padding:12px 0 20px; border-bottom:1px solid var(--border); margin-bottom:20px;
        flex-wrap:wrap; gap:10px;
    }
    .station-id { font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--ink-4); letter-spacing:0.1em; font-weight:500; }
    .station-name { font-family:'Inter',sans-serif; font-weight:700; font-size:22px; color:var(--ink); margin:0; }
    .station-clock { font-family:'JetBrains Mono',monospace; font-size:12.5px; color:var(--ink-3); text-align:right; }

    /* ---- Breathing Halo Hero ---- */
    .hero { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:36px 20px 28px; }
    .halo-wrap { position:relative; width:210px; height:210px; display:flex; align-items:center; justify-content:center; margin-bottom:4px; }
    .halo-ring {
        position:absolute; inset:0; border-radius:50%;
        border:2px solid var(--ring-color); opacity:0.3;
        animation: breathe var(--breathe-speed) ease-in-out infinite;
    }
    .halo-ring.r2 { inset:16px; animation-delay: calc(var(--breathe-speed) / -2); opacity:0.18; }
    .halo-core {
        position:relative; width:164px; height:164px; border-radius:50%;
        background: var(--white);
        border:1px solid var(--border);
        box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255,255,255,0.8);
        display:flex; flex-direction:column; align-items:center; justify-content:center; z-index:2;
    }
    .halo-value { font-family:'Inter',sans-serif; font-weight:700; font-size:54px; line-height:1; color:var(--ring-color); }
    .halo-cat { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.06em; text-transform:uppercase; color:var(--ink-3); margin-top:6px; font-weight:500; }
    .hero-eyebrow { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-4); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px; font-weight:500; }
    .hero-dominant { font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--ink-3); margin-top:10px; }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity:0.3; }
        50% { transform: scale(1.08); opacity:0.08; }
    }
    @media (prefers-reduced-motion: reduce) {
        .halo-ring { animation: none !important; }
    }

    /* ---- Weather Chips ---- */
    .chip-row { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-top:10px; }
    .chip {
        background: var(--white); border:1px solid var(--border); border-radius:14px;
        padding:14px 20px; min-width:120px; text-align:center; box-shadow:var(--shadow);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .chip:hover { box-shadow:var(--shadow-md); transform:translateY(-1px); }
    .chip-label { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--ink-4); text-transform:uppercase; letter-spacing:0.06em; font-weight:500; }
    .chip-value { font-family:'Inter',sans-serif; font-weight:600; font-size:20px; color:var(--ink); margin-top:3px; }

    /* ---- Pollutant Gauges ---- */
    .gauge-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:14px; margin-top:8px; }
    @media (max-width:900px){ .gauge-grid{ grid-template-columns:repeat(3,1fr);} }
    .gauge-cell { display:flex; flex-direction:column; align-items:center; gap:8px; }
    .gauge-name { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-4); letter-spacing:0.04em; font-weight:500; }
    .gauge-val { font-family:'Inter',sans-serif; font-size:13px; font-weight:600; }
    .gauge-status { font-family:'JetBrains Mono',monospace; font-size:9.5px; text-transform:uppercase; font-weight:500; }

    /* ---- Forecast Day Tiles ---- */
    .day-tile {
        text-align:center; padding:20px 14px; border-radius:var(--radius);
        background: var(--white); border:1px solid var(--border); box-shadow:var(--shadow);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .day-tile:hover { box-shadow:var(--shadow-md); transform:translateY(-2px); }
    .day-tile .d-label { font-family:'JetBrains Mono',monospace; color:var(--ink-4); font-size:10.5px; margin:0; letter-spacing:0.04em; font-weight:500; }
    .day-tile .d-val { font-family:'Inter',sans-serif; font-weight:700; font-size:32px; margin:8px 0 4px; }
    .day-tile .d-cat { font-family:'JetBrains Mono',monospace; font-size:11px; margin:0; text-transform:uppercase; letter-spacing:0.04em; font-weight:500; }
    .day-tile .d-range { font-family:'JetBrains Mono',monospace; color:var(--ink-4); font-size:10.5px; margin-top:10px; }

    /* ---- Guidance Strip ---- */
    .guidance {
        display:flex; gap:14px; align-items:flex-start; padding:18px 22px;
        border-radius:14px; border:1px solid var(--gl-color);
        background: color-mix(in srgb, var(--gl-color) 6%, var(--white));
        box-shadow: var(--shadow);
    }
    .guidance .g-dot { width:10px; height:10px; border-radius:50%; background:var(--gl-color); margin-top:5px; flex-shrink:0; box-shadow:0 0 8px color-mix(in srgb, var(--gl-color) 40%, transparent); }
    .guidance .g-title { font-family:'Inter',sans-serif; font-weight:600; font-size:14px; color:var(--ink); margin:0 0 4px; }
    .guidance .g-body { font-family:'Inter',sans-serif; font-size:13px; color:var(--ink-3); margin:0; line-height:1.5; }

    .stAlert { border-radius: 14px !important; background: var(--white) !important; border:1px solid var(--border) !important; box-shadow:var(--shadow) !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; }
    hr { border-color: var(--border) !important; }

    .footer-note {
        text-align:center; color:var(--ink-4);
        font-family:'JetBrains Mono',monospace; font-size:10px;
        letter-spacing:0.06em; margin-top:10px; font-weight:500;
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
                    st.session_state["model_source"] = f"Hopsworks Model Registry (v{registry_model.version})"
                    return model, scaler, features
                except Exception as version_err:
                    last_error = version_err
                    continue
            raise last_error
        except Exception as e:
            st.sidebar.warning(f"Couldn't load from Hopsworks Model Registry, using local files instead. ({e})")

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    st.session_state["model_source"] = "local file (fallback)"
    return model, scaler, features


model, scaler, feature_cols = load_model()

city_label = "Karachi"

with st.sidebar:
    st.markdown("<p class='diag-label' style='margin-top:8px'>Model</p>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='diag-row'>· source&nbsp;&nbsp;<span style='color:var(--ink-3)'>{st.session_state.get('model_source', 'unknown')}</span></div>
        <div class='diag-row'>· inputs&nbsp;&nbsp;<span style='color:var(--ink-3)'>{len(feature_cols)} features</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='diag-label' style='margin-top:24px'>Data Sources</p>", unsafe_allow_html=True)
    _has_ow = bool(st.secrets.get("OPENWEATHER_API_KEY", ""))
    _has_hw = bool(st.secrets.get("HOPSWORKS_API_KEY", ""))
    sources = [
        ("OpenWeather API", "current + forecast pollutants", _has_ow),
        ("Open-Meteo API", "weather forecast", True),
        ("Hopsworks Feature Store", "today's measured trend", _has_hw),
        ("Hopsworks Model Registry", "trained model", _has_hw),
    ]
    for name, desc, ok in sources:
        dotclass = "dot-on" if ok else "dot-off"
        state = "configured" if ok else "no key"
        st.markdown(f"""
        <div class='diag-row' style='align-items:flex-start;margin-bottom:10px'>
            <span class='dot {dotclass}' style='margin-top:5px'></span>
            <span>{name}<br><span style='color:var(--ink-4);font-size:10.5px'>{desc} · {state}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='diag-row'>tz&nbsp;&nbsp;<span style='color:var(--ink-3)'>UTC+5 (PKT)</span></div>
        <div class='diag-row'>refreshed&nbsp;&nbsp;<span style='color:var(--ink-3)'>{now_karachi.strftime('%H:%M:%S')}</span></div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Feature Store recent actuals
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
    except Exception as e:
        st.sidebar.warning(f"Feature Store read failed, today's trend may be incomplete. ({e})")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Live data fetching
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
    if val <= 50: return "Good", "4.2s", "#10B981", "Clear conditions"
    elif val <= 100: return "Moderate", "3.6s", "#F59E0B", "Acceptable"
    elif val <= 150: return "Unhealthy for Sensitive", "3.0s", "#F97316", "Sensitive groups affected"
    elif val <= 200: return "Unhealthy", "2.4s", "#EF4444", "Everyone affected"
    elif val <= 300: return "Very Unhealthy", "1.8s", "#8B5CF6", "Health alert"
    else: return "Hazardous", "1.3s", "#DC2626", "Emergency conditions"


def build_forecast(feature_df, hist_lookback_df, current_aqi, current_row, feature_cols, model, scaler, hours=72):
    df = feature_df.reset_index(drop=True)
    n = min(hours, len(df))

    aqi_history = list(hist_lookback_df["aqi"]) if hist_lookback_df is not None and not hist_lookback_df.empty else []
    pm25_history = list(hist_lookback_df["pm2_5"]) if hist_lookback_df is not None and not hist_lookback_df.empty else []
    aqi_history.append(current_aqi)
    pm25_history.append(current_row.get("pm2_5", np.nan))

    def lag(hist, k):
        if len(hist) >= k:
            return hist[-k]
        return hist[0] if hist else np.nan

    def rolling(hist, k):
        window = hist[-k:] if len(hist) >= k else hist
        return float(np.mean(window)) if window else np.nan

    preds, times = [], []
    for h in range(n):
        row = df.iloc[h]
        dt = row["datetime"]
        feat = {
            "pm2_5": row.get("pm2_5", np.nan),
            "pm10": row.get("pm10", np.nan),
            "so2": row.get("so2", np.nan),
            "co": row.get("co", np.nan),
            "no2": row.get("no2", np.nan),
            "o3": row.get("o3", np.nan),
            "pressure": row.get("pressure", np.nan),
            "wind_speed": row.get("wind_speed", np.nan),
            "humidity": row.get("humidity", np.nan),
            "temperature": row.get("temperature", np.nan),
            "month": dt.month,
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "is_weekend": int(dt.dayofweek in (5, 6)),
            "aqi_lag_1": lag(aqi_history, 1),
            "aqi_lag_3": lag(aqi_history, 3),
            "aqi_lag_24": lag(aqi_history, 24),
            "pm25_lag_1": lag(pm25_history, 1),
            "pm25_lag_24": lag(pm25_history, 24),
            "aqi_rolling_3": rolling(aqi_history, 3),
            "aqi_rolling_6": rolling(aqi_history, 6),
            "aqi_rolling_24": rolling(aqi_history, 24),
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
    hist_df = (
        hist_lookback_df[hist_lookback_df["datetime"] >= now_karachi.normalize()]
        if not hist_lookback_df.empty else hist_lookback_df
    )
    current_aqi, dominant = get_aqi(pollution)
    cat, breathe_speed, color, cat_desc = aqi_info(current_aqi)

    # ---- Header ----
    st.markdown(f"""
    <div class='station-header'>
        <div>
            <p class='station-name'>🫁 Pearl · AQI Station</p>
            <p class='station-id'>{city_label.upper()} · 24.8607°N, 67.0011°E</p>
        </div>
        <div class='station-clock'>{now_karachi.strftime('%A, %d %B %Y')}<br>{now_karachi.strftime('%I:%M:%S %p')} PKT</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Hero Halo ----
    st.markdown(f"""
    <div class='hero' style='--ring-color:{color}; --breathe-speed:{breathe_speed}'>
        <p class='hero-eyebrow'>Current Air Quality Index</p>
        <div class='halo-wrap'>
            <div class='halo-ring'></div>
            <div class='halo-ring r2'></div>
            <div class='halo-core'>
                <span class='halo-value'>{current_aqi:.0f}</span>
                <span class='halo-cat'>{cat}</span>
            </div>
        </div>
        <p class='hero-dominant'>{cat_desc} · dominant pollutant: {dominant.upper()}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Weather Chips ----
    st.markdown(f"""
    <div class='chip-row'>
        <div class='chip'><div class='chip-label'>Temperature</div><div class='chip-value'>{weather['temperature_2m']:.1f}°C</div></div>
        <div class='chip'><div class='chip-label'>Humidity</div><div class='chip-value'>{weather['relative_humidity_2m']:.0f}%</div></div>
        <div class='chip'><div class='chip-label'>Wind Speed</div><div class='chip-value'>{weather['wind_speed_10m']:.1f} km/h</div></div>
        <div class='chip'><div class='chip-label'>Pressure</div><div class='chip-value'>{weather['surface_pressure']:.0f} hPa</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Pollutant Gauges ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Pollutant Levels</p>
        <p class='panel-sub'>% of health threshold</p>
    </div>""", unsafe_allow_html=True)

    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    gauges = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#10B981" if pct < 40 else "#F59E0B" if pct < 70 else "#EF4444"
        gauges += f"""
        <div class='gauge-cell'>
            <div style='width:60px;height:60px;border-radius:50%;background:conic-gradient({gcolor} {pct:.0f}%, var(--white-3) {pct:.0f}% 100%);display:flex;align-items:center;justify-content:center;box-shadow:var(--shadow);'>
                <div style='width:46px;height:46px;border-radius:50%;background:var(--white);display:flex;align-items:center;justify-content:center;'>
                    <span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span>
                </div>
            </div>
            <span class='gauge-name'>{p.upper()}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    gauges += "</div>"
    st.markdown(gauges, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Today's Trend ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Today's AQI Trend</p>
        <p class='panel-sub'>Measured · Predicted</p>
    </div>""", unsafe_allow_html=True)
    try:
        if not hist_df.empty:
            hist_df = hist_df.copy()
            hist_df = hist_df.sort_values("datetime")

        today_end = now_karachi.replace(hour=23, minute=59, second=59, microsecond=0)
        future_times, future_preds = [], []
        if not combined_df.empty:
            future_today_mask = (combined_df["datetime"] > now_karachi) & (combined_df["datetime"] <= today_end)
            future_today_df = combined_df[future_today_mask].sort_values("datetime")
            if not future_today_df.empty:
                future_times, future_preds = build_forecast(
                    future_today_df, hist_lookback_df, current_aqi, pollution,
                    feature_cols, model, scaler, hours=len(future_today_df)
                )

        if hist_df.empty and not future_times:
            st.warning("No trend data available right now — Feature Store may not have today's data yet, or the API may be rate-limited.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4.6))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FAFBFC")
            day_start_plot = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end_plot = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([day_start_plot, day_end_plot], 0, 50, alpha=0.08, color="#10B981")
            ax.fill_between([day_start_plot, day_end_plot], 50, 100, alpha=0.08, color="#F59E0B")
            ax.fill_between([day_start_plot, day_end_plot], 100, 150, alpha=0.08, color="#F97316")
            ax.fill_between([day_start_plot, day_end_plot], 150, 200, alpha=0.08, color="#EF4444")

            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=2.5, label="Actual (measured)", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=2, linestyle="--", alpha=0.7, label="Predicted", zorder=5)
                all_vals += future_preds

            ax.scatter([now_karachi], [current_aqi], color=color, s=110, zorder=6, edgecolors="#FFFFFF", linewidths=2.5, label="Now")
            ax.axhline(current_aqi, color="#9CA3AF", linestyle="--", alpha=0.35, linewidth=1)
            ax.set_ylabel("AQI", color="#6B7280", fontsize=11, fontfamily="sans-serif", fontweight=500)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, alpha=0.15, color="#D1D5DE", linestyle="-")
            ax.tick_params(colors="#6B7280", labelsize=9)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("sans-serif")
            for spine in ax.spines.values():
                spine.set_color("#E2E6ED")
            ax.legend(facecolor="#FFFFFF", edgecolor="#E2E6ED", labelcolor="#374151", fontsize=9, loc="upper right", framealpha=0.95)
            if all_vals:
                ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- 3-Day Forecast ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>3-Day Forecast</p>
        <p class='panel-sub'>Recursive hourly model</p>
    </div>""", unsafe_allow_html=True)
    try:
        if combined_df.empty:
            st.warning("Forecast data unavailable right now — API may be rate-limited or unavailable.")
        else:
            future_df = combined_df[combined_df["datetime"] > now_karachi].sort_values("datetime")
            times, forecast_aqi = build_forecast(
                future_df, hist_lookback_df, current_aqi, pollution,
                feature_cols, model, scaler, hours=72
            )

            if not times:
                st.warning("Not enough forecast data returned by the API to build a 3-day view.")
            else:
                fdf = pd.DataFrame({"datetime": times, "aqi": forecast_aqi})
                fdf["date"] = fdf["datetime"].apply(lambda d: d.date())
                unique_dates = sorted(fdf["date"].unique())[:3]

                day_cols = st.columns(len(unique_dates))
                for d, day in enumerate(unique_dates):
                    day_vals = fdf.loc[fdf["date"] == day, "aqi"]
                    day_avg = day_vals.mean()
                    day_min, day_max = day_vals.min(), day_vals.max()
                    day_cat, _, day_color, _ = aqi_info(day_avg)
                    day_date_label = pd.Timestamp(day).strftime("%d %b · %A")
                    with day_cols[d]:
                        st.markdown(f"""
                        <div class='day-tile' style='border-color:{day_color}30'>
                            <p class='d-label'>{day_date_label}</p>
                            <p class='d-val' style='color:{day_color}'>{day_avg:.0f}</p>
                            <p class='d-cat' style='color:{day_color}'>{day_cat}</p>
                            <p class='d-range'>↓ {day_min:.0f} &nbsp;—&nbsp; {day_max:.0f} ↑</p>
                        </div>
                        """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(14, 2.8))
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#FAFBFC")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.8)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.1, color=color)
                ax.axhline(100, color="#F59E0B", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.axhline(150, color="#F97316", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.set_ylabel("AQI", color="#6B7280", fontsize=10, fontfamily="sans-serif", fontweight=500)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, alpha=0.15, color="#D1D5DE", linestyle="-")
                ax.tick_params(colors="#6B7280", labelsize=8)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily("sans-serif")
                for spine in ax.spines.values():
                    spine.set_color("#E2E6ED")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if any(a > 150 for a in forecast_aqi):
                    guide_color, guide_title, guide_body = "#EF4444", "Hazardous AQI expected", "Avoid outdoor activity over the next 3 days where possible."
                elif any(a > 100 for a in forecast_aqi):
                    guide_color, guide_title, guide_body = "#F59E0B", "Elevated AQI expected", "Sensitive groups should limit prolonged time outdoors."
                else:
                    guide_color, guide_title, guide_body = "#10B981", "Within safe range", "No elevated AQI expected in the next 3 days."
                st.markdown(f"""
                <div class='guidance' style='--gl-color:{guide_color}'>
                    <span class='g-dot'></span>
                    <div><p class='g-title'>{guide_title}</p><p class='g-body'>{guide_body}</p></div>
                </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Forecast error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Current Guidance ----
    tips = {
        "Good": ("Excellent air quality — perfect for outdoor activity.", "#10B981"),
        "Moderate": ("Acceptable. Sensitive people should limit prolonged outdoor exertion.", "#F59E0B"),
        "Unhealthy for Sensitive": ("Sensitive groups should reduce outdoor activity.", "#F97316"),
        "Unhealthy": ("Reduce outdoor physical activity for everyone.", "#EF4444"),
        "Very Unhealthy": ("Avoid outdoors — use air purifiers indoors.", "#8B5CF6"),
        "Hazardous": ("Emergency conditions. Stay indoors and seek medical help if needed.", "#DC2626"),
    }
    tip_body, tip_color = tips.get(cat, ("", color))
    st.markdown(f"""
    <div class='guidance' style='--gl-color:{tip_color}'>
        <span class='g-dot'></span>
        <div><p class='g-title'>Right now: {cat}</p><p class='g-body'>{tip_body}</p></div>
    </div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error: {e}")

st.markdown(
    "<p class='footer-note'>PEARL AQI STATION · KARACHI &nbsp;·&nbsp; "
    "Hopsworks Feature Store + Model Registry &nbsp;·&nbsp; OpenWeather + Open-Meteo</p>",
    unsafe_allow_html=True,
)
