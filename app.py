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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root{
        /* Surfaces */
        --bg: #F1F4F9;
        --bg-2: #E8EDF5;
        --white: #FFFFFF;
        --white-2: #FAFBFD;
        --white-3: #EEF1F5;
        --border: #E2E6ED;
        --border-2: #D1D5DE;
        /* Ink */
        --ink: #0B1220;
        --ink-2: #1E293B;
        --ink-3: #475569;
        --ink-4: #94A3B8;
        --ink-5: #CBD5E1;
        /* Status palette */
        --good: #059669;
        --good-soft: #D1FAE5;
        --moderate: #D97706;
        --moderate-soft: #FEF3C7;
        --uhfs: #EA580C;
        --uhfs-soft: #FFEDD5;
        --unhealthy: #DC2626;
        --unhealthy-soft: #FEE2E2;
        --very: #7C3AED;
        --very-soft: #EDE9FE;
        --hazard: #B91C1C;
        --hazard-soft: #FECACA;
        /* Brand */
        --brand: #0F172A;
        --brand-2: #1E293B;
        --accent-blue: #2563EB;
        --accent-blue-soft: #DBEAFE;
        /* Shadows — stronger, more layered */
        --shadow-xs: 0 1px 2px rgba(15,23,42,0.04);
        --shadow-sm: 0 1px 2px rgba(15,23,42,0.04), 0 2px 6px rgba(15,23,42,0.05);
        --shadow-md: 0 4px 10px rgba(15,23,42,0.06), 0 2px 6px rgba(15,23,42,0.04);
        --shadow-lg: 0 10px 24px rgba(15,23,42,0.08), 0 4px 8px rgba(15,23,42,0.04);
        --shadow-glow: 0 0 0 1px rgba(15,23,42,0.04), 0 8px 28px rgba(15,23,42,0.10);
        /* Radius */
        --radius: 14px;
        --radius-lg: 18px;
        --radius-xl: 22px;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp {
        background: var(--bg) !important;
        background-image:
            radial-gradient(1200px 600px at 0% -5%, rgba(37,99,235,0.05), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(217,119,6,0.04), transparent 55%) !important;
        background-attachment: fixed !important;
    }
    .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1240px; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.025em; }
    p, span, label, div { color: var(--ink-2); }

    /* Hide Streamlit chrome */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"], button[kind="header"],
    [data-testid="stToolbar"], [data-testid="stHeader"],
    .stApp header, .stApp > header,
    .stApp > div > div > header {
        display: none !important; width: 0 !important; height: 0 !important;
        min-width: 0 !important; min-height: 0 !important;
        padding: 0 !important; margin: 0 !important;
        overflow: hidden !important; border: none !important;
    }

    /* ===== TOP BAR ===== */
    .top-bar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 6px 4px 14px; flex-wrap: wrap; gap: 8px;
    }
    .top-bar-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .top-bar-right { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .brand {
        font-weight: 800; font-size: 18px; color: var(--ink);
        letter-spacing: -0.035em; white-space: nowrap;
        display: inline-flex; align-items: center; gap: 8px;
    }
    .brand::before {
        content: ''; width: 22px; height: 22px; border-radius: 7px;
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        box-shadow: 0 4px 10px rgba(37,99,235,0.30), inset 0 1px 0 rgba(255,255,255,0.3);
    }
    .tag {
        font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
        color: var(--ink-3); background: var(--white);
        border: 1px solid var(--border); padding: 4px 9px; border-radius: 7px;
        white-space: nowrap;
        box-shadow: var(--shadow-xs);
        display: inline-flex; align-items: center;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .tag:hover { transform: translateY(-1px); box-shadow: var(--shadow-sm); }
    .tag-dot {
        display: inline-block; width: 6px; height: 6px; border-radius: 50%;
        margin-right: 6px; vertical-align: middle;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.6);
    }
    .tag-dot-on { background: var(--good); box-shadow: 0 0 0 2px rgba(5,150,105,0.15), 0 0 8px rgba(5,150,105,0.6); }
    .tag-dot-off { background: var(--ink-4); }
    .clock {
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: var(--ink-3); text-align: right; white-space: nowrap;
        line-height: 1.45;
    }
    .clock .clock-date { color: var(--ink-2); font-weight: 600; }
    .clock .clock-time { color: var(--ink-4); font-size: 10.5px; }

    /* ===== HERO ROW ===== */
    .hero-row { display: flex; gap: 18px; margin: 4px 0 18px; align-items: stretch; }
    .hero-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 26px 30px 24px;
        box-shadow: var(--shadow-md);
        flex-shrink: 0;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-width: 220px; position: relative; overflow: hidden;
        transition: box-shadow 0.25s, transform 0.2s;
    }
    .hero-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #ffffff));
    }
    .hero-card::after {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(circle at 50% 0%, color-mix(in srgb, var(--accent) 8%, transparent), transparent 70%);
        pointer-events: none;
    }
    .hero-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }
    .hero-label {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
        color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 10px; font-weight: 600;
        position: relative; z-index: 1;
    }
    .hero-cat {
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em;
        margin-top: 6px; font-weight: 700;
        padding: 3px 10px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 12%, var(--white));
        border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
        position: relative; z-index: 1;
    }
    .hero-sub {
        font-family: 'JetBrains Mono', monospace; font-size: 10px;
        color: var(--ink-3); margin-top: 10px; font-weight: 500;
        position: relative; z-index: 1;
    }

    .weather-grid {
        flex: 1; display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
    }
    @media (max-width: 700px) { .weather-grid { grid-template-columns: repeat(2, 1fr); } }
    .w-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 16px 18px 14px;
        box-shadow: var(--shadow-sm);
        display: flex; flex-direction: column; justify-content: space-between;
        gap: 10px;
        position: relative; overflow: hidden;
        transition: box-shadow 0.25s, transform 0.2s, border-color 0.25s;
    }
    .w-card::before {
        content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
        background: var(--w-accent, var(--ink-5));
        opacity: 0.85;
    }
    .w-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); border-color: var(--w-accent, var(--border-2)); }
    .w-card-top { display: flex; align-items: center; justify-content: space-between; }
    .w-icon {
        width: 30px; height: 30px; border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        background: color-mix(in srgb, var(--w-accent, #94A3B8) 13%, var(--white));
        color: var(--w-accent, var(--ink-4));
        border: 1px solid color-mix(in srgb, var(--w-accent, #94A3B8) 18%, transparent);
    }
    .w-icon svg { width: 16px; height: 16px; display: block; }
    .w-label {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
        color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.1em;
        font-weight: 600;
    }
    .w-val {
        font-weight: 800; font-size: 24px; color: var(--ink);
        line-height: 1.05; letter-spacing: -0.025em;
        display: flex; align-items: baseline; gap: 3px;
    }
    .w-val .w-unit {
        font-size: 11px; font-weight: 500; color: var(--ink-3);
        letter-spacing: 0;
    }

    /* ===== Halo / AQI ring ===== */
    .halo-mini {
        width: 130px; height: 130px; border-radius: 50%; position: relative;
        display: flex; align-items: center; justify-content: center; margin-bottom: 10px;
    }
    .halo-mini-ring {
        position: absolute; inset: 0; border-radius: 50%;
        border: 2.5px solid var(--accent); opacity: 0.30;
        animation: breathe var(--breathe-speed) ease-in-out infinite;
    }
    .halo-mini-ring.r2 { inset: 9px; animation-delay: calc(var(--breathe-speed) / -2); opacity: 0.18; }
    .halo-mini-ring.r3 { inset: 18px; animation-delay: calc(var(--breathe-speed) / -4); opacity: 0.10; }
    .halo-mini-core {
        width: 104px; height: 104px; border-radius: 50%;
        background: var(--white);
        display: flex; align-items: center; justify-content: center; z-index: 2;
        box-shadow: 0 0 0 6px var(--white), 0 8px 24px color-mix(in srgb, var(--accent) 25%, transparent);
        border: 2px solid color-mix(in srgb, var(--accent) 80%, var(--white));
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.30; }
        50% { transform: scale(1.12); opacity: 0.06; }
    }
    @media (prefers-reduced-motion: reduce) { .halo-mini-ring { animation: none !important; } }

    /* ===== PANELS ===== */
    .panel {
        background: var(--white); border: 1px solid var(--border);
        border-radius: var(--radius-xl); padding: 22px 26px 24px;
        margin: 14px 0; box-shadow: var(--shadow-sm);
        transition: box-shadow 0.25s;
    }
    .panel:hover { box-shadow: var(--shadow-md); }
    .panel-head {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 16px; padding-bottom: 14px;
        border-bottom: 1px solid var(--white-3);
    }
    .panel-head-left { display: flex; flex-direction: column; gap: 3px; }
    .panel-title { font-weight: 800; font-size: 15px; color: var(--ink); margin: 0; letter-spacing: -0.015em; }
    .panel-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 9px;
        color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.14em; font-weight: 600;
    }
    .panel-sub {
        font-family: 'JetBrains Mono', monospace; font-size: 10px;
        color: var(--ink-4); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
    }

    /* ===== POLLUTANT GAUGES ===== */
    .gauge-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; }
    @media (max-width: 900px) { .gauge-grid { grid-template-columns: repeat(3, 1fr); } }
    .gauge-cell {
        display: flex; flex-direction: column; align-items: center; gap: 9px;
        padding: 10px 6px; border-radius: var(--radius);
        transition: background 0.2s;
    }
    .gauge-cell:hover { background: var(--white-3); }
    .gauge-ring-wrap { position: relative; width: 64px; height: 64px; }
    .gauge-ring-wrap::after {
        content: ''; position: absolute; inset: 2px; border-radius: 50%;
        box-shadow: 0 4px 12px color-mix(in srgb, var(--g-color, #94A3B8) 25%, transparent);
        opacity: 0.6;
    }
    .gauge-name {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
        color: var(--ink-3); letter-spacing: 0.06em; font-weight: 700;
    }
    .gauge-val { font-weight: 800; font-size: 14px; letter-spacing: -0.02em; }
    .gauge-status {
        font-family: 'JetBrains Mono', monospace; font-size: 9px;
        text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em;
        padding: 2px 8px; border-radius: 999px;
        background: color-mix(in srgb, var(--g-color, #94A3B8) 12%, var(--white));
    }

    /* ===== DAY TILES ===== */
    .day-tile {
        text-align: center; padding: 18px 12px 14px; border-radius: var(--radius-lg);
        background: var(--white); border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.25s, transform 0.2s, border-color 0.25s;
        position: relative; overflow: hidden;
    }
    .day-tile::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--d-color, var(--ink-5));
    }
    .day-tile:hover {
        box-shadow: var(--shadow-md); transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--d-color, var(--border-2)) 50%, var(--border));
    }
    .day-tile .d-label {
        font-family: 'JetBrains Mono', monospace; color: var(--ink-3);
        font-size: 10.5px; margin: 0; letter-spacing: 0.04em; font-weight: 600;
    }
    .day-tile .d-val { font-weight: 900; font-size: 32px; margin: 8px 0 4px; letter-spacing: -0.025em; }
    .day-tile .d-cat {
        font-family: 'JetBrains Mono', monospace; font-size: 10px; margin: 0;
        text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
        padding: 2px 8px; border-radius: 999px;
        background: color-mix(in srgb, var(--d-color, var(--ink-4)) 12%, var(--white));
        display: inline-block;
    }
    .day-tile .d-range {
        font-family: 'JetBrains Mono', monospace; color: var(--ink-4);
        font-size: 10px; margin-top: 10px; font-weight: 500;
        padding-top: 8px; border-top: 1px dashed var(--white-3);
    }

    /* ===== GUIDANCE ===== */
    .guidance {
        display: flex; gap: 14px; align-items: flex-start; padding: 16px 20px;
        border-radius: var(--radius-lg); border: 1px solid color-mix(in srgb, var(--gl-color) 30%, var(--border));
        background: linear-gradient(135deg,
            color-mix(in srgb, var(--gl-color) 10%, var(--white)) 0%,
            color-mix(in srgb, var(--gl-color) 4%, var(--white)) 100%);
        box-shadow: var(--shadow-sm);
        position: relative; overflow: hidden;
    }
    .guidance::before {
        content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
        background: var(--gl-color);
    }
    .guidance .g-dot {
        width: 32px; height: 32px; border-radius: 10px;
        background: var(--gl-color); margin-top: 1px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px color-mix(in srgb, var(--gl-color) 35%, transparent);
        position: relative;
    }
    .guidance .g-dot::after {
        content: ''; width: 8px; height: 8px; border-radius: 50%;
        background: var(--white); box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
    }
    .guidance .g-title { font-weight: 700; font-size: 13.5px; color: var(--ink); margin: 0 0 3px; letter-spacing: -0.01em; }
    .guidance .g-body { font-size: 12px; color: var(--ink-3); margin: 0; line-height: 1.5; }

    .stAlert { border-radius: var(--radius) !important; background: var(--white) !important; border: 1px solid var(--border) !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; }
    hr { border-color: var(--border) !important; }

    /* ===== Layout safety: prevent left-clipping ===== */
    .stApp [data-testid="stMain"],
    .stApp [data-testid="stMainBlockContainer"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .block-container {
        padding-top: 1.25rem !important; padding-bottom: 2.5rem !important;
        padding-left: 1.25rem !important; padding-right: 1.25rem !important;
        max-width: 1240px !important;
        margin-left: auto !important; margin-right: auto !important;
    }

    .footer-note {
        text-align: center; color: var(--ink-4);
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
        letter-spacing: 0.08em; margin-top: 16px; font-weight: 500;
        padding: 14px 0 0;
        border-top: 1px solid var(--border);
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
                    return model, scaler, features, f"Hopsworks v{registry_model.version}"
                except Exception as version_err:
                    last_error = version_err
                    continue
            raise last_error
        except Exception as e:
            st.warning(f"Using local model. ({e})")

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    return model, scaler, features, "local fallback"


model, scaler, feature_cols, model_source = load_model()

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

    # ---- Pollution (OpenWeather) ----
    curr_url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
                f"?lat={LAT}&lon={LON}&appid={API_KEY}")
    curr_resp = requests.get(curr_url, timeout=15)
    curr_resp.raise_for_status()
    pollution = curr_resp.json()["list"][0]["components"]

    # ---- Pollutant forecast (OpenWeather) ----
    poll_forecast_df = pd.DataFrame()
    try:
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
    except Exception:
        pass

    # ---- Weather (Open-Meteo) — graceful fallback if 503 ----
    weather = {"temperature_2m": 0, "relative_humidity_2m": 0, "wind_speed_10m": 0, "surface_pressure": 0}
    hourly_df = pd.DataFrame()
    weather_ok = False
    try:
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
        weather_ok = True
    except Exception:
        weather_ok = False

    # ---- Merge pollutant forecast + weather ----
    combined_df = pd.DataFrame()
    if not poll_forecast_df.empty and not hourly_df.empty:
        combined_df = pd.merge_asof(
            poll_forecast_df.sort_values("datetime"),
            hourly_df.sort_values("datetime"),
            on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"),
        ).dropna(subset=["temperature"])

    return pollution, weather, combined_df, weather_ok


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
    pollution, weather, combined_df, weather_ok = fetch_current_data()
    hist_lookback_df = fetch_recent_actuals_from_feature_store(lookback_hours=72)
    hist_df = hist_lookback_df[hist_lookback_df["datetime"] >= now_karachi.normalize()] if not hist_lookback_df.empty else hist_lookback_df
    current_aqi, dominant = get_aqi(pollution)
    cat, breathe_speed, color, cat_desc = aqi_info(current_aqi)

    # ===== TOP BAR =====
    # NOTE: switched from st.markdown(..., unsafe_allow_html=True) to st.html()
    # because st.html() bypasses Streamlit's markdown parser entirely. With
    # the old path, any 4-space-indented HTML line inside an interpolated
    # f-string could be mis-rendered as a fenced code block. This is a pure
    # UI-rendering swap; no data/prediction logic touched.
    st.html(f"""
    <div class='top-bar'>
        <div class='top-bar-left'>
            <span class='brand'>Pearl AQI</span>
            <span class='tag'><svg width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' style='margin-right:4px;color:var(--accent-blue)'><path d='M12 22s-8-7.58-8-13a8 8 0 1 1 16 0c0 5.42-8 13-8 13z'/><circle cx='12' cy='9' r='2.5'/></svg>Karachi 24.86°N</span>
            <span class='tag'><span class='tag-dot {"tag-dot-on" if _has_ow else "tag-dot-off"}'></span>OpenWeather</span>
            <span class='tag'><span class='tag-dot {"tag-dot-on" if _has_hw else "tag-dot-off"}'></span>Hopsworks</span>
            <span class='tag'>Model: {model_source}</span>
            <span class='tag'>{len(feature_cols)} features</span>
        </div>
        <div class='top-bar-right'>
            <span class='clock'><span class='clock-date'>{now_karachi.strftime('%a %d %b')}</span><br><span class='clock-time'>{now_karachi.strftime('%I:%M %p')} PKT</span></span>
        </div>
    </div>
    """)

    # ===== HERO ROW =====
    # Build weather_html as a SINGLE-LINE string. Previously this was a
    # triple-quoted f-string with a leading newline and 8-space indentation
    # (because the source code itself is indented inside the if-branch).
    # When that indented string was interpolated into the outer hero-row
    # st.markdown f-string, Streamlit's markdown parser saw the 8-space-
    # indented <div class='weather-grid'> line preceded by a blank line and
    # interpreted it as a fenced code block — so the user saw the literal
    # <div class='weather-grid'>...</div> source text on a dark background
    # instead of the four weather cards. Building it as a single concatenated
    # line removes any possibility of that misinterpretation.
    if weather_ok:
        # Inline SVG icons (Feather/Lucide style, 18x18). Each <svg> uses
        # stroke="currentColor" fill="none" so the icon inherits the card's
        # accent color via the --w-accent CSS variable set on the parent.
        icon_temp = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 4v10.5a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z'/></svg>")
        icon_hum   = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 2.69l5.66 5.66a8 8 0 1 1-11.32 0z'/></svg>")
        icon_wind  = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16h2m-6.41-7.41A2 2 0 1 1 14 12H2m16.41 4.59A2 2 0 1 1 20 20H2'/></svg>")
        icon_press = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/></svg>")
        weather_html = (
            f"<div class='weather-grid'>"
            f"<div class='w-card' style='--w-accent:#EF4444'>"
            f"<div class='w-card-top'><span class='w-label'>Temperature</span><span class='w-icon'>{icon_temp}</span></div>"
            f"<div class='w-val'>{weather['temperature_2m']:.1f}°<span class='w-unit'>C</span></div>"
            f"</div>"
            f"<div class='w-card' style='--w-accent:#0EA5E9'>"
            f"<div class='w-card-top'><span class='w-label'>Humidity</span><span class='w-icon'>{icon_hum}</span></div>"
            f"<div class='w-val'>{weather['relative_humidity_2m']:.0f}<span class='w-unit'>%</span></div>"
            f"</div>"
            f"<div class='w-card' style='--w-accent:#14B8A6'>"
            f"<div class='w-card-top'><span class='w-label'>Wind</span><span class='w-icon'>{icon_wind}</span></div>"
            f"<div class='w-val'>{weather['wind_speed_10m']:.1f}<span class='w-unit'>km/h</span></div>"
            f"</div>"
            f"<div class='w-card' style='--w-accent:#8B5CF6'>"
            f"<div class='w-card-top'><span class='w-label'>Pressure</span><span class='w-icon'>{icon_press}</span></div>"
            f"<div class='w-val'>{weather['surface_pressure']:.0f}<span class='w-unit'>hPa</span></div>"
            f"</div>"
            f"</div>"
        )
    else:
        weather_html = (
            f"<div class='weather-grid'>"
            f"<div class='w-card' style='grid-column:1/-1;align-items:center;text-align:center;justify-content:center;'>"
            f"<div class='w-label'>Weather data</div>"
            f"<div style='font-size:13px;color:var(--ink-3);font-weight:500;margin-top:4px;'>Open-Meteo temporarily unavailable</div>"
            f"</div>"
            f"</div>"
        )

    # Use st.html() instead of st.markdown(unsafe_allow_html=True) so the
    # interpolated HTML is never run through the markdown parser at all.
    # This is a UI/UX-only change: same HTML, same data, same layout — just a
    # different rendering entry point that is robust against any future
    # Streamlit markdown-parser quirks.
    st.html(f"""
    <div class='hero-row'>
        <div class='hero-card' style='--accent:{color}; --breathe-speed:{breathe_speed}'>
            <div class='halo-mini'>
                <div class='halo-mini-ring'></div>
                <div class='halo-mini-ring r2'></div>
                <div class='halo-mini-ring r3'></div>
                <div class='halo-mini-core'>
                    <span style='font-weight:900;font-size:42px;color:{color};line-height:1;letter-spacing:-0.04em'>{current_aqi:.0f}</span>
                </div>
            </div>
            <div class='hero-label'>Air Quality Index</div>
            <div class='hero-cat' style='color:{color}'>{cat}</div>
            <div class='hero-sub'>{cat_desc} · {dominant.upper()}</div>
        </div>
        {weather_html}
    </div>
    """)

    # ===== POLLUTANT GAUGES =====
    st.html("""<div class='panel'><div class='panel-head'>
        <div class='panel-head-left'><span class='panel-eyebrow'>Live readings</span><p class='panel-title'>Pollutant Levels</p></div>
    </div>""")
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    gauges = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#059669" if pct < 40 else "#D97706" if pct < 70 else "#DC2626"
        gauges += f"""
        <div class='gauge-cell' style='--g-color:{gcolor}'>
            <div class='gauge-ring-wrap'>
                <div style='width:64px;height:64px;border-radius:50%;background:conic-gradient({gcolor} {pct:.0f}%, var(--white-3) {pct:.0f}% 100%);display:flex;align-items:center;justify-content:center;position:relative;z-index:1;'>
                    <div style='width:48px;height:48px;border-radius:50%;background:var(--white);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 0 1px var(--white-3);'>
                        <span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span>
                    </div>
                </div>
            </div>
            <span class='gauge-name'>{p.upper()}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    gauges += "</div>"
    st.html(gauges)
    st.html("</div>")

    # ===== TODAY'S TREND =====
    st.html("""<div class='panel'><div class='panel-head'>
        <div class='panel-head-left'><span class='panel-eyebrow'>Today · {now_karachi.strftime('%d %b')}</span><p class='panel-title'>AQI Trend</p></div>
        <p class='panel-sub'>Measured · Predicted</p>
    </div>""")
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
            if not weather_ok:
                st.info("Trend unavailable — Open-Meteo weather service is temporarily down.")
            else:
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
    st.html("</div>")

    # ===== 3-DAY FORECAST =====
    st.html("""<div class='panel'><div class='panel-head'>
        <div class='panel-head-left'><span class='panel-eyebrow'>Next 72 hours</span><p class='panel-title'>3-Day Forecast</p></div>
        <p class='panel-sub'>Hourly model prediction</p>
    </div>""")
    try:
        if combined_df.empty:
            if not weather_ok:
                st.info("Forecast unavailable — Open-Meteo weather service is temporarily down. Current AQI and pollutant levels are still live.")
            else:
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
                        st.html(f"""
                        <div class='day-tile' style='--d-color:{dcol}'>
                            <p class='d-label'>{dl}</p>
                            <p class='d-val' style='color:{dcol}'>{da:.0f}</p>
                            <p class='d-cat' style='color:{dcol}'>{dc}</p>
                            <p class='d-range'>↓ {dmi:.0f} — {dmx:.0f} ↑</p>
                        </div>""")
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
                st.html(f"""<div class='guidance' style='--gl-color:{gc}'><span class='g-dot'></span><div><p class='g-title'>{gt}</p><p class='g-body'>{gb}</p></div></div>""")
    except Exception as e:
        st.error(f"Forecast: {e}")
    st.html("</div>")

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
    st.html(f"""<div class='guidance' style='--gl-color:{tc}'><span class='g-dot'></span><div><p class='g-title'>Right now: {cat}</p><p class='g-body'>{tb}</p></div></div>""")

except Exception as e:
    st.error(f"Error: {e}")

st.html("<p class='footer-note'>PEARL AQI STATION · KARACHI · Hopsworks + OpenWeather + Open-Meteo</p>")
