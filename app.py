import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math
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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root{
        /* Surfaces — dark control-room palette */
        --void: #0A0D13;
        --void-2: #0D1119;
        --panel: #12161F;
        --panel-2: #171C27;
        --panel-3: #1D2330;
        --line: #232A38;
        --line-soft: #1A202C;
        /* Ink */
        --ink: #EEF1F6;
        --ink-2: #C7CEDB;
        --ink-3: #8B93A6;
        --ink-4: #5A6377;
        --ink-faint: #454E60;
        /* Signature accents */
        --amber: #E8A33D;
        --amber-soft: rgba(232,163,61,0.14);
        --teal: #45D9C8;
        --teal-soft: rgba(69,217,200,0.14);
        /* Status palette (brightened for dark surfaces) */
        --good: #34D399;
        --moderate: #FBBF24;
        --uhfs: #FB923C;
        --unhealthy: #F87171;
        --very: #A78BFA;
        --hazard: #EF4444;
        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.25);
        --shadow-md: 0 6px 18px rgba(0,0,0,0.32);
        --shadow-lg: 0 14px 34px rgba(0,0,0,0.4);
        /* Radius */
        --radius: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp {
        background: var(--void) !important;
        background-image:
            radial-gradient(900px 480px at 8% -8%, rgba(232,163,61,0.07), transparent 60%),
            radial-gradient(900px 520px at 100% 0%, rgba(69,217,200,0.06), transparent 55%) !important;
        background-attachment: fixed !important;
    }
    .block-container { padding-top: 0.9rem; padding-bottom: 2rem; max-width: 1180px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.02em; }
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
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 8px 2px; flex-wrap: wrap; gap: 8px;
        border-bottom: 1px solid var(--line-soft);
        margin-bottom: 4px;
    }
    .topbar-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .brand-mark {
        width: 20px; height: 20px; border-radius: 6px; flex-shrink: 0;
        background: linear-gradient(135deg, var(--amber) 0%, #B8752A 100%);
        box-shadow: 0 3px 8px rgba(232,163,61,0.35);
    }
    .brand-word {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
        color: var(--ink); letter-spacing: -0.01em; white-space: nowrap;
    }
    .brand-tag {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--ink-4);
        letter-spacing: 0.1em; text-transform: uppercase; white-space: nowrap;
        padding-left: 8px; border-left: 1px solid var(--line); margin-left: 2px;
    }
    .chip {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; font-weight: 500;
        color: var(--ink-3); background: var(--panel);
        border: 1px solid var(--line); padding: 3px 8px; border-radius: 6px;
        white-space: nowrap; display: inline-flex; align-items: center; gap: 5px;
    }
    .chip-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
    .chip-dot-on { background: var(--good); box-shadow: 0 0 6px rgba(52,211,153,0.7); }
    .chip-dot-off { background: var(--ink-4); }

    .topbar-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .clock {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
        color: var(--ink-3); text-align: right; white-space: nowrap; line-height: 1.3;
    }
    .clock .clock-date { color: var(--ink-2); font-weight: 600; }
    .clock .clock-time { color: var(--ink-4); font-size: 9.5px; }

    .credit-divider { width: 1px; height: 14px; background: var(--line); }
    .credit-tag {
        display: inline-flex; align-items: center; gap: 6px;
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; font-weight: 500;
        color: var(--ink-4); text-decoration: none; white-space: nowrap;
        padding: 3px 8px 3px 6px; border-radius: 999px; border: 1px solid transparent;
        opacity: 0.7; transition: opacity 0.2s, background 0.2s, border-color 0.2s;
    }
    .credit-tag:hover { opacity: 1; background: var(--panel); border-color: var(--line); color: var(--teal); }
    .credit-tag .credit-icon { width: 11px; height: 11px; flex-shrink: 0; color: #4FB4E8; }
    @media (max-width: 560px) { .credit-tag span.credit-label { display: none; } }

    /* ===== PAGE STRAPLINE ===== */
    .strapline {
        font-size: 12px; color: var(--ink-3); margin: 6px 2px 16px;
        font-weight: 500; letter-spacing: -0.005em;
    }
    .strapline b { color: var(--ink-2); font-weight: 600; }

    /* ===== HERO ROW ===== */
    .hero-row { display: flex; gap: 12px; margin: 0 0 14px; align-items: stretch; flex-wrap: wrap; }
    .dial-card {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: var(--radius-xl); padding: 16px 20px 14px;
        box-shadow: var(--shadow-md); flex-shrink: 0; width: 240px;
        display: flex; flex-direction: column; align-items: center;
        position: relative;
    }
    .dial-wrap { position: relative; width: 100%; max-width: 210px; }
    .dial-readout {
        position: absolute; left: 50%; bottom: 6px; transform: translateX(-50%);
        text-align: center; width: 100%;
    }
    .dial-num {
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 34px;
        line-height: 1; letter-spacing: -0.02em; display: block;
    }
    .dial-cat {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px;
        padding: 2px 9px; border-radius: 999px; display: inline-block;
    }
    .dial-sub {
        font-size: 10px; color: var(--ink-4); margin-top: 6px; font-weight: 500;
        font-family: 'JetBrains Mono', monospace; letter-spacing: 0.02em;
    }
    .dial-pulse { animation: pulseglow var(--pulse-speed, 3s) ease-in-out infinite; }
    @keyframes pulseglow { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }
    @media (prefers-reduced-motion: reduce) { .dial-pulse { animation: none !important; } }

    .instrument-strip {
        flex: 1; min-width: 240px; background: var(--panel); border: 1px solid var(--line);
        border-radius: var(--radius-xl); box-shadow: var(--shadow-sm);
        display: flex; align-items: stretch; padding: 6px;
    }
    .instrument-item {
        flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 6px;
        padding: 14px 16px; position: relative;
    }
    .instrument-item + .instrument-item::before {
        content: ''; position: absolute; left: 0; top: 14px; bottom: 14px; width: 1px;
        background: var(--line);
    }
    .instrument-top { display: flex; align-items: center; gap: 7px; }
    .instrument-icon {
        width: 22px; height: 22px; border-radius: 7px; display: flex; align-items: center;
        justify-content: center; background: color-mix(in srgb, var(--i-accent, #5A6377) 16%, var(--panel-2));
        color: var(--i-accent, var(--ink-3)); flex-shrink: 0;
    }
    .instrument-icon svg { width: 12px; height: 12px; }
    .instrument-label {
        font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--ink-4);
        text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
    }
    .instrument-val {
        font-weight: 700; font-size: 20px; color: var(--ink); letter-spacing: -0.02em;
        display: flex; align-items: baseline; gap: 3px;
    }
    .instrument-val .instrument-unit { font-size: 10px; font-weight: 500; color: var(--ink-4); }
    @media (max-width: 720px) {
        .hero-row { flex-direction: column; }
        .dial-card { width: 100%; }
        .instrument-strip { flex-wrap: wrap; }
        .instrument-item { min-width: 45%; }
        .instrument-item + .instrument-item::before { display: none; }
    }
    @media (max-width: 420px) { .instrument-item { min-width: 100%; } .instrument-item + .instrument-item::before { display: none; } }

    /* ===== SECTION HEAD (outside card, no border box) ===== */
    .section-head {
        display: flex; align-items: baseline; justify-content: space-between;
        margin: 22px 4px 8px; flex-wrap: wrap; gap: 4px;
    }
    .section-head-left { display: flex; flex-direction: column; gap: 2px; }
    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--teal);
        text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 17px;
        color: var(--ink) !important; margin: 0; letter-spacing: -0.01em;
    }
    .section-sub {
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--ink-4);
        text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500;
    }

    /* ===== PANEL (plain card, no internal head/border) ===== */
    .panel {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: var(--radius-xl); padding: 18px 20px 20px;
        margin: 0 0 4px; box-shadow: var(--shadow-sm);
    }
    @media (max-width: 560px) { .panel { padding: 14px 14px 16px; } .section-title { font-size: 15px; } }

    /* ===== POLLUTANT GAUGES ===== */
    .gauge-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
    @media (max-width: 900px) { .gauge-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 420px) { .gauge-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; } }
    .gauge-cell {
        display: flex; flex-direction: column; align-items: center; gap: 7px;
        padding: 8px 4px; border-radius: var(--radius);
    }
    .gauge-name {
        font-family: 'JetBrains Mono', monospace; font-size: 10px;
        color: var(--ink-3); letter-spacing: 0.05em; font-weight: 700;
    }
    .gauge-val { font-weight: 700; font-size: 13px; letter-spacing: -0.01em; }
    .gauge-status {
        font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
        text-transform: uppercase; font-weight: 700; letter-spacing: 0.07em;
        padding: 2px 7px; border-radius: 999px;
        background: color-mix(in srgb, var(--g-color, #5A6377) 16%, var(--panel-2));
    }

    /* ===== DAY TILES ===== */
    .day-tile {
        text-align: center; padding: 14px 10px 12px; border-radius: var(--radius-lg);
        background: var(--panel-2); border: 1px solid var(--line);
        position: relative; overflow: hidden;
    }
    .day-tile::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2.5px;
        background: var(--d-color, var(--ink-4));
    }
    .day-tile .d-label {
        font-family: 'JetBrains Mono', monospace; color: var(--ink-3);
        font-size: 10px; margin: 0; letter-spacing: 0.03em; font-weight: 600;
    }
    .day-tile .d-val { font-weight: 800; font-size: 26px; margin: 6px 0 3px; letter-spacing: -0.02em; }
    .day-tile .d-cat {
        font-family: 'JetBrains Mono', monospace; font-size: 9px; margin: 0;
        text-transform: uppercase; letter-spacing: 0.03em; font-weight: 600;
        padding: 2px 7px; border-radius: 999px;
        background: color-mix(in srgb, var(--d-color, var(--ink-4)) 16%, var(--panel-3));
        display: inline-block;
    }
    .day-tile .d-range {
        font-family: 'JetBrains Mono', monospace; color: var(--ink-4);
        font-size: 9.5px; margin-top: 8px; font-weight: 500;
        padding-top: 6px; border-top: 1px dashed var(--line);
    }
    @media (max-width: 560px) { .day-tile .d-val { font-size: 22px; } }

    /* ===== GUIDANCE ===== */
    .guidance {
        display: flex; gap: 12px; align-items: flex-start; padding: 13px 16px;
        border-radius: var(--radius-lg); border: 1px solid color-mix(in srgb, var(--gl-color) 35%, var(--line));
        background: color-mix(in srgb, var(--gl-color) 9%, var(--panel));
        margin-top: 12px;
    }
    .guidance .g-dot {
        width: 26px; height: 26px; border-radius: 8px; background: var(--gl-color);
        margin-top: 1px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 3px 8px color-mix(in srgb, var(--gl-color) 45%, transparent);
    }
    .guidance .g-dot::after { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--void); }
    .guidance .g-title { font-weight: 700; font-size: 12.5px; color: var(--ink) !important; margin: 0 0 2px; }
    .guidance .g-body { font-size: 11.5px; color: var(--ink-3) !important; margin: 0; line-height: 1.45; }

    .stAlert { border-radius: var(--radius) !important; background: var(--panel) !important; border: 1px solid var(--line) !important; }
    .stAlert p { color: var(--ink-2) !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; }
    hr { border-color: var(--line) !important; }

    /* ===== Layout safety ===== */
    .stApp [data-testid="stMain"], .stApp [data-testid="stMainBlockContainer"] {
        margin-left: 0 !important; padding-left: 0 !important; padding-right: 0 !important;
        width: 100% !important; max-width: 100% !important;
    }
    .block-container {
        padding-top: 0.9rem !important; padding-bottom: 2rem !important;
        padding-left: 1.1rem !important; padding-right: 1.1rem !important;
        max-width: 1180px !important; margin-left: auto !important; margin-right: auto !important;
    }
    @media (max-width: 480px) {
        .block-container { padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
    }

    .footer-note {
        text-align: center; color: var(--ink-4);
        font-family: 'JetBrains Mono', monospace; font-size: 9px;
        letter-spacing: 0.08em; margin-top: 14px; font-weight: 500;
        padding: 12px 0 0; border-top: 1px solid var(--line-soft);
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
    if val <= 50: return "Good", "4.2s", "#34D399", "Clear air"
    elif val <= 100: return "Moderate", "3.6s", "#FBBF24", "Acceptable"
    elif val <= 150: return "Sensitive Groups", "3.0s", "#FB923C", "Limit outdoors"
    elif val <= 200: return "Unhealthy", "2.4s", "#F87171", "Reduce activity"
    elif val <= 300: return "Very Unhealthy", "1.8s", "#A78BFA", "Health alert"
    else: return "Hazardous", "1.3s", "#EF4444", "Stay indoors"


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


# ---------------------------------------------------------------------------
# Presentation-only helpers: build the SVG instrument dial for the hero card.
# Pure rendering (arc geometry for a semicircular gauge whose six segments
# mirror the exact thresholds used by aqi_info() above) — nothing here
# touches AQI math, the model, or any data logic.
# ---------------------------------------------------------------------------
_DIAL_SEGMENTS = [
    (0, 50, "#34D399"),
    (50, 100, "#FBBF24"),
    (100, 150, "#FB923C"),
    (150, 200, "#F87171"),
    (200, 300, "#A78BFA"),
    (300, 500, "#EF4444"),
]

def _dial_point(cx, cy, r, angle_deg):
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)

def _dial_arc(cx, cy, r, v_lo, v_hi, scale_max=500):
    a_lo = 180 * (1 - v_lo / scale_max)
    a_hi = 180 * (1 - v_hi / scale_max)
    x1, y1 = _dial_point(cx, cy, r, a_lo)
    x2, y2 = _dial_point(cx, cy, r, a_hi)
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 0 1 {x2:.2f} {y2:.2f}"

def build_dial_svg(value, scale_max=500):
    cx, cy, r = 120, 116, 96
    needle_val = max(0, min(value, scale_max))
    needle_angle = 180 * (1 - needle_val / scale_max)
    nx, ny = _dial_point(cx, cy, r + 1, needle_angle)
    tx, ty = _dial_point(cx, cy, r - 20, needle_angle)
    segs = "".join(
        f"<path d='{_dial_arc(cx, cy, r, lo, hi, scale_max)}' stroke='{c}' stroke-width='13' "
        f"stroke-linecap='butt' fill='none' opacity='0.92'/>"
        for lo, hi, c in _DIAL_SEGMENTS
    )
    return f"""<svg viewBox="0 0 240 138" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;overflow:visible;">
        {segs}
        <line x1="{tx:.2f}" y1="{ty:.2f}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="{nx:.2f}" cy="{ny:.2f}" r="9" fill="var(--ink)" opacity="0.18"/>
        <circle cx="{nx:.2f}" cy="{ny:.2f}" r="4.5" fill="var(--ink)"/>
        <text x="14" y="132" font-family="JetBrains Mono, monospace" font-size="9" fill="var(--ink-faint)" font-weight="600">0</text>
        <text x="203" y="132" font-family="JetBrains Mono, monospace" font-size="9" fill="var(--ink-faint)" font-weight="600">500</text>
    </svg>"""


try:
    pollution, weather, combined_df, weather_ok = fetch_current_data()
    hist_lookback_df = fetch_recent_actuals_from_feature_store(lookback_hours=72)
    hist_df = hist_lookback_df[hist_lookback_df["datetime"] >= now_karachi.normalize()] if not hist_lookback_df.empty else hist_lookback_df
    current_aqi, dominant = get_aqi(pollution)
    cat, breathe_speed, color, cat_desc = aqi_info(current_aqi)

    # ===== TOP BAR =====
    st.html(f"""
    <div class='topbar'>
        <div class='topbar-left'>
            <span class='brand-mark'></span>
            <span class='brand-word'>Pearl</span>
            <span class='brand-tag'>AQI Station · Karachi</span>
            <span class='chip'><span class='chip-dot {"chip-dot-on" if _has_ow else "chip-dot-off"}'></span>OpenWeather</span>
            <span class='chip'><span class='chip-dot {"chip-dot-on" if _has_hw else "chip-dot-off"}'></span>Hopsworks</span>
            <span class='chip'>{model_source}</span>
        </div>
        <div class='topbar-right'>
            <span class='clock'><span class='clock-date'>{now_karachi.strftime('%a %d %b')}</span> · <span class='clock-time'>{now_karachi.strftime('%I:%M %p')} PKT</span></span>
            <span class='credit-divider'></span>
            <a class='credit-tag' href='https://www.linkedin.com/in/warisha-siddiqui/' target='_blank' rel='noopener noreferrer' title='Warisha Arshad on LinkedIn'>
                <svg class='credit-icon' viewBox='0 0 24 24' fill='currentColor' xmlns='http://www.w3.org/2000/svg'><path d='M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.44-2.14 2.94v5.67H9.34V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.07 2.07 0 1 1 0-4.13 2.07 2.07 0 0 1 0 4.13zM7.13 20.45H3.56V9h3.57v11.45zM22.22 0H1.77C.8 0 0 .78 0 1.75v20.5C0 23.22.8 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.75V1.75C24 .78 23.2 0 22.22 0z'/></svg>
                <span class='credit-label'>by Warisha Arshad</span>
            </a>
        </div>
    </div>
    <p class='strapline'>Live pollutant readings, hourly trend and a <b>72-hour AI forecast</b> for {LAT:.2f}°N, {LON:.2f}°E.</p>
    """)

    # ===== HERO: instrument dial + weather strip =====
    if weather_ok:
        icon_temp = "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 4v10.5a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z'/></svg>"
        icon_hum   = "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 2.69l5.66 5.66a8 8 0 1 1-11.32 0z'/></svg>"
        icon_wind  = "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16h2m-6.41-7.41A2 2 0 1 1 14 12H2m16.41 4.59A2 2 0 1 1 20 20H2'/></svg>"
        icon_press = "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/></svg>"
        instrument_html = (
            f"<div class='instrument-strip'>"
            f"<div class='instrument-item' style='--i-accent:#F87171'>"
            f"<div class='instrument-top'><span class='instrument-icon'>{icon_temp}</span><span class='instrument-label'>Temp</span></div>"
            f"<div class='instrument-val'>{weather['temperature_2m']:.1f}°<span class='instrument-unit'>C</span></div>"
            f"</div>"
            f"<div class='instrument-item' style='--i-accent:#4FB4E8'>"
            f"<div class='instrument-top'><span class='instrument-icon'>{icon_hum}</span><span class='instrument-label'>Humidity</span></div>"
            f"<div class='instrument-val'>{weather['relative_humidity_2m']:.0f}<span class='instrument-unit'>%</span></div>"
            f"</div>"
            f"<div class='instrument-item' style='--i-accent:#45D9C8'>"
            f"<div class='instrument-top'><span class='instrument-icon'>{icon_wind}</span><span class='instrument-label'>Wind</span></div>"
            f"<div class='instrument-val'>{weather['wind_speed_10m']:.1f}<span class='instrument-unit'>km/h</span></div>"
            f"</div>"
            f"<div class='instrument-item' style='--i-accent:#A78BFA'>"
            f"<div class='instrument-top'><span class='instrument-icon'>{icon_press}</span><span class='instrument-label'>Pressure</span></div>"
            f"<div class='instrument-val'>{weather['surface_pressure']:.0f}<span class='instrument-unit'>hPa</span></div>"
            f"</div>"
            f"</div>"
        )
    else:
        instrument_html = (
            "<div class='instrument-strip' style='align-items:center;justify-content:center;padding:20px;'>"
            "<div style='text-align:center;'><div class='instrument-label'>Weather data</div>"
            "<div style='font-size:12px;color:var(--ink-3);font-weight:500;margin-top:4px;'>Open-Meteo temporarily unavailable</div></div>"
            "</div>"
        )

    st.html(f"""
    <div class='hero-row'>
        <div class='dial-card'>
            <div class='dial-wrap'>
                {build_dial_svg(current_aqi)}
                <div class='dial-readout'>
                    <span class='dial-num dial-pulse' style='color:{color}; --pulse-speed:{breathe_speed}'>{current_aqi:.0f}</span>
                    <span class='dial-cat' style='color:{color}; background:color-mix(in srgb, {color} 16%, var(--panel-2))'>{cat}</span>
                    <div class='dial-sub'>{cat_desc} · {dominant.upper()}</div>
                </div>
            </div>
        </div>
        {instrument_html}
    </div>
    """)

    # ===== POLLUTANT GAUGES =====
    st.html("""
    <div class='section-head'>
        <div class='section-head-left'><span class='section-eyebrow'>Live readings</span><h3 class='section-title'>Pollutant Levels</h3></div>
    </div>
    <div class='panel'>""")
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    gauges = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#34D399" if pct < 40 else "#FBBF24" if pct < 70 else "#F87171"
        gauges += f"""
        <div class='gauge-cell' style='--g-color:{gcolor}'>
            <div style='width:58px;height:58px;border-radius:50%;background:conic-gradient({gcolor} {pct:.0f}%, var(--panel-3) {pct:.0f}% 100%);display:flex;align-items:center;justify-content:center;'>
                <div style='width:44px;height:44px;border-radius:50%;background:var(--panel);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 0 1px var(--line);'>
                    <span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span>
                </div>
            </div>
            <span class='gauge-name'>{p.upper()}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    gauges += "</div>"
    st.html(gauges)
    st.html("</div>")

    # ===== TODAY'S TREND =====
    st.html(f"""
    <div class='section-head'>
        <div class='section-head-left'><span class='section-eyebrow'>Today · {now_karachi.strftime('%d %b')}</span><h3 class='section-title'>AQI Trend</h3></div>
        <span class='section-sub'>Measured · Predicted</span>
    </div>
    <div class='panel'>""")
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
            fig, ax = plt.subplots(figsize=(12, 3.4))
            fig.patch.set_facecolor("#12161F")
            ax.set_facecolor("#10141C")
            ds = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            de = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([ds, de], 0, 50, alpha=0.10, color="#34D399")
            ax.fill_between([ds, de], 50, 100, alpha=0.10, color="#FBBF24")
            ax.fill_between([ds, de], 100, 150, alpha=0.10, color="#FB923C")
            ax.fill_between([ds, de], 150, 200, alpha=0.10, color="#F87171")
            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=2, label="Measured", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=1.8, linestyle="--", alpha=0.7, label="Predicted", zorder=5)
                all_vals += future_preds
            ax.scatter([now_karachi], [current_aqi], color=color, s=70, zorder=6, edgecolors="#12161F", linewidths=2, label="Now")
            ax.axhline(current_aqi, color="#3A4152", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_ylabel("AQI", color="#8B93A6", fontsize=10, fontweight=500)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, alpha=0.10, color="#8B93A6", linestyle="-")
            ax.tick_params(colors="#8B93A6", labelsize=8)
            for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
            for s in ax.spines.values(): s.set_color("#232A38")
            ax.legend(facecolor="#171C27", edgecolor="#232A38", labelcolor="#C7CEDB", fontsize=8, loc="upper right", framealpha=0.95)
            if all_vals: ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")
    st.html("</div>")

    # ===== 3-DAY FORECAST =====
    st.html("""
    <div class='section-head'>
        <div class='section-head-left'><span class='section-eyebrow'>Next 72 hours</span><h3 class='section-title'>3-Day Forecast</h3></div>
        <span class='section-sub'>Hourly model prediction</span>
    </div>
    <div class='panel'>""")
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
                fig, ax = plt.subplots(figsize=(12, 2.2))
                fig.patch.set_facecolor("#12161F")
                ax.set_facecolor("#10141C")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.5)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.12, color=color)
                ax.axhline(100, color="#FBBF24", linestyle="--", alpha=0.5, linewidth=0.7)
                ax.axhline(150, color="#FB923C", linestyle="--", alpha=0.5, linewidth=0.7)
                ax.set_ylabel("AQI", color="#8B93A6", fontsize=9, fontweight=500)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, alpha=0.10, color="#8B93A6", linestyle="-")
                ax.tick_params(colors="#8B93A6", labelsize=7)
                for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
                for s in ax.spines.values(): s.set_color("#232A38")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                if any(a > 150 for a in forecast_aqi):
                    gc, gt, gb = "#F87171", "Hazardous AQI expected", "Avoid outdoor activity."
                elif any(a > 100 for a in forecast_aqi):
                    gc, gt, gb = "#FBBF24", "Elevated AQI expected", "Sensitive groups should limit outdoor time."
                else:
                    gc, gt, gb = "#34D399", "Within safe range", "No elevated AQI expected."
                st.html(f"""<div class='guidance' style='--gl-color:{gc}'><span class='g-dot'></span><div><p class='g-title'>{gt}</p><p class='g-body'>{gb}</p></div></div>""")
    except Exception as e:
        st.error(f"Forecast: {e}")
    st.html("</div>")

    # ===== GUIDANCE =====
    tips = {
        "Good": ("Excellent air quality — perfect for outdoor activity.", "#34D399"),
        "Moderate": ("Acceptable. Sensitive people should limit prolonged exertion.", "#FBBF24"),
        "Sensitive Groups": ("Sensitive groups should reduce outdoor activity.", "#FB923C"),
        "Unhealthy": ("Reduce outdoor physical activity for everyone.", "#F87171"),
        "Very Unhealthy": ("Avoid outdoors — use air purifiers indoors.", "#A78BFA"),
        "Hazardous": ("Emergency. Stay indoors, seek medical help if needed.", "#EF4444"),
    }
    tb, tc = tips.get(cat, ("", color))
    st.html(f"""
    <div class='section-head'><div class='section-head-left'><span class='section-eyebrow'>Right now</span><h3 class='section-title'>Health Guidance</h3></div></div>
    <div class='panel' style='padding:14px 16px;'>
        <div class='guidance' style='--gl-color:{tc}; margin-top:0;'><span class='g-dot'></span><div><p class='g-title'>{cat}</p><p class='g-body'>{tb}</p></div></div>
    </div>""")

except Exception as e:
    st.error(f"Error: {e}")

st.html("<p class='footer-note'>PEARL AQI STATION · KARACHI · Hopsworks + OpenWeather + Open-Meteo</p>")
