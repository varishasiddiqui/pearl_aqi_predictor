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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root{
        /* Surfaces — one flat dark tone, no card layers */
        --void: #0B0D11;
        --ink: #F2F4F7;
        --ink-2: #ABB2C0;
        --ink-3: #767E8F;
        --ink-4: #4B5261;
        --line: #20242D;
        /* Signature accent */
        --amber: #E8A33D;
        --teal: #45D9C8;
        /* Status palette (semantic — used only where it means something) */
        --good: #34D399;
        --moderate: #FBBF24;
        --uhfs: #FB923C;
        --unhealthy: #F87171;
        --very: #A78BFA;
        --hazard: #EF4444;
        --radius: 8px;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp { background: var(--void) !important; }
    .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 900px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
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

    /* ===== TOP BAR — plain row, one hairline underneath ===== */
    .topbar {
        display: flex; align-items: baseline; justify-content: space-between;
        padding-bottom: 10px; flex-wrap: wrap; gap: 6px 14px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 4px;
    }
    .topbar-left { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .brand-word {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 17px;
        color: var(--ink); letter-spacing: -0.01em; white-space: nowrap;
    }
    .brand-tag {
        font-size: 12px; color: var(--ink-3); white-space: nowrap;
    }
    .status-line { font-size: 11px; color: var(--ink-4); white-space: nowrap; }
    .status-dot { display: inline-block; width: 5px; height: 5px; border-radius: 50%; margin-right: 4px; }
    .status-on { background: var(--good); }
    .status-off { background: var(--ink-4); }

    .topbar-right { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
    .clock { font-size: 12px; color: var(--ink-3); white-space: nowrap; }
    .credit-tag {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 11px; color: var(--ink-4); text-decoration: none; white-space: nowrap;
        opacity: 0.75; transition: opacity 0.15s, color 0.15s;
    }
    .credit-tag:hover { opacity: 1; color: var(--teal); }
    .credit-tag .credit-icon { width: 11px; height: 11px; flex-shrink: 0; }
    @media (max-width: 560px) { .credit-tag span.credit-label { display: none; } }

    .strapline { font-size: 13px; color: var(--ink-3); margin: 12px 0 22px; line-height: 1.5; }

    /* ===== HERO — no card, just a number and inline stats ===== */
    .hero-row {
        display: flex; gap: 32px; align-items: center; flex-wrap: wrap;
        padding-bottom: 20px; margin-bottom: 20px; border-bottom: 1px solid var(--line);
    }
    .dial-wrap { position: relative; width: 190px; flex-shrink: 0; }
    .dial-readout {
        position: absolute; left: 50%; bottom: 2px; transform: translateX(-50%);
        text-align: center; width: 100%;
    }
    .dial-num {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 40px;
        line-height: 1; letter-spacing: -0.02em; display: block;
    }
    .dial-cat { font-size: 12px; font-weight: 600; margin-top: 4px; }
    .dial-sub { font-size: 11px; color: var(--ink-4); margin-top: 3px; }

    .stat-cols { display: flex; gap: 28px; flex-wrap: wrap; flex: 1; }
    .stat-item { display: flex; flex-direction: column; gap: 2px; min-width: 76px; }
    .stat-label { font-size: 11px; color: var(--ink-4); }
    .stat-val { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 19px; color: var(--ink); }
    .stat-val .stat-unit { font-size: 11px; font-weight: 500; color: var(--ink-4); margin-left: 2px; }
    .weather-note { font-size: 12px; color: var(--ink-4); }

    /* ===== SECTION HEAD — text + hairline, no box ===== */
    .section-head {
        display: flex; align-items: baseline; justify-content: space-between;
        margin: 30px 0 12px; flex-wrap: wrap; gap: 4px;
    }
    .section-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 16px; color: var(--ink) !important; margin: 0; }
    .section-note { font-size: 11px; color: var(--ink-4); }

    /* ===== POLLUTANT LIST — rows with inline bars, not donuts ===== */
    .poll-row {
        display: flex; align-items: center; gap: 12px;
        padding: 9px 0; border-bottom: 1px solid var(--line);
    }
    .poll-row:last-child { border-bottom: none; }
    .poll-name { font-size: 13px; font-weight: 600; color: var(--ink); width: 56px; flex-shrink: 0; }
    .poll-bar-track { flex: 1; height: 5px; border-radius: 3px; background: var(--line); overflow: hidden; }
    .poll-bar-fill { height: 100%; border-radius: 3px; }
    .poll-val { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--ink-2); width: 96px; text-align: right; flex-shrink: 0; }
    .poll-status { font-size: 11px; font-weight: 600; width: 68px; text-align: right; flex-shrink: 0; }
    @media (max-width: 560px) {
        .poll-name { width: 42px; }
        .poll-val { width: 72px; font-size: 11.5px; }
        .poll-status { display: none; }
    }

    /* ===== DAY ROW — plain columns split by hairlines, no tile boxes ===== */
    .day-row { display: flex; }
    .day-col {
        flex: 1; text-align: center; padding: 4px 8px 0;
        border-left: 1px solid var(--line);
    }
    .day-col:first-child { border-left: none; }
    .day-col .d-label { font-size: 12px; color: var(--ink-3); margin: 0; }
    .day-col .d-val { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 28px; margin: 6px 0 2px; letter-spacing: -0.02em; }
    .day-col .d-cat { font-size: 11.5px; font-weight: 600; margin: 0; }
    .day-col .d-range { font-size: 11px; color: var(--ink-4); margin-top: 6px; }
    @media (max-width: 560px) { .day-col .d-val { font-size: 22px; } }

    /* ===== GUIDANCE — plain text line with a colored marker, no filled box ===== */
    .guidance { display: flex; gap: 10px; align-items: flex-start; padding-top: 4px; }
    .guidance .g-bar { width: 3px; align-self: stretch; border-radius: 2px; flex-shrink: 0; margin-top: 2px; }
    .guidance .g-title { font-weight: 700; font-size: 13px; color: var(--ink) !important; margin: 0 0 2px; }
    .guidance .g-body { font-size: 12.5px; color: var(--ink-3) !important; margin: 0; line-height: 1.5; }

    .stAlert { border-radius: var(--radius) !important; background: #12151B !important; border: 1px solid var(--line) !important; }
    .stAlert p { color: var(--ink-2) !important; }
    hr { border-color: var(--line) !important; }

    /* ===== Layout safety ===== */
    .stApp [data-testid="stMain"], .stApp [data-testid="stMainBlockContainer"] {
        margin-left: 0 !important; padding-left: 0 !important; padding-right: 0 !important;
        width: 100% !important; max-width: 100% !important;
    }
    .block-container {
        padding-left: 1.1rem !important; padding-right: 1.1rem !important;
        margin-left: auto !important; margin-right: auto !important;
    }
    @media (max-width: 480px) {
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    }

    .footer-note {
        text-align: center; color: var(--ink-4); font-size: 11px;
        margin-top: 26px; padding: 16px 0 0; border-top: 1px solid var(--line);
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
    cx, cy, r = 95, 92, 78
    needle_val = max(0, min(value, scale_max))
    needle_angle = 180 * (1 - needle_val / scale_max)
    nx, ny = _dial_point(cx, cy, r + 1, needle_angle)
    tx, ty = _dial_point(cx, cy, r - 16, needle_angle)
    segs = "".join(
        f"<path d='{_dial_arc(cx, cy, r, lo, hi, scale_max)}' stroke='{c}' stroke-width='9' "
        f"stroke-linecap='butt' fill='none' opacity='0.9'/>"
        for lo, hi, c in _DIAL_SEGMENTS
    )
    return f"""<svg viewBox="0 0 190 108" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;overflow:visible;">
        {segs}
        <line x1="{tx:.2f}" y1="{ty:.2f}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="var(--ink)" stroke-width="2" stroke-linecap="round"/>
        <circle cx="{nx:.2f}" cy="{ny:.2f}" r="3.5" fill="var(--ink)"/>
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
            <span class='brand-word'>Pearl</span>
            <span class='brand-tag'>AQI Station · Karachi</span>
            <span class='status-line'><span class='status-dot {"status-on" if _has_ow else "status-off"}'></span>OpenWeather</span>
            <span class='status-line'><span class='status-dot {"status-on" if _has_hw else "status-off"}'></span>Hopsworks</span>
            <span class='status-line'>{model_source}</span>
        </div>
        <div class='topbar-right'>
            <span class='clock'>{now_karachi.strftime('%a %d %b')} · {now_karachi.strftime('%I:%M %p')} PKT</span>
            <a class='credit-tag' href='https://www.linkedin.com/in/warisha-siddiqui/' target='_blank' rel='noopener noreferrer' title='Warisha Arshad on LinkedIn'>
                <svg class='credit-icon' viewBox='0 0 24 24' fill='currentColor' xmlns='http://www.w3.org/2000/svg'><path d='M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.44-2.14 2.94v5.67H9.34V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.07 2.07 0 1 1 0-4.13 2.07 2.07 0 0 1 0 4.13zM7.13 20.45H3.56V9h3.57v11.45zM22.22 0H1.77C.8 0 0 .78 0 1.75v20.5C0 23.22.8 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.75V1.75C24 .78 23.2 0 22.22 0z'/></svg>
                <span class='credit-label'>Warisha Arshad</span>
            </a>
        </div>
    </div>
    <p class='strapline'>Live pollutant readings, hourly trend and a 72-hour forecast for {LAT:.2f}°N, {LON:.2f}°E.</p>
    """)

    # ===== HERO: dial + key stats, inline, no card =====
    if weather_ok:
        stats_html = f"""
        <div class='stat-item'><span class='stat-label'>Temperature</span><span class='stat-val'>{weather['temperature_2m']:.1f}<span class='stat-unit'>°C</span></span></div>
        <div class='stat-item'><span class='stat-label'>Humidity</span><span class='stat-val'>{weather['relative_humidity_2m']:.0f}<span class='stat-unit'>%</span></span></div>
        <div class='stat-item'><span class='stat-label'>Wind</span><span class='stat-val'>{weather['wind_speed_10m']:.1f}<span class='stat-unit'>km/h</span></span></div>
        <div class='stat-item'><span class='stat-label'>Pressure</span><span class='stat-val'>{weather['surface_pressure']:.0f}<span class='stat-unit'>hPa</span></span></div>
        """
    else:
        stats_html = "<span class='weather-note'>Weather data temporarily unavailable (Open-Meteo)</span>"

    st.html(f"""
    <div class='hero-row'>
        <div class='dial-wrap'>
            {build_dial_svg(current_aqi)}
            <div class='dial-readout'>
                <span class='dial-num' style='color:{color}'>{current_aqi:.0f}</span>
                <div class='dial-cat' style='color:{color}'>{cat}</div>
                <div class='dial-sub'>{cat_desc} · {dominant.upper()}</div>
            </div>
        </div>
        <div class='stat-cols'>{stats_html}</div>
    </div>
    """)

    # ===== POLLUTANT LEVELS =====
    st.html("""
    <div class='section-head'><h3 class='section-title'>Pollutant levels</h3></div>""")
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    names = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂", "o3": "O₃", "co": "CO"}
    rows = ""
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#34D399" if pct < 40 else "#FBBF24" if pct < 70 else "#F87171"
        unit = "µg/m³" if p != "co" else "µg/m³"
        rows += f"""
        <div class='poll-row'>
            <span class='poll-name'>{names.get(p, p.upper())}</span>
            <span class='poll-bar-track'><span class='poll-bar-fill' style='width:{pct:.0f}%; background:{gcolor}'></span></span>
            <span class='poll-val'>{val:.1f} {unit}</span>
            <span class='poll-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    st.html(rows)

    # ===== TODAY'S TREND =====
    st.html(f"""
    <div class='section-head'>
        <h3 class='section-title'>AQI trend · {now_karachi.strftime('%d %b')}</h3>
        <span class='section-note'>Measured / predicted</span>
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
            fig, ax = plt.subplots(figsize=(10, 2.8))
            fig.patch.set_facecolor("#0B0D11")
            ax.set_facecolor("#0B0D11")
            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=1.8, label="Measured", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=1.5, linestyle="--", alpha=0.65, label="Predicted", zorder=5)
                all_vals += future_preds
            ax.scatter([now_karachi], [current_aqi], color=color, s=44, zorder=6, edgecolors="#0B0D11", linewidths=1.5, label="Now")
            ax.set_ylabel("AQI", color="#767E8F", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, axis="y", alpha=0.12, color="#767E8F", linestyle="-", linewidth=0.6)
            ax.grid(False, axis="x")
            ax.tick_params(colors="#767E8F", labelsize=8)
            for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
            for s in ax.spines.values(): s.set_visible(False)
            ax.legend(frameon=False, labelcolor="#ABB2C0", fontsize=8, loc="upper right")
            if all_vals: ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")

    # ===== 3-DAY FORECAST =====
    st.html("""
    <div class='section-head'>
        <h3 class='section-title'>3-day forecast</h3>
        <span class='section-note'>Hourly prediction, next 72h</span>
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
                day_cols_html = "<div class='day-row'>"
                for day in unique_dates:
                    dv = fdf.loc[fdf["date"] == day, "aqi"]
                    da, dmi, dmx = dv.mean(), dv.min(), dv.max()
                    dc, _, dcol, _ = aqi_info(da)
                    dl = pd.Timestamp(day).strftime("%d %b · %a")
                    day_cols_html += f"""
                    <div class='day-col'>
                        <p class='d-label'>{dl}</p>
                        <p class='d-val' style='color:{dcol}'>{da:.0f}</p>
                        <p class='d-cat' style='color:{dcol}'>{dc}</p>
                        <p class='d-range'>{dmi:.0f}–{dmx:.0f}</p>
                    </div>"""
                day_cols_html += "</div>"
                st.html(day_cols_html)

                fig, ax = plt.subplots(figsize=(10, 1.9))
                fig.patch.set_facecolor("#0B0D11")
                ax.set_facecolor("#0B0D11")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.4)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.10, color=color)
                ax.axhline(100, color="#FBBF24", linestyle="--", alpha=0.4, linewidth=0.6)
                ax.axhline(150, color="#FB923C", linestyle="--", alpha=0.4, linewidth=0.6)
                ax.set_ylabel("AQI", color="#767E8F", fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, axis="y", alpha=0.12, color="#767E8F", linestyle="-", linewidth=0.6)
                ax.grid(False, axis="x")
                ax.tick_params(colors="#767E8F", labelsize=7)
                for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
                for s in ax.spines.values(): s.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if any(a > 150 for a in forecast_aqi):
                    gc, gt, gb = "#F87171", "Hazardous AQI expected", "Avoid outdoor activity."
                elif any(a > 100 for a in forecast_aqi):
                    gc, gt, gb = "#FBBF24", "Elevated AQI expected", "Sensitive groups should limit outdoor time."
                else:
                    gc, gt, gb = "#34D399", "Within safe range", "No elevated AQI expected."
                st.html(f"""<div class='guidance' style='margin-top:12px;'><span class='g-bar' style='background:{gc}'></span><div><p class='g-title'>{gt}</p><p class='g-body'>{gb}</p></div></div>""")
    except Exception as e:
        st.error(f"Forecast: {e}")

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
    <div class='section-head'><h3 class='section-title'>Health guidance</h3></div>
    <div class='guidance'><span class='g-bar' style='background:{tc}'></span><div><p class='g-title'>{cat}</p><p class='g-body'>{tb}</p></div></div>
    """)

except Exception as e:
    st.error(f"Error: {e}")

st.html("<p class='footer-note'>Pearl AQI Station · Karachi · Hopsworks + OpenWeather + Open-Meteo</p>")
