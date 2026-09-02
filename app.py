import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap

KARACHI_TZ = timezone(timedelta(hours=5))
LAT, LON = 24.8607, 67.0011
FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 2
MODEL_NAME = "aqi_predictor_karachi"

now_karachi = pd.Timestamp.now(tz="UTC").tz_convert(KARACHI_TZ)

st.set_page_config(page_title="Pearl · AQI Station Karachi", layout="wide", page_icon="🌆")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root{
        --void: #0A0C10;
        --panel: #12151C;
        --panel-2: #171B24;
        --ink: #F4F6F9;
        --ink-2: #B4BBC9;
        --ink-3: #7B8395;
        --ink-4: #4E5563;
        --line: #232833;
        --good: #34D399;
        --moderate: #FBBF24;
        --uhfs: #FB923C;
        --unhealthy: #F87171;
        --very: #A78BFA;
        --hazard: #EF4444;
    }

    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp {
        background: var(--void) !important;
        background-image: radial-gradient(1100px 520px at 12% -10%, rgba(69,217,200,0.05), transparent 60%) !important;
        background-attachment: fixed !important;
    }
    .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 880px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
    p, span, label, div { color: var(--ink-2); }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"], button[kind="header"],
    [data-testid="stToolbar"], [data-testid="stHeader"],
    .stApp header, .stApp > header, .stApp > div > div > header {
        display: none !important; width: 0 !important; height: 0 !important;
        min-width: 0 !important; min-height: 0 !important;
        padding: 0 !important; margin: 0 !important;
        overflow: hidden !important; border: none !important;
    }

    /* ===== TOP BAR ===== */
    .topbar {
        display: flex; align-items: baseline; justify-content: space-between;
        padding-bottom: 12px; flex-wrap: wrap; gap: 6px 14px;
        border-bottom: 1px solid var(--line); margin-bottom: 2px;
    }
    .topbar-left { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
    .brand-word { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 18px; color: var(--ink); letter-spacing: -0.01em; }
    .brand-tag { font-size: 12.5px; color: var(--ink-3); }
    .status-line { font-size: 11px; color: var(--ink-4); white-space: nowrap; }
    .status-dot { display: inline-block; width: 5px; height: 5px; border-radius: 50%; margin-right: 4px; }
    .status-on { background: var(--good); }
    .status-off { background: var(--ink-4); }
    .topbar-right { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
    .clock { font-size: 12.5px; color: var(--ink-3); white-space: nowrap; }
    .credit-tag { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--ink-4); text-decoration: none; opacity: 0.8; transition: opacity 0.15s, color 0.15s; }
    .credit-tag:hover { opacity: 1; color: #45D9C8; }
    .credit-tag .credit-icon { width: 11px; height: 11px; flex-shrink: 0; }
    @media (max-width: 560px) { .credit-tag span.credit-label { display: none; } }

    .strapline { font-size: 13.5px; color: var(--ink-3); margin: 14px 0 20px; line-height: 1.55; }

    /* ===== HERO — the one signature element: a tinted glow card ===== */
    .hero-card {
        position: relative; overflow: hidden;
        display: flex; align-items: center; justify-content: space-between; gap: 28px; flex-wrap: wrap;
        border-radius: 20px; border: 1px solid var(--hc-color, var(--line));
        background:
            radial-gradient(480px 240px at 0% 0%, color-mix(in srgb, var(--hc-color, #45D9C8) 20%, transparent), transparent 70%),
            var(--panel);
        padding: 26px 30px; margin-bottom: 26px;
    }
    .hero-left { display: flex; flex-direction: column; gap: 10px; min-width: 220px; }
    .hero-num-row { display: flex; align-items: baseline; gap: 12px; }
    .hero-num { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 64px; line-height: 1; letter-spacing: -0.03em; }
    .hero-cat {
        font-size: 12.5px; font-weight: 700; padding: 4px 11px; border-radius: 999px;
        background: color-mix(in srgb, var(--hc-color) 18%, var(--panel-2));
        color: var(--hc-color); white-space: nowrap;
    }
    .hero-desc { font-size: 12.5px; color: var(--ink-3); }
    .aqi-scale-wrap { max-width: 280px; margin-top: 4px; }
    .aqi-scale { position: relative; height: 7px; border-radius: 999px; overflow: visible; display: flex; width: 100%; }
    .aqi-seg { height: 100%; }
    .aqi-seg:first-child { border-radius: 999px 0 0 999px; }
    .aqi-seg:last-child { border-radius: 0 999px 999px 0; }
    .aqi-marker {
        position: absolute; top: 50%; width: 13px; height: 13px; border-radius: 50%;
        background: var(--ink); border: 2px solid var(--void); transform: translate(-50%, -50%);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--hc-color) 35%, transparent);
    }
    .aqi-scale-labels { display: flex; justify-content: space-between; font-size: 9.5px; color: var(--ink-4); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

    .hero-stats { display: grid; grid-template-columns: repeat(2, auto); gap: 16px 32px; }
    .stat-item { display: flex; flex-direction: column; gap: 3px; }
    .stat-label { font-size: 11px; color: var(--ink-4); }
    .stat-val { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 20px; color: var(--ink); }
    .stat-val .stat-unit { font-size: 11px; font-weight: 500; color: var(--ink-4); margin-left: 2px; }
    .weather-note { font-size: 12.5px; color: var(--ink-4); }
    .hero-stats { grid-template-columns: repeat(2, auto); }
    @media (max-width: 640px) {
        .hero-card { padding: 18px; flex-direction: column; align-items: flex-start; gap: 18px; }
        .hero-num { font-size: 48px; }
        .hero-stats { gap: 14px 24px; }
    }

    /* ===== SECTION HEAD ===== */
    .section-head { display: flex; align-items: baseline; justify-content: space-between; margin: 30px 0 12px; flex-wrap: wrap; gap: 4px; }
    .section-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 16px; color: var(--ink) !important; margin: 0; }
    .section-note { font-size: 11.5px; color: var(--ink-4); }

    /* ===== POLLUTANT GAUGES — compact ring grid, everything visible at a glance ===== */
    .gauge-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
    @media (max-width: 560px) { .gauge-grid { grid-template-columns: repeat(3, 1fr); gap: 10px 6px; } }
    .gauge-cell { display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .gauge-ring {
        width: 56px; height: 56px; border-radius: 50%; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
    }
    .gauge-ring-inner {
        width: 44px; height: 44px; border-radius: 50%; background: var(--panel);
        display: flex; align-items: center; justify-content: center;
        box-shadow: inset 0 0 0 1px var(--line);
    }
    .gauge-val { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 13px; }
    .gauge-name { font-size: 10.5px; color: var(--ink-3); font-weight: 600; letter-spacing: 0.02em; }
    .gauge-status { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }

    /* ===== DAY ROW ===== */
    .day-row { display: flex; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
    .day-col { flex: 1; text-align: center; padding: 16px 10px; border-left: 1px solid var(--line); }
    .day-col:first-child { border-left: none; }
    .day-col .d-label { font-size: 12px; color: var(--ink-3); margin: 0; }
    .day-col .d-val { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 30px; margin: 8px 0 3px; letter-spacing: -0.02em; }
    .day-col .d-cat { font-size: 11.5px; font-weight: 700; margin: 0; }
    .day-col .d-range { font-size: 11px; color: var(--ink-4); margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--line); }
    @media (max-width: 560px) { .day-col .d-val { font-size: 23px; } }

    /* ===== GUIDANCE ===== */
    .guidance { display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border-radius: 12px; background: color-mix(in srgb, var(--gl-color) 8%, var(--panel)); border: 1px solid color-mix(in srgb, var(--gl-color) 28%, var(--line)); }
    .guidance .g-bar { width: 4px; align-self: stretch; border-radius: 2px; flex-shrink: 0; background: var(--gl-color); }
    .guidance .g-title { font-weight: 700; font-size: 13px; color: var(--ink) !important; margin: 0 0 2px; }
    .guidance .g-body { font-size: 12.5px; color: var(--ink-3) !important; margin: 0; line-height: 1.5; }

    .stAlert { border-radius: 12px !important; background: var(--panel) !important; border: 1px solid var(--line) !important; }
    .stAlert p { color: var(--ink-2) !important; }
    hr { border-color: var(--line) !important; }

    .stApp [data-testid="stMain"] { margin-left: 0 !important; }
    .stApp [data-testid="stMainBlockContainer"], .block-container {
        max-width: 720px !important; margin-left: auto !important; margin-right: auto !important;
        padding-left: 1.1rem !important; padding-right: 1.1rem !important;
    }
    @media (max-width: 480px) { .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; } }

    .footer-note { text-align: center; color: var(--ink-4); font-size: 11px; margin-top: 26px; padding: 16px 0 0; border-top: 1px solid var(--line); }
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


# ---------------------------------------------------------------------------
# SHAP explainability — read-only, built on top of the already-loaded model.
# Does not touch training/prediction logic; just explains it.
# ---------------------------------------------------------------------------
SHAP_TIMESTEPS = 24  # matches TIMESTEPS in training_pipeline.py for the LSTM


@st.cache_resource
def build_shap_explainer(_model, _background):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    if isinstance(_model, RandomForestRegressor):
        return shap.TreeExplainer(_model), "tree"
    elif isinstance(_model, Ridge):
        return shap.LinearExplainer(_model, _background), "linear"
    else:
        # LSTM: takes a (timesteps, features) sequence, not a flat row, so it
        # needs a gradient-based explainer over sequence windows instead.
        import tensorflow as tf  # noqa: F401  (guarded, TF is heavy)
        return shap.GradientExplainer(_model, _background), "lstm"


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
            print("fetch_recent_actuals_from_feature_store: read succeeded but feature group is empty.")
            return df
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(KARACHI_TZ)
        window_start = now_karachi - pd.Timedelta(hours=lookback_hours)
        df = df[(df["datetime"] >= window_start) & (df["datetime"] <= now_karachi)]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        import traceback
        print(f"fetch_recent_actuals_from_feature_store failed: {e}")
        print(traceback.format_exc())
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
            "pm25_lag_24": lag(pm25_history, 24),
            "aqi_change_rate": aqi_history[-1] - lag(aqi_history, 2) if len(aqi_history) >= 2 else 0.0,
            "pm25_change_rate": pm25_history[-1] - lag(pm25_history, 2) if len(pm25_history) >= 2 else 0.0,
            "aqi_rolling_3": rolling(aqi_history, 3),
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
# Presentation-only helper: builds the horizontal AQI scale bar markup for
# the hero card. Segment thresholds mirror aqi_info() exactly. Pure
# rendering — no AQI math, model, or data logic lives here.
# ---------------------------------------------------------------------------
_SCALE_SEGMENTS = [
    (0, 50, "#34D399"), (50, 100, "#FBBF24"), (100, 150, "#FB923C"),
    (150, 200, "#F87171"), (200, 300, "#A78BFA"), (300, 500, "#EF4444"),
]

def build_aqi_scale_html(value, scale_max=500):
    segs = "".join(
        f"<div class='aqi-seg' style='flex:{hi - lo}; background:{c}'></div>"
        for lo, hi, c in _SCALE_SEGMENTS
    )
    pct = max(0.0, min(value, scale_max)) / scale_max * 100
    return f"""
    <div class='aqi-scale-wrap'>
        <div class='aqi-scale'>{segs}<div class='aqi-marker' style='left:{pct:.1f}%'></div></div>
        <div class='aqi-scale-labels'><span>0</span><span>150</span><span>300</span><span>500</span></div>
    </div>"""


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
            <span class='brand-word'>Pearls AQI Predictor</span>
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

    # ===== HERO CARD =====
    if weather_ok:
        stats_html = f"""
        <div class='stat-item'><span class='stat-label'>Temperature</span><span class='stat-val'>{weather['temperature_2m']:.1f}<span class='stat-unit'>°C</span></span></div>
        <div class='stat-item'><span class='stat-label'>Humidity</span><span class='stat-val'>{weather['relative_humidity_2m']:.0f}<span class='stat-unit'>%</span></span></div>
        <div class='stat-item'><span class='stat-label'>Wind</span><span class='stat-val'>{weather['wind_speed_10m']:.1f}<span class='stat-unit'>km/h</span></span></div>
        <div class='stat-item'><span class='stat-label'>Pressure</span><span class='stat-val'>{weather['surface_pressure']:.0f}<span class='stat-unit'>hPa</span></span></div>
        """
    else:
        stats_html = "<span class='weather-note'>Weather data<br>temporarily unavailable</span>"

    st.html(f"""
    <div class='hero-card' style='--hc-color:{color}'>
        <div class='hero-left'>
            <div class='hero-num-row'>
                <span class='hero-num' style='color:{color}'>{current_aqi:.0f}</span>
                <span class='hero-cat'>{cat}</span>
            </div>
            <span class='hero-desc'>{cat_desc} · dominant pollutant {dominant.upper()}</span>
            {build_aqi_scale_html(current_aqi)}
        </div>
        <div class='hero-stats'>{stats_html}</div>
    </div>
    """)

    # ===== POLLUTANT LEVELS =====
    st.html("<div class='section-head'><h3 class='section-title'>Pollutant levels</h3></div>")
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    names = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂", "o3": "O₃", "co": "CO"}
    cells = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#34D399" if pct < 40 else "#FBBF24" if pct < 70 else "#F87171"
        cells += f"""
        <div class='gauge-cell'>
            <div class='gauge-ring' style='background:conic-gradient({gcolor} {pct:.0f}%, var(--panel-2) {pct:.0f}% 100%)'>
                <div class='gauge-ring-inner'><span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span></div>
            </div>
            <span class='gauge-name'>{names.get(p, p.upper())}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    cells += "</div>"
    st.html(cells)

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
            fig.patch.set_facecolor("#0A0C10")
            ax.set_facecolor("#0A0C10")
            ds = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            de = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([ds, de], 0, 50, alpha=0.08, color="#34D399")
            ax.fill_between([ds, de], 50, 100, alpha=0.08, color="#FBBF24")
            ax.fill_between([ds, de], 100, 150, alpha=0.08, color="#FB923C")
            ax.fill_between([ds, de], 150, 200, alpha=0.08, color="#F87171")
            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=1.8, label="Measured", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=1.5, linestyle="--", alpha=0.65, label="Predicted", zorder=5)
                all_vals += future_preds
            ax.scatter([now_karachi], [current_aqi], color=color, s=48, zorder=6, edgecolors="#0A0C10", linewidths=1.5, label="Now")
            ax.set_ylabel("AQI", color="#7B8395", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, axis="y", alpha=0.12, color="#7B8395", linestyle="-", linewidth=0.6)
            ax.grid(False, axis="x")
            ax.tick_params(colors="#7B8395", labelsize=8)
            for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
            for s in ax.spines.values(): s.set_visible(False)
            ax.legend(frameon=False, labelcolor="#B4BBC9", fontsize=8, loc="upper right")
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
                fig.patch.set_facecolor("#0A0C10")
                ax.set_facecolor("#0A0C10")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.4)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.10, color=color)
                ax.axhline(100, color="#FBBF24", linestyle="--", alpha=0.4, linewidth=0.6)
                ax.axhline(150, color="#FB923C", linestyle="--", alpha=0.4, linewidth=0.6)
                ax.set_ylabel("AQI", color="#7B8395", fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, axis="y", alpha=0.12, color="#7B8395", linestyle="-", linewidth=0.6)
                ax.grid(False, axis="x")
                ax.tick_params(colors="#7B8395", labelsize=7)
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
                st.html(f"""<div class='guidance' style='--gl-color:{gc}; margin-top:14px;'><span class='g-bar'></span><div><p class='g-title'>{gt}</p><p class='g-body'>{gb}</p></div></div>""")
    except Exception as e:
        st.error(f"Forecast: {e}")

    # ===== SHAP EXPLAINABILITY =====
    st.html("""
    <div class='section-head'>
        <h3 class='section-title'>Why this forecast</h3>
        <span class='section-note'>SHAP feature contributions</span>
    </div>""")
    try:
        is_lstm = "LSTM" in type(model).__name__ or type(model).__name__ == "Sequential"
        min_rows_needed = SHAP_TIMESTEPS + 2 if is_lstm else 1
        if hist_lookback_df is None or hist_lookback_df.empty:
            if not _has_hw:
                st.info("Explainability needs the Hopsworks feature store, but HOPSWORKS_API_KEY isn't set for this app.")
            else:
                st.info("Explainability needs recent feature-store data — the read from Hopsworks came back empty (check the app logs for the underlying error).")
        elif not all(c in hist_lookback_df.columns for c in feature_cols):
            missing = [c for c in feature_cols if c not in hist_lookback_df.columns]
            st.info(f"Explainability skipped — the feature store is missing columns the model expects: {', '.join(missing)}.")
        else:
            hist_sorted = hist_lookback_df.sort_values("datetime")
            feat_hist = hist_sorted[feature_cols].dropna()
            if len(feat_hist) < min_rows_needed:
                st.info(f"Not enough recent history yet to explain this forecast — got {len(feat_hist)} complete rows, need at least {min_rows_needed} ({model_source}).")
            else:
                feature_names, shap_values = None, None

                if is_lstm:
                    scaled_all = scaler.transform(feat_hist[feature_cols])
                    windows = np.array([
                        scaled_all[i:i + SHAP_TIMESTEPS]
                        for i in range(len(scaled_all) - SHAP_TIMESTEPS + 1)
                    ])
                    X_latest_seq = windows[-1:]
                    bg_windows = windows[:-1]
                    bg_sample = bg_windows[np.random.default_rng(42).choice(
                        len(bg_windows), size=min(10, len(bg_windows)), replace=False
                    )]

                    explainer, kind = build_shap_explainer(model, bg_sample)
                    sv = explainer.shap_values(X_latest_seq)
                    sv = np.array(sv)  # shape ~ (1, 1, timesteps, features) or (1, timesteps, features)
                    sv = sv.reshape(-1, SHAP_TIMESTEPS, len(feature_cols))
                    # Sum contributions across the 24-hour window to get one
                    # value per feature (total influence, not per-hour detail).
                    per_feature = sv[0].sum(axis=0)
                    feature_names, shap_values = feature_cols, per_feature
                else:
                    bg_sample = feat_hist.sample(min(50, len(feat_hist)), random_state=42)
                    bg_scaled = scaler.transform(bg_sample[feature_cols])
                    latest_row = feat_hist.iloc[[-1]]
                    X_latest_scaled = scaler.transform(latest_row[feature_cols])

                    explainer, kind = build_shap_explainer(model, bg_scaled)
                    sv = explainer.shap_values(X_latest_scaled)
                    feature_names, shap_values = feature_cols, np.array(sv).flatten()

                shap_df = pd.DataFrame({"feature": feature_names, "shap": shap_values})
                shap_df["abs_shap"] = shap_df["shap"].abs()
                shap_df = shap_df.sort_values("abs_shap", ascending=True).tail(8)

                fig, ax = plt.subplots(figsize=(10, max(2.2, 0.32 * len(shap_df))))
                fig.patch.set_facecolor("#0A0C10")
                ax.set_facecolor("#0A0C10")
                bar_colors = ["#F87171" if v > 0 else "#34D399" for v in shap_df["shap"]]
                ax.barh(shap_df["feature"], shap_df["shap"], color=bar_colors, height=0.6)
                ax.axvline(0, color="#4E5563", linewidth=0.8)
                ax.set_xlabel("Impact on predicted AQI (SHAP value)", color="#7B8395", fontsize=9)
                ax.tick_params(colors="#7B8395", labelsize=9)
                for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_fontfamily("Inter")
                for s in ax.spines.values(): s.set_visible(False)
                ax.grid(True, axis="x", alpha=0.12, color="#7B8395", linestyle="-", linewidth=0.6)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                note = "Red pushes the forecast up, green pulls it down."
                if is_lstm:
                    note += " Summed across the model's 24-hour input window."
                st.html(f"<p class='section-note'>{note}</p>")
    except Exception as e:
        st.error(f"Explainability: {e}")

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
    <div class='guidance' style='--gl-color:{tc}'><span class='g-bar'></span><div><p class='g-title'>{cat}</p><p class='g-body'>{tb}</p></div></div>
    """)

except Exception as e:
    st.error(f"Error: {e}")

st.html("<p class='footer-note'>Pearl AQI Station · Karachi · Hopsworks + OpenWeather + Open-Meteo</p>")
