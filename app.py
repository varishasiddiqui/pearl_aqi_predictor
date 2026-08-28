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

# Use pandas-native tz-aware timestamps everywhere (NOT raw python datetime.datetime
# objects). Mixing raw datetime objects with matplotlib/pandas across a Streamlit
# rerun is what was causing "unsupported operand type(s) for /: 'datetime.datetime'
# and 'int'" — pandas Timestamps avoid that class of bug entirely.
now_karachi = pd.Timestamp.now(tz="UTC").tz_convert(KARACHI_TZ)

st.set_page_config(page_title="Pearl · AQI Station Karachi", layout="wide", initial_sidebar_state="expanded", page_icon="🫁")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root{
        --ink:#0A0E17; --panel:#121826; --panel-2:#161d2e; --line:#232B3D;
        --paper:#ECEEF3; --dim:#8891A6; --dim-2:#5B6478;
        --good:#3FCFB4; --moderate:#E3B23C; --uhfs:#E0812F;
        --unhealthy:#D2503A; --very:#9457C7; --hazard:#B23A52;
    }

    html, body, [class*="st-"] { font-family: 'IBM Plex Sans', sans-serif !important; }
    .stApp { background: var(--ink) !important; }
    .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--paper) !important; letter-spacing: -0.01em; }
    p, span, label, div { color: var(--paper); }
    .mono { font-family: 'IBM Plex Mono', monospace !important; }

    /* ---- sidebar: reads like a console diagnostics panel ---- */
    section[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] * { color: var(--paper) !important; }
    section[data-testid="stSidebar"] hr { border-color: var(--line) !important; }
    .diag-label { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--dim); letter-spacing:0.08em; text-transform:uppercase; margin:0 0 6px; }
    .diag-row { display:flex; align-items:center; gap:8px; font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:var(--paper); padding:3px 0; }
    .dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
    .dot-on { background:var(--good); box-shadow:0 0 6px var(--good); }
    .dot-off { background:var(--dim-2); }

    /* ---- generic panel/card ---- */
    .panel { background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:22px 24px; margin:14px 0; }
    .panel-head { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:14px; }
    .panel-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:16px; color:var(--paper); margin:0; }
    .panel-sub { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:0.07em; }

    /* ---- header strip ---- */
    .station-header { display:flex; align-items:center; justify-content:space-between; padding:10px 4px 18px; border-bottom:1px solid var(--line); margin-bottom:18px; flex-wrap:wrap; gap:8px; }
    .station-id { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--dim); letter-spacing:0.1em; }
    .station-name { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:22px; color:var(--paper); margin:0; }
    .station-clock { font-family:'IBM Plex Mono',monospace; font-size:13px; color:var(--dim); text-align:right; }

    /* ---- breathing halo hero ---- */
    .hero { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:38px 20px 30px; }
    .halo-wrap { position:relative; width:220px; height:220px; display:flex; align-items:center; justify-content:center; margin-bottom:6px; }
    .halo-ring { position:absolute; inset:0; border-radius:50%; border:2px solid var(--ring-color); opacity:0.55; animation: breathe var(--breathe-speed) ease-in-out infinite; }
    .halo-ring.r2 { inset:14px; animation-delay: calc(var(--breathe-speed) / -2); opacity:0.35; }
    .halo-core { position:relative; width:172px; height:172px; border-radius:50%; background: radial-gradient(circle at 50% 40%, var(--panel-2), var(--panel)); border:1px solid var(--line); display:flex; flex-direction:column; align-items:center; justify-content:center; z-index:2; }
    .halo-value { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:58px; line-height:1; color:var(--ring-color); }
    .halo-cat { font-family:'IBM Plex Mono',monospace; font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; color:var(--dim); margin-top:6px; }
    .hero-eyebrow { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--dim); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px; }
    .hero-dominant { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--dim); margin-top:10px; }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity:0.55; }
        50% { transform: scale(1.09); opacity:0.15; }
    }
    @media (prefers-reduced-motion: reduce) {
        .halo-ring { animation: none !important; }
    }

    /* ---- weather chip strip ---- */
    .chip-row { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:8px; }
    .chip { background:var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:110px; text-align:center; }
    .chip-label { font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--dim); text-transform:uppercase; letter-spacing:0.06em; }
    .chip-value { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:18px; color:var(--paper); margin-top:2px; }

    /* ---- pollutant gauges ---- */
    .gauge-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:6px; }
    @media (max-width:900px){ .gauge-grid{ grid-template-columns:repeat(3,1fr);} }
    .gauge-cell { display:flex; flex-direction:column; align-items:center; gap:6px; }
    .gauge-name { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--dim); letter-spacing:0.04em; }
    .gauge-val { font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:600; }
    .gauge-status { font-family:'IBM Plex Mono',monospace; font-size:9.5px; color:var(--dim-2); text-transform:uppercase; }

    /* ---- forecast day tiles ---- */
    .day-tile { text-align:center; padding:18px 10px; border-radius:12px; background:var(--panel-2); border:1px solid var(--line); }
    .day-tile .d-label { font-family:'IBM Plex Mono',monospace; color:var(--dim); font-size:11px; margin:0; letter-spacing:0.04em; }
    .day-tile .d-val { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:30px; margin:6px 0 2px; }
    .day-tile .d-cat { font-family:'IBM Plex Mono',monospace; font-size:11px; margin:0; text-transform:uppercase; letter-spacing:0.04em; }
    .day-tile .d-range { font-family:'IBM Plex Mono',monospace; color:var(--dim); font-size:10.5px; margin-top:8px; }

    /* ---- guidance strip ---- */
    .guidance { display:flex; gap:14px; align-items:flex-start; padding:18px 22px; border-radius:12px; border:1px solid var(--gl-color); background:color-mix(in srgb, var(--gl-color) 10%, var(--panel)); }
    .guidance .g-dot { width:10px; height:10px; border-radius:50%; background:var(--gl-color); margin-top:5px; flex-shrink:0; box-shadow:0 0 8px var(--gl-color); }
    .guidance .g-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:14px; color:var(--paper); margin:0 0 3px; }
    .guidance .g-body { font-family:'IBM Plex Sans',sans-serif; font-size:13px; color:var(--dim); margin:0; }

    .stAlert { border-radius: 12px !important; background: var(--panel) !important; border:1px solid var(--line) !important; }
    div[data-testid="stMetricValue"] { color: var(--paper) !important; }
    hr { border-color: var(--line) !important; }
    .footer-note { text-align:center; color:var(--dim-2); font-family:'IBM Plex Mono',monospace; font-size:10.5px; letter-spacing:0.04em; margin-top:6px; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading — Hopsworks Model Registry first, local .pkl as fallback
# (so the dashboard still works if Hopsworks is briefly unreachable).
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
            # Try newest version first, falling back to older ones only if a
            # given version's artifacts are missing/corrupted (e.g. leftover
            # broken registrations from earlier debugging). This is more
            # robust than get_best_model(), which can still point at a
            # broken version if its recorded metric happens to look "best".
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

with st.sidebar:
    st.markdown("<p class='panel-title' style='font-size:15px'>Station settings</p>", unsafe_allow_html=True)
    city_label = st.selectbox("City", ["Karachi"], label_visibility="collapsed")

    st.markdown("<p class='diag-label' style='margin-top:22px'>Model</p>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='diag-row'>· source&nbsp;&nbsp;<span style='color:var(--dim)'>{st.session_state.get('model_source', 'unknown')}</span></div>
        <div class='diag-row'>· inputs&nbsp;&nbsp;<span style='color:var(--dim)'>{len(feature_cols)} features</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='diag-label' style='margin-top:22px'>Data sources</p>", unsafe_allow_html=True)
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
        <div class='diag-row' style='align-items:flex-start;margin-bottom:8px'>
            <span class='dot {dotclass}' style='margin-top:5px'></span>
            <span>{name}<br><span style='color:var(--dim-2);font-size:11px'>{desc} · {state}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='diag-row'>tz&nbsp;&nbsp;<span style='color:var(--dim)'>UTC+5 (PKT)</span></div>
        <div class='diag-row'>refreshed&nbsp;&nbsp;<span style='color:var(--dim)'>{now_karachi.strftime('%H:%M:%S')}</span></div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Today's actual trend — read from Hopsworks Feature Store instead of a
# second live "history" API call, since this data already lives there once
# the feature pipeline has run for today.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_recent_actuals_from_feature_store(lookback_hours=72):
    """Pull the last `lookback_hours` of REAL measured rows from the Feature
    Store. We need more than just "today" here: aqi_lag_24 / aqi_rolling_24
    (used by whichever features the training pipeline selected) require up
    to 24h of real history to seed the recursive forecast correctly."""
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
# Live data fetching — current conditions + forward-looking forecast.
# This inherently CANNOT come from the feature store: Hopsworks only stores
# what already happened, not tomorrow's weather/pollutant forecast.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_current_data():
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    # ---- current pollution ----
    curr_url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
                f"?lat={LAT}&lon={LON}&appid={API_KEY}")
    curr_resp = requests.get(curr_url, timeout=15)
    curr_resp.raise_for_status()
    pollution = curr_resp.json()["list"][0]["components"]

    # ---- pollutant forecast, next ~4 days hourly ----
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

    # ---- weather: current + hourly forecast ----
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
    """Returns (category, breathe_speed_seconds, color, description). The
    breathe speed drives the halo animation on the dashboard — calmer air,
    slower pulse; worse air, faster pulse — so the animation itself encodes
    severity rather than just decorating the number."""
    if val <= 50: return "Good", "4.2s", "#3FCFB4", "Clear conditions"
    elif val <= 100: return "Moderate", "3.6s", "#E3B23C", "Acceptable"
    elif val <= 150: return "Unhealthy for Sensitive", "3.0s", "#E0812F", "Sensitive groups affected"
    elif val <= 200: return "Unhealthy", "2.4s", "#D2503A", "Everyone affected"
    elif val <= 300: return "Very Unhealthy", "1.8s", "#9457C7", "Health alert"
    else: return "Hazardous", "1.3s", "#B23A52", "Emergency conditions"


def build_forecast(feature_df, hist_lookback_df, current_aqi, current_row, feature_cols, model, scaler, hours=72):
    """Recursively predict AQI hour-by-hour using REAL forecasted pollutant/
    weather values for each hour (not a frozen snapshot).

    Builds the FULL candidate feature set used by training_pipeline.py's
    select_features() (raw pollutants/weather, temporal features, and
    aqi/pm2.5 lags + rolling means), then subsets to whatever `feature_cols`
    the currently-registered model actually needs. This has to mirror
    training exactly — training picks its feature set dynamically via
    correlation, so the app can't hardcode a fixed handful of columns or it
    silently drifts out of sync with whatever model gets registered next.

    aqi/pm2.5 history is seeded from real Feature Store readings (so lag_24 /
    rolling_24 are correct for the first ~24 predicted hours) and then
    extended with the model's own predictions as we step forward.
    """
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

    # ---- header strip: station identity + live clock, instrument-style ----
    st.markdown(f"""
    <div class='station-header'>
        <div>
            <p class='station-name'>🫁 Pearl · AQI Station</p>
            <p class='station-id'>{city_label.upper()} · 24.8607°N, 67.0011°E</p>
        </div>
        <div class='station-clock'>{now_karachi.strftime('%A, %d %B %Y')}<br>{now_karachi.strftime('%I:%M:%S %p')} PKT</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- hero: breathing halo. Pulse speed is set by aqi_info() itself, so
    # worse air literally breathes faster — the one bold visual idea on this
    # page, and it's tied to real severity rather than decorative. ----
    st.markdown(f"""
    <div class='hero' style='--ring-color:{color}; --breathe-speed:{breathe_speed}'>
        <p class='hero-eyebrow'>Current air quality index</p>
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

    st.markdown(f"""
    <div class='chip-row'>
        <div class='chip'><div class='chip-label'>Temperature</div><div class='chip-value'>{weather['temperature_2m']:.1f}°C</div></div>
        <div class='chip'><div class='chip-label'>Humidity</div><div class='chip-value'>{weather['relative_humidity_2m']:.0f}%</div></div>
        <div class='chip'><div class='chip-label'>Wind speed</div><div class='chip-value'>{weather['wind_speed_10m']:.1f} km/h</div></div>
        <div class='chip'><div class='chip-label'>Pressure</div><div class='chip-value'>{weather['surface_pressure']:.0f} hPa</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- pollutant readouts as small radial gauges — reads like a bank of
    # sensor dials rather than a plain text list, and each one is legible
    # on its own (value + % of the WHO-style threshold + a status word). ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Pollutant levels</p>
        <p class='panel-sub'>% of health threshold</p>
    </div>""", unsafe_allow_html=True)

    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
    gauges = "<div class='gauge-grid'>"
    for p, val in show_p.items():
        pct = min(val / threshold.get(p, 100) * 100, 100)
        status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
        gcolor = "#3FCFB4" if pct < 40 else "#E3B23C" if pct < 70 else "#D2503A"
        gauges += f"""
        <div class='gauge-cell'>
            <div style='width:58px;height:58px;border-radius:50%;background:conic-gradient({gcolor} {pct:.0f}%, var(--line) {pct:.0f}% 100%);display:flex;align-items:center;justify-content:center;'>
                <div style='width:44px;height:44px;border-radius:50%;background:var(--panel);display:flex;align-items:center;justify-content:center;'>
                    <span class='gauge-val' style='color:{gcolor}'>{val:.0f}</span>
                </div>
            </div>
            <span class='gauge-name'>{p.upper()}</span>
            <span class='gauge-status' style='color:{gcolor}'>{status}</span>
        </div>"""
    gauges += "</div>"
    st.markdown(gauges, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Today's Trend: REAL measured data from the Feature Store
    # (midnight -> now) + model forecast for the remaining hours of today. ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>Today's AQI trend</p>
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
            fig.patch.set_facecolor("#121826")
            ax.set_facecolor("#121826")
            day_start_plot = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end_plot = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([day_start_plot, day_end_plot], 0, 50, alpha=0.10, color="#3FCFB4")
            ax.fill_between([day_start_plot, day_end_plot], 50, 100, alpha=0.10, color="#E3B23C")
            ax.fill_between([day_start_plot, day_end_plot], 100, 150, alpha=0.10, color="#E0812F")
            ax.fill_between([day_start_plot, day_end_plot], 150, 200, alpha=0.10, color="#D2503A")

            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=2.5, label="Actual (measured)", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=2, linestyle="--", alpha=0.75, label="Predicted", zorder=5)
                all_vals += future_preds

            ax.scatter([now_karachi], [current_aqi], color=color, s=110, zorder=6, edgecolors="#ECEEF3", linewidths=2, label="Now")
            ax.axhline(current_aqi, color="#ECEEF3", linestyle="--", alpha=0.25, linewidth=1)
            ax.set_ylabel("AQI", color="#8891A6", fontsize=11, fontfamily="monospace")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, alpha=0.12, color="#8891A6")
            ax.tick_params(colors="#8891A6", labelsize=9)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("monospace")
            for spine in ax.spines.values():
                spine.set_color("#232B3D")
            ax.legend(facecolor="#121826", edgecolor="#232B3D", labelcolor="#ECEEF3", fontsize=9, loc="upper right")
            if all_vals:
                ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- 3-Day Forecast: recursive model prediction using REAL forecasted
    # pollutant + weather values per hour. Necessarily live-API-based, since
    # the feature store only holds data that has already happened. ----
    st.markdown("""<div class='panel'><div class='panel-head'>
        <p class='panel-title'>3-day forecast</p>
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
                        <div class='day-tile' style='border-color:{day_color}40'>
                            <p class='d-label'>{day_date_label}</p>
                            <p class='d-val' style='color:{day_color}'>{day_avg:.0f}</p>
                            <p class='d-cat' style='color:{day_color}'>{day_cat}</p>
                            <p class='d-range'>↓ {day_min:.0f} &nbsp;—&nbsp; {day_max:.0f} ↑</p>
                        </div>
                        """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(14, 2.8))
                fig.patch.set_facecolor("#121826")
                ax.set_facecolor("#121826")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.5)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.15, color=color)
                ax.axhline(100, color="#E3B23C", linestyle="--", alpha=0.4, linewidth=0.8)
                ax.axhline(150, color="#E0812F", linestyle="--", alpha=0.4, linewidth=0.8)
                ax.set_ylabel("AQI", color="#8891A6", fontsize=10, fontfamily="monospace")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, alpha=0.1, color="#8891A6")
                ax.tick_params(colors="#8891A6", labelsize=8)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily("monospace")
                for spine in ax.spines.values():
                    spine.set_color("#232B3D")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if any(a > 150 for a in forecast_aqi):
                    guide_color, guide_title, guide_body = "#D2503A", "Hazardous AQI expected", "Avoid outdoor activity over the next 3 days where possible."
                elif any(a > 100 for a in forecast_aqi):
                    guide_color, guide_title, guide_body = "#E3B23C", "Unhealthy AQI expected", "Sensitive groups should limit prolonged time outdoors."
                else:
                    guide_color, guide_title, guide_body = "#3FCFB4", "Within safe range", "No elevated AQI expected in the next 3 days."
                st.markdown(f"""
                <div class='guidance' style='--gl-color:{guide_color}'>
                    <span class='g-dot'></span>
                    <div><p class='g-title'>{guide_title}</p><p class='g-body'>{guide_body}</p></div>
                </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Forecast error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    tips = {
        "Good": ("Excellent air quality — perfect for outdoor activity.", "#3FCFB4"),
        "Moderate": ("Acceptable. Sensitive people should limit prolonged outdoor exertion.", "#E3B23C"),
        "Unhealthy for Sensitive": ("Sensitive groups should reduce outdoor activity.", "#E0812F"),
        "Unhealthy": ("Reduce outdoor physical activity for everyone.", "#D2503A"),
        "Very Unhealthy": ("Avoid outdoors — use air purifiers indoors.", "#9457C7"),
        "Hazardous": ("Emergency conditions. Stay indoors and seek medical help if needed.", "#B23A52"),
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
