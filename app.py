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
            registry_model = mr.get_model(MODEL_NAME)  # latest version
            model_dir = registry_model.download()
            import os
            model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            features = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
            st.session_state["model_source"] = f"Hopsworks Model Registry (v{registry_model.version})"
            return model, scaler, features
        except Exception as e:
            st.sidebar.warning(f"Couldn't load from Hopsworks Model Registry, using local files instead. ({e})")

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_cols.pkl")
    st.session_state["model_source"] = "local file (fallback)"
    return model, scaler, features


model, scaler, feature_cols = load_model()

with st.sidebar:
    st.markdown("## ⚙️ Dashboard Settings")
    city_label = st.selectbox("City", ["Karachi"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"- **Source:** {st.session_state.get('model_source', 'unknown')}")
    st.markdown(f"- **Features:** {len(feature_cols)} selected")
    st.markdown("---")
    st.markdown("### 🔗 Data Sources")
    st.markdown("- OpenWeather API (current / forecast)")
    st.markdown("- Open-Meteo API (weather forecast)")
    st.markdown("- Hopsworks Feature Store (today's actual trend)")
    st.markdown("- Hopsworks Model Registry (model)")
    st.markdown("---")
    st.markdown(f"**Timezone:** UTC+5 (PKT)")
    st.markdown(f"**Refreshed:** {now_karachi.strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Today's actual trend — read from Hopsworks Feature Store instead of a
# second live "history" API call, since this data already lives there once
# the feature pipeline has run for today.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_today_actuals_from_feature_store():
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
        today_start = now_karachi.normalize()
        df = df[(df["datetime"] >= today_start) & (df["datetime"] <= now_karachi)]
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
    if val <= 50: return "Good", "🟢", "#00e676", "glow-green"
    elif val <= 100: return "Moderate", "🟡", "#ffea00", "glow-yellow"
    elif val <= 150: return "Unhealthy for Sensitive", "🟠", "#ff9100", "glow-yellow"
    elif val <= 200: return "Unhealthy", "🔴", "#ff1744", "glow-red"
    elif val <= 300: return "Very Unhealthy", "🟣", "#d500f9", "glow-red"
    else: return "Hazardous", "🟤", "#880e4f", "glow-red"


def build_forecast(feature_df, current_aqi, feature_cols, model, scaler, hours=72):
    """Recursively predict AQI hour-by-hour using REAL forecasted pollutant/
    weather values for each hour (not a frozen snapshot). Only aqi_lag_24
    has to be approximated recursively, since we don't have a full rolling
    history of past AQI to look back exactly 24h for every future step."""
    preds = []
    df = feature_df.reset_index(drop=True)
    n = min(hours, len(df))
    for h in range(n):
        row = df.iloc[h]
        lag = current_aqi if h < 24 else preds[h - 24]
        feat = {
            "pm2_5": row.get("pm2_5", np.nan),
            "so2": row.get("so2", np.nan),
            "aqi_lag_24": lag,
            "pressure": row.get("pressure", np.nan),
            "month": row["datetime"].month,
            "co": row.get("co", np.nan),
            "wind_speed": row.get("wind_speed", np.nan),
            "no2": row.get("no2", np.nan),
            "o3": row.get("o3", np.nan),
        }
        X = pd.DataFrame([feat])[feature_cols]
        X_scaled = scaler.transform(X)
        pred = max(0, float(model.predict(X_scaled).flatten()[0]))
        preds.append(pred)
    return df["datetime"].iloc[:n].tolist(), preds


try:
    pollution, weather, combined_df = fetch_current_data()
    hist_df = fetch_today_actuals_from_feature_store()
    current_aqi, dominant = get_aqi(pollution)
    cat, emoji, color, glow = aqi_info(current_aqi)

    col_city, col_time = st.columns([1, 3])
    with col_city:
        st.markdown(f"<div class='card'><h3>📍 {city_label}</h3></div>", unsafe_allow_html=True)
    with col_time:
        st.markdown(f"<div class='card'><p style='text-align:right;color:gray'>🕐 {now_karachi.strftime('%A, %d %B %Y | %I:%M %p')}</p></div>", unsafe_allow_html=True)

    hex_r = int(color[1:3], 16)
    hex_g = int(color[3:5], 16)
    hex_b = int(color[5:7], 16)

    st.markdown(f"""
    <div style='text-align:center;padding:30px;border-radius:20px;background:linear-gradient(135deg,rgba({hex_r},{hex_g},{hex_b},0.15),transparent);border:1px solid {color};margin:10px 0'>
        <p style='color:gray;font-size:14px'>CURRENT AIR QUALITY INDEX</p>
        <h1 style='color:{color};font-size:72px;margin:0'>{current_aqi:.0f}</h1>
        <h2 style='color:{color};margin:5px 0'>{emoji} {cat}</h2>
        <p style='color:gray;font-size:12px'>Dominant: {dominant.upper()}</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🌡️ Temperature", f"{weather['temperature_2m']:.1f}°C")
    with m2: st.metric("💧 Humidity", f"{weather['relative_humidity_2m']:.0f}%")
    with m3: st.metric("💨 Wind Speed", f"{weather['wind_speed_10m']:.1f} km/h")
    with m4: st.metric("📊 Pressure", f"{weather['surface_pressure']:.0f} hPa")

    st.divider()

    # Pollutants
    st.subheader("☁️ Pollutant Levels")
    show_p = {k: v for k, v in pollution.items() if k not in ["no", "nh3"]}
    pcols = st.columns(3)
    for i, (p, val) in enumerate(show_p.items()):
        with pcols[i % 3]:
            threshold = {"pm2_5": 75, "pm10": 150, "no2": 100, "so2": 75, "o3": 70, "co": 10000}
            pct = min(val / threshold.get(p, 100) * 100, 100)
            status = "Low" if pct < 40 else "Moderate" if pct < 70 else "High"
            st.markdown(f"**{p.upper()}** `{val:.1f}` <span style='color:gray'>({status} {pct:.0f}%)</span>", unsafe_allow_html=True)

    st.divider()

    # ---- Today's Trend: REAL measured data from the Feature Store
    # (midnight -> now) + model forecast for the remaining hours of today. ----
    st.subheader("📈 Today's AQI Trend")
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
                    future_today_df, current_aqi, feature_cols, model, scaler,
                    hours=len(future_today_df)
                )

        if hist_df.empty and not future_times:
            st.warning("No trend data available right now — Feature Store may not have today's data yet, or the API may be rate-limited.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            day_start_plot = now_karachi.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end_plot = now_karachi.replace(hour=23, minute=59, second=0, microsecond=0)
            ax.fill_between([day_start_plot, day_end_plot], 0, 50, alpha=0.1, color="#00e676")
            ax.fill_between([day_start_plot, day_end_plot], 50, 100, alpha=0.1, color="#ffea00")
            ax.fill_between([day_start_plot, day_end_plot], 100, 150, alpha=0.1, color="#ff9100")
            ax.fill_between([day_start_plot, day_end_plot], 150, 200, alpha=0.1, color="#ff1744")

            all_vals = []
            if not hist_df.empty:
                ax.plot(hist_df["datetime"], hist_df["aqi"], color=color, linewidth=2.5, label="Actual (measured)", zorder=5)
                all_vals += hist_df["aqi"].tolist()
            if future_times:
                ax.plot(future_times, future_preds, color=color, linewidth=2, linestyle="--", alpha=0.75, label="Predicted", zorder=5)
                all_vals += future_preds

            ax.scatter([now_karachi], [current_aqi], color=color, s=110, zorder=6, edgecolors="white", linewidths=2, label="Now")
            ax.axhline(current_aqi, color="white", linestyle="--", alpha=0.3, linewidth=1)
            ax.set_ylabel("AQI", color="white", fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p", tz=KARACHI_TZ))
            ax.grid(True, alpha=0.15, color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color((1, 1, 1, 0.1))
            ax.legend(facecolor="#0e1117", edgecolor="none", labelcolor="white", fontsize=9, loc="upper right")
            if all_vals:
                ax.set_ylim(max(0, min(all_vals) - 10), max(all_vals) + 10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Trend: {e}")

    st.divider()

    # ---- 3-Day Forecast: recursive model prediction using REAL forecasted
    # pollutant + weather values per hour. Necessarily live-API-based, since
    # the feature store only holds data that has already happened. ----
    st.subheader("📅 3-Day Forecast")
    try:
        if combined_df.empty:
            st.warning("Forecast data unavailable right now — API may be rate-limited or unavailable.")
        else:
            future_df = combined_df[combined_df["datetime"] > now_karachi].sort_values("datetime")
            times, forecast_aqi = build_forecast(future_df, current_aqi, feature_cols, model, scaler, hours=72)

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
                    day_cat, day_emoji, day_color, _ = aqi_info(day_avg)
                    day_date_label = pd.Timestamp(day).strftime("%d %b %A")
                    with day_cols[d]:
                        st.markdown(f"""
                        <div style='text-align:center;padding:20px;border-radius:15px;background:rgba(255,255,255,0.03);border:1px solid {day_color}30;margin:5px 0'>
                            <p style='color:gray;font-size:12px;margin:0'>{day_date_label}</p>
                            <h2 style='color:{day_color};margin:5px 0'>{day_avg:.0f} {day_emoji}</h2>
                            <p style='color:{day_color};font-size:13px;margin:0'>{day_cat}</p>
                            <p style='color:gray;font-size:11px;margin:5px 0 0'>↓ {day_min:.0f} — {day_max:.0f} ↑</p>
                        </div>
                        """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(14, 3))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                ax.plot(fdf["datetime"], fdf["aqi"], color=color, linewidth=1.5)
                ax.fill_between(fdf["datetime"], fdf["aqi"] - 3, fdf["aqi"] + 3, alpha=0.15, color=color)
                ax.axhline(100, color="yellow", linestyle="--", alpha=0.3, linewidth=0.8)
                ax.axhline(150, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)
                ax.set_ylabel("AQI", color="white", fontsize=10)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=KARACHI_TZ))
                ax.grid(True, alpha=0.1, color="white")
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color((1, 1, 1, 0.1))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if any(a > 150 for a in forecast_aqi):
                    st.error("🚨 ALERT: Hazardous AQI expected! Avoid outdoor activities.")
                elif any(a > 100 for a in forecast_aqi):
                    st.warning("⚠️ WARNING: Unhealthy AQI expected.")
                else:
                    st.success("✅ AQI within safe range.")
    except Exception as e:
        st.error(f"Forecast error: {e}")

    st.divider()
    tips = {
        "Good": "🟢 Excellent air quality! Perfect for outdoor activities.",
        "Moderate": "🟡 Acceptable. Sensitive people should limit prolonged outdoor exertion.",
        "Unhealthy for Sensitive": "🟠 Sensitive groups: reduce outdoor activities.",
        "Unhealthy": "🔴 Reduce outdoor physical activities for everyone.",
        "Very Unhealthy": "🟣 Avoid outdoors. Use air purifiers indoors.",
        "Hazardous": "🟤 Emergency! Stay indoors. Seek medical help if needed.",
    }
    st.info(tips.get(cat, ""))

except Exception as e:
    st.error(f"Error: {e}")

st.divider()
st.markdown("<p style='text-align:center;color:gray;font-size:11px'>Pearls AQI Predictor | MLOps Project | Hopsworks Feature Store + Model Registry</p>", unsafe_allow_html=True)
