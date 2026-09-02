"""
FastAPI serving layer for the Karachi AQI predictor.

Runs as its own process (e.g. deployed separately on Render), independent
from the Streamlit dashboard (app.py). Deploying/running this does not
touch or affect the dashboard in any way.

Local dev:
    OPENWEATHER_API_KEY=xxx HOPSWORKS_API_KEY=xxx uvicorn api:app --reload
"""
import os
from datetime import timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

KARACHI_TZ = timezone(timedelta(hours=5))
LAT, LON = 24.8607, 67.0011
FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 2
MODEL_NAME = "aqi_predictor_karachi"

app = FastAPI(title="Pearl AQI Predictor API", version="1.0")

_model_cache = {}


def load_model():
    """Same resolution order as the dashboard: latest Hopsworks Model
    Registry version, falling back to local files if unavailable."""
    if "model" in _model_cache:
        return _model_cache["model"]

    hopsworks_key = os.environ.get("HOPSWORKS_API_KEY", "")
    if hopsworks_key:
        try:
            import hopsworks
            project = hopsworks.login(api_key_value=hopsworks_key)
            mr = project.get_model_registry()
            versions = sorted(mr.get_models(MODEL_NAME), key=lambda m: m.version, reverse=True)
            for registry_model in versions:
                try:
                    model_dir = registry_model.download()
                    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
                    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
                    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
                    _model_cache["model"] = (model, scaler, feature_cols, f"Hopsworks v{registry_model.version}")
                    return _model_cache["model"]
                except Exception:
                    continue
        except Exception:
            pass

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    _model_cache["model"] = (model, scaler, feature_cols, "local fallback")
    return _model_cache["model"]


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
    indices = {p: calc_aqi(p, components.get(p, 0)) for p in ["pm2_5", "pm10", "no2", "so2", "o3", "co"]}
    return max(indices.values()), max(indices, key=indices.get)


def fetch_current_pollution():
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENWEATHER_API_KEY not configured")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["list"][0]["components"]


def fetch_forecast_inputs():
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    fc_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={api_key}"
    fc_resp = requests.get(fc_url, timeout=15)
    fc_resp.raise_for_status()
    poll_df = pd.DataFrame([
        {"datetime": pd.to_datetime(i["dt"], unit="s", utc=True).tz_convert(KARACHI_TZ), **i["components"]}
        for i in fc_resp.json().get("list", [])
    ])

    w_url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
             "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&forecast_days=4")
    w_resp = requests.get(w_url, timeout=15)
    w_resp.raise_for_status()
    hourly = w_resp.json()["hourly"]
    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]).tz_localize("UTC").tz_convert(KARACHI_TZ),
        "temperature": hourly["temperature_2m"], "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"], "pressure": hourly["surface_pressure"],
    })

    combined = pd.merge_asof(
        poll_df.sort_values("datetime"), weather_df.sort_values("datetime"),
        on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"),
    ).dropna(subset=["temperature"])
    return combined


def fetch_recent_actuals(lookback_hours=72):
    hopsworks_key = os.environ.get("HOPSWORKS_API_KEY", "")
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
        now = pd.Timestamp.now(tz="UTC").tz_convert(KARACHI_TZ)
        df = df[df["datetime"] >= now - pd.Timedelta(hours=lookback_hours)]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def build_forecast(future_df, hist_df, current_aqi, current_row, feature_cols, model, scaler, hours=72):
    df = future_df.reset_index(drop=True)
    n = min(hours, len(df))
    aqi_hist = list(hist_df["aqi"]) if not hist_df.empty else []
    pm25_hist = list(hist_df["pm2_5"]) if not hist_df.empty else []
    aqi_hist.append(current_aqi)
    pm25_hist.append(current_row.get("pm2_5", np.nan))

    def lag(h, k):
        return h[-k] if len(h) >= k else (h[0] if h else np.nan)

    def rolling(h, k):
        w = h[-k:] if len(h) >= k else h
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
            "aqi_lag_1": lag(aqi_hist, 1), "aqi_lag_3": lag(aqi_hist, 3),
            "aqi_lag_24": lag(aqi_hist, 24), "pm25_lag_1": lag(pm25_hist, 1),
            "pm25_lag_24": lag(pm25_hist, 24),
            "aqi_change_rate": aqi_hist[-1] - lag(aqi_hist, 2) if len(aqi_hist) >= 2 else 0.0,
            "pm25_change_rate": pm25_hist[-1] - lag(pm25_hist, 2) if len(pm25_hist) >= 2 else 0.0,
            "aqi_rolling_3": rolling(aqi_hist, 3), "aqi_rolling_6": rolling(aqi_hist, 6),
            "aqi_rolling_24": rolling(aqi_hist, 24), "pm25_rolling_24": rolling(pm25_hist, 24),
        }
        X = pd.DataFrame([feat])[feature_cols]
        pred = max(0, float(model.predict(scaler.transform(X)).flatten()[0]))
        preds.append(pred)
        times.append(dt)
        aqi_hist.append(pred)
        pm25_hist.append(row.get("pm2_5", np.nan))
    return times, preds


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/current")
def current():
    pollution = fetch_current_pollution()
    aqi, dominant = get_aqi(pollution)
    return {"aqi": aqi, "dominant_pollutant": dominant, "components": pollution}


@app.get("/forecast")
def forecast(hours: int = 72):
    if hours < 1 or hours > 72:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 72")

    model, scaler, feature_cols, model_source = load_model()
    pollution = fetch_current_pollution()
    current_aqi, dominant = get_aqi(pollution)
    future_df = fetch_forecast_inputs()
    hist_df = fetch_recent_actuals()

    if future_df.empty:
        raise HTTPException(status_code=503, detail="Weather/pollutant forecast unavailable right now")

    times, preds = build_forecast(future_df, hist_df, current_aqi, pollution, feature_cols, model, scaler, hours=hours)
    return {
        "model_source": model_source,
        "current_aqi": current_aqi,
        "dominant_pollutant": dominant,
        "forecast": [{"datetime": t.isoformat(), "aqi": round(p, 1)} for t, p in zip(times, preds)],
    }
