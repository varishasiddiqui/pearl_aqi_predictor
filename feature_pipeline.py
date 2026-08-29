import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CITY = "Karachi"
LAT, LON = 24.8607, 67.0011
KARACHI_TZ = timezone(timedelta(hours=5))

OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 1

BREAKPOINTS = {
    "pm2_5": [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    "pm10":  [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "no2":   [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2":   [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
    "o3":    [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300)],
    "co":    [(0, 4400, 0, 50), (4401, 9400, 51, 100), (9401, 12400, 101, 150), (12401, 15400, 151, 200)],
}


def calc_sub_index(pollutant, conc):
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS.get(pollutant, []):
        if conc <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    return 500.0


def calculate_aqi(row):
    sub_indices = {p: calc_sub_index(p, row.get(p, 0)) for p in BREAKPOINTS}
    dominant = max(sub_indices, key=sub_indices.get)
    return round(max(sub_indices.values()), 1), dominant


def fetch_pollution_history(lat, lon, api_key, days):
    """OpenWeather Air Pollution History API, paginated in <=7 day chunks."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_records = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        url = (
            "http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={lat}&lon={lon}"
            f"&start={int(chunk_start.timestamp())}&end={int(chunk_end.timestamp())}"
            f"&appid={api_key}"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        all_records.extend(resp.json().get("list", []))
        chunk_start = chunk_end
        time.sleep(0.5)  # be polite to the API

    return all_records


def fetch_weather_history(lat, lon, days):
    """Open-Meteo historical weather archive (no API key needed)."""
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["hourly"]


def build_features(days):
    """Fetch raw data, merge, and compute all engineered features. Returns a
    DataFrame ready to insert into the feature store."""
    raw_pollution = fetch_pollution_history(LAT, LON, OPENWEATHER_API_KEY, days)
    print(f"Fetched {len(raw_pollution)} hourly pollution records")

    pollution_rows = []
    for item in raw_pollution:
        row = {"datetime": pd.to_datetime(item["dt"], unit="s", utc=True).tz_convert(KARACHI_TZ)}
        row.update(item["components"])
        pollution_rows.append(row)
    pollution_df = pd.DataFrame(pollution_rows).drop_duplicates(subset="datetime").sort_values("datetime")
    pollution_df = pollution_df.reset_index(drop=True)

    weather_raw = fetch_weather_history(LAT, LON, days)
    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather_raw["time"]).tz_localize("UTC").tz_convert(KARACHI_TZ),
        "temperature": weather_raw["temperature_2m"],
        "humidity": weather_raw["relative_humidity_2m"],
        "wind_speed": weather_raw["wind_speed_10m"],
        "pressure": weather_raw["surface_pressure"],
    })

    pollution_df["datetime"] = pollution_df["datetime"].dt.as_unit("ns")
    weather_df["datetime"] = weather_df["datetime"].dt.as_unit("ns")

    merged_df = pd.merge_asof(
        pollution_df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"),
    ).dropna(subset=["temperature"]).reset_index(drop=True)

    print("Merged shape:", merged_df.shape)
    if merged_df.empty:
        raise RuntimeError("No overlapping pollution+weather rows — check API responses/date range.")

    # AQI + dominant pollutant
    aqi_results = merged_df.apply(calculate_aqi, axis=1)
    merged_df["aqi"] = aqi_results.apply(lambda x: x[0])
    merged_df["dominant_pollutant"] = aqi_results.apply(lambda x: x[1])
    merged_df["city"] = CITY

    # temporal features
    merged_df = merged_df.sort_values("datetime").reset_index(drop=True)
    merged_df["hour"] = merged_df["datetime"].dt.hour
    merged_df["day_of_week"] = merged_df["datetime"].dt.dayofweek
    merged_df["day_name"] = merged_df["datetime"].dt.day_name()
    merged_df["month"] = merged_df["datetime"].dt.month
    merged_df["is_weekend"] = merged_df["day_of_week"].isin([5, 6]).astype(int)

    # lag features (backward-looking only, no leakage)
    merged_df["aqi_lag_1"] = merged_df["aqi"].shift(1)
    merged_df["aqi_lag_3"] = merged_df["aqi"].shift(3)
    merged_df["aqi_lag_24"] = merged_df["aqi"].shift(24)
    merged_df["pm25_lag_1"] = merged_df["pm2_5"].shift(1)
    merged_df["pm25_lag_24"] = merged_df["pm2_5"].shift(24)

    # rolling averages (also backward-looking only)
    merged_df["aqi_rolling_3"] = merged_df["aqi"].rolling(3).mean()
    merged_df["aqi_rolling_6"] = merged_df["aqi"].rolling(6).mean()
    merged_df["aqi_rolling_24"] = merged_df["aqi"].rolling(24).mean()
    merged_df["pm25_rolling_24"] = merged_df["pm2_5"].rolling(24).mean()

    # target: AQI 24 hours ahead of each row (only needed for training,
    # harmless to compute here too — rows with a NaN target are still valid
    # feature rows for hourly ingestion, just not usable for training yet)
    merged_df["target_aqi_24hr"] = merged_df["aqi"].shift(-24)

    before = len(merged_df)
    # Only drop rows missing the LAG/ROLLING inputs (needed for prediction).
    # We do NOT drop rows for a missing target here, since the most recent
    # ~24 hours legitimately won't have a target yet — that's expected for
    # an hourly ingestion run, not a bug.
    feature_input_cols = [
        "aqi_lag_1", "aqi_lag_3", "aqi_lag_24", "pm25_lag_1", "pm25_lag_24",
        "aqi_rolling_3", "aqi_rolling_6", "aqi_rolling_24", "pm25_rolling_24",
    ]
    merged_df = merged_df.dropna(subset=feature_input_cols).reset_index(drop=True)
    print(f"Rows before dropna: {before} -> after: {len(merged_df)}")

    return merged_df


def align_dtypes_to_schema(df, feature_group):
    """Cast df columns to whatever type the EXISTING feature group schema
    actually expects, instead of guessing. This is what caused the back-and-forth
    bugs: OpenWeather/Open-Meteo readings can come back as either whole numbers
    or decimals depending on the hour, so pandas' inferred dtype (int64 vs
    float64) varies run to run and randomly mismatches whatever type the
    feature group locked in on its very first insert. Reading the schema at
    runtime and casting to match it means this class of bug can't recur,
    regardless of which column or which direction (int<->float) it hits."""
    type_map = {
        "bigint": "int64", "int": "int32", "smallint": "int16", "tinyint": "int8",
        "double": "float64", "float": "float32",
        "boolean": "bool",
        "string": "string",
    }
    schema = {f.name: f.type for f in feature_group.features}
    for col in df.columns:
        expected = schema.get(col)
        target_dtype = type_map.get(expected)
        if target_dtype is None:
            continue  # timestamp/date/unrecognized types: leave as-is
        try:
            if target_dtype.startswith("int") and df[col].isna().any():
                # nullable Int64 so NaNs don't crash the cast
                df[col] = df[col].astype("Int64")
            else:
                df[col] = df[col].astype(target_dtype)
        except (ValueError, TypeError) as e:
            print(f"Warning: could not cast '{col}' to {target_dtype} (schema says {expected}): {e}")
    return df


def push_to_hopsworks(df):
    import hopsworks

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + engineered lag/rolling features for Karachi",
        primary_key=["datetime"],
        event_time="datetime",
        time_travel_format="HUDI",  # explicit — Colab/CI envs often auto-install
                                     # `deltalake`, which silently flips the
                                     # default to DELTA and breaks the plain
                                     # Python client. Force HUDI.
    )
    df = align_dtypes_to_schema(df, feature_group)
    feature_group.insert(df, write_options={"wait_for_job": True})
    print(f"Feature group '{FEATURE_GROUP_NAME}' updated with {len(df)} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill", type=int, default=None,
        help="Days of history to backfill (e.g. 120). Omit for a normal hourly run.",
    )
    args = parser.parse_args()

    # Normal hourly runs only need a small overlapping window (covers lag/rolling
    # windows + guards against a missed run); backfill pulls much further back.
    days = args.backfill if args.backfill is not None else 3

    df = build_features(days)
    push_to_hopsworks(df)


if __name__ == "__main__":
    main()
