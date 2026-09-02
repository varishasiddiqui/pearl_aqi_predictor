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
# v2 adds aqi_change_rate / pm25_change_rate. Bumped (not overwritten in place)
# so the existing v1 feature group and any models already registered against
# it keep working untouched — v2 just starts collecting fresh, in parallel.
FEATURE_GROUP_VERSION = 2

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

    # rate-of-change features (backward-looking only, no leakage)
    merged_df["aqi_change_rate"] = merged_df["aqi"] - merged_df["aqi_lag_1"]
    merged_df["pm25_change_rate"] = merged_df["pm2_5"] - merged_df["pm25_lag_1"]

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
        "aqi_change_rate", "pm25_change_rate",
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
    schema = {f.name: f.type for f in feature_group.features}  # still works; deprecated in favor of .columns in newer hopsworks-api, but that API's shape isn't confirmed here
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


def _extract_real_errors(log_text, max_chars=5000):
    """Filter out known-benign Spark shutdown/metrics noise and surface the
    actual failure. The Prometheus-pushgateway SocketTimeoutException seen
    during executor teardown is a documented cosmetic bug in Spark's
    banzaicloud metrics sink (it fires even on a clean, successful shutdown)
    — it is not itself a failure cause, so a plain tail of the log tends to
    show this noise instead of the real error, which is usually earlier in
    a large log."""
    noise_markers = (
        "PrometheusSink", "pushgateway", "PushGateway.java", "ScheduledReporter",
        "CoarseGrainedExecutorBackend-stop-executor", "MetricsSystem.scala",
        "com.codahale.metrics",
    )
    error_markers = (
        "ERROR", "Exception", "FAILED", "Caused by", "OutOfMemory",
        "Job aborted", "Traceback", "killed", "Killed",
    )
    real_lines = [
        line for line in log_text.splitlines()
        if not any(n in line for n in noise_markers) and any(e in line for e in error_markers)
    ]
    if real_lines:
        return "\n".join(real_lines[-100:])[-max_chars:]
    return "(no error lines found outside known Spark-metrics shutdown noise; showing raw tail)\n" + log_text[-max_chars:]


def push_to_hopsworks(df):
    import time
    import hopsworks
    from hopsworks_common.client.exceptions import JobExecutionException

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

    attempt_start_ms = int(time.time() * 1000)
    try:
        feature_group.insert(df, write_options={"wait_for_job": True})
    except JobExecutionException:
        # We've seen this specific pattern: the Hudi commit itself completes
        # (rows are visibly written — see commit_details() below), but a
        # separate server-side metadata/RPC call afterwards times out
        # (SocketTimeoutException / "Transaction marked for rollback"), and
        # Hopsworks reports the whole job as FAILED even though the data
        # already landed. Before treating this as a real failure, check
        # whether a commit newer than when we started this insert actually
        # exists — if so, the data is safe and this is a false alarm.
        print("Materialization job reported FAILED — checking if the data landed anyway...")
        try:
            commits = feature_group.commit_details()
            latest_commit_ms = max(commits.keys()) if commits else 0
            if latest_commit_ms >= attempt_start_ms - 5000:  # small clock-skew buffer
                info = commits[max(commits.keys())]
                print(
                    f"Data DID land: commit at {info.get('committedOn')} "
                    f"(inserted={info.get('rowsInserted')}, updated={info.get('rowsUpdated')}). "
                    "Treating this run as successful — the FAILED status was a server-side "
                    "timeout after the write already completed, not data loss."
                )
                print(f"Feature group '{FEATURE_GROUP_NAME}' updated with {len(df)} rows (verified via commit_details).")
                return
            else:
                print("No new commit found matching this run — this looks like a real failure.")
        except Exception as verify_err:
            print(f"Could not verify via commit_details either: {verify_err}")

        # Either verification showed no new commit, or verification itself
        # failed — pull the job logs (noise-filtered) so the actual cause is
        # visible directly in the GitHub Actions log.
        print("Fetching job logs from Hopsworks...")
        try:
            executions = feature_group.materialization_job.get_executions()
            if executions:
                out_log, err_log = executions[0].download_logs()
                if err_log:
                    print("--- stderr: real errors (Spark-metrics shutdown noise filtered out) ---")
                    print(_extract_real_errors(open(err_log).read()))
                if out_log:
                    print("--- stdout: real errors (Spark-metrics shutdown noise filtered out) ---")
                    print(_extract_real_errors(open(out_log).read(), max_chars=2000))
        except Exception as log_err:
            print(f"Could not fetch job logs automatically: {log_err}")
            print(f"Check manually: {feature_group.materialization_job.get_url()}")
        raise
    print(f"Feature group '{FEATURE_GROUP_NAME}' updated with {len(df)} rows.")


HAZARD_AQI_THRESHOLD = 150
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")


def send_hazard_alert_if_needed(df):
    """Optional hazard-AQI alert. No-op if ALERT_WEBHOOK_URL isn't set, and
    any failure here is swallowed so it can never break the feature pipeline
    run itself (the data has already landed in Hopsworks by this point)."""
    if not ALERT_WEBHOOK_URL or df.empty:
        return
    try:
        latest = df.sort_values("datetime").iloc[-1]
        if latest["aqi"] < HAZARD_AQI_THRESHOLD:
            return
        payload = {
            "text": (
                f"⚠️ Hazardous AQI in {CITY}: {latest['aqi']:.0f} "
                f"(dominant pollutant: {latest['dominant_pollutant'].upper()}) "
                f"at {latest['datetime']}"
            )
        }
        resp = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"Hazard alert sent (AQI={latest['aqi']:.0f}).")
    except Exception as e:
        print(f"Hazard alert failed (non-fatal, continuing): {e}")


AUTO_BACKFILL_ROW_THRESHOLD = 200  # below this, the group is treated as "cold" and self-backfills
AUTO_BACKFILL_DAYS = 30


def get_existing_row_count():
    """How many rows are already in the target feature group version. Used
    to decide whether this run needs to self-backfill. Any failure here
    (group doesn't exist yet, transient read issue) is treated as 0 rows —
    the safe default, since it just triggers a backfill rather than skipping
    one that was needed."""
    try:
        import hopsworks

        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        return len(fg.read())
    except Exception as e:
        print(f"Could not check existing row count ({e}); assuming 0 (cold start).")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill", type=int, default=None,
        help="Days of history to backfill (e.g. 120). Omit to let the pipeline "
             "decide automatically (self-backfills on a cold/near-empty feature group).",
    )
    args = parser.parse_args()

    if args.backfill is not None:
        days = args.backfill
    else:
        existing_rows = get_existing_row_count()
        if existing_rows < AUTO_BACKFILL_ROW_THRESHOLD:
            days = AUTO_BACKFILL_DAYS
            print(
                f"Feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} has only "
                f"{existing_rows} rows (< {AUTO_BACKFILL_ROW_THRESHOLD}) — auto-backfilling "
                f"{AUTO_BACKFILL_DAYS} days before settling into normal hourly runs."
            )
        else:
            # Normal hourly runs only need a small overlapping window (covers
            # lag/rolling windows + guards against a missed run).
            days = 3

    df = build_features(days)
    push_to_hopsworks(df)
    send_hazard_alert_if_needed(df)


if __name__ == "__main__":
    main()
