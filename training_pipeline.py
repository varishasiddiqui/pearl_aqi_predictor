import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 1
CORR_THRESHOLD = 0.10
TIMESTEPS = 24  # for LSTM sequences


def load_features():
    import hopsworks
    from hopsworks_common.client.exceptions import FeatureStoreException

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    try:
        df = fg.read(read_options={"arrow_flight_config": {"timeout": 30}})
    except FeatureStoreException as e:
        # Hopsworks' Arrow Flight "Query Service" is a separate, sometimes
        # flaky component from the offline storage itself (this is the same
        # class of transient server-side issue we've hit before — the data
        # is fine, the read path is what's unavailable). Fall back to the
        # older Hive-based read path instead of failing the whole run.
        print(f"Query Service read failed ({e}); retrying via Hive fallback...")
        df = fg.read(read_options={"use_hive": True})

    df = df.sort_values("datetime").reset_index(drop=True)
    # Rows ingested in the last ~24h legitimately have no target yet
    # (target_aqi_24hr looks 24h into the future) — drop those for training.
    df = df.dropna(subset=["target_aqi_24hr"]).reset_index(drop=True)
    print(f"Loaded {len(df)} labeled rows from Hopsworks feature group.")
    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} labeled rows available — too few to train "
            "reliably. Run feature_pipeline.py --backfill <days> first."
        )
    return df, project


def select_features(df):
    candidate_features = [
        "pm2_5", "pm10", "so2", "co", "no2", "o3", "pressure", "wind_speed",
        "humidity", "temperature", "month", "hour", "day_of_week", "is_weekend",
        "aqi_lag_1", "aqi_lag_3", "aqi_lag_24", "pm25_lag_1", "pm25_lag_24",
        "aqi_rolling_3", "aqi_rolling_6", "aqi_rolling_24", "pm25_rolling_24",
    ]
    candidate_features = [c for c in candidate_features if c in df.columns]
    corr_matrix = df[candidate_features + ["target_aqi_24hr"]].corr()
    target_corr = corr_matrix["target_aqi_24hr"].drop("target_aqi_24hr").dropna()
    ranked = target_corr.abs().sort_values(ascending=False)

    feature_cols = ranked[ranked > CORR_THRESHOLD].index.tolist()
    print(f"Selected {len(feature_cols)} features (|correlation| > {CORR_THRESHOLD}): {feature_cols}")
    if not feature_cols:
        raise RuntimeError("No features passed the correlation threshold — check data quality.")
    return feature_cols


def cross_validate_model(build_model_fn, X_tr, y_tr, scale=False):
    tscv = TimeSeriesSplit(n_splits=5)
    rmses, maes, r2s = [], [], []
    for fold_train_idx, fold_val_idx in tscv.split(X_tr):
        X_fold_train, X_fold_val = X_tr.iloc[fold_train_idx], X_tr.iloc[fold_val_idx]
        y_fold_train, y_fold_val = y_tr.iloc[fold_train_idx], y_tr.iloc[fold_val_idx]

        if scale:
            fold_scaler = StandardScaler().fit(X_fold_train)
            X_fold_train = fold_scaler.transform(X_fold_train)
            X_fold_val = fold_scaler.transform(X_fold_val)

        model = build_model_fn()
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        rmses.append(np.sqrt(mean_squared_error(y_fold_val, preds)))
        maes.append(mean_absolute_error(y_fold_val, preds))
        r2s.append(r2_score(y_fold_val, preds))

    return {
        "rmse_mean": np.mean(rmses), "mae_mean": np.mean(maes), "r2_mean": np.mean(r2s),
    }


def build_lstm_sequences(X_arr, y_arr, timesteps=TIMESTEPS):
    Xs, ys = [], []
    for i in range(timesteps, len(X_arr)):
        Xs.append(X_arr[i - timesteps:i])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)


def train_and_evaluate(df, feature_cols):
    X = df[feature_cols].reset_index(drop=True)
    y = df["target_aqi_24hr"].reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Cross-validation (TimeSeriesSplit) ---")
    cv_ridge = cross_validate_model(lambda: Ridge(alpha=100.0), X_train, y_train, scale=True)
    cv_rf = cross_validate_model(
        lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        X_train, y_train, scale=False,
    )
    print("Ridge CV:", cv_ridge)
    print("RandomForest CV:", cv_rf)

    # LSTM — single holdout eval only (5x retrain per CV fold is expensive)
    import tensorflow as tf  # noqa: F401  (import guarded here, TF is heavy)
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    X_train_lstm, y_train_lstm = build_lstm_sequences(X_train_scaled, y_train.values)
    X_test_lstm, y_test_lstm = build_lstm_sequences(X_test_scaled, y_test.values)

    lstm_model = Sequential([
        LSTM(32, activation="relu", input_shape=(TIMESTEPS, X_train.shape[1])),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=16, verbose=0, validation_split=0.15)
    lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_preds))
    lstm_mae = mean_absolute_error(y_test_lstm, lstm_preds)
    lstm_r2 = r2_score(y_test_lstm, lstm_preds)
    print(f"LSTM (single holdout): RMSE={lstm_rmse:.2f} MAE={lstm_mae:.2f} R2={lstm_r2:.3f}")

    # Final holdout comparison, same test set, all models
    ridge_final = Ridge(alpha=100.0).fit(X_train_scaled, y_train)
    ridge_test_preds = ridge_final.predict(X_test_scaled)

    rf_final = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_train)
    rf_test_preds = rf_final.predict(X_test)

    results_table = pd.DataFrame({
        "Model": ["Ridge", "RandomForest", "LSTM"],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, ridge_test_preds)),
            np.sqrt(mean_squared_error(y_test, rf_test_preds)),
            lstm_rmse,
        ],
        "MAE": [
            mean_absolute_error(y_test, ridge_test_preds),
            mean_absolute_error(y_test, rf_test_preds),
            lstm_mae,
        ],
        "R2": [
            r2_score(y_test, ridge_test_preds),
            r2_score(y_test, rf_test_preds),
            lstm_r2,
        ],
    })
    print("\n--- Final holdout comparison ---")
    print(results_table.to_string(index=False))

    best_model_name = results_table.loc[results_table["R2"].idxmax(), "Model"]
    print(f"\nBest model on this holdout: {best_model_name}")

    return best_model_name, results_table, X, y


def refit_and_save(best_model_name, X, y, feature_cols):
    model_dir = "model_dir"
    os.makedirs(model_dir, exist_ok=True)

    final_scaler = StandardScaler().fit(X)
    X_all_scaled = final_scaler.transform(X)

    if best_model_name == "Ridge":
        deployment_model = Ridge(alpha=100.0).fit(X_all_scaled, y)
        model_file = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(deployment_model, model_file)
    elif best_model_name == "RandomForest":
        deployment_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X, y)
        model_file = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(deployment_model, model_file)
    else:  # LSTM
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        X_all_lstm, y_all_lstm = build_lstm_sequences(X_all_scaled, y.values)
        deployment_model = Sequential([
            LSTM(32, activation="relu", input_shape=(TIMESTEPS, X.shape[1])),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        deployment_model.compile(optimizer="adam", loss="mse")
        deployment_model.fit(X_all_lstm, y_all_lstm, epochs=30, batch_size=16, verbose=0)
        model_file = os.path.join(model_dir, "best_model.keras")
        deployment_model.save(model_file)

    joblib.dump(final_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))
    print(f"Saved deployment artifacts for {best_model_name} in {model_dir}/ -> "
          f"{os.path.basename(model_file)}, scaler.pkl, feature_cols.pkl")
    return model_dir


def register_model(project, model_dir, best_model_name, results_table, feature_cols):
    mr = project.get_model_registry()

    final_rmse = results_table.loc[results_table["Model"] == best_model_name, "RMSE"].values[0]
    final_mae = results_table.loc[results_table["Model"] == best_model_name, "MAE"].values[0]
    final_r2 = results_table.loc[results_table["Model"] == best_model_name, "R2"].values[0]

    aqi_model = mr.python.create_model(
        name="aqi_predictor_karachi",
        metrics={"rmse": float(final_rmse), "mae": float(final_mae), "r2": float(final_r2)},
        description=(
            f"{best_model_name} AQI predictor for Karachi, 24h-ahead forecast, "
            f"{len(feature_cols)} correlation-selected features"
        ),
    )
    # Register the WHOLE folder (model + scaler + feature_cols), not just the
    # model file on its own — app.py's load_model() downloads this folder and
    # expects all three files to be inside it.
    aqi_model.save(model_dir)
    print(f"Model registered in Hopsworks Model Registry (RMSE={final_rmse:.2f}, MAE={final_mae:.2f}, R2={final_r2:.3f}).")


def main():
    df, project = load_features()
    feature_cols = select_features(df)
    best_model_name, results_table, X, y = train_and_evaluate(df, feature_cols)

    if results_table["R2"].max() < 0:
        print(
            "WARNING: best model's R2 is still negative. This usually means "
            "there isn't enough real AQI variation in the training window yet "
            "-- consider a longer backfill (feature_pipeline.py --backfill 180) "
            "rather than tuning models further. Proceeding to save/register anyway."
        )

    model_dir = refit_and_save(best_model_name, X, y, feature_cols)
    register_model(project, model_dir, best_model_name, results_table, feature_cols)


if __name__ == "__main__":
    main()
