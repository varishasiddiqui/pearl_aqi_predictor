import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless — this script runs in CI, never a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 2
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
        print(f"Query Service read failed ({e}); retrying via Hive fallback...")
        df = fg.read(read_options={"use_hive": True})

    df = df.sort_values("datetime").reset_index(drop=True)

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
        "aqi_change_rate", "pm25_change_rate",
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


def tune_ridge_alpha(X_train_scaled, y_train):
    """RidgeCV auto-selects the regularization strength (alpha) using the
    same TimeSeriesSplit CV as the rest of this file, instead of the
    previous fixed guess of alpha=100.0. Too-high an alpha over-regularizes
    Ridge until its predictions collapse toward the target's mean regardless
    of input — that was the cause of the earlier flat, poorly-fit holdout
    predictions."""
    from sklearn.linear_model import RidgeCV

    alphas = np.logspace(-2, 3, 30)  # 0.01 .. 1000, log-spaced
    ridge_cv = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    ridge_cv.fit(X_train_scaled, y_train)
    print(f"RidgeCV selected alpha={ridge_cv.alpha_:.4g} (searched {len(alphas)} values between 0.01 and 1000)")
    return float(ridge_cv.alpha_)


def tune_random_forest(X_train, y_train):
    """Small grid search over RandomForest depth/estimator count, scored
    with the same TimeSeriesSplit CV used everywhere else in this file,
    instead of the previous fixed guess of n_estimators=200, max_depth=10."""
    candidates = [
        {"n_estimators": 200, "max_depth": 8},
        {"n_estimators": 200, "max_depth": 12},
        {"n_estimators": 300, "max_depth": 12},
        {"n_estimators": 300, "max_depth": None},
    ]
    best_params, best_rmse = None, np.inf
    for params in candidates:
        scores = cross_validate_model(
            lambda p=params: RandomForestRegressor(random_state=42, **p),
            X_train, y_train, scale=False,
        )
        print(f"RandomForest candidate {params} -> CV RMSE={scores['rmse_mean']:.3f}")
        if scores["rmse_mean"] < best_rmse:
            best_rmse = scores["rmse_mean"]
            best_params = params
    print(f"RandomForest selected params: {best_params}")
    return best_params


def train_and_evaluate(df, feature_cols):
    X = df[feature_cols].reset_index(drop=True)
    y = df["target_aqi_24hr"].reset_index(drop=True)
    dates = df["datetime"].reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Hyperparameter tuning (TimeSeriesSplit CV) ---")
    best_alpha = tune_ridge_alpha(X_train_scaled, y_train)
    rf_params = tune_random_forest(X_train, y_train)

    print("\n--- Cross-validation (TimeSeriesSplit) ---")
    cv_ridge = cross_validate_model(lambda: Ridge(alpha=best_alpha), X_train, y_train, scale=True)
    cv_rf = cross_validate_model(
        lambda: RandomForestRegressor(random_state=42, **rf_params),
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

    ridge_final = Ridge(alpha=best_alpha).fit(X_train_scaled, y_train)
    ridge_test_preds = ridge_final.predict(X_test_scaled)

    rf_final = RandomForestRegressor(random_state=42, **rf_params).fit(X_train, y_train)
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

    if best_model_name == "Ridge":
        best_actual, best_predicted, best_dates = y_test.values, ridge_test_preds, dates_test.values
    elif best_model_name == "RandomForest":
        best_actual, best_predicted, best_dates = y_test.values, rf_test_preds, dates_test.values
    else:  # LSTM
        best_actual, best_predicted = y_test_lstm, lstm_preds
        best_dates = dates_test.iloc[TIMESTEPS:].values

    return best_model_name, results_table, X, y, best_actual, best_predicted, best_dates, best_alpha, rf_params


def plot_actual_vs_predicted(y_actual, y_predicted, model_name, save_path):
    """Scatter of actual vs. predicted AQI (holdout set) for the winning
    model, with a y=x reference line. Points close to the diagonal mean the
    model's predictions are close to what actually happened."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_actual, y_predicted, alpha=0.5, s=18, color="#2563EB", edgecolors="none")

    lo = min(np.min(y_actual), np.min(y_predicted))
    hi = max(np.max(y_actual), np.max(y_predicted))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.2, label="Perfect prediction (y = x)")

    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"Actual vs. Predicted AQI — {model_name} (holdout set)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved actual-vs-predicted plot to {save_path}")


def plot_actual_vs_predicted_timeseries(dates, y_actual, y_predicted, model_name, save_path, days=30):
    """Actual vs. Predicted AQI over time, as two overlaid lines, restricted
    to the last `days` days of the holdout set (falls back to the full
    holdout if it's shorter than that). This is the time-series counterpart
    to plot_actual_vs_predicted's scatter — easier to read for anyone who
    wants to see WHEN the model over/under-predicted, not just by how much."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    y_actual = pd.Series(y_actual).reset_index(drop=True)
    y_predicted = pd.Series(y_predicted).reset_index(drop=True)

    cutoff = dates.max() - pd.Timedelta(days=days)
    mask = dates >= cutoff
    if mask.sum() < 2:  # not enough points in that window — show everything instead
        mask = pd.Series(True, index=dates.index)

    plot_dates = dates[mask]
    plot_actual = y_actual[mask]
    plot_predicted = y_predicted[mask]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(plot_dates, plot_actual, color="#1F2937", linewidth=1.4, label="Actual AQI")
    ax.plot(plot_dates, plot_predicted, color="#2563EB", linewidth=1.4, linestyle="--", label="Predicted AQI")

    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.set_title(f"Actual vs. Predicted AQI over time — {model_name} (last {days} days of holdout)")
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved actual-vs-predicted time series plot to {save_path}")


def refit_and_save(best_model_name, X, y, feature_cols, best_alpha, rf_params):
    model_dir = "model_dir"
    os.makedirs(model_dir, exist_ok=True)

    final_scaler = StandardScaler().fit(X)
    X_all_scaled = final_scaler.transform(X)

    if best_model_name == "Ridge":
        deployment_model = Ridge(alpha=best_alpha).fit(X_all_scaled, y)
        model_file = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(deployment_model, model_file)
    elif best_model_name == "RandomForest":
        deployment_model = RandomForestRegressor(random_state=42, **rf_params).fit(X, y)
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


    all_metrics = {"rmse": float(final_rmse), "mae": float(final_mae), "r2": float(final_r2)}
    for _, row in results_table.iterrows():
        prefix = row["Model"].lower()  # "ridge" / "randomforest" / "lstm"
        all_metrics[f"{prefix}_rmse"] = float(row["RMSE"])
        all_metrics[f"{prefix}_mae"] = float(row["MAE"])
        all_metrics[f"{prefix}_r2"] = float(row["R2"])

    aqi_model = mr.python.create_model(
        name="aqi_predictor_karachi",
        metrics=all_metrics,
        description=(
            f"Selected: {best_model_name} (best holdout R2) for Karachi's 24h-ahead AQI "
            f"forecast, {len(feature_cols)} correlation-selected features. Full "
            f"Ridge/RandomForest/LSTM holdout comparison (RMSE, MAE, R2) logged in "
            f"training_metrics for auditability."
        ),
    )

    aqi_model.save(model_dir)
    print(f"Model registered in Hopsworks Model Registry (RMSE={final_rmse:.2f}, MAE={final_mae:.2f}, R2={final_r2:.3f}).")
    print("Full candidate comparison logged to training_metrics:")
    print(results_table.to_string(index=False))


def main():
    df, project = load_features()
    feature_cols = select_features(df)
    (best_model_name, results_table, X, y, best_actual, best_predicted,
     best_dates, best_alpha, rf_params) = train_and_evaluate(df, feature_cols)

    if results_table["R2"].max() < 0:
        print(
            "WARNING: best model's R2 is still negative. This usually means "
            "there isn't enough real AQI variation in the training window yet "
            "-- consider a longer backfill (feature_pipeline.py --backfill 180) "
            "rather than tuning models further. Proceeding to save/register anyway."
        )

    model_dir = refit_and_save(best_model_name, X, y, feature_cols, best_alpha, rf_params)
    plot_actual_vs_predicted(
        best_actual, best_predicted, best_model_name,
        os.path.join(model_dir, "actual_vs_predicted.png"),
    )
    plot_actual_vs_predicted_timeseries(
        best_dates, best_actual, best_predicted, best_model_name,
        os.path.join(model_dir, "actual_vs_predicted_timeseries.png"),
    )
    register_model(project, model_dir, best_model_name, results_table, feature_cols)


if __name__ == "__main__":
    main()
