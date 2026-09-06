# 🌍 Air Quality Index Forecasting using Machine Learning

Predicting the Air Quality Index (AQI) of **Karachi** for the next **3 days (72 hours)** using a 100% serverless MLOps pipeline.

---

## 🛠 Tech Stack

- **Data Sources**: OpenWeather API (pollutants), Open-Meteo API (weather)
- **Feature Store**: Hopsworks
- **Models**: Ridge Regression, Random Forest, LSTM (TensorFlow)
- **Model Registry**: Hopsworks Model Registry
- **Explainability**: SHAP
- **Dashboard**: Streamlit
- **API**: FastAPI
- **CI/CD**: GitHub Actions

---

## 🔄 Project Flow

```
OpenWeather API ─┐
                  ├──▶ Feature Pipeline ──▶ Hopsworks Feature Store ──▶ Training Pipeline ──▶ Model Registry ──▶ Web App / API
Open-Meteo API ───┘                              ▲
                                                  │
                                        GitHub Actions (CI/CD)
                                     (hourly + daily automation)
```

---

## 1️⃣ Feature Pipeline (`feature_pipeline.py`)

- **Step 1 →** Fetches raw pollutant data (PM2.5, PM10, NO2, SO2, O3, CO) from **OpenWeather API**, and weather data (temperature, humidity, wind speed, pressure) from **Open-Meteo API**
- **Step 2 →** Computes engineered features (model inputs) and the target (model output):
  - Time-based features: hour, day of week, month, weekend flag
  - AQI + dominant pollutant (calculated from raw pollutant concentrations)
  - Lag features: past AQI/PM2.5 values (1hr, 3hr, 24hr)
  - Derived features: `aqi_change_rate`, `pm25_change_rate`
  - Rolling averages: 3hr, 6hr, 24hr
  - Target: `target_aqi_24hr` (AQI 24 hours ahead)
- **Step 3 →** Stores all features in the **Hopsworks Feature Store** (`aqi_features_karachi`)

## 2️⃣ Backfill

- Populates the Feature Store with **historical data** so the model has enough data to train on
- Automatically triggers when the feature group has **< 200 rows** (treated as "cold start")
- Backfills **30 days** of historical pollutant + weather data
- After backfill, normal runs only pull a small 3-day overlapping window

## 3️⃣ Training Pipeline (`training_pipeline.py`)

- **Step 1 →** Fetches historical (features, targets) from the **Feature Store**
- **Step 2 →** Trains and evaluates three models:

| Model | What it does | Why chosen |
|---|---|---|
| **Ridge Regression** | Linear model with regularization to avoid overfitting | Simple, fast baseline — turned out to give the best result |
| **Random Forest** | Ensemble of decision trees averaging predictions | Robust against overfitting, captures non-linear patterns |
| **LSTM (TensorFlow)** | Deep learning model for sequences | Captures sequential/time-series dependencies in AQI trends |

  - Evaluated using **RMSE, MAE, and R²** (with `TimeSeriesSplit` cross-validation + final holdout comparison)
- **Step 3 →** Stores the trained model (+ scaler + feature list) in the **Hopsworks Model Registry**

## 4️⃣ Automated CI/CD (GitHub Actions)

- ⏱ **Feature pipeline script** → runs **every hour**
- ⏱ **Training pipeline script** → runs **every day**
- Tool used: **GitHub Actions** (free, no server management needed)

## 5️⃣ Web App (`app.py` / `api.py`)

- **Step 1 →** Loads the trained model and recent features from the **Feature Store / Model Registry**
- **Step 2 →** Computes 72-hour AQI predictions and displays them on the dashboard
- **Step 3 →** Built using:
  - **Streamlit** — interactive dashboard (live AQI, 3-day forecast, charts, alerts)
  - **FastAPI** — standalone API (`/health`, `/current`, `/forecast`)

## 6️⃣ Additional Components

- 📊 **EDA (`eda.py`)** — identifies trends: AQI time series, correlation heatmap, pollutant distributions, hourly/monthly seasonal patterns
- 🤖 Uses a **variety of models**, from statistical (Ridge) to ensemble (Random Forest) to deep learning (LSTM)
- 🔍 **SHAP** used for feature importance and explainability
- 🚨 **Hazard alerts** — Good or bad aqi range

---

## ⚠️ Limitations

- Hopsworks Query Service **timeouts** sometimes fail training reads
- No **auto-recovery** for training pipeline read failures
- Occasional **connection drops** during feature ingestion
- Model limited to **Karachi only** (no other cities)

---

## 📂 Project Structure

```
pearl_aqi_predictor/
├── feature_pipeline.py        # Hourly data ingestion + feature engineering
├── training_pipeline.py       # Daily model training + registration
├── api.py                     # FastAPI serving layer
├── app.py                     # Streamlit dashboard
├── eda.py                     # Exploratory data analysis (manual run)
├── requirements*.txt          # Dependencies
├── requirements-api*.txt      # Dependencies
├── requirements-pipeline*.txt # Dependencies
└── .github/workflows/         # CI/CD automation (GitHub Actions)
```

---

## 🚀 Deployment

- Streamlit Cloud
