"""
Exploratory Data Analysis for the Karachi AQI feature store.

Run manually (never triggered by CI/CD, so it can't affect the live
feature/training pipelines or the dashboard):

    HOPSWORKS_API_KEY=xxx python eda.py

Outputs PNGs + printed summary stats into eda_output/.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURE_GROUP_NAME = "aqi_features_karachi"
FEATURE_GROUP_VERSION = 2  # matches feature_pipeline.py / training_pipeline.py
OUTPUT_DIR = "eda_output"


def load_data():
    import hopsworks

    hopsworks_key = os.environ["HOPSWORKS_API_KEY"]
    project = hopsworks.login(api_key_value=hopsworks_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def summary_stats(df):
    print("\n--- Shape & dtypes ---")
    print(df.shape)
    print("\n--- Summary statistics ---")
    print(df.describe(include="all").T)
    print("\n--- Missing values ---")
    print(df.isna().sum().sort_values(ascending=False).head(15))


def plot_aqi_timeseries(df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime"], df["aqi"], linewidth=0.8)
    ax.set_title("AQI over time — Karachi")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aqi_timeseries.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, out_dir):
    numeric_cols = df.select_dtypes("number").columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_pollutant_distributions(df, out_dir):
    pollutants = [c for c in ["pm2_5", "pm10", "so2", "co", "no2", "o3"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, p in zip(axes.flat, pollutants):
        sns.histplot(df[p].dropna(), kde=True, ax=ax)
        ax.set_title(p)
    fig.suptitle("Pollutant concentration distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pollutant_distributions.png"), dpi=150)
    plt.close(fig)


def plot_hourly_seasonal_pattern(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    df.groupby("hour")["aqi"].mean().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Average AQI by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Mean AQI")

    df.groupby("month")["aqi"].mean().plot(kind="bar", ax=axes[1], color="orange")
    axes[1].set_title("Average AQI by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean AQI")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hourly_seasonal_pattern.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    if df.empty:
        print("Feature group is empty — run feature_pipeline.py --backfill <days> first.")
        return

    summary_stats(df)
    plot_aqi_timeseries(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_pollutant_distributions(df, OUTPUT_DIR)
    plot_hourly_seasonal_pattern(df, OUTPUT_DIR)
    print(f"\nSaved plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
