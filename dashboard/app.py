from pathlib import Path
from datetime import timedelta
import json
import sqlite3

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tinydb import TinyDB

# Professional paths for database and model
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DB_PATH = DATA_DIR / "traffic_data.db"
FINAL_CSV = DATA_DIR / "final_ai_data.csv"
ACCIDENTS_JSON = DATA_DIR / "accidents_nosql.json"
MODEL_PATH = MODEL_DIR / "traffic_lstm.pth"
METRICS_PATH = MODEL_DIR / "metrics.json"
LSTM_PREDICTIONS_PATH = MODEL_DIR / "lstm_predictions.csv"
ARIMA_PREDICTIONS_PATH = MODEL_DIR / "arima_predictions.csv"

st.set_page_config(
    page_title="Urban Traffic Congestion Forecasting",
    layout="wide"
)

# Helpers

def classify_congestion(speed, normal_speed):
    if normal_speed <= 0 or pd.isna(speed) or pd.isna(normal_speed):
        return "Unknown"

    ratio = speed / normal_speed

    if ratio >= 0.80:
        return "Low"
    elif ratio >= 0.60:
        return "Moderate"
    else:
        return "High"


def congestion_description(level):
    if level == "Low":
        return "Traffic is moving close to normal speed."
    elif level == "Moderate":
        return "Traffic is slower than normal."
    elif level == "High":
        return "Traffic is heavily slowed down."
    return "Not enough data."


def weather_code_to_text(code):
    try:
        code = int(code)
    except Exception:
        return "Unknown"

    weather_map = {
        0: "Clear",
        1: "Mostly clear",
        2: "Partly cloudy",
        3: "Cloudy",
        45: "Fog",
        48: "Fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        95: "Thunderstorm"
    }

    return weather_map.get(code, f"Code {code}")


def load_traffic():
    """Load raw traffic data from the local SQLite database."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT captured_at, current_speed, free_flow_speed
            FROM tomtom_traffic
            ORDER BY datetime(captured_at)
            """,
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Could not read traffic database: {e}")
        return pd.DataFrame()


def load_final_data():
    if not FINAL_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(FINAL_CSV)

    if "captured_at" in df.columns:
        df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")

    numeric_cols = [
        "current_speed",
        "free_flow_speed",
        "historical_crash_count",
        "hour",
        "day_of_week",
        "temperature",
        "weather_code"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "captured_at" in df.columns:
        df = df.dropna(subset=["captured_at"]).sort_values("captured_at")

    return df


def load_accidents():
    if not ACCIDENTS_JSON.exists():
        return pd.DataFrame()

    try:
        return pd.DataFrame(TinyDB(ACCIDENTS_JSON).all())
    except Exception:
        return pd.DataFrame()


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_predictions():
    lstm_df = pd.DataFrame()
    arima_df = pd.DataFrame()

    if LSTM_PREDICTIONS_PATH.exists():
        lstm_df = pd.read_csv(LSTM_PREDICTIONS_PATH)
        if "captured_at" in lstm_df.columns:
            lstm_df["captured_at"] = pd.to_datetime(lstm_df["captured_at"], errors="coerce")

    if ARIMA_PREDICTIONS_PATH.exists():
        arima_df = pd.read_csv(ARIMA_PREDICTIONS_PATH)
        if "captured_at" in arima_df.columns:
            arima_df["captured_at"] = pd.to_datetime(arima_df["captured_at"], errors="coerce")

    return lstm_df, arima_df


def red_line_chart(df, x_col, value_cols, x_title, y_title):
    if df.empty:
        st.warning("No data available for this chart.")
        return

    usable_cols = [col for col in value_cols if col in df.columns]

    if not usable_cols or x_col not in df.columns:
        st.warning("Chart columns were not found.")
        return

    chart_df = df[[x_col] + usable_cols].dropna().copy()

    if chart_df.empty:
        st.warning("Not enough data to draw this chart.")
        return

    chart_long = chart_df.melt(
        id_vars=x_col,
        value_vars=usable_cols,
        var_name="Type",
        value_name="Value"
    )

    if pd.api.types.is_datetime64_any_dtype(chart_df[x_col]):
        x_encoding = alt.X(f"{x_col}:T", title=x_title)
    else:
        x_encoding = alt.X(f"{x_col}:Q", title=x_title)

    chart = (
        alt.Chart(chart_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=x_encoding,
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    range=["#991b1b", "#fca5a5", "#dc2626", "#fecaca"]
                ),
                legend=alt.Legend(title=None)
            ),
            tooltip=[x_col, "Type", "Value"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)


class TrafficLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_lstm_model():
    model = TrafficLSTM()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def make_lstm_short_forecast(df):
    required_features = [
        "historical_crash_count",
        "hour",
        "day_of_week",
        "temperature",
        "weather_code"
    ]

    required_cols = required_features + ["current_speed", "free_flow_speed", "captured_at"]

    if df.empty or not MODEL_PATH.exists():
        return pd.DataFrame(), "missing"

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return pd.DataFrame({"Missing Columns": missing_cols}), "missing_columns"

    model_df = df.dropna(subset=required_cols).copy()

    if len(model_df) < 5:
        return pd.DataFrame(), "not_enough_data"

    try:
        X = model_df[required_features].values.astype(np.float32)
        y = model_df["current_speed"].values.astype(np.float32).reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaler_X.fit(X)
        scaler_y.fit(y)

        model = load_lstm_model()

        latest_row = model_df.iloc[-1]
        latest_time = pd.to_datetime(latest_row["captured_at"])
        normal_speed = float(latest_row["free_flow_speed"])

        future_rows = []

        for minutes_ahead in [5, 10, 15, 20, 25, 30]:
            future_time = latest_time + timedelta(minutes=minutes_ahead)

            future_input = pd.DataFrame([{
                "historical_crash_count": latest_row["historical_crash_count"],
                "hour": future_time.hour,
                "day_of_week": future_time.dayofweek,
                "temperature": latest_row["temperature"],
                "weather_code": latest_row["weather_code"]
            }])

            X_future_scaled = scaler_X.transform(
                future_input[required_features].values.astype(np.float32)
            )

            X_future_tensor = torch.FloatTensor(X_future_scaled).reshape(1, 1, len(required_features))

            with torch.no_grad():
                pred_scaled = model(X_future_tensor).numpy()

            predicted_speed = scaler_y.inverse_transform(pred_scaled)[0][0]
            predicted_speed = max(0, min(float(predicted_speed), normal_speed))

            level = classify_congestion(predicted_speed, normal_speed)

            future_rows.append({
                "Forecast Time": future_time.strftime("%I:%M %p"),
                "Minutes Ahead": minutes_ahead,
                "Predicted Speed (mph)": round(predicted_speed, 2),
                "Normal Speed (mph)": round(normal_speed, 2),
                "Congestion Level": level,
                "Meaning": congestion_description(level)
            })

        return pd.DataFrame(future_rows), "ok"

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), "error"


# Load files

traffic_df = load_traffic()
df = load_final_data()
accidents_df = load_accidents()
metrics = load_metrics()
lstm_predictions, arima_predictions = load_predictions()
forecast_df, forecast_status = make_lstm_short_forecast(df)


# Page title

st.title("Urban Traffic Congestion Predictive Forecasting")

st.write(
    "This dashboard uses traffic speed, crash history, weather data, and machine learning "
    "to estimate short-term congestion on FDR Drive."
)

if df.empty:
    st.error("No processed traffic dataset found.")
    st.write("Run these commands first:")
    st.code(
        "python scripts/feature_engineering.py\npython models/evaluate_models.py",
        language="bash"
    )
    st.stop()


# ======================================================
# Main values
# ======================================================

latest = df.iloc[-1]

current_speed = float(latest["current_speed"])
normal_speed = float(latest["free_flow_speed"])
current_level = classify_congestion(current_speed, normal_speed)

if normal_speed > 0:
    slowdown_percent = max(0, (1 - current_speed / normal_speed) * 100)
else:
    slowdown_percent = 0


overview_tab, forecast_tab, model_tab, crash_tab, pipeline_tab = st.tabs(
    ["Overview", "Future Forecast", "Model Results", "Crash History", "Project Pipeline"]
)


# Overview

with overview_tab:
    st.subheader("Current Traffic Conditions")

    st.write(
        "This section shows how traffic is moving right now compared to normal road speed."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Speed", f"{current_speed:.1f} mph")
    col2.metric("Normal Speed", f"{normal_speed:.1f} mph")
    col3.metric("Slowdown", f"{slowdown_percent:.0f}%")
    col4.metric("Congestion Level", current_level)

    st.info(congestion_description(current_level))

    if "captured_at" in df.columns:
        st.write(f"Last updated: **{latest['captured_at']}**")

    st.write(f"Traffic records in dataset: **{len(df)}**")

    st.subheader("Traffic Speed Over Time")

    st.write(
        "This chart compares current traffic speed with normal road speed. "
        "A larger gap means more congestion."
    )

    chart_df = df.rename(
        columns={
            "captured_at": "Time",
            "current_speed": "Current Speed",
            "free_flow_speed": "Normal Speed"
        }
    )

    red_line_chart(
        chart_df,
        x_col="Time",
        value_cols=["Current Speed", "Normal Speed"],
        x_title="Time",
        y_title="Speed (mph)"
    )

    st.subheader("Data Used for Prediction")

    d1, d2, d3, d4 = st.columns(4)

    crash_count = latest.get("historical_crash_count", 0)
    temperature = latest.get("temperature", None)
    weather_code = latest.get("weather_code", None)
    hour = latest.get("hour", None)

    d1.metric("Past Crashes at Similar Times", int(crash_count) if pd.notna(crash_count) else "N/A")
    d2.metric("Temperature", f"{float(temperature):.1f} °C" if pd.notna(temperature) else "N/A")
    d3.metric("Weather", weather_code_to_text(weather_code))
    d4.metric("Time of Day", f"{int(hour)}:00" if pd.notna(hour) else "N/A")

    st.write(
        "The model uses traffic speed, crash history, time of day, day of week, "
        "weather, and temperature."
    )

    with st.expander("Show merged prediction data"):
        rename_map = {
            "captured_at": "Time Collected",
            "current_speed": "Current Speed (mph)",
            "free_flow_speed": "Normal Speed (mph)",
            "historical_crash_count": "Past Crashes at Similar Time",
            "temperature": "Temperature (°C)",
            "weather_code": "Weather Code",
            "hour": "Hour of Day",
            "day_of_week": "Day of Week"
        }

        keep_cols = [col for col in rename_map if col in df.columns]
        friendly_df = df[keep_cols].rename(columns=rename_map)

        st.dataframe(friendly_df.tail(100), use_container_width=True, hide_index=True)


# Future Forecast

with forecast_tab:
    st.subheader("Future Congestion Forecast")

    if forecast_status == "ok" and not forecast_df.empty:
        next_row = forecast_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Speed in 5 Minutes",
            f"{next_row['Predicted Speed (mph)']} mph"
        )

        col2.metric(
            "Expected Congestion",
            next_row["Congestion Level"]
        )

        col3.metric(
            "Forecast Window",
            "30 minutes"
        )

        st.write(
            "The forecast estimates traffic speed for the next 30 minutes. "
            "Lower speed means higher congestion."
        )

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        red_line_chart(
            forecast_df,
            x_col="Minutes Ahead",
            value_cols=["Predicted Speed (mph)", "Normal Speed (mph)"],
            x_title="Minutes Ahead",
            y_title="Speed (mph)"
        )

    elif forecast_status == "missing":
        st.warning("The forecast cannot be shown yet because the model or dataset is missing.")
        st.code("python models/evaluate_models.py", language="bash")

    elif forecast_status == "not_enough_data":
        st.warning("More traffic rows are needed before the forecast is useful.")

    elif forecast_status == "missing_columns":
        st.warning("The forecast cannot be shown because required columns are missing.")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    else:
        st.error("The forecast could not be created.")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# Model Results

with model_tab:
    st.subheader("Model Results")

    st.write(
        "This section compares the LSTM model with the ARIMA baseline using the saved test-set predictions."
    )

    if metrics:
        lstm_metrics = metrics.get("lstm", {})
        arima_metrics = metrics.get("arima", {})
        best_model = metrics.get("best_model_by_rmse", "N/A")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("LSTM MAE", lstm_metrics.get("mae", "N/A"))
        col2.metric("LSTM RMSE", lstm_metrics.get("rmse", "N/A"))
        col3.metric("ARIMA RMSE", arima_metrics.get("rmse", "N/A"))
        col4.metric("Best Model", str(best_model).upper())

        st.write(
            "MAE and RMSE measure prediction error. Lower values mean the model predicted traffic speed more accurately."
        )

        with st.expander("Show full metrics"):
            st.json(metrics)

    else:
        st.warning("Model comparison metrics were not found.")
        st.write("Run this command to generate them:")
        st.code("python models/evaluate_models.py", language="bash")

    st.subheader("Actual Speed vs Model Predictions")

    if not lstm_predictions.empty:
        lstm_plot = lstm_predictions.rename(
            columns={
                "captured_at": "Time",
                "current_speed": "Actual Speed",
                "lstm_predicted_speed": "LSTM Prediction"
            }
        )

        red_line_chart(
            lstm_plot,
            x_col="Time",
            value_cols=["Actual Speed", "LSTM Prediction"],
            x_title="Time",
            y_title="Speed (mph)"
        )

        with st.expander("Show LSTM prediction data"):
            display_lstm = lstm_predictions.rename(
                columns={
                    "captured_at": "Time",
                    "current_speed": "Actual Speed",
                    "free_flow_speed": "Normal Speed",
                    "lstm_predicted_speed": "LSTM Prediction",
                    "absolute_error": "Prediction Error",
                    "rolling_speed_avg_3": "Rolling Speed Average",
                    "congestion_likelihood": "Congestion Likelihood"
                }
            )
            st.dataframe(display_lstm, use_container_width=True, hide_index=True)

    else:
        st.info("LSTM prediction file was not found yet.")

    st.subheader("LSTM vs ARIMA Baseline")

    if not lstm_predictions.empty and not arima_predictions.empty:
        comparison_df = pd.merge(
            lstm_predictions,
            arima_predictions,
            on=["captured_at", "current_speed"],
            how="inner"
        )

        comparison_df = comparison_df.rename(
            columns={
                "captured_at": "Time",
                "current_speed": "Actual Speed",
                "lstm_predicted_speed": "LSTM Prediction",
                "arima_predicted_speed": "ARIMA Prediction"
            }
        )

        if not comparison_df.empty:
            red_line_chart(
                comparison_df,
                x_col="Time",
                value_cols=["Actual Speed", "LSTM Prediction", "ARIMA Prediction"],
                x_title="Time",
                y_title="Speed (mph)"
            )

            with st.expander("Show LSTM and ARIMA comparison data"):
                keep_cols = [
                    "Time",
                    "Actual Speed",
                    "LSTM Prediction",
                    "ARIMA Prediction"
                ]
                st.dataframe(
                    comparison_df[keep_cols],
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info("LSTM and ARIMA predictions exist, but their timestamps did not match.")

    else:
        st.info("Run `python models/evaluate_models.py` to generate LSTM and ARIMA prediction files.")


# Crash History

with crash_tab:
    st.subheader("Historical Crash Information")

    st.write(
        "This section shows NYC crash records related to FDR Drive. "
        "Crash history can help explain congestion patterns."
    )

    if accidents_df.empty:
        st.warning("No crash data found.")
        st.code("python scripts/accident_ingestion.py", language="bash")

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Crash Records Used", len(accidents_df))

        if "hour" in accidents_df.columns:
            busiest_hour = pd.to_numeric(accidents_df["hour"], errors="coerce").dropna().mode()
            if not busiest_hour.empty:
                col2.metric("Most Common Crash Time", f"{int(busiest_hour.iloc[0])}:00")
            else:
                col2.metric("Most Common Crash Time", "N/A")
        else:
            col2.metric("Most Common Crash Time", "N/A")

        if "number_of_persons_injured" in accidents_df.columns:
            injuries = pd.to_numeric(accidents_df["number_of_persons_injured"], errors="coerce").fillna(0).sum()
            col3.metric("Total Injuries in Records", int(injuries))
        else:
            col3.metric("Total Injuries in Records", "N/A")

        if {"latitude", "longitude"}.issubset(accidents_df.columns):
            accidents_df["latitude"] = pd.to_numeric(accidents_df["latitude"], errors="coerce")
            accidents_df["longitude"] = pd.to_numeric(accidents_df["longitude"], errors="coerce")

            map_df = accidents_df.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]]
            map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})

            if not map_df.empty:
                st.subheader("Crash Locations")
                st.map(map_df)

        with st.expander("Show crash records"):
            rename_accidents = {
                "crash_date": "Crash Date",
                "crash_time": "Crash Time",
                "on_street_name": "Street",
                "cross_street_name": "Cross Street",
                "number_of_persons_injured": "People Injured",
                "number_of_persons_killed": "People Killed",
                "contributing_factor_vehicle_1": "Possible Cause",
                "hour": "Hour of Day",
                "day_of_week": "Day of Week"
            }

            keep_cols = [col for col in rename_accidents if col in accidents_df.columns]
            friendly_accidents = accidents_df[keep_cols].rename(columns=rename_accidents)

            st.dataframe(friendly_accidents.head(100), use_container_width=True, hide_index=True)


# Project Pipeline

with pipeline_tab:
    st.subheader("Project Purpose")

    st.write(
        "The purpose of this project is to estimate short-term traffic congestion on FDR Drive. "
        "The system combines traffic speed, crash history, weather, and time-based patterns."
    )

    st.subheader("How the System Works")

    st.markdown(
        """
        1. **Collect traffic data** from TomTom and seeded traffic records.
        2. **Store traffic readings** in SQLite.
        3. **Add crash history** from NYC collision records.
        4. **Add weather data** from Open-Meteo.
        5. **Train and compare models** using LSTM and ARIMA.
        6. **Display the results** in this dashboard.
        """
    )

    st.subheader("Data Sources")

    st.markdown(
        """
        - **TomTom Traffic API:** current speed and normal road speed.
        - **NYC collision records:** crash history for FDR Drive.
        - **Open-Meteo:** weather information.
        - **SQLite:** stores structured traffic readings.
        - **TinyDB:** stores crash records in a NoSQL-style format.
        """
    )

    st.subheader("What the Dashboard Shows")

    st.markdown(
        """
        - Current traffic speed
        - Normal road speed
        - Current congestion level
        - Future congestion estimate
        - LSTM model results
        - ARIMA baseline comparison
        - Crash history and crash locations
        """
    )
