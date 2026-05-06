from pathlib import Path
from datetime import timedelta
import sqlite3

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tinydb import TinyDB

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DB_PATH = DATA_DIR / "traffic_data.db"
FINAL_CSV = DATA_DIR / "final_ai_data.csv"
ACCIDENTS_JSON = DATA_DIR / "accidents_nosql.json"
MODEL_PATH = ROOT / "models" / "traffic_lstm.pth"

st.set_page_config(
    page_title="Urban Traffic Congestion Forecasting",
    layout="wide"
)


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


def load_merged_data():
    if not FINAL_CSV.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(FINAL_CSV)

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

        return df

    except Exception as e:
        st.warning(f"Could not load merged dataset: {e}")
        return pd.DataFrame()


def load_accidents():
    if not ACCIDENTS_JSON.exists():
        return pd.DataFrame()

    try:
        return pd.DataFrame(TinyDB(ACCIDENTS_JSON).all())
    except Exception:
        return pd.DataFrame()


class TrafficLSTM(nn.Module):
    def __init__(self):
        super(TrafficLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def load_model():
    model = TrafficLSTM()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def create_forecast(merged_df):
    features = [
        "historical_crash_count",
        "hour",
        "day_of_week",
        "temperature",
        "weather_code"
    ]

    if merged_df.empty or not MODEL_PATH.exists():
        return pd.DataFrame(), "missing"

    model_df = merged_df.dropna(
        subset=features + ["current_speed", "free_flow_speed"]
    ).copy()

    if len(model_df) < 5:
        return pd.DataFrame(), "not_enough_data"

    try:
        X = model_df[features].values.astype(np.float32)
        y = model_df["current_speed"].values.astype(np.float32).reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaler_X.fit(X)
        scaler_y.fit(y)

        model = load_model()

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
                future_input[features].values.astype(np.float32)
            )

            X_future_tensor = torch.FloatTensor(X_future_scaled).unsqueeze(1)

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


def red_line_chart(df, time_col, value_cols, title_x, title_y):
    chart_df = df[[time_col] + value_cols].copy()

    chart_long = chart_df.melt(
        id_vars=time_col,
        value_vars=value_cols,
        var_name="Type",
        value_name="Value"
    )

    color_range = ["#991b1b", "#fca5a5", "#ef4444", "#fecaca"]

    chart = (
        alt.Chart(chart_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X(f"{time_col}:T" if "Time" in time_col else f"{time_col}:Q", title=title_x),
            y=alt.Y("Value:Q", title=title_y),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(range=color_range),
                legend=alt.Legend(title=None)
            ),
            tooltip=[time_col, "Type", "Value"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)


traffic_df = load_traffic()
merged_df = load_merged_data()
accidents_df = load_accidents()

st.title("Urban Traffic Congestion Predictive Forecasting")

st.write(
    "This dashboard uses live TomTom traffic data, NYC crash history, weather data, "
    "and an LSTM model to estimate short-term traffic congestion."
)

if traffic_df.empty:
    st.error("No traffic data found.")
    st.write("Run the pipeline first:")
    st.code("python3 run_all.py", language="bash")
    st.stop()

traffic_df["captured_at"] = pd.to_datetime(traffic_df["captured_at"], errors="coerce")
traffic_df["current_speed"] = pd.to_numeric(traffic_df["current_speed"], errors="coerce")
traffic_df["free_flow_speed"] = pd.to_numeric(traffic_df["free_flow_speed"], errors="coerce")
traffic_df = traffic_df.dropna().sort_values("captured_at")

latest = traffic_df.iloc[-1]

current_speed = float(latest["current_speed"])
normal_speed = float(latest["free_flow_speed"])
current_level = classify_congestion(current_speed, normal_speed)

if normal_speed > 0:
    slowdown_percent = max(0, (1 - current_speed / normal_speed) * 100)
else:
    slowdown_percent = 0

forecast_df, forecast_status = create_forecast(merged_df)

overview_tab, forecast_tab, crash_tab, pipeline_tab = st.tabs(
    ["Overview", "Future Forecast", "Crash History", "Project Pipeline"]
)

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

    st.write(f"Last updated: **{latest['captured_at']}**")
    st.write(f"Traffic readings collected: **{len(traffic_df)}**")

    st.subheader("Traffic Speed Over Time")

    st.write(
        "This chart compares current traffic speed with normal road speed. "
        "A larger gap means more congestion."
    )

    chart_df = traffic_df.rename(
        columns={
            "captured_at": "Time",
            "current_speed": "Current Speed",
            "free_flow_speed": "Normal Speed"
        }
    )

    red_line_chart(
        chart_df,
        time_col="Time",
        value_cols=["Current Speed", "Normal Speed"],
        title_x="Time",
        title_y="Speed (mph)"
    )

    with st.expander("Show recent traffic readings"):
        display_traffic = traffic_df.tail(50).rename(
            columns={
                "captured_at": "Time Collected",
                "current_speed": "Current Speed (mph)",
                "free_flow_speed": "Normal Speed (mph)"
            }
        )

        st.dataframe(display_traffic, use_container_width=True, hide_index=True)

    st.subheader("Data Used for Prediction")

    if merged_df.empty:
        st.warning("Merged prediction dataset not found.")
        st.code("python3 scripts/step2_merge.py", language="bash")
    else:
        model_df = merged_df.dropna()

        if model_df.empty:
            st.warning("Merged dataset exists, but it does not have enough usable rows.")
        else:
            latest_data = model_df.iloc[-1]

            d1, d2, d3, d4 = st.columns(4)

            crash_count = latest_data.get("historical_crash_count", 0)
            temperature = latest_data.get("temperature", None)
            weather_code = latest_data.get("weather_code", None)
            hour = latest_data.get("hour", None)

            d1.metric("Past Crashes at Similar Times", int(crash_count))
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

                keep_cols = [col for col in rename_map if col in merged_df.columns]

                friendly_df = merged_df[keep_cols].rename(columns=rename_map)

                st.dataframe(friendly_df.tail(100), use_container_width=True, hide_index=True)

with forecast_tab:
    st.subheader("Future Congestion Forecast")

    if not MODEL_PATH.exists():
        st.warning("Forecast model is not ready yet.")
        st.code("python3 models/step3_train.py", language="bash")

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

        forecast_chart_df = forecast_df[[
            "Minutes Ahead",
            "Predicted Speed (mph)",
            "Normal Speed (mph)"
        ]].copy()

        forecast_long = forecast_chart_df.melt(
            id_vars="Minutes Ahead",
            value_vars=["Predicted Speed (mph)", "Normal Speed (mph)"],
            var_name="Speed Type",
            value_name="Speed"
        )

        forecast_chart = (
            alt.Chart(forecast_long)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("Minutes Ahead:Q", title="Minutes Ahead"),
                y=alt.Y("Speed:Q", title="Speed (mph)"),
                color=alt.Color(
                    "Speed Type:N",
                    scale=alt.Scale(
                        domain=["Predicted Speed (mph)", "Normal Speed (mph)"],
                        range=["#991b1b", "#fca5a5"]
                    ),
                    legend=alt.Legend(title=None)
                ),
                tooltip=["Minutes Ahead", "Speed Type", "Speed"]
            )
            .properties(height=350)
        )

        st.altair_chart(forecast_chart, use_container_width=True)

    elif forecast_status == "not_enough_data":
        st.warning("More traffic readings are needed before the forecast is useful.")
        st.write("Let the TomTom collector run longer, then rerun the merge and training steps.")

    elif forecast_status == "missing":
        st.warning("The forecast cannot be shown yet because data or model files are missing.")
        st.code(
            "python3 scripts/step2_merge.py\npython3 models/step3_train.py",
            language="bash"
        )

    else:
        st.error("The forecast could not be created.")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

with crash_tab:
    st.subheader("Historical Crash Information")

    st.write(
        "This section shows NYC crash records related to FDR Drive. "
        "Crash history can help explain congestion patterns."
    )

    if accidents_df.empty:
        st.warning("No crash data found.")
        st.code("python3 scripts/step1_mongo.py", language="bash")
    else:
        col1, col2 = st.columns(2)

        col1.metric("Crash Records Used", len(accidents_df))

        if "hour" in accidents_df.columns:
            busiest_hour = accidents_df["hour"].mode()
            if not busiest_hour.empty:
                col2.metric("Most Common Crash Time", f"{int(busiest_hour.iloc[0])}:00")
            else:
                col2.metric("Most Common Crash Time", "N/A")
        else:
            col2.metric("Most Common Crash Time", "N/A")

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

with pipeline_tab:
    st.subheader("Project Purpose")

    st.write(
        "The purpose of this project is to estimate short-term traffic congestion on FDR Drive. "
        "The system combines live traffic speed, crash history, weather, and time-based patterns."
    )

    st.subheader("How the System Works")

    st.markdown(
        """
        1. **Collect live traffic data** from the TomTom API.
        2. **Store traffic readings** in a SQLite database.
        3. **Add crash history** from NYC collision records.
        4. **Add weather data** because weather can affect traffic.
        5. **Train a forecasting model** to estimate future congestion.
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
        - Crash history
        - Model input data
        """
    )
