from pathlib import Path
from datetime import timedelta
import json
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st
from tinydb import TinyDB

try:
    import pydeck as pdk
except Exception:
    pdk = None


ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# File/path helpers
# -----------------------------

def is_ignored_path(path: Path) -> bool:
    ignored = {".git", ".venv", "venv", "__pycache__"}
    return any(part in ignored for part in path.parts)


def find_file(candidates, suffix=None, keywords=None):
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path

    if suffix is None:
        return None

    matches = []
    for path in ROOT.rglob(f"*{suffix}"):
        if is_ignored_path(path):
            continue

        text = str(path).lower()
        if keywords:
            if any(word in text for word in keywords):
                matches.append(path)
        else:
            matches.append(path)

    return matches[0] if matches else None


DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

FINAL_CSV = find_file(
    ["data/final_ai_data.csv", "final_ai_data.csv"],
    suffix=".csv",
    keywords=["final_ai_data", "final", "merged"]
)

ACCIDENTS_JSON = find_file(
    ["data/accidents_nosql.json", "accidents_nosql.json"],
    suffix=".json",
    keywords=["accident", "crash", "collision", "nosql"]
)

TRAFFIC_DB = find_file(
    ["data/traffic_data.db", "traffic_data.db"],
    suffix=".db",
    keywords=["traffic"]
)

METRICS_PATH = find_file(
    ["models/metrics.json", "metrics.json"],
    suffix=".json",
    keywords=["metrics"]
)

LSTM_PREDICTIONS_PATH = find_file(
    ["models/lstm_predictions.csv", "lstm_predictions.csv"],
    suffix=".csv",
    keywords=["lstm_predictions"]
)

ARIMA_PREDICTIONS_PATH = find_file(
    ["models/arima_predictions.csv", "arima_predictions.csv"],
    suffix=".csv",
    keywords=["arima_predictions"]
)


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="FDR Drive Traffic Forecast",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            max-width: 1180px;
        }

        h1 {
            margin-top: 0rem;
            padding-top: 0rem;
            margin-bottom: 0.2rem;
        }

        .muted {
            color: #4b5563;
            font-size: 0.95rem;
            line-height: 1.45;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Loaders
# -----------------------------

def load_final_data():
    if FINAL_CSV is None or not FINAL_CSV.exists():
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
        "weather_code",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = ["captured_at", "current_speed", "free_flow_speed"]
    existing_needed = [col for col in needed if col in df.columns]

    if existing_needed:
        df = df.dropna(subset=existing_needed)

    if "captured_at" in df.columns:
        df = df.sort_values("captured_at")

    return df


def load_traffic_from_db():
    if TRAFFIC_DB is None or not TRAFFIC_DB.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(TRAFFIC_DB)
        df = pd.read_sql_query(
            """
            SELECT captured_at, current_speed, free_flow_speed
            FROM tomtom_traffic
            ORDER BY datetime(captured_at)
            """,
            conn
        )
        conn.close()

        df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")
        df["current_speed"] = pd.to_numeric(df["current_speed"], errors="coerce")
        df["free_flow_speed"] = pd.to_numeric(df["free_flow_speed"], errors="coerce")
        df = df.dropna(subset=["captured_at", "current_speed", "free_flow_speed"])
        return df.sort_values("captured_at")

    except Exception:
        return pd.DataFrame()


def load_accidents():
    if ACCIDENTS_JSON is None or not ACCIDENTS_JSON.exists():
        return pd.DataFrame()

    try:
        records = TinyDB(ACCIDENTS_JSON).all()
        if records:
            return pd.DataFrame(records)
    except Exception:
        pass

    try:
        with open(ACCIDENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "_default" in data:
                return pd.DataFrame(list(data["_default"].values()))
            return pd.DataFrame(list(data.values()))

        return pd.DataFrame(data)

    except Exception:
        return pd.DataFrame()


def load_json(path):
    if path is None or not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_predictions():
    lstm_df = pd.DataFrame()
    arima_df = pd.DataFrame()

    if LSTM_PREDICTIONS_PATH is not None and LSTM_PREDICTIONS_PATH.exists():
        lstm_df = pd.read_csv(LSTM_PREDICTIONS_PATH)
        if "captured_at" in lstm_df.columns:
            lstm_df["captured_at"] = pd.to_datetime(lstm_df["captured_at"], errors="coerce")

    if ARIMA_PREDICTIONS_PATH is not None and ARIMA_PREDICTIONS_PATH.exists():
        arima_df = pd.read_csv(ARIMA_PREDICTIONS_PATH)
        if "captured_at" in arima_df.columns:
            arima_df["captured_at"] = pd.to_datetime(arima_df["captured_at"], errors="coerce")

    return lstm_df, arima_df


# -----------------------------
# Helpers
# -----------------------------

def classify_congestion(speed, normal_speed):
    if pd.isna(speed) or pd.isna(normal_speed) or normal_speed <= 0:
        return "Unknown"

    ratio = speed / normal_speed

    if ratio >= 0.80:
        return "Low"
    elif ratio >= 0.60:
        return "Moderate"
    return "High"


def congestion_text(level):
    if level == "Low":
        return "Traffic is close to normal."
    if level == "Moderate":
        return "Traffic is slower than normal."
    if level == "High":
        return "Traffic is heavily slowed down."
    return "Not enough data."


def weather_code_to_text(code):
    try:
        code = int(code)
    except Exception:
        return "Unknown"

    return {
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
        95: "Thunderstorm",
    }.get(code, f"Code {code}")


def make_line_chart(data, x_col, value_cols, y_title="Speed (mph)"):
    if data.empty:
        st.warning("No data available for this chart.")
        return

    use_cols = [col for col in value_cols if col in data.columns]

    if x_col not in data.columns or not use_cols:
        st.warning("Chart columns were not found.")
        return

    chart_df = data[[x_col] + use_cols].dropna().copy()

    if chart_df.empty:
        st.warning("Not enough data for this chart.")
        return

    long_df = chart_df.melt(
        id_vars=x_col,
        value_vars=use_cols,
        var_name="Line",
        value_name="Value"
    )

    x_type = "T" if pd.api.types.is_datetime64_any_dtype(chart_df[x_col]) else "Q"

    color_map = {
        "Current Speed": "#991b1b",
        "Normal Speed": "#6b7280",
        "Actual Speed": "#dc2626",
        "LSTM Prediction": "#2563eb",
        "ARIMA Prediction": "#f59e0b",
        "Predicted Speed": "#dc2626",
    }

    active_colors = [color_map.get(col, "#991b1b") for col in use_cols]

    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=3, point=True)
        .encode(
            x=alt.X(f"{x_col}:{x_type}", title=None),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color(
                "Line:N",
                scale=alt.Scale(domain=use_cols, range=active_colors),
                legend=alt.Legend(title=None)
            ),
            tooltip=[x_col, "Line", alt.Tooltip("Value:Q", format=".2f")]
        )
        .properties(height=330)
    )

    st.altair_chart(chart, width="stretch")


def make_metrics_bar(metrics_data):
    rows = []

    if metrics_data and "lstm" in metrics_data:
        rows.append({"Model": "LSTM", "Metric": "MAE", "Value": metrics_data["lstm"].get("mae")})
        rows.append({"Model": "LSTM", "Metric": "RMSE", "Value": metrics_data["lstm"].get("rmse")})

    if metrics_data and "arima" in metrics_data:
        rows.append({"Model": "ARIMA", "Metric": "MAE", "Value": metrics_data["arima"].get("mae")})
        rows.append({"Model": "ARIMA", "Metric": "RMSE", "Value": metrics_data["arima"].get("rmse")})

    metric_df = pd.DataFrame(rows).dropna()

    if metric_df.empty:
        st.warning("No model metrics found.")
        return

    chart = (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Value:Q", title="Error in mph"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=["MAE", "RMSE"], range=["#2563eb", "#f59e0b"]),
                legend=alt.Legend(title=None)
            ),
            xOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2f")]
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width="stretch")


def make_forecast(data):
    if data.empty or len(data) < 2:
        return pd.DataFrame()

    required = ["captured_at", "current_speed", "free_flow_speed"]
    if any(col not in data.columns for col in required):
        return pd.DataFrame()

    recent = data.dropna(subset=required).tail(5).copy()

    if len(recent) < 2:
        return pd.DataFrame()

    latest = recent.iloc[-1]

    current_speed = float(latest["current_speed"])
    normal_speed = float(latest["free_flow_speed"])
    latest_time = pd.to_datetime(latest["captured_at"])

    trend = recent["current_speed"].diff().mean()
    if pd.isna(trend):
        trend = 0

    rows = []

    for minutes in [5, 10, 15, 20, 25, 30]:
        predicted = current_speed + trend * (minutes / 5)
        predicted = max(0, min(predicted, normal_speed))
        level = classify_congestion(predicted, normal_speed)

        rows.append({
            "Minutes Ahead": minutes,
            "Forecast Time": (latest_time + timedelta(minutes=minutes)).strftime("%I:%M %p"),
            "Predicted Speed": round(predicted, 2),
            "Normal Speed": round(normal_speed, 2),
            "Congestion": level
        })

    return pd.DataFrame(rows)


def build_prediction_comparison(lstm_df, arima_df):
    if lstm_df.empty:
        return pd.DataFrame()

    plot_df = lstm_df.copy()

    if "captured_at" in plot_df.columns:
        plot_df["captured_at"] = pd.to_datetime(plot_df["captured_at"], errors="coerce")

    plot_df = plot_df.rename(
        columns={
            "captured_at": "Time",
            "current_speed": "Actual Speed",
            "lstm_predicted_speed": "LSTM Prediction"
        }
    )

    needed = ["Time", "Actual Speed", "LSTM Prediction"]
    if any(col not in plot_df.columns for col in needed):
        return pd.DataFrame()

    plot_df = plot_df[needed].dropna().reset_index(drop=True)

    if not arima_df.empty and "arima_predicted_speed" in arima_df.columns:
        arima_series = pd.to_numeric(arima_df["arima_predicted_speed"], errors="coerce").reset_index(drop=True)
        min_len = min(len(plot_df), len(arima_series))

        plot_df = plot_df.iloc[:min_len].copy()
        plot_df["ARIMA Prediction"] = arima_series.iloc[:min_len].values

    return plot_df


# -----------------------------
# Load all data
# -----------------------------

df = load_final_data()
traffic_db_df = load_traffic_from_db()

# If processed CSV is missing, still allow overview/forecast from SQLite.
traffic_source_df = df if not df.empty else traffic_db_df

accidents_df = load_accidents()
metrics = load_json(METRICS_PATH)
lstm_predictions, arima_predictions = load_predictions()


# -----------------------------
# Header
# -----------------------------

st.title("FDR Drive Traffic Forecast")
st.markdown(
    "<div class='muted'>Current traffic, short-term forecasts, model results, and crash patterns along FDR Drive.</div>",
    unsafe_allow_html=True
)

overview_tab, forecast_tab, model_tab, crash_tab, pipeline_tab = st.tabs(
    ["Overview", "Future Forecast", "Model Results", "Crash History", "Project Pipeline"]
)


# -----------------------------
# Overview
# -----------------------------

with overview_tab:
    st.subheader("Current Traffic Conditions")

    if traffic_source_df.empty:
        st.warning("No traffic data was found.")
        st.write("Run these commands first:")
        st.code(
            "python scripts/db_seeder.py\npython scripts/feature_engineering.py\npython models/evaluate_models.py",
            language="bash"
        )
    else:
        latest = traffic_source_df.iloc[-1]

        current_speed = float(latest["current_speed"])
        normal_speed = float(latest["free_flow_speed"])
        slowdown = max(0, (1 - current_speed / normal_speed) * 100) if normal_speed > 0 else 0
        level = classify_congestion(current_speed, normal_speed)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Current Speed", f"{current_speed:.1f} mph")
        col2.metric("Normal Speed", f"{normal_speed:.1f} mph")
        col3.metric("Slowdown", f"{slowdown:.0f}%")
        col4.metric("Congestion", level)

        st.caption(
            "Current Speed: higher is better. Normal Speed: expected clear-road speed. "
            "Slowdown: higher is worse. Congestion: Low is best, High is worst."
        )

        st.info(congestion_text(level))

        st.write(f"Traffic records loaded: **{len(traffic_source_df)}**")

        if "captured_at" in traffic_source_df.columns:
            st.write(f"Last updated: **{latest['captured_at']}**")

        st.subheader("Traffic Speed Trend")

        st.write(
            "This graph compares actual traffic speed with normal road speed. "
            "If the red line is far below the gray line, traffic is more congested."
        )

        chart_df = traffic_source_df.tail(80).rename(
            columns={
                "captured_at": "Time",
                "current_speed": "Current Speed",
                "free_flow_speed": "Normal Speed"
            }
        )

        make_line_chart(
            chart_df,
            "Time",
            ["Current Speed", "Normal Speed"]
        )

        if not df.empty:
            st.subheader("Data Used for Prediction")

            d1, d2, d3, d4 = st.columns(4)

            d1.metric("Crash History Match", int(latest.get("historical_crash_count", 0)))
            d2.metric("Temperature", f"{float(latest.get('temperature', 0)):.1f} °C")
            d3.metric("Weather", weather_code_to_text(latest.get("weather_code", None)))
            d4.metric("Hour", f"{int(latest.get('hour', 0))}:00")

            st.caption(
                "Crash History Match: higher can mean more accident risk for similar times. "
                "Weather and time help explain traffic changes."
            )


# -----------------------------
# Future Forecast
# -----------------------------

with forecast_tab:
    st.subheader("Future Congestion Forecast")

    if traffic_source_df.empty:
        st.warning("Forecast unavailable because traffic data is missing.")
    else:
        forecast_df = make_forecast(traffic_source_df)

        if forecast_df.empty:
            st.warning("Not enough traffic rows to create a forecast.")
        else:
            next_row = forecast_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric("Speed in 5 Minutes", f"{next_row['Predicted Speed']} mph")
            col2.metric("Expected Congestion", next_row["Congestion"])
            col3.metric("Forecast Window", "30 minutes")

            st.caption(
                "Predicted Speed: higher is better. Expected Congestion: Low is best, High is worst."
            )

            st.write(
                "This forecast uses the latest speed trend to estimate short-term traffic changes."
            )

            st.dataframe(forecast_df, width="stretch", hide_index=True)

            st.write(
                "This graph shows the predicted traffic speed over the next 30 minutes. "
                "Higher speed means traffic is expected to move better."
            )

            max_speed = max(
                forecast_df["Predicted Speed"].max(),
                forecast_df["Normal Speed"].max()
            )

            y_max = max_speed + 10

            base_chart = alt.Chart(forecast_df).encode(
                x=alt.X(
                    "Minutes Ahead:Q",
                    title="Minutes Ahead",
                    scale=alt.Scale(domain=[5, 30])
                ),
                y=alt.Y(
                    "Predicted Speed:Q",
                    title="Predicted Speed (mph)",
                    scale=alt.Scale(domain=[0, y_max])
                ),
                tooltip=[
                    "Minutes Ahead",
                    "Forecast Time",
                    alt.Tooltip("Predicted Speed:Q", format=".2f"),
                    "Congestion"
                ]
            )

            forecast_line = base_chart.mark_line(
                color="#dc2626",
                strokeWidth=4
            )

            forecast_points = base_chart.mark_point(
                color="#dc2626",
                filled=True,
                size=90
            )

            forecast_labels = base_chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-12,
                color="#dc2626",
                fontSize=13,
                fontWeight="bold"
            ).encode(
                text=alt.Text("Predicted Speed:Q", format=".1f")
            )

            normal_line = alt.Chart(forecast_df).mark_line(
                color="#6b7280",
                strokeWidth=2,
                strokeDash=[6, 4]
            ).encode(
                x=alt.X("Minutes Ahead:Q"),
                y=alt.Y("Normal Speed:Q")
            )

            forecast_chart = (
                normal_line + forecast_line + forecast_points + forecast_labels
            ).properties(height=350)

            st.altair_chart(forecast_chart, width="stretch")


# -----------------------------
# Model Results
# -----------------------------

with model_tab:
    st.subheader("Model Results")

    if metrics:
        lstm_metrics = metrics.get("lstm", {})
        arima_metrics = metrics.get("arima", {})
        best_model = str(metrics.get("best_model_by_rmse", "N/A")).upper()

        col1, col2, col3 = st.columns(3)

        col1.metric("LSTM RMSE", lstm_metrics.get("rmse", "N/A"))
        col2.metric("ARIMA RMSE", arima_metrics.get("rmse", "N/A"))
        col3.metric("Best Model", best_model)

        st.caption(
            "RMSE measures prediction error in mph. Lower is better. "
            "Best Model is the one with the lower RMSE."
        )

        st.subheader("Model Error Comparison")
        make_metrics_bar(metrics)

    else:
        st.warning("Model metrics were not found.")
        st.code("python models/evaluate_models.py", language="bash")

    st.subheader("Actual Speed vs Predictions")

    comparison_df = build_prediction_comparison(lstm_predictions, arima_predictions)

    if comparison_df.empty:
        st.info("Run `python models/evaluate_models.py` to generate prediction files.")
    else:
        model_cols = ["Actual Speed", "LSTM Prediction"]
        if "ARIMA Prediction" in comparison_df.columns:
            model_cols.append("ARIMA Prediction")

        st.write(
            "The model results are split into two graphs to keep the lines easier to read."
        )

        # Smooth the graph slightly so the trend is easier to see.
        smooth_df = comparison_df.sort_values("Time").copy()

        rolling_window = 5 if len(smooth_df) >= 10 else 3

        for col in model_cols:
            smooth_df[col] = (
                pd.to_numeric(smooth_df[col], errors="coerce")
                .rolling(window=rolling_window, min_periods=1, center=True)
                .mean()
            )

        st.markdown("#### Actual Speed vs LSTM Prediction")

        st.write(
            "This graph compares the real traffic speed against the LSTM model. "
            "The closer the blue line is to the red line, the better the LSTM prediction is."
        )

        make_line_chart(
            smooth_df,
            "Time",
            ["Actual Speed", "LSTM Prediction"]
        )

        st.caption(
            f"This graph uses a {rolling_window}-point rolling average to make the trend easier to read. "
            "Red = actual speed. Blue = LSTM prediction."
        )

        if "ARIMA Prediction" in smooth_df.columns:
            st.markdown("#### Actual Speed vs ARIMA Prediction")

            st.write(
                "This graph compares the real traffic speed against the ARIMA baseline model. "
                "The closer the orange line is to the red line, the better the ARIMA prediction is."
            )

            make_line_chart(
                smooth_df,
                "Time",
                ["Actual Speed", "ARIMA Prediction"]
            )

            st.caption(
                f"This graph uses a {rolling_window}-point rolling average to make the trend easier to read. "
                "Red = actual speed. Orange = ARIMA prediction."
            )

        with st.expander("Show prediction data"):
            st.dataframe(
                comparison_df[["Time"] + model_cols],
                width="stretch",
                hide_index=True
            )


# -----------------------------
# Crash History
# -----------------------------

with crash_tab:
    st.subheader("Historical Crash Information")

    if accidents_df.empty:
        st.warning("Crash data was not found.")
        st.write("Make sure this file exists:")
        st.code("data/accidents_nosql.json")
        st.write("Detected accident file:")
        st.code(str(ACCIDENTS_JSON) if ACCIDENTS_JSON else "Not found")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Crash Records", len(accidents_df))

        if "hour" in accidents_df.columns:
            busiest = pd.to_numeric(accidents_df["hour"], errors="coerce").dropna().mode()
            col2.metric("Most Common Crash Hour", f"{int(busiest.iloc[0])}:00" if not busiest.empty else "N/A")
        else:
            col2.metric("Most Common Crash Hour", "N/A")

        if "number_of_persons_injured" in accidents_df.columns:
            injuries = pd.to_numeric(accidents_df["number_of_persons_injured"], errors="coerce").fillna(0).sum()
            col3.metric("Total Injuries", int(injuries))
        else:
            col3.metric("Total Injuries", "N/A")

        st.caption(
            "Crash Records: more records give more crash history. "
            "Total Injuries: higher is worse."
        )

        if {"latitude", "longitude"}.issubset(accidents_df.columns):
            accidents_df["latitude"] = pd.to_numeric(accidents_df["latitude"], errors="coerce")
            accidents_df["longitude"] = pd.to_numeric(accidents_df["longitude"], errors="coerce")

            map_df = accidents_df.dropna(subset=["latitude", "longitude"]).copy()

            if map_df.empty:
                st.warning("Crash records exist, but no valid latitude/longitude values were found.")
            else:
                st.subheader("Crash Location Map")

                st.write(
                    "This is the original point map. Each point represents one crash record."
                )

                point_map = map_df[["latitude", "longitude"]].rename(
                    columns={"latitude": "lat", "longitude": "lon"}
                )

                st.map(point_map, width="stretch")

                st.subheader("Crash Density Heatmap")

                if pdk is None:
                    st.warning("PyDeck is not installed. Run: pip install pydeck")
                else:
                    st.write(
                        "This heatmap shows where crashes are concentrated. "
                        "Darker areas mean more crash records are clustered together."
                    )

                    heatmap_layer = pdk.Layer(
                        "HeatmapLayer",
                        data=map_df,
                        get_position="[longitude, latitude]",
                        opacity=0.75,
                        radiusPixels=45,
                        intensity=1,
                        threshold=0.05,
                    )

                    view_state = pdk.ViewState(
                        latitude=float(map_df["latitude"].mean()),
                        longitude=float(map_df["longitude"].mean()),
                        zoom=10,
                        pitch=35,
                    )

                    deck = pdk.Deck(
                        layers=[heatmap_layer],
                        initial_view_state=view_state,
                        tooltip={"text": "Crash concentration area"}
                    )

                    st.pydeck_chart(deck, width="stretch")

        st.subheader("Crash Time Heatmap")

        if {"hour", "day_of_week"}.issubset(accidents_df.columns):
            time_df = accidents_df.copy()
            time_df["hour"] = pd.to_numeric(time_df["hour"], errors="coerce")
            time_df["day_of_week"] = pd.to_numeric(time_df["day_of_week"], errors="coerce")
            time_df = time_df.dropna(subset=["hour", "day_of_week"])

            if not time_df.empty:
                count_df = (
                    time_df
                    .groupby(["day_of_week", "hour"])
                    .size()
                    .reset_index(name="Crash Count")
                )

                count_df["Day"] = count_df["day_of_week"].map({
                    0: "Mon",
                    1: "Tue",
                    2: "Wed",
                    3: "Thu",
                    4: "Fri",
                    5: "Sat",
                    6: "Sun"
                })

                heat_chart = (
                    alt.Chart(count_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("hour:O", title="Hour of Day"),
                        y=alt.Y("Day:N", title="Day of Week", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
                        color=alt.Color(
                            "Crash Count:Q",
                            scale=alt.Scale(scheme="reds"),
                            title="Crash Count"
                        ),
                        tooltip=["Day", "hour", "Crash Count"]
                    )
                    .properties(height=260)
                )

                st.altair_chart(heat_chart, width="stretch")

                st.caption("Darker cells mean more crashes happened during that day and hour.")

        with st.expander("Show crash records"):
            st.dataframe(accidents_df.head(100), width="stretch", hide_index=True)


# -----------------------------
# Project Pipeline
# -----------------------------

with pipeline_tab:
    st.subheader("Project Purpose")

    st.write(
        "The project estimates short-term congestion on FDR Drive by combining traffic speed, crash history, weather, and time-based patterns."
    )

    st.subheader("How It Works")

    st.markdown(
        """
        1. Collect or seed traffic speed records.
        2. Store traffic data in SQLite.
        3. Add crash history from NYC collision records.
        4. Add weather information.
        5. Train and compare LSTM and ARIMA models.
        6. Display traffic, forecasts, model results, and crash patterns.
        """
    )

    st.subheader("Data Sources")

    st.markdown(
        """
        - TomTom Traffic API for live speed data.
        - NYC collision records for crash history.
        - Open-Meteo for weather information.
        - SQLite for structured traffic records.
        - TinyDB/JSON for crash records.
        """
    )

    with st.expander("Detected local files"):
        st.write("Processed dataset:")
        st.code(str(FINAL_CSV) if FINAL_CSV else "Not found")

        st.write("Traffic database:")
        st.code(str(TRAFFIC_DB) if TRAFFIC_DB else "Not found")

        st.write("Crash data:")
        st.code(str(ACCIDENTS_JSON) if ACCIDENTS_JSON else "Not found")

        st.write("Metrics:")
        st.code(str(METRICS_PATH) if METRICS_PATH else "Not found")

        st.write("LSTM predictions:")
        st.code(str(LSTM_PREDICTIONS_PATH) if LSTM_PREDICTIONS_PATH else "Not found")

        st.write("ARIMA predictions:")
        st.code(str(ARIMA_PREDICTIONS_PATH) if ARIMA_PREDICTIONS_PATH else "Not found")
