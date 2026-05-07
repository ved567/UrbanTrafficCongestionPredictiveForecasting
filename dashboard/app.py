from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

FINAL_CSV = DATA_DIR / "final_ai_data.csv"
MODEL_PATH = MODEL_DIR / "traffic_lstm.pth"
METRICS_PATH = MODEL_DIR / "metrics.json"


class TrafficLSTM(nn.Module):
    """
    Identical LSTM architecture used for loading the trained model.
    """
    def __init__(self):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def load_model():
    model = TrafficLSTM()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_dashboard():
    """
    Streamlit application to visualize real-time traffic speeds
    and model predictions.
    """
    st.title("Urban Traffic Congestion Forecast")

    if not FINAL_CSV.exists():
        st.warning("No data found. Please run the pipeline first.")
        st.code("python run_all.py", language="bash")
        return

    df = pd.read_csv(FINAL_CSV)

    st.write("Current Data Preview (Last 5 Records):")
    st.dataframe(df.tail(5), width="stretch")

    if "current_speed" in df.columns:
        st.line_chart(df["current_speed"])

    st.subheader("Model Status")

    if MODEL_PATH.exists():
        st.success("LSTM model found.")
    else:
        st.warning("LSTM model not found. Run the ML workflow first.")
        st.code("python models/evaluate_models.py", language="bash")

    if METRICS_PATH.exists():
        st.subheader("Model Evaluation Metrics")
        st.json(pd.read_json(METRICS_PATH).to_dict())

    required_features = [
        "historical_crash_count",
        "hour",
        "day_of_week",
        "temperature",
        "weather_code"
    ]

    if MODEL_PATH.exists() and all(col in df.columns for col in required_features):
        st.subheader("Latest LSTM Prediction")

        try:
            model_df = df.dropna(subset=required_features + ["current_speed"]).copy()

            X = model_df[required_features].values.astype(np.float32)
            y = model_df["current_speed"].values.astype(np.float32).reshape(-1, 1)

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(X)
            scaler_y.fit(y)

            latest_X = X_scaled[-1].reshape(1, 1, len(required_features))
            latest_tensor = torch.FloatTensor(latest_X)

            model = load_model()

            with torch.no_grad():
                pred_scaled = model(latest_tensor).numpy()

            predicted_speed = scaler_y.inverse_transform(pred_scaled)[0][0]

            st.metric("Predicted Current Speed", f"{predicted_speed:.2f} mph")

        except Exception as e:
            st.error(f"Could not generate prediction: {e}")


if __name__ == "__main__":
    run_dashboard()