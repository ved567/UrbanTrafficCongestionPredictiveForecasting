"""Train and evaluate the LSTM traffic forecasting model.

Outputs:
- models/traffic_lstm.pth       Saved PyTorch model weights and metadata
- models/lstm_predictions.csv   Test-set actual vs predicted speeds
- models/lstm_metrics.json      LSTM MAE/RMSE and split metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Prevent tiny CPU training jobs from hanging on some local/macOS setups.
torch.set_num_threads(1)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from traffic_model import TrafficLSTM
except ImportError:
    from models.traffic_model import TrafficLSTM


FEATURES = [
    "historical_crash_count",
    "hour",
    "day_of_week",
    "temperature",
    "weather_code",
]
TARGET = "current_speed"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")

    required = ["captured_at", TARGET, "free_flow_speed", *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in [TARGET, "free_flow_speed", *FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).sort_values("captured_at").reset_index(drop=True)

    # Lightweight feature engineering for reporting/explainability.
    # The deployed dashboard still uses the original 5 model features, so these
    # engineered fields do not change the model input shape.
    df["rolling_speed_avg_3"] = df[TARGET].rolling(window=3, min_periods=1).mean()
    df["congestion_likelihood"] = 1 - (df[TARGET] / df["free_flow_speed"].replace(0, np.nan))
    df["congestion_likelihood"] = df["congestion_likelihood"].clip(lower=0, upper=1).fillna(0)

    if len(df) < 5:
        raise ValueError("Need at least 5 clean rows to train and evaluate the model.")

    return df


def chronological_train_test_split(df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * train_ratio))
    if split_idx >= len(df):
        split_idx = len(df) - 1
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def to_tensors(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train_df[FEATURES].values.astype(np.float32))
    y_train = scaler_y.fit_transform(train_df[[TARGET]].values.astype(np.float32))
    X_test = scaler_X.transform(test_df[FEATURES].values.astype(np.float32))

    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    return X_train_t, y_train_t, X_test_t, scaler_X, scaler_y


def train_lstm(csv_path: Path, models_dir: Path, epochs: int = 250, learning_rate: float = 0.01) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    train_df, test_df = chronological_train_test_split(df)
    X_train_t, y_train_t, X_test_t, scaler_X, scaler_y = to_tensors(train_df, test_df)

    model = TrafficLSTM(input_size=len(FEATURES), hidden_size=50, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).numpy()

    predictions = scaler_y.inverse_transform(pred_scaled).reshape(-1)
    actuals = test_df[TARGET].values.astype(float)

    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
        "target": TARGET,
        "input_size": len(FEATURES),
        "hidden_size": 50,
        "num_layers": 2,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "final_training_loss": float(loss_history[-1]),
    }
    torch.save(checkpoint, models_dir / "traffic_lstm.pth")

    prediction_df = test_df[["captured_at", TARGET, "free_flow_speed", "rolling_speed_avg_3", "congestion_likelihood"]].copy()
    prediction_df["lstm_predicted_speed"] = np.round(predictions, 3)
    prediction_df["absolute_error"] = np.round(np.abs(actuals - predictions), 3)
    prediction_df.to_csv(models_dir / "lstm_predictions.csv", index=False)

    metrics = {
        "model": "LSTM",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": FEATURES,
        "target": TARGET,
        "train_test_split": "chronological 80/20",
        "final_training_loss": round(float(loss_history[-1]), 6),
    }
    with open(models_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_root() / "data" / "final_ai_data.csv"))
    parser.add_argument("--models-dir", default=str(project_root() / "models"))
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    metrics = train_lstm(Path(args.csv), Path(args.models_dir), epochs=args.epochs)
    print("Saved LSTM model to models/traffic_lstm.pth")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
