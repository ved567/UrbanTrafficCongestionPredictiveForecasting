"""Train and evaluate an ARIMA historical baseline.

Outputs:
- models/arima_predictions.csv
- models/arima_metrics.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

TARGET = "current_speed"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_series(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["captured_at", TARGET]).sort_values("captured_at").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError("Need at least 5 clean rows to evaluate ARIMA.")

    return df


def chronological_train_test_split(df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * train_ratio))
    if split_idx >= len(df):
        split_idx = len(df) - 1
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_arima(csv_path: Path, models_dir: Path) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_series(csv_path)
    train_df, test_df = chronological_train_test_split(df)

    train_y = train_df[TARGET].astype(float).values
    test_y = test_df[TARGET].astype(float).values

    # Small datasets can make ARIMA order selection fragile. This sequence gives
    # a real baseline while falling back gracefully when there are very few rows.
    candidate_orders = [(2, 1, 1), (1, 1, 1), (1, 0, 0), (0, 1, 0)]
    fitted = None
    selected_order = None
    last_error = None

    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(train_y, order=order).fit()
            selected_order = order
            break
        except Exception as exc:
            last_error = exc

    if fitted is None:
        raise RuntimeError(f"ARIMA failed for all candidate orders. Last error: {last_error}")

    forecast = np.asarray(fitted.forecast(steps=len(test_y)), dtype=float)
    mae = float(mean_absolute_error(test_y, forecast))
    rmse = float(np.sqrt(mean_squared_error(test_y, forecast)))

    prediction_df = test_df[["captured_at", TARGET]].copy()
    prediction_df["arima_predicted_speed"] = np.round(forecast, 3)
    prediction_df["absolute_error"] = np.round(np.abs(test_y - forecast), 3)
    prediction_df.to_csv(models_dir / "arima_predictions.csv", index=False)

    metrics = {
        "model": "ARIMA",
        "order": list(selected_order),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target": TARGET,
        "train_test_split": "chronological 80/20",
    }
    with open(models_dir / "arima_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_root() / "data" / "final_ai_data.csv"))
    parser.add_argument("--models-dir", default=str(project_root() / "models"))
    args = parser.parse_args()

    metrics = train_arima(Path(args.csv), Path(args.models_dir))
    print("Saved ARIMA baseline metrics and predictions")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
