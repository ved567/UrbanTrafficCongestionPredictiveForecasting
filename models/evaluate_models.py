"""Run the complete ML workflow: LSTM + ARIMA + comparison metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from train_lstm import train_lstm
    from train_arima import train_arima
except ImportError:
    from models.train_lstm import train_lstm
    from models.train_arima import train_arima


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_root() / "data" / "final_ai_data.csv"))
    parser.add_argument("--models-dir", default=str(project_root() / "models"))
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    lstm_metrics = train_lstm(csv_path, models_dir, epochs=args.epochs)
    arima_metrics = train_arima(csv_path, models_dir)

    winner = "lstm" if lstm_metrics["rmse"] <= arima_metrics["rmse"] else "arima"
    comparison = {
        "dataset": str(csv_path),
        "split": "chronological 80/20 train/test split to avoid leakage",
        "lstm": lstm_metrics,
        "arima": arima_metrics,
        "best_model_by_rmse": winner,
    }

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Saved combined model comparison to models/metrics.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
