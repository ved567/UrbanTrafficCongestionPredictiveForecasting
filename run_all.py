import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(step_name, command):
    print("\n" + "=" * 70)
    print(step_name)
    print("=" * 70)

    try:
        subprocess.run(command, check=True)
        print(f"{step_name} completed successfully.")
    except subprocess.CalledProcessError:
        print(f"\nPipeline failed during: {step_name}")
        print(f"Command failed: {' '.join(command)}")
        sys.exit(1)


def main():
    print("\nStarting Urban Traffic Congestion Predictive Forecasting Pipeline")

    data_dir = ROOT / "data"
    models_dir = ROOT / "models"

    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    run_step(
        "Step 1: Creating traffic database",
        [sys.executable, "scripts/db_seeder.py"]
    )

    run_step(
        "Step 2: Downloading and processing NYC collision data",
        [sys.executable, "scripts/accident_ingestion.py"]
    )

    run_step(
        "Step 3: Creating final AI dataset",
        [sys.executable, "scripts/feature_engineering.py"]
    )

    run_step(
        "Step 4: Training and evaluating LSTM and ARIMA models",
        [sys.executable, "models/evaluate_models.py"]
    )

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/traffic_data.db")
    print("data/nyc_collisions_final.json")
    print("data/accidents_nosql.json")
    print("data/final_ai_data.csv")
    print("models/traffic_lstm.pth")
    print("models/metrics.json")
    print("models/lstm_metrics.json")
    print("models/arima_metrics.json")
    print("models/lstm_predictions.csv")
    print("models/arima_predictions.csv")

    print("\nTo open the dashboard, run:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
