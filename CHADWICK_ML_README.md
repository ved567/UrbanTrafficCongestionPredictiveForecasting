# Chadwick ML Workflow

This folder adds the complete ML portion of the urban traffic congestion project.

## What was added

- `models/train_lstm.py`: trains the LSTM traffic-speed predictor.
- `models/train_arima.py`: trains an ARIMA historical baseline.
- `models/evaluate_models.py`: runs both models and writes comparison metrics.
- `models/traffic_model.py`: shared PyTorch model class.
- `models/step3_train.py`: compatibility wrapper so `python run_all.py` still works.
- `models/traffic_lstm.pth`: saved model weights for the Streamlit dashboard.
- `models/metrics.json`: MAE/RMSE comparison for LSTM and ARIMA.
- `models/lstm_predictions.csv` and `models/arima_predictions.csv`: test-set predictions.

## How to run only the ML workflow

```bash
python models/evaluate_models.py
```

## How to run from the full pipeline

```bash
python run_all.py
```

## Evaluation design

The workflow uses a chronological 80/20 train/test split. This avoids data leakage because the models train on earlier traffic records and test on later records.

## Metrics

Both models are evaluated with:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error

## Chadwick contribution line

Chadwick implemented the machine learning workflow, including LSTM traffic prediction, ARIMA baseline comparison, chronological 80/20 train/test evaluation, MAE/RMSE metrics, lightweight feature engineering, saved model weights, and dashboard-ready model artifacts.
