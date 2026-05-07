"""Compatibility entry point for run_all.py.

This replaces the original single-model training script with the complete
Chadwick ML workflow: LSTM model training + ARIMA baseline + metrics.json.
"""

from evaluate_models import main

if __name__ == "__main__":
    main()