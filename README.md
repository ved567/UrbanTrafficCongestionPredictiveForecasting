# Urban Traffic Congestion Predictive Forecasting

This project aims to predict urban traffic congestion using real-time traffic data, historical accident data, and weather information.

## Project Structure

- `/data`: Stores traffic databases (`.db`), accident records (`.json`), and processed datasets (`.csv`).
- `/scripts`: Collection and seeding scripts for gathering data.
- `/models`: LSTM/ARIMA model code and saved `.pth` model files.
- `/dashboard`: Streamlit UI code for visualizing predictions.
- `run_all.py`: Orchestration script to run the entire pipeline.

## Setup

1.  Clone the repository.
2.  Install dependencies (e.g., `pip install pandas requests tinydb torch streamlit`).
3.  Copy `.env.example` to `.env` and add your TomTom API key.
4.  Run the system using `python run_all.py`.

## Branching
- `main`: Stable base folder structure.
- `kirollos-db-optimization`: Ongoing database and structure optimizations.
