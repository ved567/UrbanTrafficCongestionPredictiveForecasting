# Urban Traffic Congestion Predictive Forecasting

This project predicts short-term traffic congestion on FDR Drive using traffic speed data, historical NYC crash records, weather information, and machine learning models.

The system builds a full data pipeline that generates traffic data, downloads and processes NYC collision records, adds weather features, trains LSTM and ARIMA forecasting models, and displays the results in a Streamlit dashboard.

## Project Demo Video

https://youtu.be/f8Nt18n-TxI?si=S4idHr-O1JtlaIUj

## Project Features

- Generates traffic records for an FDR Drive location
- Downloads NYC collision records
- Filters crash records related to FDR Drive
- Adds weather data from Open-Meteo
- Creates a final model-ready dataset
- Trains and compares LSTM and ARIMA forecasting models
- Displays traffic conditions, future forecasts, model results, crash history, weather, and maps in a dashboard

## Project Structure

UrbanTrafficCongestionPredictiveForecasting/
├── dashboard/
│   └── app.py
├── data/
│   └── generated data files appear here after running the pipeline
├── models/
│   ├── evaluate_models.py
│   ├── train_lstm.py
│   ├── train_arima.py
│   ├── traffic_model.py
│   └── generated model files appear here after running the pipeline
├── scripts/
│   ├── db_seeder.py
│   ├── accident_ingestion.py
│   ├── feature_engineering.py
│   └── traffic_collector.py
├── run_all.py
├── requirements.txt
├── .env.example
└── README.md
# Data Sources
- TomTom Traffic API: live traffic speed data
- NYC Open Data: historical motor vehicle collision records
- Open-Meteo: weather information
- SQLite: storage for traffic records
- TinyDB / JSON: storage for processed crash records

## How to Run the Project
1. Run 'pip install -r requirements.txt'
2. Run 'python run_all.py'
3. Run 'streamlit run dashboard/app.py'


## TomTom API

The TomTom Traffic API provides live traffic information such as current speed, free-flow speed, and traffic confidence for a selected road location. In this project, it is used to collect traffic data for FDR Drive and compare real traffic speed with normal road speed.

You can get a TomTom API key by creating a free developer account on the TomTom Developer Portal. After creating an account, generate an API key and add it to the `.env` file

