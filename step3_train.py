import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

try:
    df = pd.read_csv("final_ai_data.csv")
except FileNotFoundError:
    exit()

features = ['historical_crash_count', 'hour', 'day_of_week', 'temperature', 'weather_code']
X_data = df[features].values
y_data = df['current_speed'].values

split_point = int(len(X_data) * 0.8)
X_train_raw, X_test_raw = X_data[:split_point], X_data[split_point:]
y_train, y_test = y_data[:split_point], y_data[split_point:]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error')

model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)