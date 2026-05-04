import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/final_ai_data.csv")
if len(df) < 5:
    print("Error: Need more rows in traffic_data.db")
    exit()

features = ['historical_crash_count', 'hour', 'day_of_week', 'temperature', 'weather_code']
X = df[features].values.astype(np.float32)
y = df['current_speed'].values.astype(np.float32).reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train = torch.FloatTensor(X_scaled).unsqueeze(1)
y_train = torch.FloatTensor(y_scaled)

class TrafficLSTM(nn.Module):
    def __init__(self):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = TrafficLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), "models/traffic_lstm.pth")
print("Model saved to models/traffic_lstm.pth")

model.eval()
predictions = model(X_train).detach().numpy()
predictions_rescaled = scaler_y.inverse_transform(predictions)
mae = np.mean(np.abs(y - predictions_rescaled))
print(f"Final MAE: {mae}")