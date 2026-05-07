import streamlit as st
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class TrafficLSTM(nn.Module):
    """
    Identical LSTM architecture used for loading the trained model.
    """
    def __init__(self):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def run_dashboard():
    """
    Streamlit application to visualize real-time traffic speeds 
    and model predictions.
    """
    st.title("Urban Traffic Congestion Forecast")
    
    try:
        df = pd.read_csv("data/final_ai_data.csv")
        st.write("Current Data Preview (Last 5 Records):")
        st.dataframe(df.tail(5))

        st.line_chart(df['current_speed'])
    except Exception:
        st.warning("No data found. Please run the pipeline first.")

if __name__ == "__main__":
    run_dashboard()
