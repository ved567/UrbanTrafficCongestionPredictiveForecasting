"""Shared PyTorch model definition for traffic-speed forecasting.

This file is imported by training/evaluation scripts and mirrors the model
architecture expected by the Streamlit dashboard.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """LSTM regression model that predicts current traffic speed in mph."""

    def __init__(self, input_size: int = 5, hidden_size: int = 50, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
