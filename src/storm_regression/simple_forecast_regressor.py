"""
SimpleForecastRegressor: A fully connected feedforward neural network for multistep time series forecasting.

This model takes three separate multivariate time series inputs:
- v: solar wind speed features
- vgrad: gradients of the solar wind speed
- vomni: other solar wind or space weather features (e.g., IMF, density)

Each input stream is flattened across the time and feature dimensions and concatenated
to form a single input vector. The concatenated vector is passed through a multi-layer
perceptron (MLP) consisting of two hidden layers with ReLU activations, followed by
an output layer that predicts `forecast_steps` future values.

This architecture is simple yet effective when modeling low-resolution or aggregated
temporal dependencies, where convolutional or recurrent processing may not be necessary.
It is suitable for baselines or when interpretability and training speed are prioritized.
"""
import torch
import torch.nn as nn

class SimpleForecastRegressor(nn.Module):
    def __init__(self, v_features, vgrad_features, vomni_features, historic_target_features):
        super().__init__()
        # Flatten inputs and concatenate all features along feature dimension
        input_size = v_features + vgrad_features + vomni_features + historic_target_features

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, v, vgrad, vomni, historic_target):
        # v, vgrad, vomni shapes: [batch, time, features]
        # Flatten time and features for each input and concatenate along feature axis
        v_flat = v.view(v.size(0), -1)
        vgrad_flat = vgrad.view(vgrad.size(0), -1)
        vomni_flat = vomni.view(vomni.size(0), -1)
        historic_target_flat = historic_target.view(historic_target.size(0), -1)
        
        x = torch.cat([v_flat, vgrad_flat, vomni_flat, historic_target_flat], dim=1)
        return self.model(x).squeeze()