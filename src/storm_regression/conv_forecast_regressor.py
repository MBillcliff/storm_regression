"""
ConvForecastRegressor: A convolutional encoder-based regression model for time series forecasting.

This model takes four separate multivariate time series inputs:
- v: solar wind speed features
- vgrad: gradients of the solar wind speed
- vomni: difference between observed omni flow speed and v
- historic_target: the past values of the target variable being forecasted

Each input stream is passed through a shared architecture of 1D convolutional layers
(ConvEncoder), which apply temporal feature extraction and global average pooling.

The encoded feature vectors from all four streams are concatenated and passed through
a feedforward head (MLP) to produce a scalar regression output, typically representing
a future value or severity index.

This model is designed for scenarios where spatially-independent, temporally-structured
data streams are available and can be encoded separately before fusion for final prediction.
"""
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling over time dimension
        )

    def forward(self, x):
        # x shape: [batch, time, features]
        x = x.permute(0, 2, 1)  # -> [batch, features, time] for Conv1d
        x = self.encoder(x)
        return x.squeeze(-1)  # -> [batch, out_channels]


class ConvForecastRegressor(nn.Module):
    def __init__(self, v_feat, vgrad_feat, vomni_feat, historic_target_feat=None):
        super().__init__()
        self.use_historic_target = historic_target_feat is not None

        self.v_encoder = ConvEncoder(in_channels=v_feat)
        self.vgrad_encoder = ConvEncoder(in_channels=vgrad_feat)
        self.vomni_encoder = ConvEncoder(in_channels=vomni_feat)

        if self.use_historic_target:
            self.historic_target_encoder = ConvEncoder(in_channels=historic_target_feat)

        num_streams = 3 + int(self.use_historic_target)
        combined_features = num_streams * 32

        self.head = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, v, vgrad, vomni, historic_target=None):
        v_encoded = self.v_encoder(v)
        vgrad_encoded = self.vgrad_encoder(vgrad)
        vomni_encoded = self.vomni_encoder(vomni)

        features = [v_encoded, vgrad_encoded, vomni_encoded]

        if self.use_historic_target:
            assert historic_target is not None, "historic_target input is required but not provided"
            historic_target_encoded = self.historic_target_encoder(historic_target)
            features.append(historic_target_encoded)

        x = torch.cat(features, dim=1)
        return self.head(x).squeeze()
