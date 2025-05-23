import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from scipy.signal import cwt, ricker
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.stattools import adfuller


class StructuralBreakDetector(nn.Module):
    def __init__(
        self,
        wavelet_widths=np.arange(1, 31),
        input_length=60,
        hidden_dim=64,
        dropout=0.3,
    ):
        super().__init__()
        self.wavelet_widths = wavelet_widths
        self.hidden_dim = hidden_dim
        self.input_length = input_length

        # CNN for wavelet-transformed input
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (B, 1, F, T)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically determine CNN output size
        self.cnn_output_dim = self._get_cnn_output_dim()

        # LSTM for raw spread/price
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, batch_first=True)

        # Fully connected layers
        self.fc = None  # on la créera dans forward()

    def _get_cnn_output_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, len(self.wavelet_widths), self.input_length)
            dummy_input = dummy_input.to(
                next(self.cnn.parameters()).device
            )  # 🔥 important
            out = self.cnn(dummy_input)
            return out.view(1, -1).shape[1]

    def forward(self, spread, y_price, x_price):
        B, T = spread.shape

        # CWT + CNN
        wavelet_imgs = []
        for i in range(B):
            wt = cwt(spread[i].cpu().numpy(), ricker, self.wavelet_widths)
            wavelet_imgs.append(wt)
        wavelet_imgs = np.stack(wavelet_imgs)
        wavelet_imgs = (
            torch.tensor(wavelet_imgs, dtype=torch.float32)
            .unsqueeze(1)
            .to(spread.device)
        )

        cnn_out = self.cnn(wavelet_imgs)
        cnn_out_flat = cnn_out.view(B, -1)

        # LSTM
        raw_input = torch.stack([spread, y_price, x_price], dim=2)
        _, (h_n, _) = self.lstm(raw_input)
        lstm_out = h_n[-1]

        combined = torch.cat([cnn_out_flat, lstm_out], dim=1)

        # Construire dynamiquement la couche fully connected
        if self.fc is None:
            input_dim = combined.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(
                combined.device
            )  # Important : placer sur le bon device

        return self.fc(combined)


def generate_structural_break_dataset(
    train_data: pd.DataFrame,
    cointegrated_pairs: list,
    input_length: int = 60,
    future_window: int = 30,
    adf_threshold: float = 0.05,
):
    X_spread, X_y, X_x, labels = [], [], [], []

    ones_vec = np.ones(input_length)

    for ticker_y, ticker_x, _ in cointegrated_pairs:
        y = train_data[ticker_y].dropna().values
        x = train_data[ticker_x].dropna().values
        labels_local = []

        min_len = min(len(y), len(x))
        y = y[:min_len]
        x = x[:min_len]

        for t in range(input_length, min_len - future_window):
            y_window = y[t - input_length : t]
            x_window = x[t - input_length : t]

            # Régression linéaire
            X_mat = np.column_stack((x_window, ones_vec))
            beta, intercept = np.linalg.lstsq(X_mat, y_window, rcond=None)[0]
            spread_window = y_window - (beta * x_window + intercept)
            spread_window = (spread_window - spread_window.mean()) / (
                spread_window.std() + 1e-8
            )
            x_window = (x_window - x_window.mean()) / (x_window.std() + 1e-8)
            y_window = (y_window - y_window.mean()) / (y_window.std() + 1e-8)

            # Fenêtre future
            y_future = y[t : t + future_window]
            x_future = x[t : t + future_window]
            spread_future = y_future - (beta * x_future + intercept)

            # Skip fenêtre plate (variance nulle)
            std_spread = np.std(spread_window)
            if std_spread < 1e-8:
                continue

            try:
                pval = adfuller(spread_future, maxlag=1, regression="c", autolag=None)[
                    1
                ]
            except Exception:
                pval = 1.0

            mean_drift = abs(np.mean(spread_future) - np.mean(spread_window))
            label = int(pval > adf_threshold and mean_drift > 0.5 * std_spread)

            X_spread.append(spread_window)
            X_y.append(y_window)
            X_x.append(x_window)
            labels.append(label)
            labels_local.append(label)

        instability_rate = np.mean(labels_local) if labels_local else 0
        print(
            f"Taux de cibles instables pour {ticker_y} et {ticker_x}: {instability_rate:.2%}"
        )

    # Conversion en tenseurs
    X_spread = torch.tensor(X_spread, dtype=torch.float32)
    X_y = torch.tensor(X_y, dtype=torch.float32)
    X_x = torch.tensor(X_x, dtype=torch.float32)
    y_labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return TensorDataset(X_spread, X_y, X_x, y_labels)
