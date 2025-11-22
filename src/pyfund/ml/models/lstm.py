# src/pyfundlib/ml/models/lstm.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ...utils.logger import get_logger
from .base_model import BaseMLModel

logger = get_logger(__name__)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len])


class LSTMModel(BaseMLModel, nn.Module):
    """
    State-of-the-art LSTM for financial time series prediction.
    Supports regression (next return) and classification (direction).
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        output_size: int = 1,
        task: Literal["regression", "classification"] = "regression",
        seq_len: int = 60,
        name: str = "lstm_predictor",
        version: str = "1.0",
    ):
        # Init BaseMLModel for metadata, save/load
        BaseMLModel.__init__(self, name=name, version=version, tags=["lstm", "deep-learning", task])
        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.task = task
        self.seq_len = seq_len

        self.scaler = StandardScaler()

        # Architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        # Activation for classification
        if task == "classification":
            self.output_activation = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=-1)
        else:
            self.output_activation = nn.Identity()

        self.to(self._get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last timestep
        out = self.layer_norm(out)
        out = self.dropout_layer(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return self.output_activation(out)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 15,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> LSTMModel:
        X = np.array(X)
        y = np.array(y).reshape(-1, self.output_size)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        dataset = TimeSeriesDataset(X_scaled, y, self.seq_len)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss() if self.task == "regression" else nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._get_device()), batch_y.to(self._get_device())

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = self._validate(val_loader, criterion)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                patience_counter = 0
                self.best_state_dict = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if hasattr(self, "best_state_dict"):
            self.load_state_dict(self.best_state_dict)

        self._is_fitted = True
        self.feature_names_in_ = (
            list(range(X.shape[1] if len(X.shape) > 1 else 1))
            if isinstance(X, np.ndarray)
            else list(X.columns)
        )
        self.metadata.training_samples = len(y)
        self.metadata.performance_metrics = {"final_val_loss": best_loss}
        self.metadata.status = "trained"

        return self

    def _validate(self, loader: DataLoader, criterion) -> float:
        self.eval()
        loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._get_device()), batch_y.to(self._get_device())
                outputs = self(batch_x)
                loss += criterion(outputs, batch_y).item()
        return loss / len(loader)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        dataset = TimeSeriesDataset(X_scaled, np.zeros((len(X), self.output_size)), self.seq_len)
        loader = DataLoader(dataset, batch_size=256)

        preds = []
        self.eval()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self._get_device())
                out = self(batch_x)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("predict_proba only for classification")
        return self.predict(X)

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
