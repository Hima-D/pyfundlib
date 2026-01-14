from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy


class TimeSeriesMomentumStrategy(BaseStrategy):
    default_params: dict[str, Any] = {
        "lookback_days": 252,
        "vol_lookback_days": 63,
        "vol_target": 0.20,
        "neutral_threshold": 0.0,
    }

    def __init__(self, params: Optional[dict[str, Any]] = None):
        super().__init__(params)
        self.lookback_days = int(self.params["lookback_days"])
        self.vol_lookback_days = int(self.params["vol_lookback_days"])
        self.vol_target = float(self.params["vol_target"])
        self.neutral_threshold = float(self.params["neutral_threshold"])

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "Close" not in data.columns:
            raise ValueError("Data must contain 'Close' column")

        close = data["Close"].astype(float)
        returns = close.pct_change().fillna(0.0)

        rolling_ret = close.pct_change(self.lookback_days)

        vol = returns.rolling(self.vol_lookback_days).std() * np.sqrt(252)
        vol = vol.replace(0.0, np.nan).ffill().fillna(method="bfill")

        raw_signal = np.sign(rolling_ret)

        scaled_signal = raw_signal * (self.vol_target / vol.clip(lower=1e-6))
        scaled_signal = scaled_signal.clip(-1.0, 1.0)

        if self.neutral_threshold > 0.0:
            mask = scaled_signal.abs() < self.neutral_threshold
            scaled_signal = scaled_signal.where(~mask, 0.0)

        signals = scaled_signal.replace(0.0, np.nan).ffill().fillna(0.0)
        return pd.Series(signals, index=data.index)

