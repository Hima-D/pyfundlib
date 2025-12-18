# src/pyfund/strategies/hft_micro_reversion.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.logger import logger
from .base import BaseStrategy


class HFTMicroReversionStrategy(BaseStrategy):
    """
    High-Frequency Micro-Price Mean Reversion Strategy

    Core idea:
    - MICRO-PRICE = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
    - When micro-price deviates significantly from mid => expect reversion
    - Amplified by order flow imbalance and volume spikes
    - Ultra-short holding period (seconds to minutes)
    """

    default_params: dict[str, Any] = {
        "lookback_ticks": 50,  # Number of recent ticks for stats
        "z_entry": 2.2,  # Enter when |z| > 2.2
        "z_exit": 0.3,  # Exit when |z| < 0.3
        "max_position": 1000,  # Max contracts/shares per signal
        "holding_seconds": 30,  # Force exit after 30s
        "volume_filter": 1.5,  # Only trade if volume > 1.5x avg
        "imbalance_threshold": 0.6,  # Order flow imbalance trigger
    }

    def __init__(self, ticker: str = "ES", params: dict[str, Any] | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        self.position: int = 0
        self.entry_price: float = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.tick_history: list[dict[str, Any]] = []

    @staticmethod
    def _ensure_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and ensure required columns exist."""
        df = data.copy()
        # Accept either 'Close' or 'close'
        if "Close" in df.columns and "close" not in df.columns:
            df["close"] = df["Close"].astype(float)
        elif "close" in df.columns:
            df["close"] = df["close"].astype(float)
        else:
            # fallback synthetic close if missing
            df["close"] = 0.0

        for col in ["volume", "bid_price", "ask_price", "bid_size", "ask_size"]:
            if col not in df.columns:
                # sensible defaults (avoid None operations)
                if col == "volume":
                    df[col] = 0.0
                elif col in ("bid_price", "ask_price"):
                    # set to close +/- tiny spread
                    df[col] = df["close"] * (1.0 + (0.0001 if col == "ask_price" else -0.0001))
                else:  # sizes
                    df[col] = 1.0
            # ensure types
            df[col] = df[col].astype(float)
        return df

    def _vectorized_micro_price(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized micro price:
        micro = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        Return numpy array of floats.
        """
        bid_p = df["bid_price"].to_numpy(dtype=float)
        ask_p = df["ask_price"].to_numpy(dtype=float)
        bid_s = df["bid_size"].to_numpy(dtype=float)
        ask_s = df["ask_size"].to_numpy(dtype=float)

        total = bid_s + ask_s
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            micro = (bid_s * ask_p + ask_s * bid_p) / total
            # when total is zero, fallback to mid
            mid = (bid_p + ask_p) / 2.0
            micro = np.where(np.isfinite(micro), micro, mid)
        return micro.astype(float)

    def _order_flow_imbalance(self, df: pd.DataFrame) -> float:
        """
        Compute a simple order flow imbalance (OFI) as:
        OFI = sum(last_k (volume * sign(price change)))
        Returns float
        """
        if len(df) < 2:
            return 0.0
        price = df["close"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        # price diff as numpy
        diff = np.diff(price, prepend=price[0])
        # sign: +1 if diff>0, -1 if diff<0, 0 otherwise
        sign = np.sign(diff)
        signed_vol = sign * volume
        # return last 10 ticks sum (guard in case array smaller)
        return float(signed_vol[-10:].sum())

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate HFT micro-reversion signals on tick data.

        Expected columns: ['close' or 'Close', 'volume', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        Returns pd.Series of integers: -1 (short), 0 (flat), 1 (long)
        """
        # Defensive copy & ensure datetime index
        df_in = data.copy()
        if not isinstance(df_in.index, pd.DatetimeIndex):
            try:
                df_in.index = pd.DatetimeIndex(df_in.index)
            except Exception:
                # create a synthetic datetime index if impossible
                df_in.index = pd.date_range("1970-01-01", periods=len(df_in), freq="S")

        df = self._ensure_columns(df_in)

        lookback = int(self.params["lookback_ticks"])
        if len(df) < lookback:
            return pd.Series(0, index=df.index, dtype=int)

        # Vectorized micro price & mid
        micro_arr = self._vectorized_micro_price(df)
        mid_arr = ((df["bid_price"].to_numpy(dtype=float) + df["ask_price"].to_numpy(dtype=float)) / 2.0).astype(float)
        deviation = (micro_arr - mid_arr).astype(float)

        # rolling mean/std on deviation
        deviation_series = pd.Series(deviation, index=df.index)
        dev_mean = deviation_series.rolling(window=lookback, min_periods=lookback).mean()
        dev_std = deviation_series.rolling(window=lookback, min_periods=lookback).std()
        # avoid division by zero: where std==0 -> z = nan
        z_score = (deviation_series - dev_mean) / dev_std
        z_score = z_score.replace([np.inf, -np.inf], np.nan)

        # volume spike metric
        vol_series = df["volume"]
        avg_volume = vol_series.rolling(20, min_periods=1).mean()
        volume_spike = vol_series / (avg_volume + 1e-12)  # avoid div by zero

        # precompute OFI (global for simplicity, could be rolling)
        ofi = self._order_flow_imbalance(df)

        signals = pd.Series(0, index=df.index, dtype=int)

        # iterate from lookback to end (we require lookback window)
        for i in range(lookback, len(df)):
            idx = df.index[i]
            # safe float extraction for z and volume spike
            z_val = float(z_score.iloc[i]) if not pd.isna(z_score.iloc[i]) else np.nan
            vol_val = float(volume_spike.iloc[i]) if not pd.isna(volume_spike.iloc[i]) else np.nan

            # volume filter boolean: allow if vol unknown (nan) or vol_val > threshold
            vol_ok = np.isnan(vol_val) or (vol_val > float(self.params["volume_filter"]))

            # imbalance normalized to recent volume (10 ticks window)
            recent_vol = float(df["volume"].iloc[max(0, i - 10) : i].sum() + 1e-9)
            imbalance = float(abs(ofi)) / recent_vol if recent_vol > 0 else 0.0

            current_time = pd.Timestamp(idx)

            # Force exit on holding time
            if self.position != 0 and self.entry_time is not None:
                try:
                    held_seconds = (current_time - self.entry_time).total_seconds()
                except Exception:
                    held_seconds = 0.0
                if held_seconds > float(self.params["holding_seconds"]):
                    # exit
                    signals.iloc[i] = 0
                    self.position = 0
                    self.entry_price = 0.0
                    self.entry_time = None
                    continue

            # Exit condition by z-score threshold
            if self.position != 0 and not np.isnan(z_val) and abs(z_val) < float(self.params["z_exit"]):
                signals.iloc[i] = 0
                self.position = 0
                self.entry_price = 0.0
                self.entry_time = None
                continue

            # Entry conditions (only enter when flat)
            if self.position == 0 and not np.isnan(z_val) and vol_ok:
                if z_val > float(self.params["z_entry"]):
                    # micro-price too high => short to capture reversion
                    signals.iloc[i] = -1
                    self.position = -1
                    self.entry_price = float(df["close"].iloc[i])
                    self.entry_time = current_time
                elif z_val < -float(self.params["z_entry"]):
                    # micro-price too low => long for reversion up
                    signals.iloc[i] = 1
                    self.position = 1
                    self.entry_price = float(df["close"].iloc[i])
                    self.entry_time = current_time

            # If already in position, propagate it (holding)
            elif self.position != 0:
                signals.iloc[i] = self.position

        # final log line
        try:
            last_z = float(z_score.iloc[-1]) if not pd.isna(z_score.iloc[-1]) else np.nan
        except Exception:
            last_z = np.nan
        logger.info(
            f"HFT Micro Reversion | Z-score(last): {last_z:.2f} | Position: {self.position} | Signal(last): {int(signals.iloc[-1])}"
        )

        return signals

    def __repr__(self) -> str:
        return f"HFTMicroReversion({self.ticker}, pos={self.position})"


# Live test (simulated tick data)
if __name__ == "__main__":
    # Try to fetch 1-day 1-min data; if DataFetcher fails, simulate
    try:
        ticker = "BTC-USD"
        df = DataFetcher.get_price(ticker, period="1d", interval="1m")
        # Ensure datetimes
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)
        # If the API returns 'Close' column, keep it; strategy _ensure_columns handles it
    except Exception:
        # Simulate 1 minute ticks for a trading day (~1440)
        n = 1440
        idx = pd.date_range("2025-01-01 09:15:00", periods=n, freq="T")  # minute frequency
        price = 30000.0 + np.cumsum(np.random.normal(scale=5.0, size=n))
        df = pd.DataFrame({"Close": price, "volume": np.abs(np.random.normal(20, 10, size=n))}, index=idx)

    # Inject simple order book columns if missing (small random spreads/sizes)
    if "bid_price" not in df.columns or "ask_price" not in df.columns:
        close = df["Close"] if "Close" in df.columns else df["close"]
        df["bid_price"] = (close * (1 - np.random.uniform(0.0001, 0.0006, size=len(df))))
        df["ask_price"] = (close * (1 + np.random.uniform(0.0001, 0.0006, size=len(df))))
    if "bid_size" not in df.columns:
        df["bid_size"] = np.random.uniform(0.1, 10.0, size=len(df))
    if "ask_size" not in df.columns:
        df["ask_size"] = np.random.uniform(0.1, 10.0, size=len(df))
    if "volume" not in df.columns:
        df["volume"] = np.abs(np.random.normal(20, 10, size=len(df)))

    strategy = HFTMicroReversionStrategy("BTC-USD", {"z_entry": 2.0, "holding_seconds": 60})
    signals = strategy.generate_signals(df)
    # count non-zero signals as entries/exits (not exact trade count but indicative)
    trades = (signals.astype(int).diff().abs() > 0).sum()
    print(f"Total signals generated (changes): {int(trades)}")
    print(f"Final position: {int(signals.iloc[-1])}")
    print(signals.value_counts())
