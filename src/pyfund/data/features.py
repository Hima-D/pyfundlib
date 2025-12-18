# src/pyfund/data/features.py
from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from ..indicators.macd import macd
from ..indicators.rsi import rsi
from ..indicators.sma import sma


class FeatureEngineer:
    """
    Highly flexible feature engineering class.

    Allows users to:
    - Enable/disable individual features
    - Add custom features easily
    - Control parameters per feature
    - Chain multiple feature sets
    """

    @staticmethod
    def add_technical_features(
        df: pd.DataFrame,
        *,
        # RSI
        rsi_period: int = 14,
        add_rsi: bool = True,
        # SMA
        sma_periods: Sequence[int] = (20, 50, 200),
        add_sma: bool = True,
        # MACD
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        add_macd: bool = True,
        # Returns & Volatility
        returns_periods: Sequence[int] = (1, 5, 10),
        add_returns: bool = True,
        volatility_windows: Sequence[int] = (20, 60),
        add_volatility: bool = True,
        # Volume-based (if Volume column exists)
        add_volume_features: bool = True,
        # Price-based
        add_price_ratios: bool = True,
        add_high_low: bool = True,
    ) -> pd.DataFrame:
        """
        Add a comprehensive set of technical features with full control.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least 'Close', optionally 'High', 'Low', 'Volume'

        Returns
        -------
        pd.DataFrame
            Original dataframe with new feature columns added and NaNs dropped at the end.
        """
        if not {"Close"}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'Close' column")

        close = df["Close"]
        high = df.get("High")
        low = df.get("Low")
        volume = df.get("Volume")

        # === 1. RSI ===
        if add_rsi:
            df[f"rsi_{rsi_period}"] = rsi(close, window=rsi_period)

        # === 2. Simple Moving Averages ===
        if add_sma:
            for period in sma_periods:
                df[f"sma_{period}"] = sma(close, window=period)
                if period >= 20:  # Common crossovers
                    df[f"close_vs_sma_{period}"] = close / df[f"sma_{period}"] - 1.0

        # === 3. MACD ===
        if add_macd:
            df["macd"], df["macd_signal"], df["macd_hist"] = macd(
                close, fast=macd_fast, slow=macd_slow, signal=macd_signal
            )
            df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int).diff().fillna(0)

        # === 4. Returns ===
        if add_returns:
            for p in returns_periods:
                df[f"return_{p}d"] = close.pct_change(p)

        # === 5. Volatility ===
        if add_volatility:
            daily_ret = close.pct_change()
            for window in volatility_windows:
                df[f"volatility_{window}d"] = daily_ret.rolling(window).std() * (
                    252**0.5
                )  # annualized

        # === 6. Volume Features ===
        if add_volume_features and volume is not None:
            df["volume_sma_20"] = volume.rolling(20).mean()
            df["volume_ratio"] = volume / df["volume_sma_20"]

        # === 7. High/Low/Range Features ===
        if add_high_low and high is not None and low is not None:
            df["hl_range"] = (high - low) / close
            df["close_to_high"] = (close - low) / (
                high - low + 1e-8
            )  # where price is in daily range
            df["close_to_low"] = (high - close) / (high - low + 1e-8)

        # === 8. Price Ratios & Momentum ===
        if add_price_ratios:
            for p in [5, 20, 60]:
                df[f"price_ratio_{p}d"] = close / close.shift(p) - 1.0

        return df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------------
    # Bonus: Let users register completely custom features easily
    # ------------------------------------------------------------------------
    @staticmethod
    def apply_custom_features(
        df: pd.DataFrame, features: dict[str, Callable[[pd.DataFrame], pd.Series]]
    ) -> pd.DataFrame:
        """
        Apply arbitrary user-defined features.

        Example:
            features = {
                "bb_upper": lambda df: bollinger_bands(df['Close'])[0],
                "eom": lambda df: ease_of_movement(df),
            }
            df = FeatureEngineer.apply_custom_features(df, features)
        """
        for name, func in features.items():
            df[name] = func(df)
        return df
