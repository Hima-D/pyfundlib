# src/pyfund/strategies/pairs_trading.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from ..data.fetcher import DataFetcher
from .base import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    """
    Classic Statistical Arbitrage: Pairs Trading

    Finds cointegrated pairs and trades the spread mean reversion.
    - Long the underperformer, short the overperformer when spread diverges
    - Exit when spread reverts to mean
    - Fully hedge-ratio adjusted (beta-neutral)

    Works well on ETFs, stocks in same sector, futures, crypto.
    """

    default_params = {
        "ticker_a": "XOM",
        "ticker_b": "CVX",
        "lookback_window": 252,  # Formation period (days)
        "entry_zscore": 2.0,  # Enter when |z| > 2.0
        "exit_zscore": 0.5,  # Exit when |z| < 0.5
        "stop_zscore": 4.0,  # Hard stop if |z| > 4.0
        "adf_pvalue_threshold": 0.05,  # Cointegration test threshold
        "min_hedge_ratio": 0.1,  # Avoid extreme ratios
        "rolling_window": 20,  # window for live zscore
        "min_periods": 10,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker_a: str = self.params["ticker_a"]
        self.ticker_b: str = self.params["ticker_b"]
        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std: float = 1.0
        self.is_cointegrated: bool = False

    def _check_cointegration(self, price_a: pd.Series, price_b: pd.Series) -> Tuple[bool, float]:
        """
        Test for cointegration using Engle-Granger (ADF on OLS residuals).
        Returns (is_cointegrated: bool, hedge_ratio: float).
        """
        # Align and drop NaNs
        df = pd.concat([price_a, price_b], axis=1).dropna()
        if df.shape[0] < max(100, int(self.params["lookback_window"] * 0.5)):
            return False, 1.0

        df.columns = ["A", "B"]
        # OLS: A = alpha + beta * B
        X = sm.add_constant(df["B"])
        y = df["A"]
        try:
            ols = sm.OLS(y, X).fit()
            hedge_ratio = float(ols.params.get("B", ols.params.iloc[1]))
        except Exception:
            # fallback: use simple ratio of means
            hedge_ratio = float((y.mean() / (df["B"].mean() + 1e-12)))

        # Avoid extreme hedge ratios
        if abs(hedge_ratio) < float(self.params["min_hedge_ratio"]) or not np.isfinite(hedge_ratio):
            return False, 1.0

        # Residual / spread
        spread = y - hedge_ratio * df["B"]

        # ADF test on spread residuals
        try:
            adf_res = adfuller(spread, autolag="AIC")
            p_value = float(adf_res[1])
        except Exception:
            p_value = 1.0

        is_coint = p_value < float(self.params["adf_pvalue_threshold"])
        return bool(is_coint), float(hedge_ratio)

    def _calculate_spread_zscore(self, data_a: pd.Series, data_b: pd.Series) -> pd.Series:
        """
        Calculate rolling z-score of the spread using the current hedge_ratio.
        Returns a Series aligned to data_a.index (filled with NaN where unavailable).
        """
        df = pd.concat([data_a, data_b], axis=1).dropna()
        df.columns = ["A", "B"]

        spread = df["A"] - self.hedge_ratio * df["B"]
        window = int(self.params.get("rolling_window", 20))
        minp = int(self.params.get("min_periods", 10))

        rolling_mean = spread.rolling(window=window, min_periods=minp).mean()
        rolling_std = spread.rolling(window=window, min_periods=minp).std(ddof=0)

        zscore = (spread - rolling_mean) / (rolling_std.replace(0, np.nan))
        # Reindex to original index to align with caller's data; missing values -> NaN
        zscore = zscore.reindex(data_a.index)
        return zscore

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate pair trading signals

        Returns a pd.Series indexed like `data` with values:
            +1 → Long A, Short B (spread too low)
            -1 → Short A, Long B (spread too high)
             0 → Neutral
        """
        # If caller provided Close_A/Close_B use them; else fetch historical series
        if "Close_A" in data.columns and "Close_B" in data.columns:
            price_a = data["Close_A"].astype(float)
            price_b = data["Close_B"].astype(float)
        else:
            try:
                df_a = DataFetcher.get_price(self.ticker_a, period="5y")["Close"].astype(float)
                df_b = DataFetcher.get_price(self.ticker_b, period="5y")["Close"].astype(float)
            except Exception:
                # Can't fetch → return flat signals
                return pd.Series(0, index=data.index, dtype=int)

            common_index = df_a.index.intersection(df_b.index)
            if len(common_index) == 0:
                return pd.Series(0, index=data.index, dtype=int)

            price_a = df_a.reindex(common_index)
            price_b = df_b.reindex(common_index)

        # Formation period: last lookback_window days before the final 100 trading days
        lookback = int(self.params["lookback_window"])
        reserve = 100  # reserve for out-of-sample trading
        if len(price_a) <= lookback + reserve:
            # Not enough history to form model
            self.is_cointegrated = False
            return pd.Series(0, index=data.index, dtype=int)

        formation_start = - (lookback + reserve)
        formation_end = - reserve
        form_a = price_a.iloc[formation_start:formation_end]
        form_b = price_b.iloc[formation_start:formation_end]

        # Check cointegration and set hedge ratio
        is_coint, hedge = self._check_cointegration(form_a, form_b)
        self.is_cointegrated = bool(is_coint)
        self.hedge_ratio = float(hedge) if self.is_cointegrated else 1.0

        if not self.is_cointegrated:
            # No cointegration -> stay flat
            return pd.Series(0, index=data.index, dtype=int)

        # Calculate z-score across the available price series (aligned to price_a)
        zscore_series = self._calculate_spread_zscore(price_a, price_b)

        # For the output, align zscore to the incoming `data` index.
        # If data contains Close_A/Close_B we assume same index; else map trading period indices to `data`.
        # We'll produce signals indexed by `data.index` — fill with 0 where zscore not available.
        signals = pd.Series(0, index=data.index, dtype=int)

        # Map zscore values into the data index when possible
        if "Close_A" in data.columns and "Close_B" in data.columns:
            z_aligned = zscore_series.reindex(data.index)
        else:
            # If the caller didn't provide Close_A/Close_B, we only traded on the common_index
            # Align zscore to data.index where possible (intersection), others remain NaN -> treated as 0
            z_aligned = zscore_series.reindex(data.index)

        entry_z = float(self.params["entry_zscore"])
        exit_z = float(self.params["exit_zscore"])
        stop_z = float(self.params["stop_zscore"])

        position = 0
        # Iterate through aligned zscore index
        for i, idx in enumerate(z_aligned.index):
            z_val_raw = z_aligned.iloc[i]
            z_val = float(z_val_raw) if pd.notna(z_val_raw) and np.isfinite(z_val_raw) else np.nan

            if position == 0:
                if not np.isnan(z_val) and z_val > entry_z:
                    # spread is high -> short A, long B
                    signals.iloc[i] = -1
                    position = -1
                elif not np.isnan(z_val) and z_val < -entry_z:
                    # spread is low -> long A, short B
                    signals.iloc[i] = 1
                    position = 1
            else:
                # Exit if spread back to mean or stop-loss triggered
                if np.isnan(z_val):
                    # missing z: hold (or optionally flatten); here we hold
                    signals.iloc[i] = position
                elif abs(z_val) < exit_z or abs(z_val) > stop_z:
                    signals.iloc[i] = 0
                    position = 0
                else:
                    signals.iloc[i] = position

        return signals

    def __repr__(self) -> str:
        return f"PairsTrading({self.ticker_a}-{self.ticker_b}, hedge={self.hedge_ratio:.3f}, cointegrated={self.is_cointegrated})"


# Quick test (only when run as script)
if __name__ == "__main__":
    strategy = PairsTradingStrategy(
        {"ticker_a": "KO", "ticker_b": "PEP", "entry_zscore": 2.0, "exit_zscore": 0.75}
    )

    # Try to fetch data; if unavailable, test will gracefully exit or return flat signals
    try:
        df_a = DataFetcher.get_price("KO", period="5y")["Close"]
        df_b = DataFetcher.get_price("PEP", period="5y")["Close"]
        data = pd.concat([df_a, df_b], axis=1).dropna()
        data.columns = ["Close_A", "Close_B"]
    except Exception:
        # synthetic test: correlated random walks
        idx = pd.date_range("2020-01-01", periods=1000, freq="D")
        x = np.cumsum(np.random.normal(0, 1, len(idx))) + 50
        y = x * 0.8 + np.cumsum(np.random.normal(0, 0.5, len(idx))) + 10
        data = pd.DataFrame({"Close_A": x, "Close_B": y}, index=idx)

    signals = strategy.generate_signals(data)
    print(f"Cointegrated: {strategy.is_cointegrated}")
    print(f"Hedge Ratio: {strategy.hedge_ratio:.3f}")
    print(f"Total trades (changes): {(signals.diff().abs().ne(0)).sum()}")
    print(signals.value_counts())
