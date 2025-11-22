# src/pyfundlib/ml/pipelines/feature_pipeline.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ...indicators.macd import macd
from ...indicators.rsi import rsi
from ...indicators.sma import sma
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStep:
    """A single transform step in the pipeline"""

    name: str
    func: Callable[[pd.DataFrame], pd.DataFrame]
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df
        logger.debug(f"Applying pipeline step: {self.name} {self.params}")
        return self.func(df, **self.params)


class FeaturePipeline:
    """
    The most powerful, flexible, and reusable feature engineering pipeline in quant finance.
    Chain any indicators, ratios, lags, rolling stats, or custom functions.
    """

    def __init__(self, name: str = "default_pipeline", version: str = "1.0"):
        self.name = name
        self.version = version
        self.steps: list[PipelineStep] = []
        self.fitted = False
        self.feature_names_: list[str] | None = None

    def add_step(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        params: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> FeaturePipeline:
        """Add a custom step"""
        step = PipelineStep(name=name, func=func, params=params or {}, enabled=enabled)
        self.steps.append(step)
        return self

    def add_technical_indicators(
        self,
        rsi_periods: list[int | None = None,
        sma_periods: list[int | None = None,
        macd_params: tuple = (12, 26, 9),
        include_volume: bool = True,
    ) -> FeaturePipeline:
        """Add standard technical indicators"""
        rsi_periods = rsi_periods or [14]
        sma_periods = sma_periods or [20, 50, 200]

        def tech_features(df: pd.DataFrame) -> pd.DataFrame:
            close = df["Close"]
            if "Volume" in df.columns and include_volume:
                volume = df["Volume"]

            for p in rsi_periods:
                df[f"rsi_{p}"] = rsi(close, window=p)

            for p in sma_periods:
                df[f"sma_{p}"] = sma(close, window=p)
                df[f"close_sma_ratio_{p}"] = close / df[f"sma_{p}"]

            df["macd"], df["macd_signal"], df["macd_hist"] = macd(close, *macd_params)

            if include_volume:
                df["volume_sma_20"] = volume.rolling(20).mean()
                df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

            df["hl_ratio"] = (df["High"] - df["Low"]) / df["Close"]
            df["oc_ratio"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)

            return df

        self.add_step("technical_indicators", tech_features)
        return self

    def add_returns_and_volatility(
        self,
        return_periods: list[int | None = None,
        vol_windows: list[int | None = None,
        annualize_vol: bool = True,
    ) -> FeaturePipeline:
        return_periods = return_periods or [1, 5, 10, 20]
        vol_windows = vol_windows or [20, 60]

        def returns_vol(df: pd.DataFrame) -> pd.DataFrame:
            close = df["Close"]
            daily_ret = close.pct_change()

            for p in return_periods:
                df[f"ret_{p}d"] = close.pct_change(p)

            for w in vol_windows:
                vol = daily_ret.rolling(w).std()
                if annualize_vol:
                    vol = vol * np.sqrt(252)
                df[f"vol_{w}d"] = vol

            df["skew_60d"] = daily_ret.rolling(60).skew()
            df["kurt_60d"] = daily_ret.rolling(60).kurt()

            return df

        self.add_step("returns_volatility", returns_vol)
        return self

    def add_lags(
        self,
        columns: list[str],
        lags: list[int | None = None,
    ) -> FeaturePipeline:
        lags = lags or [1, 2, 3, 5, 10]

        def lag_features(df: pd.DataFrame) -> pd.DataFrame:
            for col in columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            return df

        self.add_step(f"lags_{'_'.join(columns)}", lag_features)
        return self

    def add_rolling_stats(
        self,
        columns: list[str],
        windows: list[int | None = None,
        stats: list[str | None = None,
    ) -> FeaturePipeline:
        windows = windows or [5, 10, 20, 60]
        stats = stats or ["mean", "std", "min", "max"]

        def rolling(df: pd.DataFrame) -> pd.DataFrame:
            for col in columns:
                series = df[col]
                for w in windows:
                    rolling = series.rolling(w)
                    if "mean" in stats:
                        df[f"{col}_roll_mean_{w}"] = rolling.mean()
                    if "std" in stats:
                        df[f"{col}_roll_std_{w}"] = rolling.std()
                    if "min" in stats:
                        df[f"{col}_roll_min_{w}"] = rolling.min()
                    if "max" in stats:
                        df[f"{col}_roll_max_{w}"] = rolling.max()
            return df

        self.add_step("rolling_stats", rolling)
        return self

    def add_zscore(
        self,
        columns: list[str] | None = None,
        window: int = 60,
    ) -> FeaturePipeline:
        columns = columns or ["Close"]

        def zscore_features(df: pd.DataFrame) -> pd.DataFrame:
            for col in columns:
                rolling_mean = df[col].rolling(window).mean()
                rolling_std = df[col].rolling(window).std()
                df[f"{col}_zscore_{window}"] = (df[col] - rolling_mean) / rolling_std
            return df

        self.add_step(f"zscore_{window}", zscore_features)
        return self

    def add_custom_step(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> FeaturePipeline:
        """Add any custom function"""
        self.add_step(name, func, params)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps and store feature names"""
        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)

        df_out = df_out.dropna() if len(df_out) > 0 else df_out
        self.feature_names_ = [c for c in df_out.columns if c not in df.columns]
        self.fitted = True

        logger.info(
            f"FeaturePipeline '{self.name}' applied | {len(self.feature_names_)} features created"
        )
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pipeline (assumes already fitted)"""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted first with fit_transform()")

        df_out = df.copy()
        for step in self.steps:
            df_out = step(df_out)
        return df_out[list(df.columns) + self.feature_names_]

    def get_feature_names(self) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")
        return self.feature_names_

    def summary(self) -> None:
        print(f"\n=== FeaturePipeline: {self.name} v{self.version} ===")
        print(f"Steps: {len(self.steps)}")
        for step in self.steps:
            status = "ON" if step.enabled else "OFF"
            print(f"  [{status}] {step.name}")
        print(f"Features generated: {len(self.feature_names_) if self.fitted else 0}")
        if self.fitted:
            print(f"Sample features: {self.feature_names_[:10]}...")
        print("=" * 60)


# Pre-built elite pipelines
def get_default_pipeline() -> FeaturePipeline:
    return (
        FeaturePipeline(name="alpha_v1")
        .add_technical_indicators(rsi_periods=[7, 14, 21], sma_periods=[10, 20, 50, 200])
        .add_returns_and_volatility(return_periods=[1, 3, 5, 10, 20], vol_windows=[10, 20, 60])
        .add_lags(["Close", "Volume"], lags=[1, 2, 3, 5])
        .add_rolling_stats(["Close"], windows=[10, 20, 60], stats=["mean", "std"])
        .add_zscore(["Close", "Volume"], window=60)
    )
