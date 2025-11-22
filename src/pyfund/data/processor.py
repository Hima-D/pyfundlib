# src/pyfund/data/processor.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

OhlcvAgg = Literal["open", "high", "low", "close", "volume"]
Rule = str  # e.g., "1W", "1d", "30min", "1M", etc.


class DataProcessor:
    """
    Comprehensive OHLCV data processing utilities.
    Handles resampling, cleaning, alignment, and feature-ready preparation.
    """

    @staticmethod
    def resample(
        df: DataFrame,
        rule: Rule = "1W",
        *,
        closed: Literal["left", "right"] = "left",
        label: Literal["left", "right"] = "left",
        custom_agg: dict[str | str] | None = None,
        min_periods: int = 1,
    ) -> DataFrame:
        """
        Resample OHLCV data with smart defaults.

        Parameters
        ----------
        df : pd.DataFrame
            Must have DatetimeIndex and OHLCV columns
        rule : str
            Pandas offset alias: '1W', '1d', '30min', '1M', etc.
        closed / label : str
            Interval closed side and label position
        custom_agg : dict, optional
            Override default aggregation, e.g.:
            {'Close': 'last', 'Volume': 'mean'}
        min_periods : int
            Minimum observations required per bin

        Returns
        -------
        pd.DataFrame
            Resampled and aggregated OHLCV DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        default_agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }

        if custom_agg:
            default_agg.update(custom_agg)

        # Only aggregate columns that exist
        agg_dict = {col: agg for col, agg in default_agg.items() if col in df.columns}

        return df.resample(rule, closed=closed, label=label).agg(agg_dict).dropna(how="all")

    @staticmethod
    def clean_and_fill(
        df: DataFrame,
        *,
        method: Literal["ffill", "bfill", "both", "interpolate", "drop"] = "both",
        limit: int | None = None,
        freq: str | None = None,
        max_gap_fill: int | None = 5,
    ) -> DataFrame:
        """
        Clean missing data with multiple strategies.

        Parameters
        ----------
        method :
            - 'ffill': forward fill only
            - 'bfill': backward fill (good for leading NaNs)
            - 'both': ffill + bfill (most common)
            - 'interpolate': linear/time interpolation
            - 'drop': drop rows with any NaN
        limit : int, optional
            Max consecutive periods to fill
        freq : str, optional
            If provided, reindex to exact frequency first
        max_gap_fill : int
            Never fill gaps larger than this (in periods)

        Examples
        --------
        df = DataProcessor.clean_and_fill(df, method="both")
        df = DataProcessor.clean_and_fill(df, method="interpolate")
        df = DataProcessor.clean_and_fill(df, freq="1d", method="ffill", max_gap_fill=3)
        """
        df = df.copy()

        if freq:
            # Reindex to perfect calendar (business days, etc.)
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            df = df.reindex(date_range)

        if method == "drop":
            return df.dropna()

        if method == "interpolate":
            return df.interpolate(method="time", limit=limit, limit_direction="both")

        # Forward/backward fill with gap protection
        if max_gap_fill is not None:
            # Fill only small gaps
            df = df.where(df.notnull(), np.nan)  # ensure clean NaNs
            mask = df.notnull()
            for col in df.columns:
                # Find large gaps and prevent filling across them
                gap_mask = mask[col].diff().fillna(True).cumsum()
                df[col] = df.groupby(gap_mask)[col].transform(
                    lambda x: x.ffill(limit=max_gap_fill) if method in ["ffill", "both"] else x
                )
                if method in ["bfill", "both"]:
                    df[col] = df[col].bfill(limit=max_gap_fill)

        # Simple fill methods
        if method == "ffill":
            df = df.ffill(limit=limit)
        elif method == "bfill":
            df = df.bfill(limit=limit)
        elif method == "both":
            df = df.ffill(limit=limit).bfill(limit=limit)

        return df

    @staticmethod
    def align_multiple(dfs: dict[str, DataFrame], fill_method: str = "ffill") -> DataFrame:
        """
        Align multiple ticker DataFrames on the same index (e.g., for multi-asset backtesting).

        Parameters
        ----------
        dfs : dict
            {ticker: df} mapping
        fill_method : str
            How to fill missing dates per ticker

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (ticker, field)
        """
        aligned = pd.concat(
            {
                ticker: DataProcessor.clean_and_fill(df, method=fill_method)
                for ticker, df in dfs.items()
            },
            axis=1,
        )
        return aligned.sort_index()

    @staticmethod
    def add_log_returns(
        df: DataFrame, price_col: str = "Close", name: str = "log_return"
    ) -> DataFrame:
        """Add log returns column."""
        df = df.copy()
        df[name] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    @staticmethod
    def ensure_business_days(df: DataFrame) -> DataFrame:
        """Reindex to business days only (removes weekends/holidays if present)."""
        df = df.copy()
        df = df[df.index.dayofweek < 5]  # Mon-Fri only
        return df.asfreq("B")  # Business days
