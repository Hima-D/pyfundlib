# src/pyfund/data/fetcher.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from ..core.broker_registry import (
    broker_registry,  # You'll create this next if not exists
)
from ..utils.cache import cached_function

Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]

Source = Literal["yfinance", "alpaca", "zerodha", "ibkr", "binance", "polygon"]


class DataFetcher:
    """
    Unified, broker-agnostic price data fetcher with caching and fallbacks.
    """

    @staticmethod
    @cached_function(
        dir_name="price_data",
        key_lambda=lambda *a, **kw: f"{kw.get('source','yf')}_{a[0] if isinstance(a[0], str) else '_'.join(a[0])}_{kw.get('interval','1d')}_{kw.get('period','2y') or kw.get('start','')}",
    )
    def get_price(
        ticker: str | Sequence[str],
        *,
        period: str | None = "2y",
        start: str | None = None,
        end: str | None = None,
        interval: Interval = "1d",
        source: Source = "yfinance",
        prepost: bool = False,
        auto_adjust: bool = True,
        keep_na: bool = False,
        **source_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch price data from any supported source with automatic caching.

        Parameters
        ----------
        ticker : str or list[str]
            Single ticker or list of tickers
        period : str, optional
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        start/end : str, optional
            Override period with exact dates (YYYY-MM-DD)
        interval : str
            Data interval
        source : str
            Data source: 'yfinance' (default), 'alpaca', 'zerodha', etc.
        prepost : bool
            Include pre/post market hours (where supported)
        auto_adjust : bool
            Adjust OHLC for splits/dividends
        source_kwargs : dict
            Extra arguments passed directly to the broker fetcher

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex
            Columns: Open, High, Low, Close, Volume, (Adj Close if not auto_adjusted)
        """
        # Resolve the actual fetcher function from registry
        fetch_func = broker_registry.get_data_fetcher(source)

        df = fetch_func(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            interval=interval,
            prepost=prepost,
            auto_adjust=auto_adjust,
            **source_kwargs,
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker} from {source}")

        # Standardize columns and index
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns MultiIndex for multiple tickers
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
            if auto_adjust:
                df = df.xs("Close", axis=1, level=1, drop_level=True).pct_change().add(1).cumprod()
                # Better to re-fetch adjusted properly — but this is fallback

        # Final cleanup
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if not keep_na:
            df = df.dropna()

        df.name = ticker if isinstance(ticker, str) else "_".join(ticker)
        return df

    @staticmethod
    def get_multiple(
        tickers: Sequence[str],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convenience: fetch multiple tickers and return clean Close-only DataFrame.
        """
        df = DataFetcher.get_price(tickers, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].sort_index(axis=1)
        return df["Close"]


# ———————————————————————— Default yfinance implementation ————————————————————————
def _fetch_yfinance(
    ticker: str | Sequence[str],
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    prepost: bool = False,
    auto_adjust: bool = True,
    **kwargs,
) -> pd.DataFrame:
    return yf.download(
        tickers=ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        prepost=prepost,
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
        **kwargs,
    )


# Register built-in sources (you can move this to broker_registry init)
broker_registry.register_data_fetcher("yfinance", _fetch_yfinance)
# Later you’ll do:
# broker_registry.register_data_fetcher("alpaca", alpaca_fetch_function)
# broker_registry.register_data_fetcher("zerodha", zerodha_fetch_function)
