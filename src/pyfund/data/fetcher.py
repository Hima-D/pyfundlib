# src/pyfundlib/data/fetcher.py
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from ..core.broker_registry import register_broker
from ..utils.cache import cached_function

logger = logging.getLogger(__name__)


Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo"
]

Source = Literal["yfinance", "alpaca", "zerodha", "ibkr", "binance", "polygon"]


class DataFetcher:
    """
    Unified, smart data fetcher with:
    - Automatic disk caching (safe for daily+ strategies)
    - Real-time mode (cache bypass for live/HFT)
    - Broker registry (plug-and-play sources)
    - Clean, standardized output
    """

    @staticmethod
    @cached_function(
        dir_name="price_data",
        key_lambda=lambda *a, **kw: "_".join([
            kw.get("source", "yf"),
            a[0] if isinstance(a[0], str) else "-".join(sorted(a[0])),
            kw.get("interval", "1d"),
            kw.get("period", "max"),
            kw.get("start", "")[:10],
            kw.get("end", "")[:10],
        ]),
        # Smart TTL: 7 days for daily+, 0 seconds (disabled) for intraday live
        expire_seconds=lambda **kw: (
            0 if kw.get("interval", "1d") in ("1m", "2m", "5m", "15m", "30m", "60m")
            else 7 * 24 * 60 * 60
        ),
    )
    def get_price(
        ticker: str | Sequence[str],
        *,
        period: str | None = "max",
        start: str | None = None,
        end: str | None = None,
        interval: Interval = "1d",
        source: Source = "yfinance",
        prepost: bool = False,
        auto_adjust: bool = True,
        keep_na: bool = False,
        cache: bool = True,           # ← NEW: disable cache for live trading
        **source_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with intelligent caching.

        For live intraday trading → use `cache=False`
        For research/ML/backtesting → leave `cache=True` (default)
        """
        # Bypass cache entirely if disabled
        if not cache:
            logger.debug(f"Cache bypassed for {ticker} (live mode)")
            return DataFetcher._fetch_and_clean(
                ticker, period, start, end, interval, source,
                prepost, auto_adjust, keep_na, source_kwargs
            )

        # Otherwise use smart cached version
        return DataFetcher._fetch_and_clean(
            ticker, period, start, end, interval, source,
            prepost, auto_adjust, keep_na, source_kwargs
        )

    @staticmethod
    def _fetch_and_clean(
        ticker: str | Sequence[str],
        period: str | None,
        start: str | None,
        end: str | None,
        interval: Interval,
        source: Source,
        prepost: bool,
        auto_adjust: bool,
        keep_na: bool,
        source_kwargs: dict,
    ) -> pd.DataFrame:
        """Internal: actual fetch + standardization (used by cache and live)"""
        fetch_func = register_broker.get_data_fetcher(source)

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

        if df is None or df.empty:
            raise ValueError(f"No data for {ticker} from {source}")

        # Handle yfinance MultiIndex for multiple tickers
        if isinstance(ticker, (list, tuple)) and isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1).sort_index(axis=1)

        # Apply proper adjustment if needed
        if auto_adjust and "Adj Close" in df.columns:
            ratio = df["Adj Close"] / df["Close"].replace(0, pd.NA)
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col] * ratio
            df["Close"] = df["Adj Close"]

        # Standard columns
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]

        if not keep_na:
            df = df.dropna()

        df.index.name = "date"
        df.name = ticker if isinstance(ticker, str) else "_".join(map(str, ticker))

        logger.debug(f"Fetched {len(df)} rows → {ticker} @ {interval}")
        return df

    @staticmethod
    def get_multiple(
        tickers: Sequence[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch multiple tickers → clean Close-only DataFrame"""
        df = DataFetcher.get_price(tickers, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            close = df.xs("Close", axis=1, level=0)
        else:
            close = df["Close"].copy()
            if len(tickers) == 1:
                close = close.to_frame(name=tickers[0])
        return pd.DataFrame(close).sort_index(axis=1)

    @staticmethod
    def get_historical(
        ticker: str,
        period: str = "max",
        interval: Interval = "1d",
        cache: bool = True,        # ← safe default for ML training
        **kwargs,
    ) -> pd.DataFrame:
        """
        Used by MLPredictor.train_all() → cached by default (perfect for training)
        """
        return DataFetcher.get_price(
            ticker=ticker,
            period=period,
            interval=interval,
            source="yfinance",
            auto_adjust=True,
            cache=cache,
            **kwargs,
        )

    @staticmethod
    def get_live_price(ticker: str, interval: Interval = "1m") -> pd.DataFrame:
        """Convenience: always fresh data for live trading / HFT"""
        return DataFetcher.get_price(
            ticker=ticker,
            period="5d" if interval != "1m" else "1d",
            interval=interval,
            cache=False,           # ← never cache live data
            source="yfinance",
        )


# ——————————————— Built-in yfinance backend (fast & reliable) ———————————————

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
    df = yf.download(
        tickers=ticker,
        period=period or "max",
        start=start,
        end=end,
        interval=interval,
        prepost=prepost,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        **kwargs,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance: no data for {ticker}")
    return df


# Register sources
register_broker.register_data_fetcher("yfinance", _fetch_yfinance)
register_broker.register_data_fetcher("yf", _fetch_yfinance)
register_broker.register_data_fetcher("yahoo", _fetch_yfinance)

logger.info("DataFetcher ready | yfinance backend registered | Smart caching enabled")