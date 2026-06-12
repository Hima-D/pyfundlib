"""Data fetching and caching module for PyFundLib."""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetching."""

    def __init__(self, source: str = "yfinance", cache_dir: str = "./cache"):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_ohlcv(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass


class YFinanceFetcher(DataFetcher):
    """Yahoo Finance data fetcher."""

    def fetch_ohlcv(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV from yfinance."""
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise


class DataCached(YFinanceFetcher):
    """Data fetcher with Parquet caching and compression."""

    def __init__(
        self,
        source: str = "yfinance",
        cache_dir: str = "./cache",
        compression: str = "zstd",
    ):
        super().__init__(source, cache_dir)
        self.compression = compression

    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        cache_name = f"{ticker}_{start_date}_{end_date}.parquet"
        return self.cache_dir / cache_name

    def fetch_ohlcv(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV with caching."""
        if isinstance(tickers, str):\n            tickers = [tickers]\n\n        all_data = {}\n        for ticker in tickers:\n            cache_path = self._get_cache_path(ticker, start_date, end_date)\n\n            # Try to load from cache\n            if use_cache and cache_path.exists():\n                try:\n                    df = pd.read_parquet(cache_path)\n                    logger.info(f\"Loaded {ticker} from cache\")\n                    all_data[ticker] = df\n                    continue\n                except Exception as e:\n                    logger.warning(f\"Cache read failed: {e}\")\n\n            # Fetch fresh data\n            df = super().fetch_ohlcv(ticker, start_date, end_date, interval)\n            df.index.name = \"date\"\n            if isinstance(df.index, pd.MultiIndex):\n                df = df.reset_index()\n            else:\n                df[\"ticker\"] = ticker\n\n            # Save to cache\n            try:\n                df.to_parquet(cache_path, compression=self.compression)\n                logger.info(f\"Cached {ticker}\")\n            except Exception as e:\n                logger.warning(f\"Cache write failed: {e}\")\n\n            all_data[ticker] = df\n\n        # Combine all data\n        if len(all_data) == 1:\n            return list(all_data.values())[0]\n        return pd.concat(all_data.values(), ignore_index=False)\n\n\nclass UniverseManager:\n    \"\"\"Manages trading universe and asset metadata.\"\"\"\n\n    def __init__(self, universe: Optional[List[str]] = None):\n        self.universe = list(set(universe or []))  # Remove duplicates\n        self._metadata: Dict[str, Dict[str, Any]] = {}\n\n    def add_assets(self, assets: List[str]) -> None:\n        \"\"\"Add assets to universe.\"\"\"\n        self.universe.extend(assets)\n        self.universe = list(set(self.universe))\n        logger.info(f\"Added assets. Universe size: {len(self.universe)}\")\n\n    def remove_assets(self, assets: List[str]) -> None:\n        \"\"\"Remove assets from universe.\"\"\"\n        self.universe = [a for a in self.universe if a not in assets]\n        logger.info(f\"Removed assets. Universe size: {len(self.universe)}\")\n\n    def get_universe(self) -> List[str]:\n        \"\"\"Get current universe.\"\"\"\n        return self.universe.copy()\n\n    def set_metadata(self, ticker: str, **kwargs) -> None:\n        \"\"\"Set metadata for asset.\"\"\"\n        self._metadata[ticker] = kwargs\n\n    def get_metadata(self, ticker: str) -> Dict[str, Any]:\n        \"\"\"Get metadata for asset.\"\"\"\n        return self._metadata.get(ticker, {})\n\n    def size(self) -> int:\n        \"\"\"Get universe size.\"\"\"\n        return len(self.universe)
