# src/pyfund/core/broker.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]
BrokerMode = Literal["live", "paper", "backtest"]


class BrokerError(Exception):
    """Base exception for all broker-related errors"""


class AuthenticationError(BrokerError):
    """Raised when broker authentication fails"""


class OrderError(BrokerError):
    """Raised when order placement fails"""


class Broker(ABC):
    """
    Universal Broker Interface - The Gold Standard

    One interface → works with Zerodha, Alpaca, Interactive Brokers, Binance, Upstox, etc.
    Zero code changes when switching brokers.
    """

    def __init__(
        self,
        mode: BrokerMode = "paper",
        name: str = "generic",
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        self.mode = mode
        self.name = name.lower()
        self.max_retries = max_retries
        self.timeout = timeout
        self.is_connected = False
        logger.info(f"{self.__class__.__name__} initialized in {mode.upper()} mode")

    @abstractmethod
    def connect(self) -> None:
        """Establish connection and authenticate"""

    @abstractmethod
    def disconnect(self) -> None:
        """Clean shutdown"""

    @abstractmethod
    def get_price(
        self,
        ticker: str,
        period: str | None = None,
        interval: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data

        Returns standardized DataFrame with columns:
        ['Open', 'High', 'Low', 'Close', 'Volume']
        Index: DatetimeIndex (timezone-aware if possible)
        """

    @abstractmethod
    def get_balance(self) -> dict[str | float]:
        """Return cash + margin balances"""

    @abstractmethod
    def get_positions(self) -> dict[str, dict[str, Any]]:
        """
        Return current positions with full details
        Example: {"RELIANCE": {"qty": 100, "avg_price": 2500.0, "pnl": 5000.0}}
        """

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = "market",
        price: float | None = None,
        time_in_force: TimeInForce = "day",
        tag: str | None = None,
    ) -> dict[str, Any]:
        """
        Place order and return order confirmation
        Should raise OrderError on failure
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders, return count canceled"""

    @abstractmethod
    def get_open_orders(self) -> list[dict[str, Any]]:
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _validate_ticker(self, ticker: str) -> str:
        """Normalize ticker format"""
        return ticker.upper().strip()

    def _safe_call(self, func, *args, **kwargs):
        """Retry wrapper with exponential backoff"""
        import time

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    raise
                wait = (2**attempt) * 0.5
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    # Convenience methods
    def buy(self, ticker: str, qty: float, **kwargs):
        return self.place_order(ticker, qty, "buy", **kwargs)

    def sell(self, ticker: str, qty: float, **kwargs):
        return self.place_order(ticker, qty, "sell", **kwargs)

    def flatten_position(self, ticker: str):
        positions = self.get_positions()
        if ticker in positions and positions[ticker]["qty"] != 0:
            qty = -positions[ticker]["qty"]
            side = "sell" if qty > 0 else "buy"
            return self.place_order(ticker, abs(qty), side, order_type="market")

    def flatten_all(self):
        count = 0
        for ticker in self.get_positions().keys():
            if self.flatten_position(ticker):
                count += 1
        self.cancel_all_orders()
        return count


# Example usage in live trading
if __name__ == "__main__":
    # This will work with ANY broker that implements this interface
    with Broker(mode="paper") as broker:
        df = broker.get_price("RELIANCE")
        print(df.tail())
        balance = broker.get_balance()
        print(f"Cash: ₹{balance.get('cash', 0):,.2f}")
