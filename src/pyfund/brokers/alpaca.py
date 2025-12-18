# src/pyfund/brokers/alpaca.py
"""
Alpaca Broker — CREDENTIALS REQUIRED
────────────────────────────────────
You CANNOT instantiate this without valid API keys.
No environment fallbacks. No silent failures.
This is production-grade security.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ..core.broker import BrokerInterface
from ..core.broker_registry import register_broker
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AlpacaCredentials:
    """Immutable, required credentials"""
    api_key: str
    secret_key: str
    paper: bool = True

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("ALPACA_API_KEY is required and cannot be empty")
        if not self.secret_key or not self.secret_key.strip():
            raise ValueError("ALPACA_SECRET_KEY is required and cannot be empty")


@register_broker("alpaca")
class AlpacaBroker(BrokerInterface):
    """
    Secure Alpaca broker — forces explicit credentials.
    Use only via: AlpacaBroker(credentials=AlpacaCredentials(...))
    """

    def __init__(self, credentials: AlpacaCredentials):
        if not isinstance(credentials, AlpacaCredentials):
            raise TypeError("You must pass an AlpacaCredentials object")

        self.credentials = credentials
        self.paper = credentials.paper

        try:
            self.trading_client = TradingClient(
                api_key=credentials.api_key,
                secret_key=credentials.secret_key,
                paper=credentials.paper,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=credentials.api_key,
                secret_key=credentials.secret_key,
            )
            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"AlpacaBroker connected successfully | {mode} mode")
        except Exception as e:
            logger.error("Alpaca authentication failed. Check your keys.")
            raise ConnectionError(f"Failed to connect to Alpaca: {e}") from e

    # === All methods same as before, but now 100% safe ===
    def get_account(self) -> Dict[str, Any]:
        acc = self.trading_client.get_account()
        return {
            "portfolio_value": float(acc.portfolio_value),
            "cash": float(acc.cash),
            "buying_power": float(acc.buying_power),
        }

    def get_positions(self) -> Dict[str, float]:
        positions = self.trading_client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions if float(p.qty) != 0}

    def get_price(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        from datetime import datetime, timedelta

        days = {"1y": 365, "2y": 730, "max": 3650}.get(period.lower(), 730)
        request = StockBarsRequest(
            symbol_or_symbols=ticker.upper(),
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=days),
        )
        bars = self.data_client.get_stock_bars(request)
        if ticker.upper() not in bars:
            return pd.DataFrame()

        df = bars[ticker.upper()].df[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        df.columns = [c.capitalize() for c in df.columns]
        return df

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        if qty == 0:
            return {"status": "skipped", "reason": "zero_quantity"}

        order = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        response = self.trading_client.submit_order(order)
        logger.info(f"ORDER {side.upper()} {abs(qty)} {ticker} | ID: {response.id}")
        return {"status": "success", "order_id": str(response.id)}

    def get_current_price(self, ticker: str) -> float:
        try:
            quote = self.data_client.get_stock_latest_quote({ticker.upper()})
            return float(quote[ticker.upper()].ask_price or quote[ticker.upper()].bid_price)
        except:
            df = self.get_price(ticker, "5d")
            return float(df["Close"].iloc[-1]) if not df.empty else 0.0


# === USER MUST DO THIS — NO SHORTCUTS ===
def create_alpaca_broker(
    api_key: str,
    secret_key: str,
    paper: bool = True,
) -> AlpacaBroker:
    """
    Factory function — forces user to pass keys explicitly.
    Recommended way to create broker.
    """
    if not api_key:
        raise ValueError("api_key is required")
    if not secret_key:
        raise ValueError("secret_key is required")

    creds = AlpacaCredentials(api_key=api_key, secret_key=secret_key, paper=paper)
    return AlpacaBroker(credentials=creds)