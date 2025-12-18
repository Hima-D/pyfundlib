# src/pyfund/brokers/zerodha.py
"""
Zerodha Kite Connect Broker — CREDENTIALS REQUIRED
────────────────────────────────────────────────────
Same security & design as AlpacaBroker.
You CANNOT use this without valid API key + access token.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from kiteconnect import KiteConnect

from ..core.broker import Broker
from ..core.broker_registry import register_broker
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ZerodhaCredentials:
    """Immutable, required credentials"""
    api_key: str
    access_token: str

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("ZERODHA_API_KEY is required")
        if not self.access_token or not self.access_token.strip():
            raise ValueError("ZERODHA_ACCESS_TOKEN is required (login and generate daily)")


@register_broker("zerodha")
class ZerodhaBroker(Broker):
    """
    Secure Zerodha Kite Connect broker.
    Forces explicit credentials — no fallbacks.
    """

    def __init__(self, credentials: ZerodhaCredentials):
        if not isinstance(credentials, ZerodhaCredentials):
            raise TypeError("You must pass a ZerodhaCredentials object")

        self.credentials = credentials

        try:
            self.kite = KiteConnect(api_key=credentials.api_key)
            self.kite.set_access_token(credentials.access_token)

            # Test connection
            profile = self.kite.profile()
            logger.info(f"ZerodhaBroker connected | User: {profile['user_name']} | LIVE mode")
        except Exception as e:
            logger.error("Zerodha connection failed. Invalid API key or access token.")
            raise ConnectionError(f"Failed to connect to Zerodha: {e}") from e

    def get_account(self) -> Dict[str, Any]:
        """Get margins and balance"""
        try:
            margins = self.kite.margins()
            equity = margins.get("equity", {})
            return {
                "cash": float(equity.get("available", {}).get("cash", 0)),
                "portfolio_value": float(equity.get("net", 0)),
                "buying_power": float(equity.get("available", {}).get("cash", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise

    def get_positions(self) -> Dict[str, float]:
        """Return {tradingsymbol: quantity}"""
        try:
            positions = self.kite.positions()["net"]
            return {
                pos["tradingsymbol"]: float(pos["quantity"])
                for pos in positions
                if pos["quantity"] != 0
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_price(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Fetch historical data from Zerodha.
        Supported: "1y", "2y", "max"
        """
        try:
            days_map = {"1y": 365, "2y": 730, "max": 2000}
            days = days_map.get(period.lower(), 730)

            to_date = datetime.today()
            from_date = to_date - timedelta(days=days)

            # Handle indices
            instrument = f"NSE:{ticker}"
            if ticker in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                instrument = f"NFO:{ticker}50"  # Approximate

            data = self.kite.historical_data(
                instrument=instrument,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )

            if not data:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df.set_index("date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = [c.capitalize() for c in df.columns]
            return df.sort_index()

        except Exception as e:
            logger.error(f"Price fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self, ticker: str) -> float:
        """Get latest LTP"""
        try:
            quote = self.kite.quote(f"NSE:{ticker}")
            return float(quote[f"NSE:{ticker}"]["last_price"])
        except:
            df = self.get_price(ticker, period="5d")
            return float(df["Close"].iloc[-1]) if not df.empty else 0.0

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Dict[str, Any]:
        """Place order with strict validation"""
        try:
            if abs(qty) < 1:
                logger.info(f"Skip fractional/small order: {ticker} {qty}")
                return {"status": "skipped", "reason": "small_qty"}

            qty = int(abs(qty))
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if side.lower() == "buy" else self.kite.TRANSACTION_TYPE_SELL
            order_type_kite = self.kite.ORDER_TYPE_MARKET if order_type == "market" else self.kite.ORDER_TYPE_LIMIT

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange="NSE",
                tradingsymbol=ticker,
                transaction_type=transaction_type,
                quantity=qty,
                product=self.kite.PRODUCT_CNC,
                order_type=order_type_kite,
                price=limit_price,
                validity=self.kite.VALIDITY_DAY,
            )

            logger.info(f"ORDER {side.upper()} {qty} {ticker} | ID: {order_id}")
            return {"status": "success", "order_id": order_id}

        except Exception as e:
            logger.error(f"Order failed {ticker} {side} {qty}: {e}")
            return {"status": "failed", "error": str(e)}

    def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        try:
            orders = self.kite.orders()
            for order in orders:
                if order["status"] not in ["COMPLETE", "REJECTED", "CANCELLED"]:
                    self.kite.cancel_order(order["variety"], order["order_id"])
                    logger.info(f"Cancelled order {order['order_id']}")
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")


# === USER MUST PASS CREDENTIALS ===
def create_zerodha_broker(api_key: str, access_token: str) -> ZerodhaBroker:
    """
    Factory function — forces explicit credentials.
    """
    if not api_key:
        raise ValueError("api_key is required")
    if not access_token:
        raise ValueError("access_token is required — generate daily via login")

    creds = ZerodhaCredentials(api_key=api_key, access_token=access_token)
    return ZerodhaBroker(credentials=creds)