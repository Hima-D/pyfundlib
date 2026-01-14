from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from pyfundlib.core.broker import Broker, BrokerMode, OrderSide, OrderType, TimeInForce
from pyfundlib.core.broker_registry import register_broker
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IBKRCredentials:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: Optional[str] = None


@register_broker("ibkr")
class IBKRBroker(Broker):
    def __init__(
        self,
        credentials: Optional[IBKRCredentials] = None,
        mode: BrokerMode = "paper",
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        super().__init__(mode=mode, name="ibkr", max_retries=max_retries, timeout=timeout)
        self.credentials = credentials or IBKRCredentials()
        self._ib = None

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as e:
            raise ImportError("ib_insync is required for IBKRBroker") from e

        ib = IB()
        ib.connect(self.credentials.host, self.credentials.port, clientId=self.credentials.client_id)
        self._ib = ib
        self.is_connected = True
        logger.info("IBKR connected")

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self.is_connected = False
        logger.info("IBKR disconnected")

    def _ensure_connected(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("IBKRBroker is not connected. Call connect() first.")

    def get_price(
        self,
        ticker: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        self._ensure_connected()

        try:
            from ib_insync import Stock, util
        except ImportError as e:
            raise ImportError("ib_insync is required for IBKRBroker") from e

        contract = Stock(ticker.upper(), "SMART", "USD")

        duration = period or "2 Y"
        bar_size = interval or "1 day"

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_date or "",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            return pd.DataFrame()

        df = util.df(bars)
        if "date" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        columns = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in columns):
            return pd.DataFrame()

        out = df[columns].copy()
        out.columns = [c.capitalize() for c in out.columns]
        out.index.name = "date"
        return out.sort_index()

    def get_balance(self) -> Dict[str, float]:
        self._ensure_connected()
        summary = self._ib.accountSummary()

        cash = 0.0
        equity = 0.0
        for item in summary:
            if item.tag == "TotalCashValue":
                cash = float(item.value)
            elif item.tag == "NetLiquidation":
                equity = float(item.value)

        return {
            "cash": cash,
            "equity": equity,
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_connected()
        positions = {}
        for pos in self._ib.positions():
            symbol = pos.contract.symbol.upper()
            positions[symbol] = {
                "qty": float(pos.position),
                "avg_price": float(pos.avgCost),
            }
        return positions

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = "market",
        price: Optional[float] = None,
        time_in_force: TimeInForce = "day",
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_connected()

        if qty == 0:
            return {"status": "skipped", "reason": "zero_quantity"}

        try:
            from ib_insync import LimitOrder, MarketOrder, Stock
        except ImportError as e:
            raise ImportError("ib_insync is required for IBKRBroker") from e

        contract = Stock(ticker.upper(), "SMART", "USD")

        if order_type == "market":
            order = MarketOrder(side.upper(), abs(qty))
        else:
            if price is None:
                raise ValueError("price is required for limit and stop orders")
            order = LimitOrder(side.upper(), abs(qty), price)

        order.tif = time_in_force.upper()
        if tag:
            order.orderRef = tag

        trade = self._ib.placeOrder(contract, order)
        logger.info(f"IBKR order placed {side} {qty} {ticker}")

        return {
            "status": "submitted",
            "order_id": str(trade.order.orderId),
            "ticker": ticker.upper(),
            "qty": float(qty),
            "side": side,
        }

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()

        open_orders = self._ib.openOrders()
        for o in open_orders:
            if str(o.orderId) == str(order_id):
                self._ib.cancelOrder(o)
                return True
        return False

    def cancel_all_orders(self) -> int:
        self._ensure_connected()

        count = 0
        for o in list(self._ib.openOrders()):
            self._ib.cancelOrder(o)
            count += 1
        return count

    def get_open_orders(self) -> list[Dict[str, Any]]:
        self._ensure_connected()

        orders = []
        for o in self._ib.openOrders():
            orders.append(
                {
                    "order_id": str(o.orderId),
                    "symbol": o.contract.symbol,
                    "side": o.action,
                    "qty": float(o.totalQuantity),
                    "order_type": o.orderType,
                    "tif": o.tif,
                }
            )
        return orders

