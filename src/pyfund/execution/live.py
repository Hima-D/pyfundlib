# src/pyfund/execution/live.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..core.broker_registry import broker_registry
from ..utils.logger import get_logger

logger = get_logger(__name__)

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "opg", "ioc", "fok"]


@dataclass
class OrderRequest:
    ticker: str
    qty: float | int
    side: Side
    type: OrderType = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = "day"
    client_order_id: str | None = None
    extended_hours: bool = False


@dataclass
class OrderResponse:
    order_id: str
    status: str
    filled_qty: float
    filled_price: float | None = None
    submitted_at: str | None = None
    raw: dict[str, Any] | None = None


class LiveExecutor:
    """
    Unified live trading executor with dry-run support.
    Automatically routes to the correct broker via broker_registry.
    """

    def __init__(
        self,
        broker: str = "alpaca",
        dry_run: bool = True,
        default_account: str | None = None,
    ):
        self.broker_name = broker.lower()
        self.dry_run = dry_run
        self.default_account = default_account

        # Resolve actual broker client
        self.client = broker_registry.get_broker_client(self.broker_name)

        if self.dry_run:
            logger.warning(
                f"LiveExecutor initialized in DRY-RUN mode for {self.broker_name.upper()}"
            )
        else:
            logger.info(f"LiveExecutor connected to LIVE {self.broker_name.upper()} broker")

    def place_order(self, order: OrderRequest | dict[str, Any]) -> OrderResponse:
        """
        Place a single order (sync). Returns OrderResponse.
        """
        if isinstance(order, dict):
            order = OrderRequest(**order)

        if self.dry_run:
            logger.info(
                f"[DRY RUN] {order.side.upper()} {order.qty} {order.ticker} "
                f"@ {order.type.upper()} (TIF: {order.time_in_force})"
            )
            return OrderResponse(
                order_id=f"dry_{int(time.time()*1e6)}",
                status="filled",
                filled_qty=order.qty,
                filled_price=None,
                submitted_at=pd.Timestamp.now().isoformat(),
            )

        try:
            resp = self.client.submit_order(order)
            logger.info(f"Order placed: {order.side} {order.qty} {order.ticker} → {resp.status}")
            return resp
        except Exception as e:
            logger.error(f"Order failed for {order.ticker}: {e}")
            raise

    def batch_place_orders(self, orders: list[OrderRequest]) -> list[OrderResponse]:
        """Submit multiple orders (with basic rate limiting respect)."""
        results = []
        for order in orders:
            results.append(self.place_order(order))
            time.sleep(0.1)  # Be gentle to the API
        return results

    def cancel_order(self, order_id: str) -> bool:
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel order {order_id}")
            return True
        try:
            result = self.client.cancel_order(order_id)
            logger.info(f"Order {order_id} canceled")
            return result
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        if self.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True
        try:
            self.client.cancel_all_orders()
            logger.info("All orders canceled")
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False

    def get_positions(self) -> dict[str | float]:
        """Return dict of ticker → quantity (plus CASH)"""
        if self.dry_run:
            return {"CASH": 100_000.0, "SPY": 100.0, "AAPL": 50.0}

        try:
            positions = self.client.get_positions()
            logger.debug(f"Retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account(self) -> dict[str, Any]:
        if self.dry_run:
            return {
                "cash": 100_000.0,
                "portfolio_value": 150_000.0,
                "buying_power": 400_000.0,
                "currency": "USD",
            }
        try:
            return self.client.get_account()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_open_orders(self) -> list[OrderResponse]:
        if self.dry_run:
            return []
        try:
            return self.client.list_orders(status="open")
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
