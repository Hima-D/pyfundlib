# src/pyfundlib/utils/monitor.py
from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from threading import Event, Thread
from typing import Any

import pandas as pd
import psutil

from .logger import get_logger

logger = get_logger(__name__)


class Monitor:
    """
    Real-time monitoring dashboard for live trading systems.
    Tracks:
    - Strategy P&L, positions, signals
    - Broker connection & orders
    - ML model drift/predictions
    - System resources (CPU, RAM, disk)
    - Custom metrics
    """

    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.metrics: dict[str, Any] = {}
        self.history: pd.DataFrame = pd.DataFrame()
        self.stop_event = Event()
        self.thread: Thread | None = None

        # Built-in system metrics
        self.register_metric("cpu_percent", psutil.cpu_percent)
        self.register_metric("ram_percent", psutil.virtual_memory().percent)
        self.register_metric("disk_usage", lambda: psutil.disk_usage("/").percent)
        self.register_metric("timestamp", datetime.now)

    def register_metric(self, name: str, func: Callable[[], Any]) -> None:
        """Register a custom metric (e.g., portfolio value, active orders)"""
        self.metrics[name] = func
        logger.info(f"Monitor metric registered: {name}")

    def update(self) -> None:
        """Collect all metrics once"""
        row = {}
        for name, func in self.metrics.items():
            try:
                row[name] = func() if callable(func) else func
            except Exception as e:
                row[name | None = None
                logger.warning(f"Metric {name} failed: {e}")
        row["datetime"] = datetime.now()
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
        self._log_summary(row)

    def _log_summary(self, row: dict[str, Any]) -> None:
        """Pretty console dashboard"""
        print("\n" + "=" * 60)
        print(f" PYFUNDLIB MONITOR | {row.get('datetime', '')}")
        print("=" * 60)
        for k, v in row.items():
            if k == "datetime":
                continue
            print(f" {k:20} | {v}")
        print("-" * 60)

    def start(self, daemon: bool = True) -> None:
        """Start background monitoring thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Monitor already running")
            return

        def _run():
            logger.info(f"Monitor started (refresh every {self.refresh_interval}s)")
            while not self.stop_event.wait(self.refresh_interval):
                self.update()

        self.thread = Thread(target=_run, daemon=daemon)
        self.thread.start()
        logger.info("Real-time monitoring ACTIVE")

    def stop(self) -> None:
        """Stop monitoring"""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        logger.info("Monitor stopped")

    def get_latest(self) -> dict[str, Any]:
        """Get most recent metrics"""
        if self.history.empty:
            return {}
        return self.history.iloc[-1].to_dict()

    def plot(self, metric: str, window: int = 100) -> None:
        """Quick plot of a metric over time"""
        if metric not in self.history.columns:
            logger.error(f"Metric {metric} not found")
            return
        import matplotlib.pyplot as plt

        data = self.history[metric].tail(window)
        plt.figure(figsize=(12, 6))
        data.plot(title=f"{metric} Over Time")
        plt.grid(True, alpha=0.3)
        plt.show()


# Global monitor instance (easy access)
monitor = Monitor(refresh_interval=10)


# Example: How users extend it
def track_portfolio_value():
    from pyfundlib.execution.live import LiveExecutor

    try:
        executor = LiveExecutor(dry_run=True)
        return executor.get_account().get("portfolio_value", 0)
    except:
        return 0


def track_active_signals():
    # Replace with your strategy's current signal
    return "RSI Oversold → BUY AAPL"


# Register in your main/live script
monitor.register_metric("portfolio_value", track_portfolio_value)
monitor.register_metric("current_signal", track_active_signals)
monitor.register_metric(
    "uptime_seconds",
    lambda: time.time() - monitor.start_time if hasattr(monitor, "start_time") else 0,
)

# Start it!
monitor.start()
