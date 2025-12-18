# src/pyfund/strategies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Union, Optional

import numpy as np
import pandas as pd

from ..utils.logger import logger

Signal = Union[int, float]  # 1.0 = full long, -1.0 = full short, 0.5 = half long, etc.
PositionSide = Literal["long", "short", "flat"]


@dataclass
class SignalResult:
    """Rich signal output with metadata"""

    signals: pd.Series  # Core signal series (index-aligned with data)
    positions: pd.Series | None = None  # Position after entry/exit rules
    entries: pd.Series | None = None  # True on new entries
    exits: pd.Series | None = None  # True on exits
    metadata: dict[str, Any] | None = None  # Extra info (confidence, z-score, etc.)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies in pyfund

    Features:
    - Clean parameter handling with validation
    - Standardized signal output (rich SignalResult)
    - Built-in logging and error handling
    - Position tracking and state management
    - Ready for vectorized backtesting and live trading
    """

    # Class-level default parameters (can be overridden by subclasses)
    default_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize strategy with parameters

        Args:
            params: Strategy parameters. Will be merged with default_params.
        """
        # Merge user params with defaults (user overrides defaults)
        merged_params = {**self.__class__.default_params, **(params or {})}

        # Optional: validate parameters
        self._validate_params(merged_params)

        self.params = merged_params

        # State variables (reset on each run)
        self.current_position: Signal = 0.0
        self.last_signal_date: pd.Timestamp | None = None
        self.trade_count = 0

        logger.debug(f"{self.__class__.__name__} initialized with params: {self.params}")

    def _validate_params(self, params: dict[str, Any]) -> None:
        """Override in subclass for custom validation"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> SignalResult:
        """
        Core method: generate trading signals from price data

        Args:
            data: DataFrame with at minimum ['Open', 'High', 'Low', 'Close', 'Volume']
                  Index must be datetime

        Returns:
            SignalResult containing aligned signal series and metadata
        """

    # ===================================================================
    # Helper methods that most strategies will want to use or override
    # ===================================================================

    def _apply_position_sizing(self, raw_signals: pd.Series) -> pd.Series:
        """
        Convert raw directional signals to position sizes
        Subclasses can override for Kelly, volatility targeting, etc.
        """
        # Default: full size on signal, hold until opposite
        positioned = raw_signals.replace(0, np.nan).ffill().fillna(0)
        return positioned

    def _detect_entries_exits(self, positions: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Detect entry and exit points"""
        entries = (positions != 0) & (positions.shift(1) == 0)
        exits = (positions == 0) & (positions.shift(1) != 0)
        return entries, exits

    def _log_trade(self, date: pd.Timestamp, signal: Signal, price: float, reason: str = ""):
        """Standardized trade logging"""
        side = "LONG" if signal > 0 else "SHORT" if signal < 0 else "FLAT"
        self.trade_count += 1
        logger.info(
            f"TRADE #{self.trade_count} | {date.date()} | {side} | Price: {price:,.2f} | {reason}".strip()
        )

    # ===================================================================
    # Optional: convenience methods
    # ===================================================================

    def __repr__(self) -> str:
        name = self.__class__.__name__
        params_str = ", ".join(
            f"{k}={v}" for k, v in self.params.items() if k in self.default_params
        )
        return f"{name}({params_str})"

    def reset(self):
        """Reset strategy state (useful for walk-forward testing)"""
        self.current_position = 0.0
        self.last_signal_date = None
        self.trade_count = 0


# Example concrete strategy using the new base
if __name__ == "__main__":

    class DummyStrategy(BaseStrategy):
        default_params = {"threshold": 0.5}

        def generate_signals(self, data: pd.DataFrame) -> SignalResult:
            returns = data["Close"].pct_change()
            raw = np.where(
                returns > self.params["threshold"] / 100,
                1,
                np.where(returns < -self.params["threshold"] / 100, -1, 0),
            )
            raw_signals = pd.Series(raw, index=data.index)

            positions = self._apply_position_sizing(raw_signals)
            entries, exits = self._detect_entries_exits(positions)

            return SignalResult(
                signals=raw_signals,
                positions=positions,
                entries=entries,
                exits=exits,
                metadata={"description": "Simple momentum threshold strategy"},
            )

    # Test
    from ..data.fetcher import DataFetcher

    df = DataFetcher.get_price("AAPL", period="1y")
    strat = DummyStrategy({"threshold": 1.0})
    result = strat.generate_signals(df)
    print(result.signals.tail(10))