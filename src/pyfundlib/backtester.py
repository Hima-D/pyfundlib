"""Vectorized backtesting engine for PyFundLib."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""

    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1.0
    side: str = "long"  # long or short
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None

    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_date is not None


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtest."""

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


class Portfolio:
    """Manages portfolio state during backtesting."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # ticker -> quantity
        self.commission = commission
        self.slippage = slippage
        self.equity_curve: List[float] = [initial_capital]
        self.trades: List[Trade] = []

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current equity."""
        position_value = sum(
            self.positions.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in self.positions
        )
        return self.cash + position_value

    def execute_trade(
        self, ticker: str, quantity: float, price: float, side: str = "buy"
    ) -> bool:
        """Execute a trade."""
        cost_with_slippage = price * (1 + self.slippage)
        total_cost = abs(quantity) * cost_with_slippage * (1 + self.commission)

        if side == "buy":
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {ticker}")
                return False
            self.cash -= total_cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:  # sell
            if abs(quantity) > self.positions.get(ticker, 0):
                logger.warning(f"Insufficient position for {ticker}")
                return False
            self.cash += abs(quantity) * cost_with_slippage * (1 - self.commission)
            self.positions[ticker] -= quantity

        return True

    def record_equity(self, prices: Dict[str, float]) -> None:
        """Record current equity."""
        self.equity_curve.append(self.get_equity(prices))


class Backtester:
    """Vectorized backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.portfolio = Portfolio(initial_capital, commission, slippage)
        self.metrics = PerformanceMetrics()

    def run(
        self,
        ohlcv_data: pd.DataFrame,
        signals: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PerformanceMetrics:
        """Run backtest with trading signals."""
        # Filter data by date range
        if start_date:\n            ohlcv_data = ohlcv_data[ohlcv_data.index >= start_date]\n        if end_date:\n            ohlcv_data = ohlcv_data[ohlcv_data.index <= end_date]\n\n        # Execute trades based on signals\n        for date_idx, (date, row) in enumerate(ohlcv_data.iterrows()):\n            if date not in signals.index:\n                continue\n\n            signal_row = signals.loc[date]\n            prices = row.to_dict()\n\n            # Process buy/sell signals\n            for ticker in signal_row.index:\n                signal = signal_row[ticker]\n                if signal == 1:  # Buy signal\n                    self.portfolio.execute_trade(ticker, 100, prices.get(ticker, 0), \"buy\")\n                elif signal == -1:  # Sell signal\n                    self.portfolio.execute_trade(ticker, 100, prices.get(ticker, 0), \"sell\")\n\n            # Record equity\n            self.portfolio.record_equity(prices)\n\n        # Calculate metrics\n        self._calculate_metrics()\n        return self.metrics\n\n    def _calculate_metrics(self) -> None:\n        \"\"\"Calculate performance metrics.\"\"\"\n        equity = np.array(self.portfolio.equity_curve)\n        returns = np.diff(equity) / equity[:-1]\n\n        # Basic metrics\n        self.metrics.total_return = (equity[-1] - equity[0]) / equity[0]\n        self.metrics.annual_return = (1 + self.metrics.total_return) ** (252 / len(returns)) - 1\n\n        # Risk metrics\n        if len(returns) > 0:\n            self.metrics.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)\n            downside_returns = returns[returns < 0]\n            if len(downside_returns) > 0:\n                self.metrics.sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)\n\n        # Drawdown\n        cumulative = np.cumprod(1 + returns)\n        running_max = np.maximum.accumulate(cumulative)\n        drawdown = (cumulative - running_max) / running_max\n        self.metrics.max_drawdown = np.min(drawdown)\n\n    def get_equity_curve(self) -> List[float]:\n        \"\"\"Get equity curve.\"\"\"\n        return self.portfolio.equity_curve

    def get_metrics(self) -> PerformanceMetrics:\n        \"\"\"Get performance metrics.\"\"\"\n        return self.metrics
