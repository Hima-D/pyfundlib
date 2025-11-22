# src/pyfundlib/backtester/engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..reporting.perf_report import PerformanceReport
from ..risk import RiskManager
from ..strategies.base import BaseStrategy
from ..utils.logger import get_logger
from ..utils.plotter import Plotter

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for all backtest outputs"""

    equity_curve: pd.Series
    signals: pd.Series
    trades: pd.DataFrame
    metrics: dict[str | float]
    data: pd.DataFrame


class Backtester:
    """
    The most powerful, flexible, and beautiful backtester in open-source finance.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        ticker: str | None = None,
        data: pd.DataFrame | None = None,
        initial_capital: float = 100_000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 5 bps
        position_sizing: float = 1.0,  # 100% exposure
        risk_per_trade: float = 0.01,  # Kelly/volatility parity later
        name: str = "backtest",
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.name = name

        # Load data
        if data is not None:
            self.data = data.copy()
        elif ticker:
            self.data = DataFetcher.get_price(ticker.upper())
        else:
            raise ValueError("Must provide ticker or data")

        self.data["returns"] = self.data["Close"].pct_change()

    def run(self) -> BacktestResult:
        """Run the full backtest"""
        logger.info(f"Starting backtest: {self.name} on {self.strategy.__class__.__name__}")

        # Generate signals (-1, 0, +1)
        signals = self.strategy.generate_signals(self.data)
        signals = signals.shift(1).fillna(0)  # No lookahead!

        # Apply position sizing
        position = signals * self.position_sizing

        # Calculate returns with costs
        raw_returns = position * self.data["returns"]
        transaction_costs = (position.diff().abs() * (self.commission + self.slippage)).fillna(0)
        strategy_returns = raw_returns - transaction_costs

        # Equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Extract trades
        trades = self._extract_trades(position, strategy_returns)

        # Full metrics
        metrics = RiskManager.risk_metrics(equity)
        metrics.update(
            {
                "total_return": (equity.iloc[-1] / equity.iloc[0]) - 1,
                "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1,
                "win_rate": (trades["pnl"] > 0).mean() if len(trades) > 0 else 0,
                "num_trades": len(trades),
                "avg_trade_duration": trades["duration"].mean() if len(trades) > 0 else 0,
            }
        )

        result = BacktestResult(
            equity_curve=equity,
            signals=position,
            trades=trades,
            metrics=metrics,
            data=self.data,
        )

        logger.info(
            f"Backtest complete | CAGR: {metrics['cagr']:+.1%} | Sharpe: {metrics.get('sharpe', 0):.2f} | MaxDD: {metrics['max_drawdown']:.1%}"
        )

        return result

    def _extract_trades(self, position: pd.Series, returns: pd.Series) -> pd.DataFrame:
        """Convert position series into trade log"""
        trades = []
        entry_price = entry_date = None
        direction = 0

        for date, (pos, _ret) in enumerate(zip(position, returns)):
            if pos != direction and pos != 0:
                # Entry
                if entry_price is not None:
                    # Close previous
                    pnl = direction * (self.data["Close"].iloc[date] / entry_price - 1) - 2 * (
                        self.commission + self.slippage
                    )
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": self.data.index[date],
                            "direction": "long" if direction > 0 else "short",
                            "pnl": pnl,
                            "return": pnl,
                            "duration": (self.data.index[date] - entry_date).days,
                        }
                    )
                # New entry
                entry_price = self.data["Close"].iloc[date]
                entry_date = self.data.index[date]
                direction = np.sign(pos)

        # Close final trade
        if entry_price is not None and len(self.data) > 0:
            last_date = self.data.index[-1]
            last_price = self.data["Close"].iloc[-1]
            pnl = direction * (last_price / entry_price - 1) - 2 * (self.commission + self.slippage)
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": last_date,
                    "direction": "long" if direction > 0 else "short",
                    "pnl": pnl,
                    "return": pnl,
                    "duration": (last_date - entry_date).days,
                }
            )

        return pd.DataFrame(trades)

    def report(
        self,
        result: BacktestResult | None = None,
        save_dir: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        """Generate beautiful report"""
        if result is None:
            result = self.run()

        save_dir = save_dir or Path("reports") / self.name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Equity plot
        Plotter.equity_curve(
            result.equity_curve,
            title=f"{self.name} | CAGR: {result.metrics['cagr']:+.1%} | MaxDD: {result.metrics['max_drawdown']:.1%}",
            save_path=save_dir / "equity_curve.png",
        )

        # Full performance report
        perf = PerformanceReport(
            equity_curve=result.equity_curve,
            trades=result.trades,
            title=self.name,
        )
        perf.generate_report(save_dir / "performance_report.png")

        # Save CSV
        result.equity_curve.to_csv(save_dir / "equity_curve.csv")
        result.trades.to_csv(save_dir / "trades.csv", index=False)

        if show_plot:
            Plotter.equity_curve(result.equity_curve)

        logger.info(f"Full report saved → {save_dir}")

    def summary(self) -> dict[str | float]:
        result = self.run()
        return result.metrics
