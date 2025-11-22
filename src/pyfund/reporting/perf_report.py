# src/pyfund/reporting/perf_report.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PerformanceReport:
    """
    Generate beautiful, publication-ready performance reports for any strategy.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.04,  # 4% annual
        title: str = "Strategy Performance Report",
    ):
        """
        Parameters
        ----------
        equity_curve : pd.Series
            Index: date, Values: portfolio equity (or cumulative returns)
        trades : pd.DataFrame, optional
            Columns: ['entry_date', 'exit_date', 'return', 'duration_days', 'direction']
        benchmark : pd.Series, optional
            e.g., SPY returns for comparison
        risk_free_rate : float
            Annual risk-free rate
        """
        self.equity = equity_curve.sort_index()
        self.returns = self.equity.pct_change().fillna(0)
        self.trades = trades
        self.benchmark = benchmark.pct_change().fillna(0) if benchmark is not None else None
        self.rf = risk_free_rate / 252  # Daily risk-free rate
        self.title = title

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def generate_report(self, output_path: Path | None = None) -> None:
        """Generate full multi-page report."""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. Equity Curve + Benchmark
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1)

        # 2. Underwater (Drawdown)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(ax2)

        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_monthly_returns(ax3)

        # 4. Daily Returns Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax4)

        # 5. Metrics Table
        ax5 = fig.add_subplot(gs[2:, 1])
        ax5.axis("off")
        self._plot_metrics_table(ax5)

        fig.suptitle(self.title, fontsize=24, fontweight="bold", y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Report saved to {output_path}")
        plt.show()

    def summary(self) -> dict[str | float]:
        """Return dictionary of all key performance metrics."""
        return {
            "Total Return": self._total_return(),
            "CAGR": self._cagr(),
            "Sharpe Ratio": self._sharpe(),
            "Sortino Ratio": self._sortino(),
            "Calmar Ratio": self._calmar(),
            "Max Drawdown": self._max_drawdown(),
            "Win Rate": self._win_rate(),
            "Profit Factor": self._profit_factor(),
            "Avg Win / Avg Loss": self._avg_win_over_loss(),
            "Volatility (Ann.)": self._annual_volatility(),
            "Skewness": self.returns.skew(),
            "Kurtosis": self.returns.kurtosis(),
        }

    # ======================= Plotting Helpers =======================
    def _plot_equity_curve(self, ax):
        cum_returns = (1 + self.returns).cumprod()
        cum_returns.plot(ax=ax, lw=2, label="Strategy")
        if self.benchmark is not None:
            (1 + self.benchmark).cumprod().plot(ax=ax, lw=2, alpha=0.8, label="Benchmark")
        ax.set_title("Equity Curve", fontsize=16, fontweight="bold")
        ax.legend()
        ax.set_ylabel("Growth of $1")

    def _plot_drawdown(self, ax):
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        drawdown.plot(area=True, ax=ax, color="red", alpha=0.6)
        ax.set_title("Underwater Plot (Drawdown)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Drawdown")

    def _plot_monthly_returns(self, ax):
        monthly = self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_heatmap = monthly.to_frame("Return").pivot_table(
            values="Return", index=monthly.index.year, columns=monthly.index.month
        )
        sns.heatmap(
            monthly_heatmap * 100,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "Return %"},
        )
        ax.set_title("Monthly Returns (%)", fontsize=16, fontweight="bold")

    def _plot_returns_distribution(self, ax):
        self.returns.hist(bins=60, ax=ax, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(
            self.returns.mean(),
            color="green",
            linestyle="--",
            label=f"Mean: {self.returns.mean():.2%}",
        )
        ax.axvline(
            self.returns.median(),
            color="orange",
            linestyle="--",
            label=f"Median: {self.returns.median():.2%}",
        )
        ax.set_title("Daily Returns Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("Daily Return")
        ax.legend()

    def _plot_metrics_table(self, ax):
        metrics = self.summary()
        rows = []
        for k, v in metrics.items():
            if isinstance(v, float):
                rows.append([k, f"{v:,.2f}" if abs(v) > 100 else f"{v:,.2%}"])
            else:
                rows.append([k, str(v)])

        table = ax.table(cellText=rows, colWidths=[0.6, 0.4], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.2)
        ax.set_title("Performance Metrics", fontsize=18, fontweight="bold", pad=20)

    # ======================= Metrics =======================
    def _total_return(self) -> float:
        return (1 + self.returns).prod() - 1

    def _cagr(self) -> float:
        days = (self.equity.index[-1] - self.equity.index[0]).days
        return (1 + self._total_return()) ** (365 / days) - 1

    def _sharpe(self) -> float:
        excess = self.returns - self.rf
        return np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0.0

    def _sortino(self) -> float:
        downside = self.returns[self.returns < 0]
        return np.sqrt(252) * (self.returns.mean() - self.rf) / (downside.std() or 1)

    def _max_drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        return ((cum.cummax() - cum) / cum.cummax()).max()

    def _calmar(self) -> float:
        return self._cagr() / self._max_drawdown() if self._max_drawdown() > 0 else np.inf

    def _annual_volatility(self) -> float:
        return self.returns.std() * np.sqrt(252)

    def _win_rate(self) -> float:
        if self.trades is None or len(self.trades) == 0:
            return np.nan
        return (self.trades["return"] > 0).mean()

    def _profit_factor(self) -> float:
        if self.trades is None:
            return np.nan
        wins = self.trades[self.trades["return"] > 0]["return"].sum()
        losses = abs(self.trades[self.trades["return"] < 0]["return"].sum())
        return wins / losses if losses > 0 else np.inf

    def _avg_win_over_loss(self) -> float:
        if self.trades is None:
            return np.nan
        wins = self.trades[self.trades["return"] > 0]["return"]
        losses = self.trades[self.trades["return"] < 0]["return"]
        return wins.mean() / abs(losses.mean()) if len(losses) > 0 else np.inf
