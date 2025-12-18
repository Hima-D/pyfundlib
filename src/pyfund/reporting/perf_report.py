from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
        # Ensure we operate on a copy and sorted index
        self.equity = equity_curve.copy().sort_index()
        if not isinstance(self.equity.index, pd.DatetimeIndex):
            self.equity.index = pd.DatetimeIndex(self.equity.index)

        # returns as daily pct change; if the series already looks like cumulative equity (prices), pct_change() is correct
        self.returns = self.equity.pct_change().fillna(0).astype(float)

        self.trades = trades.copy() if trades is not None else None

        if benchmark is not None:
            b = benchmark.copy().sort_index()
            if not isinstance(b.index, pd.DatetimeIndex):
                b.index = pd.DatetimeIndex(b.index)
            self.benchmark = b.pct_change().fillna(0).astype(float)
        else:
            self.benchmark = None

        # Daily risk-free rate
        self.rf = float(risk_free_rate) / 252.0
        self.title = title

        # Set style (optional; requires seaborn installed)
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
            sns.set_palette("husl")
        except Exception:
            # fallback to matplotlib defaults if seaborn not available
            plt.style.use("default")

    def generate_report(self, output_path: Path | None = None) -> None:
        """Generate full multi-panel report (single figure)."""
        fig = plt.figure(figsize=(18, 24))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

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

        fig.suptitle(self.title, fontsize=22, fontweight="bold", y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Report saved to {output_path}")
        plt.show()

    def summary(self) -> dict[str, float]:
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
            "Skewness": float(self.returns.skew()),
            "Kurtosis": float(self.returns.kurtosis()),
        }

    # ======================= Plotting Helpers =======================
    def _plot_equity_curve(self, ax: Axes) -> None:
        cum_returns = (1 + self.returns).cumprod()
        ax.plot(cum_returns.index, cum_returns.values, lw=2, label="Strategy")
        if self.benchmark is not None:
            bench_cum = (1 + self.benchmark).cumprod()
            ax.plot(bench_cum.index, bench_cum.values, lw=2, alpha=0.8, label="Benchmark")
        ax.set_title("Equity Curve", fontsize=16, fontweight="bold")
        ax.legend()
        ax.set_ylabel("Growth of $1")
        ax.grid(True)

    def _plot_drawdown(self, ax: Axes) -> None:
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        # fill between for underwater plot
        ax.fill_between(drawdown.index, drawdown.values, 0, where=drawdown.values < 0, interpolate=True, color="red", alpha=0.5)
        ax.set_title("Underwater Plot (Drawdown)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.grid(True)

    def _monthly_return(self, x: pd.Series) -> float:
        # helper to compute monthly compounded return from daily returns
        return float((1 + x).prod() - 1)

    def _plot_monthly_returns(self, ax: Axes) -> None:
        monthly = self.returns.resample("M").apply(self._monthly_return)
        monthly_df = monthly.to_frame("Return")
        # Ensure index is DatetimeIndex
        if not isinstance(monthly_df.index, pd.DatetimeIndex):
            monthly_df.index = pd.DatetimeIndex(monthly_df.index)
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        monthly_heatmap = monthly_df.pivot_table(
            values="Return", index="year", columns="month", aggfunc="first"
        )
        sns.heatmap(
            monthly_heatmap * 100,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "Return %"},
            linewidths=0.5,
        )
        ax.set_title("Monthly Returns (%)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Year")

    def _plot_returns_distribution(self, ax: Axes) -> None:
        # use matplotlib histogram for control
        data = self.returns.dropna().values.astype(float)
        ax.hist(data, bins=60, alpha=0.7, edgecolor="black")
        mean_val = float(np.mean(data)) if len(data) > 0 else 0.0
        median_val = float(np.median(data)) if len(data) > 0 else 0.0
        ax.axvline(mean_val, linestyle="--", label=f"Mean: {mean_val:.2%}")
        ax.axvline(median_val, linestyle="--", label=f"Median: {median_val:.2%}")
        ax.set_title("Daily Returns Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("Daily Return")
        ax.legend()
        ax.grid(True)

    def _plot_metrics_table(self, ax: Axes) -> None:
        metrics = self.summary()
        rows: list[list[Any]] = []
        for k, v in metrics.items():
            # Convert to float where possible for consistent formatting
            try:
                v_float = float(v)
            except Exception:
                rows.append([k, str(v)])
                continue

            if np.isnan(v_float):
                rows.append([k, "NaN"])
            elif np.isinf(v_float):
                rows.append([k, "Inf"])
            elif abs(v_float) > 1:
                rows.append([k, f"{v_float:,.2f}"])
            else:
                rows.append([k, f"{v_float:.2%}"])

        table = ax.table(cellText=rows, colWidths=[0.6, 0.4], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 1.8)
        ax.set_title("Performance Metrics", fontsize=18, fontweight="bold", pad=20)

    # ======================= Metrics =======================
    def _total_return(self) -> float:
        return float((1 + self.returns).prod() - 1)

    def _cagr(self) -> float:
        # Ensure datetime index for subtraction; cast days to float
        start_date = self.equity.index[0]
        end_date = self.equity.index[-1]
        delta = end_date - start_date
        days = float(delta.days) if delta.days > 0 else 0.0
        if days <= 0:
            return 0.0
        return (1 + self._total_return()) ** (365.0 / days) - 1

    def _sharpe(self) -> float:
        excess = self.returns - self.rf
        std = float(excess.std())
        return (np.sqrt(252.0) * float(excess.mean()) / std) if std != 0 else 0.0

    def _sortino(self) -> float:
        downside = self.returns[self.returns < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        denom = downside_std if downside_std != 0 else 1.0
        return np.sqrt(252.0) * (float(self.returns.mean()) - self.rf) / denom

    def _max_drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        dd = ((cum.cummax() - cum) / cum.cummax()).max()
        return float(dd)

    def _calmar(self) -> float:
        dd = self._max_drawdown()
        return float(self._cagr() / dd) if dd > 0 else np.inf

    def _annual_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(252.0))

    def _win_rate(self) -> float:
        if self.trades is None or len(self.trades) == 0:
            return float("nan")
        return float((self.trades["return"] > 0).mean())

    def _profit_factor(self) -> float:
        if self.trades is None or len(self.trades) == 0:
            return float("nan")
        wins = float(self.trades[self.trades["return"] > 0]["return"].sum())
        losses = float(abs(self.trades[self.trades["return"] < 0]["return"].sum()))
        return float(wins / losses) if losses > 0 else np.inf

    def _avg_win_over_loss(self) -> float:
        if self.trades is None or len(self.trades) == 0:
            return float("nan")
        wins = self.trades[self.trades["return"] > 0]["return"]
        losses = self.trades[self.trades["return"] < 0]["return"]
        if len(losses) == 0:
            return np.inf
        return float(float(wins.mean()) / float(abs(losses.mean())))
