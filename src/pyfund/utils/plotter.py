# src/pyfund/utils/plotter.py
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["axes.prop_cycle"] = cycler(
    color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#4C9A2A"]
)



class Plotter:
    """
    Unified plotting utility for pyfund — beautiful charts out of the box.
    """

    @staticmethod
    def equity_curve(
        equity: pd.Series,
        benchmark: pd.Series | None = None,
        trades: pd.DataFrame | None = None,
        title: str = "Strategy Equity Curve",
        save_path: Path | None = None,
        show: bool = True,
    ) -> figure.Figure:
        """
        Plot equity curve with optional benchmark and trade markers.
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Normalize to starting value = 100 or 1.0
        equity_norm = equity / equity.iloc[0] * 100
        equity_norm.plot(ax=ax, lw=2.5, label="Strategy", alpha=0.9)

        if benchmark is not None:
            bench_norm = benchmark / benchmark.iloc[0] * 100
            bench_norm.plot(ax=ax, lw=2, label="Benchmark", alpha=0.8, linestyle="--")

        # Plot trade entry/exit markers
        if trades is not None:
            buys = trades[trades["side"] == "buy"]
            sells = trades[trades["side"] == "sell"]

            if not buys.empty:
                ax.scatter(
                    buys["entry_date"],
                    equity_norm.loc[buys["entry_date"]],
                    marker="^",
                    color="green",
                    s=100,
                    label="Buy",
                    zorder=5,
                )
            if not sells.empty:
                ax.scatter(
                    sells["exit_date"],
                    equity_norm.loc[sells["exit_date"]],
                    marker="v",
                    color="red",
                    s=100,
                    label="Sell",
                    zorder=5,
                )

        ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
        ax.set_ylabel("Portfolio Value (%)", fontsize=14)
        ax.set_xlabel("")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Clean date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Equity curve saved → {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def drawdown(
        equity: pd.Series,
        title: str = "Drawdown Waterfall",
        save_path: Path | None = None,
    ) -> figure.Figure:
        cum_ret = (1 + equity.pct_change()).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max

        fig, ax = plt.subplots(figsize=(16, 6))
        drawdown.plot(area=True, color="crimson", alpha=0.7, ax=ax)
        ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def returns_distribution(
        returns: pd.Series,
        title: str = "Daily Returns Distribution",
        save_path: Path | None = None,
    ) -> figure.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))
        returns.hist(bins=80, alpha=0.8, color="#2E86AB", edgecolor="white", ax=ax)
        ax.axvline(returns.mean(), color="green", linewidth=2, label=f"Mean: {returns.mean():.2%}")
        ax.axvline(
            returns.median(),
            color="orange",
            linewidth=2,
            linestyle="--",
            label=f"Median: {returns.median():.2%}",
        )

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend()

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 252,
        title: str = "Rolling Sharpe Ratio (252d)",
        save_path: Path | None = None,
    ) -> figure.Figure:
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )

        fig, ax = plt.subplots(figsize=(16, 6))
        rolling_sharpe.plot(ax=ax, lw=2, color="#4C9A2A")
        ax.axhline(1.0, color="red", linestyle="--", label="Sharpe = 1.0")
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig
