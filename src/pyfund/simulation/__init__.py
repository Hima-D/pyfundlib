# src/pyfundlib/simulation/__init__.py
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import genextreme, norm, t

from ..data.processor import DataProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarketSimulator:
    """
    Advanced market simulation engine:
    - Geometric Brownian Motion (GBM)
    - Bootstrapped historical simulation
    - Jump-diffusion (Merton model)
    - Regime-switching
    - Stress scenarios (2008, COVID, Flash Crash)
    """

    @staticmethod
    def gbm(
        S0: float,
        mu: float,
        sigma: float,
        T: int = 252,
        dt: float = 1 / 252,
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Geometric Brownian Motion simulation (Black-Scholes world).

        Parameters
        ----------
        S0 : float
            Initial price
        mu : float
            Annual drift (expected return)
        sigma : float
            Annual volatility
        T : int
            Number of trading days
        n_paths : int
            Number of simulated paths

        Returns
        -------
        pd.DataFrame
            Index: trading days, Columns: simulated price paths
        """
        rng = np.random.default_rng(seed)
        steps = int(T / dt)
        Z = rng.normal(0, 1, size=(steps, n_paths))
        dW = Z * np.sqrt(dt)

        # Drift + diffusion
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * dW

        log_returns = drift + diffusion
        log_returns = np.vstack([np.zeros((1, n_paths)), log_returns])
        price_paths = S0 * np.exp(np.cumsum(log_returns, axis=0))

        dates = pd.bdate_range(start=pd.Timestamp.today(), periods=steps + 1)
        return pd.DataFrame(price_paths, index=dates, columns=range(n_paths))

    @staticmethod
    def bootstrap(
        historical_returns: pd.Series,
        n_paths: int = 1000,
        horizon: int = 252,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Block bootstrap simulation preserving autocorrelation.
        """
        rng = np.random.default_rng(seed)
        returns = historical_returns.dropna().values
        n = len(returns)
        paths = []

        for _ in range(n_paths):
            # Random blocks to preserve short-term correlation
            path = []
            remaining = horizon
            idx = 0
            while remaining > 0:
                block_size = min(rng.integers(5, 30), remaining, n - idx)
                block = returns[idx : idx + block_size]
                path.extend(block)
                idx = rng.integers(0, n)
                remaining -= block_size
            paths.append(np.cumprod(1 + np.array(path))[-1])  # final equity multiplier

        return pd.Series(paths, name="final_equity_multiplier")

    @staticmethod
    def jump_diffusion(
        S0: float,
        mu: float,
        sigma: float,
        lambd: float = 0.1,  # jumps per year
        jump_mean: float = -0.05,
        jump_std: float = 0.1,
        T: int = 252,
        n_paths: int = 1000,
    ) -> pd.DataFrame:
        """
        Merton Jump-Diffusion model (real markets have fat tails!).
        """
        rng = np.random.default_rng()
        dt = 1 / 252
        steps = T

        paths = np.zeros((steps + 1, n_paths))
        paths[0] = S0

        for t in range(1, steps + 1):
            # Diffusion component
            dW = rng.normal(0, np.sqrt(dt), n_paths)
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW

            # Jump component
            jumps = rng.poisson(lambd * dt, n_paths)
            jump_sizes = np.exp(rng.normal(jump_mean, jump_std, n_paths)) - 1
            jump_component = jumps * jump_sizes

            paths[t] = paths[t - 1] * np.exp(diffusion + jump_component)

        dates = pd.bdate_range(start="today", periods=steps + 1)
        return pd.DataFrame(paths, index=dates)

    @staticmethod
    def stress_scenario(
        returns: pd.Series,
        scenarios: dict[str, Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """
        Pre-defined historical stress scenarios.
        """
        if scenarios is None:
            scenarios = {
                "2008 Financial Crisis": [-0.07, -0.11, -0.08, -0.12, -0.09, -0.10, -0.05, -0.06],
                "COVID Crash (Mar 2020)": [-0.12, -0.09, -0.34, -0.12, 0.12, 0.09, 0.11, 0.05],
                "2022 Bear Market": [-0.04, -0.03, -0.09, -0.05, -0.08, -0.06, -0.04, -0.07],
                "Flash Crash (May 2010)": [-0.03, -0.02, -0.09, 0.08, -0.04, 0.06, 0.05, -0.01],
                "Bull Market (2023-2024)": [0.03, 0.05, 0.04, 0.06, 0.07, 0.05, 0.08, 0.04],
            }

        results = {}
        # equity = 1.0
        base_# equity = (1 + returns).cumprod()

        for name, shocks in scenarios.items():
            shocked_returns = returns.copy()
            for i, shock in enumerate(shocks):
                if i < len(shocked_returns):
                    shocked_returns.iloc[i] += shock
            equity_curve = (1 + shocked_returns).cumprod()
            results[name] = {
                "final_return": equity_curve.iloc[-1] - 1,
                "max_drawdown": (
                    (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
                ).max(),
                "vs_baseline": equity_curve.iloc[-1] / base_equity.iloc[-1] - 1,
            }

        return pd.DataFrame(results).T


# Bonus: Monte Carlo VaR / Strategy Simulator
class StrategySimulator:
    """Run thousands of strategy paths with parameter jitter"""

    @staticmethod
    def monte_carlo_backtest(
        strategy_class,
        df: pd.DataFrame,
        param_distributions: dict[str, Any],
        n_trials: int = 1000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Example:
            param_distributions = {
                "rsi_period": (10, 20),
                "entry_threshold": (20, 40),
                "exit_threshold": (60, 80),
            }
        """
        rng = np.random.default_rng(seed)
        results = []

        for _ in range(n_trials):
            params = {}
            for param, dist in param_distributions.items():
                if isinstance(dist, tuple):
                    params[param] = rng.integers(dist[0], dist[1] + 1)
                else:
                    params[param] = rng.choice(dist)

            strategy = strategy_class(**params)
            signals = strategy.generate_signals(df)
            # equity = (1 + signals.shift(1) * df["Close"].pct_change()).cumprod()

            results.append(
                {
                    **params,
                    "total_return": equity.iloc[-1] - 1,
                    "cagr": (equity.iloc[-1]) ** (252 / len(equity)) - 1,
                    "max_dd": ((equity.cummax() - equity) / equity.cummax()).max(),
                    "sharpe": equity.pct_change().mean() / equity.pct_change().std() * np.sqrt(252),
                }
            )

        return pd.DataFrame(results)
