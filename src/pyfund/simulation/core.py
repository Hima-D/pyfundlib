# src/pyfundlib/simulation/core.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Type aliases for clarity
PricePath = pd.DataFrame  # index: dates, columns: paths
ReturnPath = pd.Series  # single path of returns
SimulatorFunc = Callable[..., PricePath]


@dataclass
class Regime:
    """Define a market regime (bull, bear, sideways, crisis, etc.)"""

    name: str
    mu: float  # Annual drift
    sigma: float  # Annual volatility
    weight: float = 1.0  # Probability weight in regime-switching
    jump_lambda: float = 0.0  # Jumps per year (optional)
    jump_mean: float = 0.0
    jump_std: float = 0.0


@dataclass
class StressShock:
    """Single shock event"""

    date_offset: int  # Days from start
    return_shock: float  # e.g., -0.20 for -20% crash


class CustomSimulator:
    """
    The ultimate flexible simulation engine.
    Users can:
    - Register custom simulators
    - Define regime-switching models
    - Apply stress shocks
    - Mix historical + synthetic paths
    """

    def __init__(self):
        self._simulators: dict[str, SimulatorFunc] = {}
        self.register_default_simulators()

    def register_simulator(self, name: str, func: SimulatorFunc) -> None:
        """Allow users to add their own simulation methods"""
        self._simulators[name.lower()] = func
        logger.info(f"Registered custom simulator: {name}")

    def register_default_simulators(self):
        """Built-in simulators"""
        from bootstrap import bootstrap_simulator
        from gbm import gbm_simulator
        from heston import heston_simulator
        from jump_diffusion import jump_diffusion_simulator

        self.register_simulator("gbm", gbm_simulator)
        self.register_simulator("jump_diffusion", jump_diffusion_simulator)
        self.register_simulator("heston", heston_simulator)
        self.register_simulator("bootstrap", bootstrap_simulator)

    def simulate(
        self,
        method: str = "gbm",
        S0: float = 100.0,
        regimes: Sequence[Regime] | None = None,
        shocks: Sequence[StressShock] | None = None,
        T: int = 252 * 5,
        n_paths: int = 1000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> PricePath:
        """
        One API to rule all simulations — fully customizable.

        Examples
        --------
        # Simple GBM
        paths = sim.simulate("gbm", S0=500, mu=0.10, sigma=0.18)

        # Regime-switching: 70% bull, 20% bear, 10% crisis
        regimes = [
            Regime("bull", mu=0.15, sigma=0.15, weight=7),
            Regime("bear", mu=-0.10, sigma=0.30, weight=2),
            Regime("crisis", mu=-0.30, sigma=0.60, weight=1, jump_lambda=2.0, jump_mean=-0.25),
        ]
        paths = sim.simulate("regime_switch", regimes=regimes)

        # With COVID-style shock on day 30
        shocks = [StressShock(date_offset=30, return_shock=-0.34)]
        paths = sim.simulate("gbm", shocks=shocks)
        """
        method = method.lower()
        rng = np.random.default_rng(seed)

        if method not in self._simulators:
            raise ValueError(
                f"Unknown simulation method: {method}. Available: {list(self._simulators.keys())}"
            )

        # Base simulation
        base_paths = self._simulators[method](
            S0=S0,
            T=T,
            n_paths=n_paths,
            rng=rng,
            regimes=regimes,
            **kwargs,
        )

        # Apply stress shocks if any
        if shocks:
            paths = base_paths.copy()
            for shock in shocks:
                idx = min(shock.date_offset, len(paths) - 1)
                shock_factor = 1 + shock.return_shock
                paths.iloc[idx:] = paths.iloc[idx:] * shock_factor
            return paths

        return base_paths


# Global instance — users can customize it!
simulator = CustomSimulator()


# Example: Let users easily add their own!
def my_crypto_simulator(S0: float, T: int, n_paths: int, rng: np.random.Generator, **kwargs):
    """Extreme volatility + fat tails for crypto"""
    mu = kwargs.get("mu", 0.8)
    sigma = kwargs.get("sigma", 1.2)
    # Use Student-t for fat tails
    df = 3
    steps = T
    dt = 1 / 252
    Z = rng.standard_t(df, size=(steps, n_paths))
    dW = Z * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW
    log_returns = np.vstack([np.zeros(n_paths), drift + diffusion])
    paths = S0 * np.exp(np.cumsum(log_returns, axis=0))
    dates = pd.bdate_range(start="today", periods=steps + 1)
    return pd.DataFrame(paths, index=dates)


# Users can do this in their code:
# simulator.register_simulator("crypto_madness", my_crypto_simulator)
# paths = simulator.simulate("crypto_madness", S0=60000, mu=1.5, sigma=0.9)
