# src/pyfundlib/simulation/gbm.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

def gbm_simulator(
    S0: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
    mu: float = 0.10,
    sigma: float = 0.18,
    dt: float | None = None,
    freq: str = "B",  # Business day frequency
) -> pd.DataFrame:
    """
    Simulate asset price paths using Geometric Brownian Motion (GBM).

    Parameters
    ----------
    S0 : float
        Initial asset price.
    T : float
        Time horizon in years.
    n_paths : int
        Number of simulation paths.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
    mu : float, default 0.10
        Annual drift (expected return).
    sigma : float, default 0.18
        Annual volatility.
    dt : float, optional
        Time step size. If None, uses 1/252 for daily.
    freq : str, default "B"
        Pandas frequency for dates ("B" for business days, "D" for calendar).

    Returns
    -------
    pd.DataFrame
        Simulated paths with dates as index and paths as columns.

    Notes
    -----
    - Assumes risk-neutral or real-world measure based on mu.
    - Paths start at S0 and evolve daily (or as per dt/freq).
    - For production, consider antithetic variates or other variance reductions.
    """
    if rng is None:
        rng = np.random.default_rng()

    if dt is None:
        dt = 1 / 252  # Daily steps

    steps = int(T / dt)
    Z = rng.normal(0, 1, (steps, n_paths))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = np.cumsum(drift + diffusion, axis=0)
    paths = S0 * np.exp(np.vstack([np.zeros(n_paths), log_returns]))

    # Generate dates
    start_date = pd.Timestamp.now().normalize()  # Start from today at midnight
    dates = pd.bdate_range(start=start_date, periods=steps + 1, freq=freq)
    if len(dates) != paths.shape[0]:
        dates = pd.date_range(start=start_date, periods=steps + 1, freq="D")  # Fallback to calendar if mismatch

    df = pd.DataFrame(paths.T, index=dates, columns=[f"Path_{i}" for i in range(n_paths)])
    return df