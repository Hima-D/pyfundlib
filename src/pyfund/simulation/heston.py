# src/pyfundlib/simulation/heston.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def heston_simulator(
    S0: float,
    v0: float,
    T: float,
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    mu: float = 0.10,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    dt: float | None = None,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Simulate asset price paths using the Heston Stochastic Volatility Model.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance (volatility squared). Example: sigma=20% → v0 = 0.20**2 = 0.04
    T : float
        Time horizon in years.
    n_paths : int
        Number of simulated paths.
    rng : np.random.Generator, optional
        Random number generator.
    mu : float, default 0.10
        Drift (expected return).
    kappa : float, default 2.0
        Speed of mean reversion for variance.
    theta : float, default 0.04
        Long-run average variance.
    xi : float, default 0.3
        Volatility of volatility (vol-of-vol).
    rho : float, default -0.7
        Correlation between price and volatility shocks.
    dt : float, optional
        Time step. Defaults to 1/252 (daily).
    freq : str, default "B"
        Pandas date frequency ("B" = business days, "D" = calendar days).

    Returns
    -------
    pd.DataFrame
        Price simulation with DatetimeIndex and one column per simulation path.

    Notes
    -----
    - Uses Euler discretization for the CIR variance process.
    - Ensures variance stays non-negative (full truncation scheme).
    - Produces volatility clustering and skew, unlike GBM.
    """

    # ------------------------
    # RNG setup
    # ------------------------
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------
    # Time step setup
    # ------------------------
    if dt is None:
        dt = 1 / 252  # daily

    steps = int(T / dt)
    if steps <= 0:
        raise ValueError("T / dt must yield at least one simulation step.")

    # ------------------------
    # Allocate arrays
    # ------------------------
    S = np.zeros((steps + 1, n_paths))
    v = np.zeros((steps + 1, n_paths))

    S[0] = S0
    v[0] = max(v0, 1e-8)

    # Correlated Brownian increments
    Z1 = rng.normal(size=(steps, n_paths))
    Z2 = rng.normal(size=(steps, n_paths))
    Z2_corr = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    # ------------------------
    # Simulation loop (vectorized over paths)
    # ------------------------
    for t in range(steps):
        v_t = v[t]

        # Variance (CIR process with full truncation)
        dv = kappa * (theta - np.maximum(v_t, 0)) * dt + xi * np.sqrt(np.maximum(v_t, 0)) * np.sqrt(dt) * Z2_corr[t]
        v[t + 1] = np.maximum(v_t + dv, 0)  # keep non-negative

        # Price
        S[t + 1] = S[t] * np.exp(
            (mu - 0.5 * v_t) * dt + np.sqrt(np.maximum(v_t, 0)) * np.sqrt(dt) * Z1[t]
        )

    # ------------------------
    # Date index
    # ------------------------
    start_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(start=start_date, periods=steps + 1, freq=freq)

    if len(dates) != steps + 1:
        # fallback to calendar if mismatch
        dates = pd.date_range(start=start_date, periods=steps + 1, freq="D")

    # ------------------------
    # Build DataFrame
    # ------------------------
    df = pd.DataFrame(
        S.T,
        index=dates,
        columns=[f"Path_{i}" for i in range(n_paths)]
    )
    return df
