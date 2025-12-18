from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union


def jump_diffusion_simulator(
    S0: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
    mu: float = 0.10,
    sigma: float = 0.18,
    lambda_jump: float = 0.5,         # avg number of jumps per year
    mu_jump: float = -0.10,           # avg jump size (lognormal mean, negative = crashes)
    sigma_jump: float = 0.20,         # jump size volatility
    dt: float | None = None,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Simulate asset price paths using Merton Jump-Diffusion model.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    T : float
        Time horizon in years.
    n_paths : int
        Number of simulation paths.
    rng : np.random.Generator, optional
        Random generator. If None, uses np.random.default_rng().
    mu : float, default 0.10
        Annual drift.
    sigma : float, default 0.18
        Annual volatility.
    lambda_jump : float, default 0.5
        Expected number of jumps per year (Poisson intensity).
    mu_jump : float, default -0.10
        Mean jump size (lognormal). Negative means downward crashes.
    sigma_jump : float, default 0.20
        Std deviation of jump size.
    dt : float, optional
        Time step. If None, uses 1/252.
    freq : str, default "B"
        Date frequency.

    Returns
    -------
    pd.DataFrame
        Simulated price paths with jumps.
    """
    if rng is None:
        rng = np.random.default_rng()

    if dt is None:
        dt = 1 / 252  # Daily steps

    steps = int(T / dt)

    # === Diffusion part ===
    Z = rng.normal(0, 1, (steps, n_paths))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # === Jump part ===
    # Poisson(λ * dt) for whether a jump occurs
    N_jumps = rng.poisson(lambda_jump * dt, (steps, n_paths))

    # Jump sizes (lognormal)
    jump_sizes = rng.normal(mu_jump, sigma_jump, (steps, n_paths))
    J = N_jumps * jump_sizes  # jumps applied when Poisson draws > 0

    # === Combine diffusion + jumps ===
    log_returns = np.cumsum(drift + diffusion + J, axis=0)

    # Start path at S0
    paths = S0 * np.exp(np.vstack([np.zeros(n_paths), log_returns]))

    # === Dates ===
    start_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(start=start_date, periods=steps + 1, freq=freq)
    if len(dates) != paths.shape[0]:
        dates = pd.date_range(start=start_date, periods=steps + 1, freq="D")

    df = pd.DataFrame(paths.T, index=dates, columns=[f"Path_{i}" for i in range(n_paths)])
    return df
