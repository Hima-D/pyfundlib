# src/pyfundlib/simulation/bootstrap.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union


def bootstrap_simulator(
    price_series: Union[pd.Series, pd.DataFrame],
    T: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
    dt: float | None = None,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Simulate price paths using non-parametric bootstrap resampling of historical returns.

    Parameters
    ----------
    price_series : pd.Series or DataFrame
        Historical price data. If a DataFrame is given, the first column is used.
    T : float
        Time horizon in years.
    n_paths : int
        Number of simulation paths.
    rng : np.random.Generator, optional
        Random number generator. If None, np.random.default_rng() is used.
    dt : float, optional
        Time step size. Default: 1/252 (daily).
    freq : str, default "B"
        Pandas frequency ("B" = business days, "D" = calendar days).

    Returns
    -------
    pd.DataFrame
        Simulated bootstrap paths with a DatetimeIndex and one column per path.
    """
    # ------------------------
    # RNG setup
    # ------------------------
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------
    # Extract price series
    # ------------------------
    if isinstance(price_series, pd.DataFrame):
        if price_series.shape[1] == 0:
            raise ValueError("DataFrame price_series must contain at least one column.")
        price_series = price_series.iloc[:, 0]

    price_series = price_series.dropna()

    if price_series.size < 2:
        raise ValueError("Price series must have at least 2 valid observations.")

    # ------------------------
    # Compute historical returns
    # ------------------------
    returns = price_series.pct_change().dropna().values  # ndarray
    n_hist = len(returns)

    # ------------------------
    # Time step setup
    # ------------------------
    if dt is None:
        dt = 1 / 252  # daily steps

    steps = int(T / dt)
    if steps <= 0:
        raise ValueError("T / dt must yield at least 1 simulation step.")

    # ------------------------
    # Bootstrap sampling
    # ------------------------
    # Random indices into historical returns (shape: [steps, paths])
    sample_idx = rng.integers(0, n_hist, size=(steps, n_paths))
    sampled_returns = returns[sample_idx]

    # ------------------------
    # Convert returns to price paths
    # ------------------------
    start_price = float(price_series.iloc[-1])

    # (1 + r_t)
    # ensure sampled_returns is a plain NumPy array (not a pandas ExtensionArray) so scalar addition is valid
    gross_returns = 1.0 + np.asarray(sampled_returns, dtype=float)

    # cumulative product → price evolution (shape: steps x paths)
    path_matrix = np.cumprod(gross_returns, axis=0)

    # prepend starting price row
    full_paths = np.vstack([
        np.full(n_paths, start_price),
        start_price * path_matrix,
    ])

    # ------------------------
    # Date index generation
    # ------------------------
    start_date = pd.Timestamp.now().normalize()

    dates = pd.bdate_range(start=start_date, periods=steps + 1, freq=freq)
    if len(dates) != full_paths.shape[0]:
        # fallback to calendar days
        dates = pd.date_range(start=start_date, periods=steps + 1, freq="D")

    # ------------------------
    # Build DataFrame (paths as columns)
    # ------------------------
    df = pd.DataFrame(
        full_paths.T,
        index=dates,
        columns=[f"Path_{i}" for i in range(n_paths)],
    )

    return df
