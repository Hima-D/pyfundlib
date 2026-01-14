from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def bootstrap_sharpe_ci(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    if r.size < 10:
        return 0.0, 0.0

    def sharpe_stat(x: np.ndarray, axis: int = 0) -> np.ndarray:
        m = np.mean(x, axis=axis)
        s = np.std(x, axis=axis, ddof=1)
        out = np.where(s > 0, m / s * np.sqrt(252.0), 0.0)
        return np.atleast_1d(out)

    data = (r,)
    res = bootstrap(
        data,
        sharpe_stat,
        confidence_level=confidence,
        n_resamples=5000,
        method="basic",
        vectorized=False,
        random_state=42,
    )

    lo = float(res.confidence_interval.low[0])
    hi = float(res.confidence_interval.high[0])
    return lo, hi


def bootstrap_maxdd_ci(equity: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    eq = np.asarray(equity).flatten()
    eq = eq[~np.isnan(eq)]
    if eq.size < 10:
        return 0.0, 0.0

    returns = np.diff(eq) / eq[:-1]

    def maxdd_stat(x: np.ndarray, axis: int = 0) -> np.ndarray:
        r = x
        eq_path = np.cumprod(1 + r, axis=axis)
        roll_max = np.maximum.accumulate(eq_path, axis=axis)
        dd = (roll_max - eq_path) / roll_max
        out = np.max(dd, axis=axis)
        return np.atleast_1d(out)

    data = (returns,)
    res = bootstrap(
        data,
        maxdd_stat,
        confidence_level=confidence,
        n_resamples=3000,
        method="basic",
        vectorized=False,
        random_state=123,
    )
    lo = float(res.confidence_interval.low[0])
    hi = float(res.confidence_interval.high[0])
    return lo, hi


@dataclass
class RegimeLabelResult:
    regimes: pd.Series
    transitions: pd.DataFrame


def simple_vol_regime_label(
    returns: pd.Series,
    low_vol_threshold: float = 0.01,
    high_vol_threshold: float = 0.03,
    window: int = 21,
) -> RegimeLabelResult:
    r = returns.dropna()
    vol = r.rolling(window).std() * np.sqrt(252.0)
    regimes = pd.Series(index=vol.index, dtype="object")
    regimes[vol <= low_vol_threshold] = "low_vol"
    regimes[(vol > low_vol_threshold) & (vol < high_vol_threshold)] = "normal_vol"
    regimes[vol >= high_vol_threshold] = "high_vol"

    transitions: Dict[Tuple[str, str], int] = {}
    prev = None
    for v in regimes.dropna():
        cur = str(v)
        if prev is not None and cur != prev:
            key = (prev, cur)
            transitions[key] = transitions.get(key, 0) + 1
        prev = cur

    if transitions:
        idx = pd.MultiIndex.from_tuples(transitions.keys(), names=["from", "to"])
        trans_df = pd.DataFrame({"count": list(transitions.values())}, index=idx)
    else:
        trans_df = pd.DataFrame(columns=["count"])

    return RegimeLabelResult(regimes=regimes, transitions=trans_df)

