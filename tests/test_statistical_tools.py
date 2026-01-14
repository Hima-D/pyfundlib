from __future__ import annotations

import numpy as np
import pandas as pd

from pyfundlib.utils.statistical_tools import (
    bootstrap_sharpe_ci,
    bootstrap_maxdd_ci,
    simple_vol_regime_label,
)


def _make_random_walk(n: int = 252 * 3, mu: float = 0.0005, sigma: float = 0.01):
    rng = np.random.default_rng(7)
    r = rng.normal(mu, sigma, size=n)
    eq = 100 * np.cumprod(1 + r)
    return r, eq


def test_bootstrap_sharpe_ci_shapes_and_order():
    r, _ = _make_random_walk()
    lo, hi = bootstrap_sharpe_ci(r, confidence=0.9)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo <= hi


def test_bootstrap_maxdd_ci_shapes_and_range():
    r, eq = _make_random_walk()
    lo, hi = bootstrap_maxdd_ci(eq, confidence=0.9)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi


def test_simple_vol_regime_label_outputs_regimes_and_transitions():
    r, _ = _make_random_walk()
    idx = pd.bdate_range("2020-01-01", periods=len(r))
    series = pd.Series(r, index=idx)

    res = simple_vol_regime_label(series, low_vol_threshold=0.01, high_vol_threshold=0.03, window=21)
    assert isinstance(res.regimes, pd.Series)
    assert res.regimes.dropna().isin(["low_vol", "normal_vol", "high_vol"]).all()
    assert hasattr(res, "transitions")
