from __future__ import annotations

import numpy as np
import pandas as pd

from pyfundlib.reporting.validation import validate_equity_curve, summarize_validation


def _make_equity_curve(n: int = 252 * 3, mu: float = 0.0005, sigma: float = 0.01) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(11)
    r = rng.normal(mu, sigma, size=n)
    eq = 100 * np.cumprod(1 + r)
    return pd.Series(eq, index=idx)


def test_validate_equity_curve_produces_consistent_structures():
    equity = _make_equity_curve()
    result = validate_equity_curve(equity)

    assert isinstance(result.core, dict)
    assert isinstance(result.sharpe_ci, tuple)
    assert len(result.sharpe_ci) == 2
    assert isinstance(result.maxdd_ci, tuple)
    assert len(result.maxdd_ci) == 2
    assert hasattr(result, "regimes")
    assert hasattr(result, "regime_transitions")

    summary = summarize_validation(result)
    assert isinstance(summary, str)
    assert "Sharpe" in summary or "Validation error" in summary

