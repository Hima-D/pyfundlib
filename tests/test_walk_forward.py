from __future__ import annotations

import pandas as pd
import numpy as np

from pyfundlib.backtester.walk_forward import WalkForwardBacktester
from pyfundlib.strategies.sma_crossover import SMACrossoverStrategy


def _make_price_series(days: int = 252 * 5) -> pd.DataFrame:
    idx = pd.bdate_range(start="2020-01-01", periods=days)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, 0.01, size=len(idx))
    prices = 100 * (1 + pd.Series(returns, index=idx)).cumprod()
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1_000}, index=idx)
    return df


def test_walk_forward_runs_and_returns_slices():
    df = _make_price_series()

    wf = WalkForwardBacktester(
        strategy_class=SMACrossoverStrategy,
        params={"short_window": 20, "long_window": 60},
        window_train=252,
        window_test=126,
        min_length=252 * 2,
    )

    result = wf.run(df)

    assert len(result.slices) >= 1
    mf = result.metrics_frame
    assert not mf.empty
    assert "cagr" in mf.columns
    assert "sharpe" in mf.columns
    assert isinstance(result.mean_sharpe, float)
    assert isinstance(result.mean_cagr, float)

