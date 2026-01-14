from __future__ import annotations

import pandas as pd
import numpy as np

from pyfundlib.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from pyfundlib.backtester.engine import Backtester


def _make_trending_series(days: int = 252 * 3) -> pd.DataFrame:
    idx = pd.bdate_range(start="2020-01-01", periods=days)
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0005, 0.01, size=len(idx))
    prices = 100 * (1 + pd.Series(returns, index=idx)).cumprod()
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1_000}, index=idx)
    return df


def test_time_series_momentum_generates_signals_and_backtests():
    df = _make_trending_series()
    strategy = TimeSeriesMomentumStrategy(
        {
            "lookback_days": 126,
            "vol_lookback_days": 63,
            "vol_target": 0.20,
            "neutral_threshold": 0.05,
        }
    )

    signals = strategy.generate_signals(df)
    assert len(signals) == len(df)
    assert signals.dtype.kind in ("f", "i")

    bt = Backtester(strategy=strategy, data=df, name="ts_momentum_test")
    result = bt.run()

    assert result.equity_curve.iloc[0] > 0
    assert len(result.trades) >= 0
    assert "cagr" in result.metrics

