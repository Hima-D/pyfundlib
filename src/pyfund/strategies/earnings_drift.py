# src/pyfund/strategies/earning_drift.py
from datetime import timedelta

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.logger import logger
from .base import BaseStrategy


class EarningDriftStrategy(BaseStrategy):
    """
    Post-Earnings Announcement Drift (PEAD) + Earnings Surprise Strategy

    One of the strongest and most persistent market anomalies:
    - Stocks with positive earnings surprises outperform
    - Negative surprises underperform
    - Effect lasts 30-90 days (sometimes longer)
    - Amplified by estimate revisions and analyst sentiment

    Ball & Brown (1968), Foster et al., Bernard & Thomas — still works in 2025!
    """

    default_params = {
        "surprise_threshold": 0.10,  # |EPS surprise| > 10% to trigger
        "lookback_days": 90,  # Hold period (classic PEAD window)
        "min_market_cap": 500_000_000,  # Avoid micro-caps
        "volume_filter": 1.0,  # Avg volume > $1M/day
        "revision_momentum": True,  # Require upward/downward estimate revisions
        "z_score_threshold": 1.5,  # Earnings surprise z-score
        "max_positions": 20,  # Portfolio concentration
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.active_positions: dict[str, dict] = {}  # ticker -> entry info

    def _get_earnings_surprise(self, ticker: str) -> dict | None:
        """Fetch latest earnings surprise (simulated — in real use: Alpha Vantage, Polygon, etc.)"""
        try:
            # In real implementation, use proper financial data API
            # Here we simulate realistic surprise distribution
            df = DataFetcher.get_price(ticker, period="2y")
            if len(df) < 100:
                return None

            # Simulate quarterly earnings dates (every ~90 days)
            last_date = df.index[-1]
            earnings_dates = pd.date_range(last_date - timedelta(days=365), last_date, freq="90D")[
                -4:
            ]

            # Simulate EPS actual vs expected
            surprises = []
            for ed in earnings_dates:
                if ed > last_date:
                    break
                surprise_pct = np.random.normal(0.02, 0.15)  # Mean +2%, std 15%
                surprises.append(
                    {
                        "date": ed,
                        "surprise_pct": surprise_pct,
                        "z_score": surprise_pct / 0.10,  # Normalize
                    }
                )

            latest = surprises[-1]
            if abs(latest["surprise_pct"]) > self.params["surprise_threshold"]:
                return {
                    "ticker": ticker,
                    "earnings_date": latest["date"],
                    "surprise_pct": latest["surprise_pct"],
                    "z_score": latest["z_score"],
                    "direction": 1 if latest["surprise_pct"] > 0 else -1,
                }
        except Exception:
            pass
        return None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate PEAD signals across a universe
        In practice: scan 1000+ stocks daily after earnings season
        """
        signals = pd.Series(0, index=data.index)

        # In real use: loop over universe
        universe = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "JPM", "V", "MA", "PYPL", "BAC"]

        active_tickers = []
        for ticker in universe:
            surprise = self._get_earnings_surprise(ticker)
            if not surprise:
                continue

            days_since_earnings = (data.index[-1] - surprise["earnings_date"]).days

            # Entry window: 1-5 days after earnings
            if 1 <= days_since_earnings <= 5:
                if abs(surprise["z_score"]) > self.params["z_score_threshold"]:
                    position_size = surprise["direction"]
                    self.active_positions[ticker] = {
                        "entry_date": data.index[-1],
                        "direction": position_size,
                        "surprise": surprise["surprise_pct"],
                    }
                    active_tickers.append(ticker)
                    logger.info(
                        f"PEAD Signal: {ticker} | Surprise: {surprise['surprise_pct']:+.1%} | Direction: {'LONG' if position_size > 0 else 'SHORT'}"
                    )

            # Hold up to lookback_days
            elif ticker in self.active_positions:
                entry = self.active_positions[ticker]
                if days_since_earnings > self.params["lookback_days"]:
                    del self.active_positions[ticker]
                else:
                    active_tickers.append(ticker)

        # In real backtest: assign equal weight
        if active_tickers:
            weight = 1.0 / len(active_tickers)
            for t in active_tickers:
                direction = self.active_positions[t]["direction"]
                # Simplified: assign signal based on direction
                signals.iloc[-1] = direction * weight  # For multi-asset overlay

        return signals

    def __repr__(self):
        return f"EarningDriftStrategy(positions={len(self.active_positions)})"


# Live test
if __name__ == "__main__":
    strategy = EarningDriftStrategy({"surprise_threshold": 0.08, "lookback_days": 60})

    # Simulate daily scan
    data = pd.DataFrame(index=pd.date_range("2025-01-01", periods=200))
    signals = strategy.generate_signals(data)

    print(f"Active PEAD positions: {len(strategy.active_positions)}")
    for t, info in strategy.active_positions.items():
        print(f"  {t}: {info['direction']:+} | Surprise: {info['surprise']:+.1%}")
