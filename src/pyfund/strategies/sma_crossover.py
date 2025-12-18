# src/pyfund/strategies/sma_crossover.py

import pandas as pd

from ..indicators.sma import sma
from .base import BaseStrategy          # FIXED IMPORT


class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy

    Golden Cross: Short SMA crosses above Long SMA → Buy (1)
    Death Cross: Short SMA crosses below Long SMA → Sell (-1)
    Flat otherwise (0)

    Classic trend-following strategy used by CTAs, retail, and institutions.
    """

    default_params = {
        "short_window": 50,
        "long_window": 200,
        "use_trailing_stop": False,
        "trailing_stop_pct": 0.10,  # 10%
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.short_window = self.params["short_window"]
        self.long_window = self.params["long_window"]

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on SMA crossover

        Returns:
            pd.Series with values:
                1  → Long (golden cross)
                -1 → Short (death cross)
                0  → Flat
        """
        close = data["Close"]

        short_sma = sma(close, self.short_window)
        long_sma = sma(close, self.long_window)

        # Previous values for crossover detection
        prev_short = short_sma.shift(1)
        prev_long = long_sma.shift(1)

        signals = pd.Series(0, index=data.index)

        # Golden Cross: short crosses above long
        golden_cross = (prev_short <= prev_long) & (short_sma > long_sma)
        signals[golden_cross] = 1

        # Death Cross: short crosses below long
        death_cross = (prev_short >= prev_long) & (short_sma < long_sma)
        signals[death_cross] = -1

        # Optional: forward-fill to hold position until opposite cross
        signals = signals.replace(0, pd.NA).ffill().fillna(0)

        return signals.astype(int)

    def __repr__(self):
        return f"SMACrossoverStrategy({self.short_window}/{self.long_window})"


# Quick test when run directly
if __name__ == "__main__":
    from ..data.fetcher import DataFetcher

    df = DataFetcher.get_price("SPY", period="5y")
    strategy = SMACrossoverStrategy({"short_window": 50, "long_window": 200})
    signals = strategy.generate_signals(df)

    print("Last 10 signals:")
    print(signals.tail(10))
    print(
        f"\nCurrent position: {'LONG' if signals.iloc[-1] == 1 else 'SHORT' if signals.iloc[-1] == -1 else 'FLAT'}"
    )
