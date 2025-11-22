# src/pyfund/strategies/pair_trading.py

import pandas as pd
import statsmodels.api as sm

from ..data.fetcher import DataFetcher
from .base import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    """
    Classic Statistical Arbitrage: Pairs Trading

    Finds cointegrated pairs and trades the spread mean reversion.
    - Long the underperformer, short the overperformer when spread diverges
    - Exit when spread reverts to mean
    - Fully hedge-ratio adjusted (beta-neutral)

    Works amazingly on ETFs, stocks in same sector, futures, crypto.
    """

    default_params = {
        "ticker_a": "XOM",
        "ticker_b": "CVX",
        "lookback_window": 252,  # Formation period
        "entry_zscore": 2.0,  # Enter when |z| > 2.0
        "exit_zscore": 0.5,  # Exit when |z| < 0.5
        "stop_zscore": 4.0,  # Hard stop if |z| > 4.0
        "adf_pvalue_threshold": 0.05,  # Cointegration test threshold
        "min_hedge_ratio": 0.1,  # Avoid extreme ratios
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker_a = self.params["ticker_a"]
        self.ticker_b = self.params["ticker_b"]
        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std: float = 1.0
        self.is_cointegrated: bool = False

    def _check_cointegration(self, price_a: pd.Series, price_b: pd.Series) -> tuple[bool | float]:
        """Test for cointegration using Engle-Granger"""
        if len(price_a) != len(price_b):
            return False, 1.0

        # Align indices
        df = pd.concat([price_a, price_b], axis=1).dropna()
        if len(df) < 100:
            return False, 1.0

        # Run OLS: A ~ B
        X = sm.add_constant(df.iloc[:, 1])
        y = df.iloc[:, 0]
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params[1]

        # Avoid extreme hedge ratios
        if abs(hedge_ratio) < self.params["min_hedge_ratio"]:
            return False, 1.0

        spread = y - hedge_ratio * X.iloc[:, 1]
        adf_result = sm.tsa.stattools.adfuller(spread)
        p_value = adf_result[1]

        is_coint = p_value < self.params["adf_pvalue_threshold"]
        return is_coint, hedge_ratio

    def _calculate_spread_zscore(self, data_a: pd.Series, data_b: pd.Series) -> pd.Series:
        """Calculate normalized spread z-score"""
        df = pd.concat([data_a, data_b], axis=1).dropna()
        df.columns = ["A", "B"]

        # Use rolling window for dynamic mean/std
        spread = df["A"] - self.hedge_ratio * df["B"]
        rolling_mean = spread.rolling(window=20, min_periods=10).mean()
        rolling_std = spread.rolling(window=20, min_periods=10).std()

        zscore_series = (spread - rolling_mean) / rolling_std
        return zscore_series.reindex(data_a.index).fillna(0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate pair trading signals
        Returns:
            +1 → Long A, Short B (spread too low)
            -1 → Short A, Long B (spread too high)
             0 → Neutral
        """
        if "Close_A" not in data.columns or "Close_B" not in data.columns:
            # Fetch and prepare data
            try:
                df_a = DataFetcher.get_price(self.ticker_a, period="5y")["Close"]
                df_b = DataFetcher.get_price(self.ticker_b, period="5y")["Close"]
            except Exception:
                return pd.Series(0, index=data.index)

            common_index = df_a.index.intersection(df_b.index)
            price_a = df_a.loc[common_index]
            price_b = df_b.loc[common_index]
        else:
            price_a = data["Close_A"]
            price_b = data["Close_B"]

        # Step 1: Check cointegration on formation period
        formation_end = len(price_a) - 100  # Last 100 days for trading
        if formation_end > self.params["lookback_window"]:
            form_a = price_a.iloc[-self.params["lookback_window"] : -100]
            form_b = price_b.iloc[-self.params["lookback_window"] : -100]
            self.is_cointegrated, self.hedge_ratio = self._check_cointegration(form_a, form_b)
        else:
            self.is_cointegrated = False

        if not self.is_cointegrated:
            print(f"No cointegration between {self.ticker_a} and {self.ticker_b}")
            return pd.Series(0, index=data.index)

        # Step 2: Calculate live z-score on trading period
        zscore_series = self._calculate_spread_zscore(price_a, price_b)

        signals = pd.Series(0, index=data.index)

        position = 0
        for i in range(len(zscore_series)):
            z = zscore_series.iloc[i]

            if position == 0:
                if z > self.params["entry_zscore"]:
                    signals.iloc[i] = -1  # Short spread
                    position = -1
                elif z < -self.params["entry_zscore"]:
                    signals.iloc[i] = 1  # Long spread
                    position = 1
            else:
                # Exit conditions
                if abs(z) < self.params["exit_zscore"] or abs(z) > self.params["stop_zscore"]:
                    signals.iloc[i] = 0
                    position = 0
                else:
                    signals.iloc[i] = position  # Hold

        return signals

    def __repr__(self):
        return f"PairsTrading({self.ticker_a}-{self.ticker_b}, hedge={self.hedge_ratio:.3f})"


# Quick test
if __name__ == "__main__":
    strategy = PairsTradingStrategy(
        {"ticker_a": "KO", "ticker_b": "PEP", "entry_zscore": 2.0, "exit_zscore": 0.75}
    )

    # Simulate data
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=1000))
    df_a = DataFetcher.get_price("KO", period="5y")["Close"]
    df_b = DataFetcher.get_price("PEP", period="5y")["Close"]
    data = pd.concat([df_a, df_b], axis=1).dropna()
    data.columns = ["Close_A", "Close_B"]

    signals = strategy.generate_signals(data)
    print(f"Cointegrated: {strategy.is_cointegrated}")
    print(f"Hedge Ratio: {strategy.hedge_ratio:.3f}")
    print(f"Total trades: {(signals.diff().abs() > 0).sum()}")
    print(signals.value_counts())
