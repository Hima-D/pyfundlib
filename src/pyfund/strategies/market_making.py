# src/pyfund/strategies/market_making.py

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.logger import logger
from .base import BaseStrategy


class MarketMakingStrategy(BaseStrategy):
    """
    High-Frequency Style Market Making Strategy (for sim/live crypto/stocks)

    Features:
    - Dynamic bid/ask quoting based on fair value + skew
    - Inventory risk control (skew quotes to flatten)
    - Volatility-adaptive spread
    - Adverse selection protection (widen on fast moves)
    - Order book imbalance detection
    - P&L tracking
    """

    default_params = {
        "base_spread_bps": 15,  # 15 bps base spread (0.15%)
        "skew_factor": 0.0005,  # How much to skew per $1 inventory
        "inventory_limit": 1000.0,  # Max absolute inventory in base currency
        "vol_window": 60,  # Volatility lookback (minutes)
        "gamma": 0.001,  # Risk aversion (higher = wider spreads)
        "kappa": 0.1,  # Speed of mean reversion for skew
        "order_size": 0.1,  # Size per quote (in base currency, e.g., 0.1 BTC)
        "max_position": 5.0,  # Hard position limit
    }

    def __init__(self, ticker: str = "BTC-USD", params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        self.inventory = 0.0
        self.cash = 100000.0
        self.fair_value = 0.0
        self.pnl = 0.0
        self.quote_history = []

    def _estimate_volatility(self, returns: pd.Series) -> float:
        """Annualized volatility from recent returns"""
        vol = returns.rolling(self.params["vol_window"]).std().iloc[-1]
        return vol * np.sqrt(252 * 1440) if not pd.isna(vol) else 0.3  # 1440 minutes/day

    def _order_book_imbalance(self, data: pd.DataFrame) -> float:
        """Simple proxy using volume delta if real book unavailable"""
        if "Volume" not in data.columns:
            return 0.0
        recent = data["Volume"].tail(10)
        return (recent.pct_change().fillna(0)).mean()

    def _calculate_quotes(
        self, mid_price: float, volatility: float
    ) -> tuple[float, float, float | float]:
        """Calculate bid/ask prices and sizes with inventory skew"""
        base_spread = self.params["base_spread_bps"] / 10000
        vol_spread = self.params["gamma"] * volatility**2

        # Dynamic spread
        spread = base_spread + vol_spread
        half_spread = spread / 2

        # Inventory skew: if long inventory → quote lower to sell
        skew = -self.params["skew_factor"] * self.inventory * mid_price

        bid_price = mid_price - half_spread * mid_price + skew
        ask_price = mid_price + half_spread * mid_price + skew

        # Size adjustment: reduce size as inventory approaches limit
        risk_factor = 1.0 - (abs(self.inventory) / self.params["inventory_limit"])
        size = self.params["order_size"] * max(risk_factor, 0.1)

        return bid_price, ask_price, size

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        In market making, we don't generate directional signals.
        Instead, we continuously quote bid/ask.
        This method returns current quote state.
        """
        if len(data) < 10:
            return pd.Series(
                {"action": "WAIT", "reason": "insufficient_data"}, index=data.index[-1:]
            )

        close = data["Close"]
        returns = np.log(close / close.pct_change().shift(1))

        self.fair_value = close.iloc[-1]
        volatility = self._estimate_volatility(returns)
        imbalance = self._order_book_imbalance(data)

        bid, ask, size = self._calculate_quotes(self.fair_value, volatility)

        # Hard limits
        if abs(self.inventory) >= self.params["max_position"]:
            action = "FLATTEN"
            bid = ask = self.fair_value  # Quote at mid to exit
        else:
            action = "QUOTE"

        quote = {
            "timestamp": data.index[-1],
            "ticker": self.ticker,
            "fair_value": self.fair_value,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "spread_bps": round((ask - bid) / self.fair_value * 10000, 1),
            "size": round(size, 4),
            "inventory": round(self.inventory, 4),
            "pnl": round(self.pnl, 2),
            "action": action,
            "volatility": round(volatility, 3),
        }

        self.quote_history.append(quote)
        logger.info(
            f"MM Quote: {quote['bid']} / {quote['ask']} | Inv: {quote['inventory']} | P&L: {quote['pnl']}"
        )

        return pd.Series(quote)

    def on_fill(self, side: str, price: float, qty: float):
        """Callback when a quote is filled (simulate or live)"""
        if side == "buy":
            self.inventory += qty
            self.cash -= price * qty
        else:  # sell
            self.inventory -= qty
            self.cash += price * qty

        self.pnl = self.cash + self.inventory * self.fair_value - 100000.0
        logger.info(
            f"FILL {side.upper()} {qty} @ {price} | New Inv: {self.inventory:.4f} | P&L: ${self.pnl:,.2f}"
        )

    def __repr__(self):
        return (
            f"MarketMakingStrategy({self.ticker}, inv={self.inventory:.2f}, pnl=${self.pnl:,.0f})"
        )


# Live demo
if __name__ == "__main__":
    mm = MarketMakingStrategy("BTC-USD", {"base_spread_bps": 10, "order_size": 0.01})

    print("Starting Market Maker for BTC-USD...")
    data = DataFetcher.get_price("BTC-USD", period="1d", interval="1m")

    for i in range(len(data) - 100, len(data)):
        quote = mm.generate_signals(data.iloc[: i + 1])
        print(quote.to_dict() if isinstance(quote, pd.Series) else quote)

        # Simulate random fill
        if np.random.rand() < 0.1:
            side = "buy" if np.random.rand() < 0.5 else "sell"
            price = quote["bid"] if side == "buy" else quote["ask"]
            mm.on_fill(side, price, quote["size"] * np.random.uniform(0.5, 1.0))
