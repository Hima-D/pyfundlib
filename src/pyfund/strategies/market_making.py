# src/pyfund/strategies/market_making.py
from __future__ import annotations

from typing import Any, Tuple

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

    default_params: dict[str, Any] = {
        "base_spread_bps": 15,  # 15 bps base spread (0.15%)
        "skew_factor": 0.0005,  # How much to skew per $1 inventory
        "inventory_limit": 1000.0,  # Max absolute inventory in base currency
        "vol_window": 60,  # Volatility lookback (minutes)
        "gamma": 0.001,  # Risk aversion (higher = wider spreads)
        "kappa": 0.1,  # Speed of mean reversion for skew
        "order_size": 0.1,  # Size per quote (in base currency, e.g., 0.1 BTC)
        "max_position": 5.0,  # Hard position limit
    }

    def __init__(self, ticker: str = "BTC-USD", params: dict[str, Any] | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        self.inventory: float = 0.0
        self.cash: float = 100000.0
        self.fair_value: float = 0.0
        self.pnl: float = 0.0
        self.quote_history: list[dict[str, Any]] = []

    def _estimate_volatility(self, returns: pd.Series | np.ndarray) -> float:
        """
        Estimate (annualized) volatility from recent returns.

        Accepts either a pd.Series or a numpy array. Returns a float.
        For minute-level returns we scale by sqrt(252 * 1440) to annualize.
        """
        # Convert to Series for rolling convenience
        if isinstance(returns, np.ndarray):
            try:
                returns = pd.Series(returns)
            except Exception:
                returns = pd.Series(dtype=float)

        if not isinstance(returns, pd.Series):
            returns = pd.Series(dtype=float)

        window = int(self.params.get("vol_window", 60))
        # Use sample std (ddof=1) consistent with pandas default
        rolling_std = returns.rolling(window=window, min_periods=1).std(ddof=1).iloc[-1]
        try:
            vol = float(rolling_std) if not pd.isna(rolling_std) else 0.0
        except Exception:
            vol = 0.0

        # If returns are minute-based, annualize by sqrt(252 * 1440)
        minutes_per_day = 1440.0
        annualizer = np.sqrt(252.0 * minutes_per_day)
        return float(vol * annualizer) if vol > 0 else 0.3

    def _order_book_imbalance(self, data: pd.DataFrame) -> float:
        """Simple proxy using volume delta if real book unavailable"""
        # Accept "Volume" or "volume" columns
        if "Volume" in data.columns:
            recent = data["Volume"].tail(10).astype(float)
        elif "volume" in data.columns:
            recent = data["volume"].tail(10).astype(float)
        else:
            return 0.0

        # percent-change average as a crude imbalance proxy
        return float(recent.pct_change().fillna(0).mean())

    def _calculate_quotes(self, mid_price: float, volatility: float) -> Tuple[float, float, float]:
        """Calculate bid/ask prices and sizes with inventory skew"""
        base_spread = float(self.params.get("base_spread_bps", 15)) / 10000.0
        vol_spread = float(self.params.get("gamma", 0.001)) * (float(volatility) ** 2)

        # Dynamic spread
        spread = base_spread + vol_spread
        half_spread = spread / 2.0

        # Inventory skew: if long inventory → quote lower to sell; short → quote higher to buy
        skew_factor = float(self.params.get("skew_factor", 0.0005))
        skew = -skew_factor * float(self.inventory) * float(mid_price)

        bid_price = float(mid_price) - half_spread * float(mid_price) + skew
        ask_price = float(mid_price) + half_spread * float(mid_price) + skew

        # Size adjustment: reduce size as inventory approaches limit
        inventory_limit = float(self.params.get("inventory_limit", 1000.0))
        risk_factor = 1.0 - (abs(float(self.inventory)) / max(inventory_limit, 1.0))
        size = float(self.params.get("order_size", 0.1)) * max(risk_factor, 0.1)

        return bid_price, ask_price, size

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        In market making, we don't generate directional signals.
        Instead, we continuously quote bid/ask.
        This method returns a pd.Series containing the quote payload for the latest index.
        """
        # Defensive checks
        if len(data) < 2:
            # return a one-row Series with action WAIT for caller convenience
            last_idx = data.index[-1] if len(data.index) > 0 else pd.Timestamp.now()
            return pd.Series({"action": "WAIT", "reason": "insufficient_data"}, index=[last_idx])

        df = data.copy()

        # Ensure 'Close' exists; accept 'close' as alias
        if "Close" not in df.columns and "close" in df.columns:
            df["Close"] = df["close"]
        if "Close" not in df.columns:
            # fallback: use previous fair value or synthetic
            df["Close"] = self.fair_value if self.fair_value > 0 else 100.0

        # Compute log returns properly: log(close / close.shift(1))
        close = pd.Series(df["Close"].astype(float), index=df.index)
        returns = pd.Series(np.log(close / close.shift(1)), index=close.index).fillna(0.0)

        # Update fair value to last close
        self.fair_value = float(close.iloc[-1])

        # Estimate volatility (pass a pandas Series so _estimate_volatility's rolling works)
        volatility = self._estimate_volatility(returns)

        # Order book imbalance (optional)
        imbalance = self._order_book_imbalance(df)

        # Compute quotes
        bid, ask, size = self._calculate_quotes(self.fair_value, volatility)

        # Hard limits: if inventory too large, prefer flattening behavior
        if abs(self.inventory) >= float(self.params.get("max_position", 5.0)):
            action = "FLATTEN"
            # quote at mid to encourage fills that reduce inventory
            bid = ask = float(self.fair_value)
        else:
            action = "QUOTE"

        quote = {
            "timestamp": df.index[-1],
            "ticker": self.ticker,
            "fair_value": float(self.fair_value),
            "bid": round(float(bid), 8),
            "ask": round(float(ask), 8),
            "spread_bps": round(((float(ask) - float(bid)) / float(self.fair_value)) * 10000.0, 2),
            "size": round(float(size), 8),
            "inventory": round(float(self.inventory), 8),
            "pnl": round(float(self.pnl), 2),
            "action": action,
            "volatility": round(float(volatility), 6),
            "imbalance": round(float(imbalance), 6),
        }

        self.quote_history.append(quote)
        logger.info(
            f"MM Quote: {quote['bid']} / {quote['ask']} | Inv: {quote['inventory']} | P&L: {quote['pnl']}"
        )

        return pd.Series(quote)

    def on_fill(self, side: str, price: float, qty: float) -> None:
        """Callback when a quote is filled (simulate or live)"""
        side_l = side.lower()
        price_f = float(price)
        qty_f = float(qty)

        if side_l == "buy":
            self.inventory += qty_f
            self.cash -= price_f * qty_f
        elif side_l == "sell":
            self.inventory -= qty_f
            self.cash += price_f * qty_f
        else:
            logger.warning(f"on_fill received unknown side: {side}")

        # Update P&L: portfolio value + cash minus starting cash (100k)
        self.pnl = float(self.cash + self.inventory * float(self.fair_value) - 100000.0)
        logger.info(
            f"FILL {side_l.upper()} {qty_f} @ {price_f} | New Inv: {self.inventory:.4f} | P&L: ${self.pnl:,.2f}"
        )

    def __repr__(self) -> str:
        return f"MarketMakingStrategy({self.ticker}, inv={self.inventory:.2f}, pnl=${self.pnl:,.0f})"


# Live demo
if __name__ == "__main__":
    mm = MarketMakingStrategy("BTC-USD", {"base_spread_bps": 10, "order_size": 0.01})

    print("Starting Market Maker for BTC-USD...")
    try:
        data = DataFetcher.get_price("BTC-USD", period="1d", interval="1m")
    except Exception:
        # fallback simulated minute data
        n = 1440
        idx = pd.date_range("2025-01-01 00:00:00", periods=n, freq="T")
        price = 30000.0 + np.cumsum(np.random.normal(scale=5.0, size=n))
        data = pd.DataFrame({"Close": price}, index=idx)

    # iterate the last 100 points as a streaming demo
    start = max(0, len(data) - 100)
    for i in range(start, len(data)):
        window = data.iloc[: i + 1]
        quote = mm.generate_signals(window)
        # print as dict for readability
        print(quote.to_dict() if isinstance(quote, pd.Series) else quote)

        # Simulate random fills with small probability
        if np.random.rand() < 0.05:
            side = "buy" if np.random.rand() < 0.5 else "sell"
            price = quote["bid"] if side == "buy" else quote["ask"]
            mm.on_fill(side, price, quote["size"] * np.random.uniform(0.5, 1.0))
