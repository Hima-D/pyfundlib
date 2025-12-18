# src/pyfund/strategies/delta_hedged_straddle.py
from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional, Tuple
from typing import Dict

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.logger import logger
from .base import BaseStrategy


class DeltaHedgedStraddleStrategy(BaseStrategy):
    """
    Delta-Hedged Straddle (Volatility Arbitrage)

    Core idea:
    - Buy ATM straddle (call + put) → long vega, short gamma
    - Continuously delta-hedge with underlying
    - P&L ≈ (Realized Vol² - Implied Vol²) × Vega
    - Profits when realized volatility > implied volatility
    """

    default_params = {
        "dte_target": 30,  # Target days to expiration
        "dte_tolerance": 7,  # Acceptable range
        "hedge_frequency": "daily",  # daily, intraday, continuous
        "rebalance_threshold": 0.10,  # Rehedge when delta > 10%
        "straddle_type": "atm",  # atm, 5delta, etc.
        "max_position_size": 100,  # Max straddles
        "stop_loss_pct": 0.50,  # 50% loss → close
        "take_profit_pct": 1.00,  # 100% gain → close
        "include_dividends": True,
    }

    def __init__(self, ticker: str = "SPY", params: dict[str, Any] | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        # Use explicit types for position keys so Pylance understands values
        self.position: dict[str, Any] = {
            "entry_date": None,  # Optional[pd.Timestamp]
            "expiration": None,  # Optional[pd.Timestamp]
            "strike": 0.0,
            "call_premium": 0.0,
            "put_premium": 0.0,
            "straddle_cost": 0.0,
            "shares_hedged": 0.0,
            "total_pnl": 0.0,
            "hedge_pnl": 0.0,
            "gamma_pnl": 0.0,
            "vega_pnl": 0.0,
            "theta_decay": 0.0,
        }

    def _get_atm_options_chain(self, date: pd.Timestamp) -> dict | None:
        """Simulate fetching ATM straddle (in real use: yfinance, polygon, tastytrade, etc.)"""
        try:
            # DataFetcher.get_price should return DataFrame with DatetimeIndex and "Close" column
            price_df = DataFetcher.get_price(self.ticker, period="10d")
            # Make sure index is DatetimeIndex
            if not isinstance(price_df.index, pd.DatetimeIndex):
                price_df.index = pd.DatetimeIndex(price_df.index)
            price = float(price_df.loc[date]["Close"])
            iv = 0.20 + float(np.random.normal(0, 0.05))  # Simulate IV
            dte = int(self.params["dte_target"] + np.random.randint(-3, 4))

            strike = float(round(price / 5) * 5)  # Round to nearest $5
            # Option premium heuristic (very simplified)
            call_premium = float(price * iv * np.sqrt(dte / 365.0) / 4.0)
            put_premium = float(call_premium * 1.05)  # Slight put skew

            return {
                "date": date,
                "expiration": pd.Timestamp(date + timedelta(days=dte)),
                "strike": strike,
                "call_premium": round(call_premium, 2),
                "put_premium": round(put_premium, 2),
                "iv": round(float(iv), 3),
                "dte": dte,
            }
        except Exception:
            return None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate delta-hedged straddle signals
        - Look for entry near month-end or after vol crush
        - Hold ~30 days with daily delta hedging
        """
        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.DatetimeIndex(data.index)

        # Ensure Close column exists (caller should provide, but be defensive)
        if "Close" not in data.columns:
            data = data.copy()
            # Backfill Close with forward-filled values of synthetic price 100 if absent
            data["Close"] = 100.0

        signals = pd.Series(0, index=data.index, dtype=int)

        # Entry condition: new month or vol crush (simplified)
        if self.position["entry_date"] is None:
            last_day = data.index[-1]
            # safe access to 30 days back
            lookback_index = -30 if len(data.index) >= 30 else 0
            try:
                older_day = data.index[lookback_index]
            except Exception:
                older_day = data.index[0]

            if last_day.day >= 25 or (last_day - older_day).days >= 30:
                chain = self._get_atm_options_chain(last_day)
                if (
                    chain
                    and self.params["dte_target"] - self.params["dte_tolerance"]
                    <= chain["dte"]
                    <= self.params["dte_target"] + self.params["dte_tolerance"]
                ):
                    # Initialize position explicitly with floats to avoid Optional typing
                    call_p = float(chain["call_premium"])
                    put_p = float(chain["put_premium"])
                    cost = call_p + put_p
                    self.position.update(
                        {
                            "entry_date": pd.Timestamp(last_day),
                            "expiration": pd.Timestamp(chain["expiration"]),
                            "strike": float(chain["strike"]),
                            "call_premium": call_p,
                            "put_premium": put_p,
                            "straddle_cost": float(cost),
                            "shares_hedged": 0.0,
                            "total_pnl": -cost * 100.0,  # per contract
                            "hedge_pnl": 0.0,
                            "gamma_pnl": 0.0,
                            "vega_pnl": 0.0,
                            "theta_decay": 0.0,
                        }
                    )
                    signals.loc[last_day] = 1  # Enter straddle
                    logger.info(
                        f"DELTA-HEDGED STRADDLE ENTRY | {self.ticker} | Strike: {chain['strike']} | Cost: ${cost:.2f} | DTE: {chain['dte']}"
                    )

        # Daily delta hedging and P&L update
        if self.position["entry_date"] is not None:
            # Guard: ensure entry_date is Timestamp
            entry_date = self.position.get("entry_date")
            if entry_date is None:
                entry_date = data.index[-1]
                self.position["entry_date"] = pd.Timestamp(entry_date)

            current_price = float(data["Close"].iloc[-1])
            # days_held: guard in case entry_date is not a Timestamp
            try:
                days_held = int((pd.Timestamp(data.index[-1]) - pd.Timestamp(entry_date)).days)
            except Exception:
                days_held = 0

            # Simulate theta decay and gamma scalping P&L
            remaining_dte = max(0, int(self.params["dte_target"]) - days_held)
            decay_factor = float(np.exp(-float(days_held) / 30.0)) if days_held >= 0 else 1.0

            straddle_cost = float(self.position.get("straddle_cost") or 0.0)
            # current straddle value simplified: decayed cost plus gamma-related bump
            current_straddle_value = float(straddle_cost * decay_factor * 1.1)

            # Delta hedge: buy/sell shares to neutralize
            delta_per_straddle = 0.5  # ATM approx
            target_hedge = float(delta_per_straddle * 100.0)  # per contract
            hedge_change = float(target_hedge - float(self.position.get("shares_hedged") or 0.0))
            hedge_cost = float(hedge_change * current_price)

            # Update hedge P&L and shares hedged
            # If we buy shares (positive hedge_cost), that reduces cash -> we record as negative hedge_pnl here
            prev_hedge_pnl = float(self.position.get("hedge_pnl") or 0.0)
            self.position["hedge_pnl"] = prev_hedge_pnl - hedge_cost
            self.position["shares_hedged"] = target_hedge

            # Gamma P&L: simplified as change in option value * contract multiplier
            prev_gamma = float(self.position.get("gamma_pnl") or 0.0)
            self.position["gamma_pnl"] = prev_gamma + (current_straddle_value - straddle_cost) * 100.0

            # Total P&L: gamma + hedge - initial cost
            self.position["total_pnl"] = float(
                float(self.position.get("gamma_pnl") or 0.0)
                + float(self.position.get("hedge_pnl") or 0.0)
                - (straddle_cost * 100.0)
            )

            # Exit conditions
            pnl_pct = (
                float(self.position["total_pnl"]) / (straddle_cost * 100.0)
                if straddle_cost > 0.0
                else 0.0
            )
            if days_held >= int(self.params["dte_target"]) or remaining_dte <= 3:
                signals.loc[data.index[-1]] = -1
                logger.info(
                    f"STRADDLE EXIT | P&L: ${self.position['total_pnl']:,.0f} ({pnl_pct:+.1%}) | Gamma: ${self.position['gamma_pnl']:,.0f} | Hedge: ${self.position['hedge_pnl']:,.0f}"
                )
                # Reset position but keep types consistent
                self.position = {
                    "entry_date": None,
                    "expiration": None,
                    "strike": 0.0,
                    "call_premium": 0.0,
                    "put_premium": 0.0,
                    "straddle_cost": 0.0,
                    "shares_hedged": 0.0,
                    "total_pnl": 0.0,
                    "hedge_pnl": 0.0,
                    "gamma_pnl": 0.0,
                    "vega_pnl": 0.0,
                    "theta_decay": 0.0,
                }

        return signals

    def _calculate_greeks(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> Tuple[float, float, float, float]:
        """Black-Scholes greeks (simplified). Returns (call_delta, gamma, vega, theta_call)"""
        from scipy.stats import norm

        # guard T and sigma
        T_safe = float(max(T, 1e-8))
        sigma_safe = float(max(sigma, 1e-8))

        d1 = (np.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
        d2 = d1 - sigma_safe * np.sqrt(T_safe)
        call_delta = float(norm.cdf(d1))
        gamma = float(norm.pdf(d1) / (S * sigma_safe * np.sqrt(T_safe)))
        vega = float(S * norm.pdf(d1) * np.sqrt(T_safe))
        # Simplified theta (per day)
        theta_call = float(
            -(S * norm.pdf(d1) * sigma_safe) / (2.0 * np.sqrt(T_safe))
            - r * K * np.exp(-r * T_safe) * norm.cdf(d2)
        )
        return call_delta, gamma, vega, theta_call

    def __repr__(self) -> str:
        entry = self.position.get("entry_date")
        if entry:
            try:
                days_left = int(self.params["dte_target"] - (pd.Timestamp.now() - pd.Timestamp(entry)).days)
            except Exception:
                days_left = int(self.params["dte_target"])
            strike = float(self.position.get("strike") or 0.0)
            return f"DeltaHedgedStraddle({self.ticker} @ {strike}, DTE={days_left})"
        return "DeltaHedgedStraddle(flat)"


# Live test
if __name__ == "__main__":
    strategy = DeltaHedgedStraddleStrategy("SPY")

    # Create a simple DF with Close prices so generate_signals is runnable
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    # build a toy close price series with small random walk
    prices = 400.0 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    df = pd.DataFrame({"Close": prices}, index=dates)

    signals = strategy.generate_signals(df)
    print(f"Straddle signals: {signals.abs().sum()} entries/exits")
    print(f"Current position: {strategy.position['entry_date'] or 'FLAT'}")
