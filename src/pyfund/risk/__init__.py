# src/pyfundlib/risk/__init__.py
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t


class RiskManager:
    """
    Comprehensive risk management suite for portfolios and individual positions.
    """

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.99,
        method: str = "parametric",
        window: int | None = None,
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Parameters
        ----------
        returns : pd.Series
            Daily returns
        confidence_level : float
            e.g., 0.95, 0.99, 0.995
        method : str
            "parametric" (Gaussian), "historical", "cornish-fisher"
        window : int, optional
            Rolling window for dynamic VaR

        Returns
        -------
        float or pd.Series
            VaR value (negative = loss)
        """
        if window:
            returns = returns.rolling(window=window)

        if method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            z = norm.ppf(1 - confidence_level)
            return mu + z * sigma

        elif method == "historical":
            return returns.quantile(1 - confidence_level)

        elif method == "cornish-fisher":
            z = norm.ppf(1 - confidence_level)
            skew = returns.skew()
            kurt = returns.kurtosis()
            z_adj = (
                z
                + (z**2 - 1) * skew / 6
                + (z**3 - 3 * z) * (kurt - 3) / 24
                - (2 * z**3 - 5 * z) * (skew**2) / 36
            )
            return returns.mean() + z_adj * returns.std()

        else:
            raise ValueError("method must be 'parametric', 'historical', or 'cornish-fisher'")

    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.99) -> float:
        """Expected Shortfall / Conditional VaR"""
        var = RiskManager.value_at_risk(returns, confidence_level, method="historical")
        return returns[returns <= var].mean()

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown"""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()

    @staticmethod
    def calmar_ratio(cagr: float, max_dd: float) -> float:
        """CAGR / Max Drawdown"""
        return cagr / abs(max_dd) if max_dd != 0 else np.inf

    @staticmethod
    def ulcer_index(equity: pd.Series) -> float:
        """Measures pain of drawdowns (for Ulcer Performance Index)"""
        drawdown = (equity - equity.cummax()) / equity.cummax()
        return np.sqrt(np.mean(drawdown**2))

    @staticmethod
    def pain_index(equity: pd.Series) -> float:
        """Average depth of drawdowns"""
        drawdown = (equity - equity.cummax()) / equity.cummax()
        return np.mean(np.abs(drawdown))

    @staticmethod
    def position_sizing(
        account_value: float,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.05,
        volatility: float | None = None,
    ) -> float:
        """
        Kelly / Volatility-adjusted position sizing.

        Parameters
        ----------
        risk_per_trade : float
            Max % of account to risk per trade (e.g., 0.01 = 1%)
        stop_loss_pct : float
            Expected stop distance
        volatility : float, optional
            Use volatility parity instead of fixed stop
        """
        if volatility:
            risk_amount = account_value * risk_per_trade
            return risk_amount / volatility
        else:
            return (risk_per_trade * account_value) / stop_loss_pct

    @staticmethod
    def volatility_target_scaling(
        current_vol: float,
        target_vol: float = 0.15,
        leverage_cap: float = 4.0,
    ) -> float:
        """Scale position to hit target annualized volatility"""
        scaling = target_vol / current_vol
        return min(scaling, leverage_cap)

    @staticmethod
    def risk_metrics(equity_curve: pd.Series) -> dict[str | float]:
        """One-call risk summary"""
        returns = equity_curve.pct_change().dropna()
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
        max_dd = RiskManager.max_drawdown(equity_curve)

        return {
            "VaR_95%": RiskManager.value_at_risk(returns, 0.95),
            "VaR_99%": RiskManager.value_at_risk(returns, 0.99),
            "CVaR_99%": RiskManager.conditional_var(returns, 0.99),
            "Max Drawdown": max_dd,
            "Calmar Ratio": RiskManager.calmar_ratio(cagr, max_dd),
            "Ulcer Index": RiskManager.ulcer_index(equity_curve),
            "Annual Volatility": returns.std() * np.sqrt(252),
        }
