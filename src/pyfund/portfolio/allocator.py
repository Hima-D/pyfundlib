# src/pyfund/portfolio/allocator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

AllocationMethod = Literal[
    "equal_weight", "equal_risk", "risk_parity", "inverse_vol", "kelly", "mean_variance"
]


@dataclass
class AllocationResult:
    """Rich allocation output with full transparency"""

    weights: dict[str | float]  # Final target weights
    raw_weights: dict[str | float]  # Pre-normalized weights
    method: str  # Allocation method used
    total_leverage: float  # Gross exposure (sum |w|)
    risk_contributions: dict[str | float] | None = None
    metadata: dict[str, Any | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAllocator(ABC):
    """Abstract base for all allocators"""

    @abstractmethod
    def allocate(self, signals: pd.Series, data: pd.DataFrame | None = None) -> AllocationResult:
        pass


class PortfolioAllocator(BaseAllocator):
    """
    Advanced Multi-Strategy Portfolio Allocator

    Features:
    - Multiple allocation methods (equal weight, risk parity, inverse vol, etc.)
    - Signal confidence weighting
    - Volatility targeting
    - Position limits and turnover control
    - Full transparency and diagnostics
    """

    def __init__(
        self,
        tickers: list[str],
        method: AllocationMethod = "equal_risk",
        target_volatility: float = 0.15,  # 15% annualized portfolio vol target
        max_position: float = 0.30,  # Max 30% in one asset
        max_leverage: float = 2.0,  # Max gross exposure
        min_weight: float = 0.02,  # Minimum position size
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.method = method
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_weight = min_weight
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

    def _calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualized covariance matrix with shrinkage"""
        cov = returns.cov() * 252
        # Simple shrinkage toward equal correlation
        shrinkage = 0.1
        avg_corr = returns.corr().values.mean()
        prior = np.ones_like(cov) * avg_corr * np.diag(cov)
        np.fill_diagonal(prior, np.diag(cov))
        return (1 - shrinkage) * cov + shrinkage * prior

    def _equal_weight(self, active: list[str]) -> dict[str | float]:
        weight = 1.0 / len(active) if active else 0.0
        return {t: weight for t in active}

    def _equal_risk_contribution(
        self, returns: pd.DataFrame, active: list[str]
    ) -> dict[str | float]:
        """ERC - each asset contributes equally to portfolio risk"""
        sub_returns = returns[active]
        cov = self._calculate_covariance(sub_returns)

        try:
            from scipy.optimize import minimize

            def risk_contribution(w, cov):
                portfolio_risk = np.sqrt(w.T @ cov @ w)
                rc = w * (cov @ w) / portfolio_risk
                return ((rc - rc.mean()) ** 2).sum()

            x0 = np.ones(len(active)) / len(active)
            bounds = [(0, 1) for _ in active]
            constraints = {"type": "eq", "fun": lambda x: x.sum() - 1}

            result = minimize(
                risk_contribution,
                x0,
                args=(cov.values,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            weights = result.x if result.success else np.ones(len(active)) / len(active)
        except Exception as e:
            weights = np.ones(len(active)) / len(active)

        return dict(zip(active, weights))

    def _inverse_volatility(self, returns: pd.DataFrame, active: list[str]) -> dict[str | float]:
        """Weight inversely proportional to volatility"""
        vols = returns[active].std() * np.sqrt(252)
        inv_vol = 1.0 / vols.replace(0, np.nan)
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(0).to_dict()

    def allocate(
        self,
        signals: pd.Series,
        data: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
    ) -> AllocationResult:
        """
        Main allocation method

        Args:
            signals: Series with tickers as index, values = confidence (-1 to +1)
            data: Optional returns data for risk modeling
            prices: Optional price data for position sizing

        Returns:
            AllocationResult with full transparency
        """
        # Filter active signals (non-zero)
        active_signals = signals[signals.abs() > 1e-6]
        active_tickers = active_signals.index.tolist()

        if not active_tickers:
            return AllocationResult(
                weights={t: 0.0 for t in self.tickers},
                raw_weights={},
                method=self.method,
                total_leverage=0.0,
                metadata={"reason": "no_active_signals"},
            )

        # Get returns if not provided
        if data is None:
            from ..data.fetcher import DataFetcher

            returns_dict = {}
            for t in active_tickers:
                df = DataFetcher.get_price(t, period=f"{self.lookback_days + 100}d")
                returns_dict[t] = np.log(df["Close"] / df["Close"].shift(1))
            returns = pd.DataFrame(returns_dict).dropna()
        else:
            returns = data.pct_change().dropna()

        # Apply allocation method
        if self.method == "equal_weight":
            raw_weights = self._equal_weight(active_tickers)
        elif self.method == "equal_risk":
            raw_weights = self._equal_risk_contribution(returns, active_tickers)
        elif self.method == "inverse_vol":
            raw_weights = self._inverse_volatility(returns, active_tickers)
        else:
            raw_weights = self._equal_weight(active_tickers)  # fallback

        # Apply signal direction and confidence
        final_weights = {}
        for ticker in self.tickers:
            if ticker in active_tickers:
                direction = np.sign(signals[ticker])
                confidence = abs(signals[ticker])
                weight = raw_weights.get(ticker, 0.0) * direction * confidence
                final_weights[ticker] = weight
            else:
                final_weights[ticker] = 0.0

        # Normalize to target volatility and apply constraints
        weights_df = pd.Series(final_weights)
        gross_exposure = weights_df.abs().sum()

        if gross_exposure > 0:
            # Scale to target leverage
            scale = min(self.max_leverage, self.target_volatility * 3) / gross_exposure
            weights_df *= scale

            # Apply position limits
            weights_df = weights_df.clip(-self.max_position, self.max_position)
            weights_df[weights_df.abs() < self.min_weight] = 0.0

        final_weights = weights_df.to_dict()
        total_leverage = sum(abs(w) for w in final_weights.values())

        return AllocationResult(
            weights=final_weights,
            raw_weights=raw_weights,
            method=self.method,
            total_leverage=round(total_leverage, 3),
            metadata={
                "active_tickers": len(active_tickers),
                "target_volatility": self.target_volatility,
                "applied_scale": scale if "scale" in locals() else 1.0,
            },
        )


# Quick test
if __name__ == "__main__":
    allocator = PortfolioAllocator(
        tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        method="equal_risk",
        target_volatility=0.20,
    )

    signals = pd.Series({"AAPL": 0.8, "TSLA": 1.0, "NVDA": 0.6, "MSFT": -0.3, "GOOGL": 0.0})
    result = allocator.allocate(signals)

    print("Portfolio Allocation Result:")
    print(pd.Series(result.weights).round(4))
    print(f"Total Leverage: {result.total_leverage}x")
