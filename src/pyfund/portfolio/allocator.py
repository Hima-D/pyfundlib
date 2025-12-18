# src/pyfund/portfolio/allocator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

AllocationMethod = Literal[
    "equal_weight",
    "equal_risk",
    "risk_parity",
    "inverse_vol",
    "kelly",
    "mean_variance",
]


@dataclass
class AllocationResult:
    """Rich allocation output with full transparency."""

    weights: dict[str, float]  # Final target weights
    raw_weights: dict[str, float]  # Pre-normalized weights
    method: str  # Allocation method used
    total_leverage: float  # Gross exposure (sum |w|)
    risk_contributions: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.risk_contributions is None:
            self.risk_contributions = {}


class BaseAllocator(ABC):
    """Abstract base for all allocators."""

    @abstractmethod
    def allocate(
        self, signals: pd.Series, data: pd.DataFrame | None = None
    ) -> AllocationResult:
        """Perform allocation based on signals and optional data."""
        pass


class PortfolioAllocator(BaseAllocator):
    """
    Advanced Multi-Strategy Portfolio Allocator.

    Features:
    - Multiple allocation methods (equal weight, risk parity, inverse vol, etc.)
    - Signal confidence weighting
    - Volatility targeting via estimated portfolio volatility
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
        shrinkage: float = 0.1,  # Covariance shrinkage intensity
    ):
        self.tickers = [t.upper() for t in tickers]
        self.method = method
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_weight = min_weight
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        self.shrinkage = shrinkage

    def _calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Annualized covariance matrix with Ledoit-Wolf-style shrinkage to equal correlation."""
        if len(returns.columns) == 0:
            raise ValueError("No returns data provided for covariance calculation.")

        cov_emp = returns.cov() * 252
        if len(returns) < 2:
            raise ValueError("Insufficient data for covariance estimation.")

        # Empirical covariance
        cov = cov_emp.values

        # Shrinkage to equal-correlation prior
        vars_ = np.diag(cov)
        sqrt_vars = np.sqrt(vars_)
        corr_matrix = returns.corr()
        # Fix: Use cov (array) instead of cov.shape for triu_indices_from
        triu_indices = np.triu_indices_from(cov, k=1)
        avg_corr = np.mean(corr_matrix.values[triu_indices])
        if np.isnan(avg_corr):
            avg_corr = 0.0  # Fallback for degenerate cases

        # Prior: diagonal = empirical vars, off-diagonal = avg_corr * sqrt(var_i * var_j)
        prior = avg_corr * np.outer(sqrt_vars, sqrt_vars)
        np.fill_diagonal(prior, vars_)

        # Shrunk covariance
        shrunk_cov = (1 - self.shrinkage) * cov + self.shrinkage * prior
        return shrunk_cov

    def _equal_weight(self, active: list[str]) -> dict[str, float]:
        """Equal weight allocation."""
        if not active:
            return {}
        weight = 1.0 / len(active)
        return {t: weight for t in active}

    def _equal_risk_contribution(
        self, returns: pd.DataFrame, active: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Equal Risk Contribution (ERC) - each asset contributes equally to portfolio risk.

        Returns:
            Tuple of (weights, risk_contributions)
        """
        if len(active) == 0:
            return {}, {}
        if len(active) == 1:
            return {active[0]: 1.0}, {active[0]: 1.0}

        sub_returns = returns[active].dropna()
        if sub_returns.empty or len(sub_returns) < 10:  # Minimum data check
            return self._equal_weight(active), {}

        cov = self._calculate_covariance(sub_returns)

        def risk_contribution(w: np.ndarray, cov: np.ndarray) -> float:
            port_risk = np.sqrt(w.T @ cov @ w)
            if port_risk == 0:
                return np.inf
            rc = (cov @ w) * w / port_risk
            return np.sum((rc - np.mean(rc)) ** 2)

        x0 = np.ones(len(active)) / len(active)
        bounds = [(0.0, 1.0) for _ in active]
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        result = minimize(
            risk_contribution,
            x0,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "maxiter": 1000},
        )

        if result.success:
            weights = result.x
        else:
            weights = x0  # Fallback

        # Normalize if needed (should sum to 1)
        weights /= np.sum(weights)

        # Compute risk contributions
        port_risk = np.sqrt(weights.T @ cov @ weights)
        rc = dict(zip(active, (cov @ weights * weights / port_risk) if port_risk > 0 else np.ones(len(active)) / len(active)))

        return dict(zip(active, weights)), rc

    def _inverse_volatility(self, returns: pd.DataFrame, active: list[str]) -> dict[str, float]:
        """Weight inversely proportional to volatility."""
        if not active:
            return {}
        sub_returns = returns[active].dropna()
        if sub_returns.empty:
            return self._equal_weight(active)

        vols = sub_returns.std() * np.sqrt(252)
        vols = vols.replace(0, np.finfo(float).eps)  # Avoid div by zero
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()

    def _kelly_criterion(self, returns: pd.DataFrame, active: list[str], signals: pd.Series) -> dict[str, float]:
        """Kelly Criterion: Optimal bet sizing based on expected edge."""
        if len(active) == 0:
            return {}
        sub_returns = returns[active].dropna()
        if sub_returns.empty:
            return self._equal_weight(active)

        # Assume signals proportional to expected excess return (simple heuristic)
        mu = pd.Series({t: signals.get(t, 0.0) * sub_returns[t].mean() for t in active})
        cov = self._calculate_covariance(sub_returns)

        # Kelly: f = mu / (lambda * cov), but for multi-asset, solve max mu'w - (1/2) w' cov w with sum w=1, w>0
        # Here, approximate as inverse variance weighted by mu
        prec = np.linalg.pinv(cov)  # Precision matrix
        ones = np.ones(len(active))
        try:
            # Fix: Ensure b is 1D array of floats
            b = mu.values.astype(np.float64)
            a = prec.astype(np.float64)
            w_opt = np.linalg.solve(a, b)
            w_opt /= np.sum(np.abs(w_opt))  # Normalize to gross=1
        except np.linalg.LinAlgError:
            w_opt = ones / len(active)

        return dict(zip(active, w_opt))

    def _mean_variance(self, returns: pd.DataFrame, active: list[str], signals: pd.Series) -> dict[str, float]:
        """Mean-Variance optimization: Maximize Sharpe Ratio."""
        if len(active) == 0:
            return {}
        sub_returns = returns[active].dropna()
        if sub_returns.empty:
            return self._equal_weight(active)

        # Expected returns from signals
        mu = pd.Series({t: signals.get(t, 0.0) * sub_returns[t].mean() + self.risk_free_rate / 252 for t in active})
        cov = self._calculate_covariance(sub_returns)

        def neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
            # Fix: Scalarize both port_ret and port_risk
            port_ret = float((w.T @ mu).item())
            port_risk = float(np.sqrt(w.T @ cov @ w).item())
            if port_risk == 0:
                return np.inf
            return - (port_ret - self.risk_free_rate / 252) / port_risk

        x0 = np.ones(len(active)) / len(active)
        bounds = [(0.0, 1.0) for _ in active]
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        result = minimize(
            neg_sharpe,
            x0,
            args=(mu.values, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )

        if result.success:
            weights = result.x
        else:
            weights = x0

        weights /= np.sum(weights)
        return dict(zip(active, weights))

    def allocate(
        self,
        signals: pd.Series,
        data: pd.DataFrame | None = None,  # Prices or returns DataFrame
        prices: pd.DataFrame | None = None,  # Deprecated; use data for prices
    ) -> AllocationResult:
        """
        Main allocation method.

        Args:
            signals: Series with tickers as index, values = confidence (-1 to +1)
            data: Optional DataFrame of prices (will compute returns) or returns

        Returns:
            AllocationResult with full transparency
        """
        if prices is not None:
            import warnings
            warnings.warn("prices arg deprecated; use data instead.", DeprecationWarning)
            data = prices

        # Ensure signals indexed by tickers
        signals = signals.reindex(self.tickers, fill_value=0.0)

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

        # Prepare returns data
        if data is None:
            # In production, fetch via DataFetcher; here, raise or mock for tests
            try:
                from ..data.fetcher import DataFetcher
                returns_dict = {}
                for t in active_tickers:
                    price_df = DataFetcher.get_price(t, period=f"{self.lookback_days + 100}d")
                    returns_dict[t] = np.log(price_df["Close"] / price_df["Close"].shift(1))
                returns = pd.DataFrame(returns_dict).dropna(how="all")
            except ImportError:
                raise ImportError("DataFetcher not available; provide data or install pyfundlib[data]")
        else:
            # Assume data is prices; compute log returns and ensure DataFrame
            log_returns = np.log(data / data.shift(1))
            returns = pd.DataFrame(log_returns, columns=data.columns).dropna(how="all")

        if returns.empty or len(returns.columns) == 0:
            return AllocationResult(
                weights={t: 0.0 for t in self.tickers},
                raw_weights=self._equal_weight(active_tickers),
                method=self.method,
                total_leverage=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Apply allocation method
        method_map = {
            "equal_weight": self._equal_weight,
            "equal_risk": self._equal_risk_contribution,
            "inverse_vol": self._inverse_volatility,
            "kelly": lambda r, a, s: self._kelly_criterion(r, a, signals),
            "mean_variance": lambda r, a, s: self._mean_variance(r, a, signals),
        }
        alloc_func = method_map.get(self.method, self._equal_weight)

        if self.method == "equal_risk":
            raw_weights, risk_contribs = alloc_func(returns, active_tickers)
        else:
            raw_weights = alloc_func(returns, active_tickers)
            risk_contribs = {}

        # Apply signal direction and confidence to raw weights
        final_weights = {t: 0.0 for t in self.tickers}
        for ticker in active_tickers:
            if ticker in raw_weights:
                direction = np.sign(signals[ticker])
                confidence = abs(signals[ticker])
                final_weights[ticker] = raw_weights[ticker] * direction * confidence

        # Convert to Series for easier manipulation
        weights_df = pd.Series(final_weights)
        gross_exposure = weights_df.abs().sum()

        # Volatility targeting: scale to achieve target portfolio vol
        scale = 1.0
        if gross_exposure > 0 and not returns.empty:
            # Estimate current portfolio vol (absolute weights for risk calc)
            abs_weights = weights_df.abs() / gross_exposure  # Normalize to 1 for estimation
            sub_returns = returns[active_tickers]
            # Fix: Use to_numpy() for ndarray compatibility
            abs_weights_arr = abs_weights.to_numpy().reshape(-1, 1)
            cov_matrix = self._calculate_covariance(sub_returns)
            port_vol = float(np.sqrt(abs_weights_arr.T @ cov_matrix @ abs_weights_arr).item())
            if port_vol > 0:
                scale = self.target_volatility / port_vol
            scale = min(scale, self.max_leverage / gross_exposure)  # Cap leverage
        else:
            scale = min(1.0, self.max_leverage / gross_exposure) if gross_exposure > 0 else 1.0

        # Apply scaling
        weights_df *= scale * gross_exposure  # Scale back to original exposure level, then apply vol scale

        # Enforce constraints
        weights_df = weights_df.clip(-self.max_position, self.max_position)
        weights_df[weights_df.abs() < self.min_weight] = 0.0
        # Re-normalize after clipping (optional, but preserves intent)
        remaining = 1.0 - weights_df[weights_df.abs() >= self.min_weight].sum()
        if remaining > 0 and len(active_tickers) > 1:
            active_mask = weights_df.abs() >= self.min_weight
            active_df = weights_df[active_mask]
            if not active_df.empty:
                active_df *= (1.0 / active_df.sum()) * remaining
                weights_df.update(active_df)

        final_weights = weights_df.to_dict()
        total_leverage = sum(abs(w) for w in final_weights.values())

        # Fix: Scalarize for round; use .item() for ndarray scalars
        est_port_vol = 0.0
        if gross_exposure > 0:
            # Reuse abs_weights
            abs_weights_arr = abs_weights.to_numpy().reshape(-1, 1)
            cov_matrix = self._calculate_covariance(returns[active_tickers])
            est_port_vol = float((np.sqrt(abs_weights_arr.T @ cov_matrix @ abs_weights_arr) * scale * gross_exposure).item())

        metadata = {
            "active_tickers": len(active_tickers),
            "target_volatility": self.target_volatility,
            "applied_scale": round(float(scale), 4),
            "estimated_port_vol": round(est_port_vol, 4),
        }

        return AllocationResult(
            weights=final_weights,
            raw_weights=raw_weights,
            method=self.method,
            total_leverage=round(total_leverage, 3),
            risk_contributions=risk_contribs,
            metadata=metadata,
        )


# Quick test
if __name__ == "__main__":
    # Mock DataFetcher for standalone test
    class MockDataFetcher:
        @staticmethod
        def get_price(ticker: str, period: str) -> pd.DataFrame:
            np.random.seed(hash(ticker) % (2**32))  # Deterministic per ticker
            dates = pd.date_range(end=pd.Timestamp.now(), periods=400, freq="D")
            daily_returns = np.random.normal(0, 0.02, len(dates))  # ~32% ann vol base
            price = 100 * np.exp(np.cumsum(daily_returns))
            return pd.DataFrame({"Close": price}, index=dates)

    # Patch for test: Use setattr to avoid type issues
    import sys
    from types import ModuleType
    mock_module = ModuleType("pyfund.data.fetcher")
    setattr(mock_module, "DataFetcher", MockDataFetcher)
    sys.modules["pyfund.data.fetcher"] = mock_module

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
    print(f"Metadata: {result.metadata}")
    if result.risk_contributions:
        print("Risk Contributions:")
        print(pd.Series(result.risk_contributions).round(4))