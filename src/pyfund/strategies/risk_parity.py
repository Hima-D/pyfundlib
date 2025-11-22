# src/pyfund/portfolio/risk_parity.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.fetcher import DataFetcher


class RiskParityAllocator:
    """
    Risk Parity Portfolio Allocator

    Allocates capital so each asset contributes equally to total portfolio risk.
    True "Holy Grail" of portfolio construction — maximum diversification.

    Methods:
    - Inverse Volatility (heuristic, fast)
    - Full Risk Parity (exact, using covariance + optimization)
    - Hierarchical Risk Parity (HRP) coming soon
    """

    def __init__(
        self,
        tickers: list[str],
        lookback_days: int = 252,
        risk_budget: list[float] | None = None,  # Custom risk targets per asset
        method: str = "full",  # "inverse_vol", "full", "erc"
        max_leverage: float = 3.0,
        rebalance_freq: str = "monthly",
    ):
        self.tickers = [t.upper() for t in tickers]
        self.lookback_days = lookback_days
        self.risk_budget = np.array(risk_budget) if risk_budget else None
        self.method = method.lower()
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq

        self.weights: dict[str | float] = {}
        self.risk_contributions: dict[str | float] = {}

    def _fetch_returns(self) -> pd.DataFrame:
        """Fetch price data and compute log returns"""
        data = {}
        for ticker in self.tickers:
            try:
                df = DataFetcher.get_price(ticker, period=f"{self.lookback_days + 100}d")
                data[ticker] = df["Close"]
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
        prices = pd.DataFrame(data).dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def _inverse_volatility_weights(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Heuristic: weight inversely proportional to volatility"""
        vol = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        return weights

    def _risk_contribution(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Compute marginal and total risk contribution"""
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_risk
        return risk_contrib

    def _full_risk_parity_objective(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Minimize difference between actual and target risk contributions"""
        rc = self._risk_contribution(weights, cov_matrix)
        if self.risk_budget is not None:
            target = self.risk_budget
        else:
            target = np.ones(len(weights)) / len(weights)
        diff = rc - target * rc.sum()
        return (diff**2).sum()

    def allocate(self) -> dict[str | float]:
        """Main allocation function"""
        returns = self._fetch_returns()
        if returns.empty or len(returns.columns) < 2:
            print("Not enough data for risk parity")
            return {t: 1.0 / len(self.tickers) for t in self.tickers}

        cov_matrix = returns.cov() * 252  # Annualized

        if self.method == "inverse_vol":
            raw_weights = self._inverse_volatility_weights(cov_matrix)
        else:
            # Full Risk Parity optimization
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "ineq", "fun": lambda w: w},  # long-only
            ]
            bounds = [(0.0, 1.0) for _ in range(len(self.tickers))]
            initial = np.ones(len(self.tickers)) / len(self.tickers)

            result = minimize(
                fun=self._full_risk_parity_objective,
                x0=initial,
                args=(cov_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "disp": False},
            )

            if not result.success:
                print("Optimization failed, falling back to inverse volatility")
                raw_weights = self._inverse_volatility_weights(cov_matrix)
            else:
                raw_weights = result.x

        # Normalize and apply leverage cap
        weights = raw_weights / raw_weights.sum()
        total_leverage = np.abs(weights).sum()
        if total_leverage > self.max_leverage:
            weights = weights * (self.max_leverage / total_leverage)

        # Final weights
        self.weights = {ticker: float(w) for ticker, w in zip(self.tickers, weights)}

        # Risk contributions
        rc = self._risk_contribution(weights, cov_matrix)
        rc_pct = rc / rc.sum()
        self.risk_contributions = {ticker: float(r) for ticker, r in zip(self.tickers, rc_pct)}

        return self.weights

    def summary(self) -> pd.DataFrame:
        """Return beautiful allocation summary"""
        if not self.weights:
            self.allocate()

        df = pd.DataFrame(
            {"Weight": self.weights, "Risk_Contribution": self.risk_contributions}
        ).round(4)
        df = df.sort_values("Risk_Contribution", ascending=False)
        df.loc["TOTAL"] = [df["Weight"].abs().sum(), 1.0]
        return df


# Quick test
if __name__ == "__main__":
    rp = RiskParityAllocator(
        tickers=["SPY", "TLT", "GLD", "DBC", "VNQ"], method="full", max_leverage=2.0
    )

    weights = rp.allocate()
    print("Risk Parity Portfolio Weights:")
    print(rp.summary())
