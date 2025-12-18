# src/pyfund/risk/constraints.py
"""
Dynamic Portfolio Risk Constraints Engine
────────────────────────────────────────
100% user-configurable — no hardcoded tickers or sectors.
Used by:
- Signal generator
- Portfolio rebalancer
- Live trading
- Backtester
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


class RiskConstraints:
    """
    Fully dynamic risk engine.
    Users can pass custom limits + sector mapping at runtime.
    """

    def __init__(
        self,
        max_position: float = 0.25,
        max_sector: float = 0.50,
        max_leverage: float = 3.0,
        min_position: float = 0.005,
        max_volatility: Optional[float] = None,
        max_drawdown: Optional[float] = None,
    ):
        self.config = {
            "max_position": float(max_position),
            "max_sector": float(max_sector),
            "max_leverage": float(max_leverage),
            "min_position": float(min_position),
            "max_volatility": max_volatility,
            "max_drawdown": max_drawdown,
        }

    def check(
        self,
        weights: pd.Series | Dict[str, float],
        sector_map: Optional[Mapping[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Universal check function — works with any stocks.

        Args:
            weights: dict or Series of {ticker: weight}
            sector_map: Optional {ticker: sector} mapping
            metadata: Extra info (e.g., current vol, drawdown)

        Returns:
            Full compliance report
        """
        # Convert to Series
        if isinstance(weights, dict):
            weights = pd.Series(weights, dtype=float)
        weights = weights.copy()
        weights = weights.dropna()
        weights = weights[weights.abs() >= self.config["min_position"]]

        if weights.empty:
            return self._empty_report()

        violations: List[str] = []
        details: Dict[str, Any] = {
            "total_positions": len(weights),
            "gross_exposure": float(weights.abs().sum()),
            "net_exposure": float(weights.sum()),
        }

        # 1. Max position
        max_pos = weights.abs().max()
        ticker_max = weights.abs().idxmax()
        details["largest_position"] = {"ticker": ticker_max, "weight": float(max_pos)}

        if max_pos > self.config["max_position"]:
            violations.append(
                f"Position too large: {ticker_max} {max_pos:.1%} > {self.config['max_position']:.0%}"
            )

        # 2. Leverage
        gross = weights.abs().sum()
        if gross > self.config["max_leverage"]:
            violations.append(f"Gross leverage {gross:.2f}x > {self.config['max_leverage']:.1f}x")

        # 3. Sector exposure
        if sector_map:
            sector_series = pd.Series({
                t: sector_map.get(t, "Unknown") for t in weights.index
            }, index=weights.index)

            sector_exp = weights.groupby(sector_series).sum().abs()
            max_sector_val = sector_exp.max()
            max_sector_name = sector_exp.idxmax()

            details["sector_exposure"] = sector_exp.to_dict()
            details["largest_sector"] = {"sector": max_sector_name, "exposure": float(max_sector_val)}

            if max_sector_val > self.config["max_sector"]:
                violations.append(
                    f"Sector concentration: {max_sector_name} {max_sector_val:.1%} > {self.config['max_sector']:.0%}"
                )
        else:
            details["sector_exposure"] = None

        # 4. Optional: volatility/drawdown from metadata
        if metadata:
            if self.config["max_volatility"] and metadata.get("annual_vol") > self.config["max_volatility"]:
                violations.append(f"Volatility {metadata['annual_vol']:.1%} > {self.config['max_volatility']:.0%}")

            if self.config["max_drawdown"] and metadata.get("drawdown", 0) > self.config["max_drawdown"]:
                violations.append(f"Drawdown {metadata['drawdown']:.1%} > {self.config['max_drawdown']:.0%}")

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "details": details,
            "config": self.config,
        }

    def clip(self, weights: pd.Series | Dict[str, float], sector_map: Optional[Mapping[str, str]] = None) -> pd.Series:
        """Return safe, clipped weights"""
        w = pd.Series(weights, dtype=float) if isinstance(weights, dict) else weights.copy()

        # Clip individual positions
        w = w.clip(-self.config["max_position"], self.config["max_position"])

        # Clip leverage
        gross = w.abs().sum()
        if gross > self.config["max_leverage"]:
            w *= self.config["max_leverage"] / gross

        # Clip sectors
        if sector_map:
            sectors = pd.Series({t: sector_map.get(t, "Other") for t in w.index}, index=w.index)
            for sector in sectors.unique():
                mask = sectors == sector
                sector_gross = w[mask].abs().sum()
                if sector_gross > self.config["max_sector"]:
                    w[mask] *= self.config["max_sector"] / sector_gross

        return w.round(6)

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "compliant": True,
            "violations": [],
            "details": {"note": "No active positions"},
            "config": self.config,
        }


# === QUICK USER EXAMPLES ===

if __name__ == "__main__":
    rc = RiskConstraints(max_position=0.20, max_sector=0.40, max_leverage=2.0)

    # User inputs ANY stocks
    my_portfolio = {
        "BTC-USD": 0.35,
        "ETH-USD": 0.30,
        "SOL-USD": 0.25,
        "AAPL": 0.18,
        "GOOGL": -0.15,
    }

    my_sectors = {
        "BTC-USD": "Crypto",
        "ETH-USD": "Crypto",
        "SOL-USD": "Crypto",
        "AAPL": "Technology",
        "GOOGL": "Technology",
    }

    report = rc.check(my_portfolio, sector_map=my_sectors)
    print("Compliance Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    safe_weights = rc.clip(my_portfolio, my_sectors)
    print("\nSafe weights:")
    print(safe_weights)