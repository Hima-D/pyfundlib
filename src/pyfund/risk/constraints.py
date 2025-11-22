# src/pyfundlib/risk/constraints.py
from __future__ import annotations

import pandas as pd


class RiskConstraints:
    """Enforce portfolio-level risk limits"""

    def __init__(
        self,
        max_position: float = 0.20,  # 20% max per ticker
        max_sector: float = 0.40,  # 40% max per sector
        max_drawdown: float = 0.20,  # Hard stop at -20%
        max_volatility: float = 0.25,  # 25% annual vol target
        leverage_limit: float = 3.0,
    ):
        self.max_position = max_position
        self.max_sector = max_sector
        self.max_drawdown = max_drawdown
        self.max_volatility = max_volatility
        self.leverage_limit = leverage_limit

    def check_compliance(self, weights: pd.Series, sector_map: dict[str | str]) -> dict[str, Any]:
        violations = []

        # Position concentration
        if weights.abs().max() > self.max_position:
            violations.append(f"Position > {self.max_position:.0%}")

        # Sector exposure
        sector_exposure = weights.groupby(sector_map).sum().abs()
        if sector_exposure.max() > self.max_sector:
            violations.append(f"Sector exposure > {self.max_sector:.0%}")

        # Leverage
        if weights.abs().sum() > self.leverage_limit:
            violations.append(f"Leverage > {self.leverage_limit:.1f}x")

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "max_position": weights.abs().max(),
            "max_sector": sector_exposure.max(),
            "total_leverage": weights.abs().sum(),
        }
