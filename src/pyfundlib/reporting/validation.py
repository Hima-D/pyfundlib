from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pyfundlib.utils.statistical_tests import StatisticalValidator
from pyfundlib.utils.statistical_tools import (
    bootstrap_sharpe_ci,
    bootstrap_maxdd_ci,
    simple_vol_regime_label,
)


@dataclass
class ValidationResult:
    core: Dict[str, Any]
    sharpe_ci: Tuple[float, float]
    maxdd_ci: Tuple[float, float]
    regimes: pd.Series
    regime_transitions: pd.DataFrame


def validate_equity_curve(
    equity_curve: pd.Series,
    annualization_factor: int = 252,
    confidence: float = 0.95,
    random_state: Optional[int] = 42,
) -> ValidationResult:
    eq = equity_curve.dropna()
    if eq.empty or len(eq) < 10:
        empty_core = {"error": "Not enough observations for validation"}
        return ValidationResult(
            core=empty_core,
            sharpe_ci=(0.0, 0.0),
            maxdd_ci=(0.0, 0.0),
            regimes=pd.Series(dtype="object"),
            regime_transitions=pd.DataFrame(columns=["count"]),
        )

    returns = eq.pct_change().dropna().to_numpy()

    validator = StatisticalValidator(
        annualization_factor=annualization_factor,
        random_state=random_state,
    )
    core = validator.validate(returns)

    sharpe_ci = bootstrap_sharpe_ci(returns, confidence=confidence)
    maxdd_ci = bootstrap_maxdd_ci(eq.to_numpy(), confidence=confidence)

    ret_series = pd.Series(returns, index=eq.index[1:])
    regime_result = simple_vol_regime_label(ret_series)

    return ValidationResult(
        core=core,
        sharpe_ci=sharpe_ci,
        maxdd_ci=maxdd_ci,
        regimes=regime_result.regimes,
        regime_transitions=regime_result.transitions,
    )


def summarize_validation(result: ValidationResult) -> str:
    core = result.core
    if "error" in core:
        return f"Validation error: {core['error']}"

    cagr = core.get("cagr_percent", 0.0)
    sharpe = core.get("sharpe", 0.0)
    deflated_sharpe = core.get("deflated_sharpe", 0.0)
    pbo = core.get("pbo", 0.0)
    wf = core.get("walk_forward", {})
    wf_mean = wf.get("mean", 0.0)
    wf_periods = wf.get("periods", 0)
    robust = core.get("robust", False)

    s_lo, s_hi = result.sharpe_ci
    dd_lo, dd_hi = result.maxdd_ci

    parts = [
        f"CAGR: {cagr:.3f}%",
        f"Sharpe: {sharpe:.3f} (CI {s_lo:.3f}–{s_hi:.3f})",
        f"Deflated Sharpe: {deflated_sharpe:.3f}",
        f"PBO: {pbo:.3f}",
        f"Walk-forward Sharpe: {wf_mean:.3f} over {wf_periods} periods",
        f"Max drawdown CI: {dd_lo:.1%}–{dd_hi:.1%}",
        f"Robust: {'YES' if robust else 'NO'}",
    ]
    return " | ".join(parts)

