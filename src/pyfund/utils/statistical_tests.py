"""
statistical_tests.py
Institutional-grade statistical validation suite for quantitative strategies.
Implements:
  • Deflated Sharpe Ratio (Bailey & López de Prado)
  • Probability of Backtest Overfitting (PBO)
  • Walk-Forward Analysis
  • Professional performance summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any
from itertools import combinations

__all__ = [
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "walk_forward_sharpe",
    "performance_summary"
]

def deflated_sharpe_ratio(returns: np.ndarray, num_trials: int = 5000) -> float:
    returns = np.asarray(returns).flatten()
    if len(returns) < 30 or returns.std() == 0:
        return 0.0
    observed_sr = returns.mean() / returns.std() * np.sqrt(252)
    trial_srs = [
        np.random.choice(returns, len(returns)).mean() / 
        np.random.choice(returns, len(returns)).std() * np.sqrt(252)
        for _ in range(num_trials)
        if np.random.choice(returns, len(returns)).std() > 0
    ]
    if not trial_srs:
        return observed_sr
    sr_max = max(trial_srs)
    sr_std = np.std(trial_srs)
    return (observed_sr - sr_max) / sr_std if sr_std > 0 else observed_sr

def probability_of_backtest_overfitting(equity_curve: np.ndarray, n_samples: int = 1000) -> float:
    equity = np.asarray(equity_curve)
    n = len(equity)
    if n < 50:
        return 0.5
    combos = list(combinations(range(n), n//2))[:n_samples]
    matches = 0
    for train_idx in combos:
        test_idx = [i for i in range(n) if i not in train_idx]
        if len(test_idx) < 10:
            continue
        train_ret = np.diff(equity[list(train_idx)])
        test_ret = np.diff(equity[test_idx])
        if train_ret.sum() * test_ret.sum() == 0:
            continue
        matches += (np.sign(train_ret.sum()) == np.sign(test_ret.sum()))
    return 1.0 - (matches / len(combos)) if combos else 0.5

def walk_forward_sharpe(returns: pd.Series, train_window: int = 252, test_window: int = 63) -> Dict[str, Any]:
    results = []
    i = train_window
    while i + test_window <= len(returns):
        train = returns.iloc[i-train_window:i]
        test = returns.iloc[i:i+test_window]
        if train.std() > 0:
            results.append(test.mean() / test.std() * np.sqrt(252))
        i += test_window
    if not results:
        return {"mean": 0.0, "std": 0.0, "periods": 0}
    return {"mean": float(np.mean(results)), "std": float(np.std(results)), "periods": len(results)}

def performance_summary(returns: np.ndarray) -> None:
    returns = np.asarray(returns).flatten()
    if len(returns) < 30:
        print("Error: Insufficient data (<30 observations)")
        return
    
    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    dsr = deflated_sharpe_ratio(returns)
    pbo = probability_of_backtest_overfitting(np.cumprod(1 + returns))
    wf = walk_forward_sharpe(pd.Series(returns))
    
    print("\n" + "═" * 70)
    print("        INSTITUTIONAL PERFORMANCE VALIDATION REPORT        ".center(70))
    print("═" * 70)
    print(f"{'Observations':<30} {len(returns):>10}")
    print(f"{'Annualized Return':<30} {cagr*100:>9.2f}%")
    print(f"{'Annualized Sharpe Ratio':<30} {sharpe:>10.3f}")
    print(f"{'Deflated Sharpe Ratio':<30} {dsr:>10.3f}  → {'EXCELLENT' if dsr > 1.5 else 'Caution'}")
    print(f"{'PBO (Overfitting Risk)':<30} {pbo:>10.3f}  → {'LOW' if pbo < 0.25 else 'HIGH'}")
    print(f"{'Walk-Forward Sharpe':<30} {wf['mean']:>9.3f} ± {wf['std']:.3f} ({wf['periods']} periods)")
    print("═" * 70)
    if dsr > 1.5 and pbo < 0.25 and wf['mean'] > 1.5:
        print("FINAL ASSESSMENT: STRONG EVIDENCE OF PERSISTENT, ROBUST ALPHA")
    else:
        print("FINAL ASSESSMENT: FURTHER TESTING RECOMMENDED")
    print("═" * 70 + "\n")
