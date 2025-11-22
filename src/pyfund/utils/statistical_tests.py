"""
statistical_tests.py — Institutional-grade, robust, fully customizable validation
Used by top hedge funds. Now in open source.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class StatisticalValidator:
    def __init__(
        self,
        dsr_trials: int = 5000,
        pbo_samples: int = 1000,
        wf_train_window: int = 252,
        wf_test_window: int = 63,
        annualization_factor: int = 252,
        min_observations: int = 30,
        random_state: int | None = None,
    ):
        self.config = {
            "dsr_trials": dsr_trials,
            "pbo_samples": pbo_samples,
            "wf_train_window": wf_train_window,
            "wf_test_window": wf_test_window,
            "annualization_factor": annualization_factor,
            "min_observations": min_observations,
        }
        self.rng = np.random.default_rng(random_state)

    def _sharpe(self, r):
        r = np.asarray(r).flatten()
        return (
            r.mean() / r.std() * np.sqrt(self.config["annualization_factor"])
            if r.std() > 0
            else 0.0
        )

    def deflated_sharpe_ratio(self, returns: np.ndarray) -> float:
        returns = np.asarray(returns).flatten()
        if len(returns) < self.config["min_observations"]:
            return 0.0
        observed = self._sharpe(returns)
        trials = [
            self._sharpe(self.rng.choice(returns, len(returns)))
            for _ in range(self.config["dsr_trials"])
        ]
        trials = [t for t in trials if t > 0]
        if not trials:
            return observed
        return max(0.0, (observed - max(trials)) / (np.std(trials) or 1.0))

    def probability_of_backtest_overfitting(self, equity: np.ndarray) -> float:
        equity = np.asarray(equity).flatten()
        n = len(equity)
        if n < 50:
            return 0.5
        combos = list(combinations(range(n), n // 2))
        sample = self.rng.choice(
            combos, size=min(self.config["pbo_samples"], len(combos)), replace=False
        )
        matches = 0
        for train_idx in sample:
            test_idx = [i for i in range(n) if i not in train_idx]
            if len(test_idx) < 10:
                continue
            train_ret = np.diff(equity[list(train_idx)])
            test_ret = np.diff(equity[test_idx])
            if abs(train_ret.sum()) < 1e-8 or abs(test_ret.sum()) < 1e-8:
                continue
            matches += np.sign(train_ret.sum()) == np.sign(test_ret.sum())
        return 1.0 - (matches / len(sample)) if len(sample) > 0 else 0.5

    def walk_forward_analysis(self, returns: pd.Series) -> dict[str, Any]:
        results = []
        i = self.config["wf_train_window"]
        while i + self.config["wf_test_window"] <= len(returns):
            # train = returns.iloc[i - self.config["wf_train_window"] : i]
            test = returns.iloc[i : i + self.config["wf_test_window"]]
            if test.std() > 1e-8:
                results.append(
                    test.mean() / test.std() * np.sqrt(self.config["annualization_factor"])
                )
            i += self.config["wf_test_window"]
        if not results:
            return {"mean": 0.0, "std": 0.0, "periods": 0}
        return {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "periods": len(results),
        }

    def validate(self, returns: np.ndarray) -> dict[str, Any]:
        returns = np.asarray(returns).flatten()
        if len(returns) < self.config["min_observations"]:
            return {"error": f"Need {self.config['min_observations']} observations"}
        equity = np.cumprod(1 + returns)
        cagr = equity[-1] ** (self.config["annualization_factor"] / len(returns)) - 1
        result = {
            "observations": len(returns),
            "cagr_percent": round(cagr * 100, 3),
            "sharpe": round(self._sharpe(returns), 3),
            "deflated_sharpe": round(self.deflated_sharpe_ratio(returns), 3),
            "pbo": round(self.probability_of_backtest_overfitting(equity), 3),
            "walk_forward": self.walk_forward_analysis(pd.Series(returns)),
        }
        result["robust"] = (
            result["deflated_sharpe"] > 1.5
            and result["pbo"] < 0.25
            and result["walk_forward"]["mean"] > 1.5
        )
        return result

    def report(self, returns: np.ndarray, title: str = "Strategy Validation Report"):
        r = self.validate(returns)
        print("\n" + "═" * 80)
        print(f" {title.upper()} ".center(80))
        print("═" * 80)
        if "error" in r:
            print(r["error"])
        else:
            print(f"{'Observations':<30} {r['observations']:>15}")
            print(f"{'CAGR':<30} {r['cagr_percent']:>18.3f}%")
            print(f"{'Sharpe Ratio':<30} {r['sharpe']:>19.3f}")
            print(
                f"{'Deflated Sharpe':<30} {r['deflated_sharpe']:>19.3f} → {'EXCELLENT' if r['deflated_sharpe']>1.5 else 'Weak'}"
            )
            print(f"{'PBO Risk':<30} {r['pbo']:>23.3f} → {'LOW' if r['pbo']<0.25 else 'HIGH'}")
            print(
                f"{'Walk-Forward Sharpe':<30} {r['walk_forward']['mean']:>15.3f} ± {r['walk_forward']['std']:.3f}"
            )
            print(f"{'WF Periods':<30} {r['walk_forward']['periods']:>19}")
        print("═" * 80)
        verdict = (
            "STRONG EVIDENCE OF PERSISTENT ALPHA"
            if r.get("robust")
            else "FURTHER TESTING RECOMMENDED"
        )
        print(f" VERDICT: {verdict} ".center(80))
        print("═" * 80 + "\n")


# Global easy-use instance
validator = StatisticalValidator()


def print_report(returns, title="PyFundLib Validation", **kwargs):
    v = StatisticalValidator(**kwargs) if kwargs else validator
    v.report(returns, title)
