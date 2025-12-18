# src/pyfund/optimization/bayesian.py
"""
Bayesian Optimization Module for pyfund

Features:
- Hyperparameter tuning (ML models, indicators, risk params)
- Position sizing optimization (Kelly, volatility targeting)
- Strategy parameter search (RSI thresholds, lookbacks, etc.)
- Uses Scikit-Optimize (skopt) — lightweight, powerful, no heavy dependencies
- Full MLflow logging + visualization
- Works with any objective function (backtest Sharpe, Calmar, win rate, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_score: float
    all_scores: List[float]
    all_params: List[Dict[str, Any]]
    convergence_trace: List[float]
    n_calls: int
    time_elapsed: float


class BayesianOptimizer:
    """
    Bayesian optimization engine using Gaussian Processes.
    Optimized for financial strategy tuning.
    """

    def __init__(
        self,
        dimensions: List[Any],
        n_initial_points: int = 10,
        n_calls: int = 50,
        noise: float = 1e-5,
        acq_func: str = "gp_hedge",
        random_state: int | None = 42,
        mlflow_run_name: str | None = None,
    ):
        self.dimensions = dimensions
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.acq_func = acq_func
        self.random_state = random_state

        # Create optimizer
        self.optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",
            acq_func=acq_func,
            acq_optimizer="lbfgs",
            random_state=random_state,
        )

        self.results: List[Tuple[float, Dict[str, Any]]] = []
        self.mlflow_run_name = mlflow_run_name or f"bayesian_opt_{datetime.now():%Y%m%d_%H%M}"

        logger.info(f"BayesianOptimizer initialized | {n_calls} calls | {acq_func}")

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        minimize: bool = False,
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            objective: Function that takes params dict and returns score (higher = better)
                       To minimize, set minimize=True (e.g., for drawdown)
            minimize: If True, flips score so lower is better

        Returns:
            OptimizationResult with best params and history
        """
        start_time = datetime.now()

        @use_named_args(self.dimensions)
        def wrapped_objective(**params):
            score = objective(params)
            if minimize:
                score = -score
            return score

        logger.info(f"Starting Bayesian optimization ({self.n_calls} iterations)...")

        with mlflow.start_run(run_name=self.mlflow_run_name):
            mlflow.log_param("n_calls", self.n_calls)
            mlflow.log_param("n_initial_points", self.n_initial_points)
            mlflow.log_param("acq_func", self.acq_func)

            for i in range(self.n_calls):
                x = self.optimizer.ask()
                y = wrapped_objective(*x)

                self.optimizer.tell(x, y)
                self.results.append((y, dict(zip(self._param_names(), x))))

                best_so_far = max(r[0] for r in self.results)
                logger.info(f"Iter {i+1}/{self.n_calls} | Current: {y:.4f} | Best: {best_so_far:.4f}")

                mlflow.log_metric("current_score", y, step=i)
                mlflow.log_metric("best_score", best_so_far, step=i)

        # Extract results
        if minimize:
            best_idx = np.argmin([r[0] for r in self.results])
        else:
            best_idx = np.argmax([r[0] for r in self.results])

        best_score = self.results[best_idx][0]
        if minimize:
            best_score = -best_score

        result = OptimizationResult(
            best_params=self.results[best_idx][1],
            best_score=best_score,
            all_scores=[r[0] if not minimize else -r[0] for r in self.results],
            all_params=[r[1] for r in self.results],
            convergence_trace=[max(self.results[:i+1], key=lambda x: x[0] if not minimize else -x[0])[0]
                              if not minimize else -min(self.results[:i+1], key=lambda x: x[0])[0]
                              for i in range(len(self.results))],
            n_calls=len(self.results),
            time_elapsed=(datetime.now() - start_time).total_seconds(),
        )

        logger.info(f"Optimization complete! Best score: {result.best_score:.4f}")
        logger.info(f"Best params: {result.best_params}")

        self._log_final_results(result)
        self.plot_convergence(result)

        return result

    def _param_names(self) -> List[str]:
        return [dim.name for dim in self.dimensions if hasattr(dim, "name")]

    def _log_final_results(self, result: OptimizationResult) -> None:
        mlflow.log_metric("final_best_score", result.best_score)
        for k, v in result.best_params.items():
            mlflow.log_param(f"best_{k}", v)

    def plot_convergence(self, result: OptimizationResult, save_path: str | None = None) -> None:
        """Plot optimization convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(result.convergence_trace, marker="o")
        plt.title("Bayesian Optimization Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.grid(True, alpha=0.3)

        if save_path or True:
            path = Path("reports/bayesian_convergence.png")
            path.parent.mkdir(exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(path))
            logger.info(f"Convergence plot saved: {path}")

        plt.close()


# === Pre-built Templates ===

def optimize_rsi_strategy(
    backtest_func: Callable[[int, int], float],
    rsi_low_range: Tuple[int, int] = (20, 40),
    rsi_high_range: Tuple[int, int] = (60, 80),
) -> OptimizationResult:
    """Optimize RSI mean reversion thresholds"""
    dimensions = [
        Integer(low=rsi_low_range[0], high=rsi_low_range[1], name="rsi_low"),
        Integer(low=rsi_high_range[0], high=rsi_high_range[1], name="rsi_high"),
    ]

    optimizer = BayesianOptimizer(dimensions, n_calls=30)
    return optimizer.optimize(
        objective=lambda params: backtest_func(params["rsi_low"], params["rsi_high"])
    )


def optimize_position_sizing(
    returns: pd.Series,
    methods: List[str] = None,
) -> OptimizationResult:
    """Optimize risk per trade using Kelly/vol targeting"""
    methods = methods or ["kelly", "volatility_parity", "fixed_fraction", "risk_parity"]

    dimensions = [
        Real(0.001, 0.2, name="risk_per_trade"),
        Categorical(methods, name="method"),
    ]

    def objective(params):
        # Simulate PnL with given sizing
        risk = params["risk_per_trade"]
        # Simple simulation
        vol = returns.std() * np.sqrt(252)
        target_vol = risk
        scaled_returns = returns * (target_vol / vol)
        return scaled_returns.mean() / scaled_returns.std() * np.sqrt(252)  # Sharpe

    optimizer = BayesianOptimizer(dimensions, n_calls=40)
    return optimizer.optimize(objective)