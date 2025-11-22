# src/pyfundlib/ml/pipelines/training_pipeline.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration import MLflowCallback
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ...utils.logger import get_logger
from ..models.base_model import BaseMLModel
from .feature_pipeline import FeaturePipeline

logger = get_logger(__name__)

MetricFunc = Callable[[np.ndarray, np.ndarray] | float]


@dataclass
class TrainingResult:
    best_model: BaseMLModel
    best_params: dict[str, Any]
    best_score: float
    trial_history: pd.DataFrame
    feature_importance: pd.Series | None = None


class TrainingPipeline:
    """
    The most advanced ML training pipeline for financial time series.
    Zero boilerplate. Full power.
    """

    def __init__(
        self,
        model_class: type[BaseMLModel],
        feature_pipeline: FeaturePipeline,
        task: str = "classification",  # or "regression"
        metric: str = "auc",  # "accuracy", "rmse", "sharpe", etc.
        cv_splits: int = 5,
        n_trials: int = 100,
        timeout: int | None = 3600,
        mlflow_experiment: str = "pyfundlib",
        random_state: int = 42,
    ):
        self.model_class = model_class
        self.feature_pipeline = feature_pipeline
        self.task = task
        self.cv_splits = cv_splits
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

        self.metric_func = self._get_metric_func(metric)
        self.is_higher_better = metric not in ["rmse", "mse"]

        mlflow.set_experiment(mlflow_experiment)

    def _get_metric_func(self, metric_name: str) -> MetricFunc:
        if metric_name == "auc":
            return lambda y, p: roc_auc_score(y, p)
        elif metric_name == "accuracy":
            return lambda y, p: accuracy_score(y, p.round() if self.task == "regression" else p)
        elif metric_name in ["rmse", "mse"]:
            return lambda y, p: mean_squared_error(y, p, squared=(metric_name == "rmse"))
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        # Suggest hyperparameters based on model type
        if "xgboost" in self.model_class.__name__.lower():
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
        elif "lstm" in self.model_class.__name__.lower():
            params = {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
                "num_layers": trial.suggest_int("num_layers", 1, 5),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            }
        elif "random_forest" in self.model_class.__name__.lower():
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5, 0.8]
                ),
            }
        else:
            params = {}

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.model_class(**params)
            model.fit(X_train, y_train)

            if self.task == "classification":
                pred = (
                    model.predict_proba(X_val)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(X_val)
                )
            else:
                pred = model.predict(X_val)

            score = self.metric_func(y_val, pred)
            scores.append(score)

        return np.mean(scores)

    def run(
        self,
        X_raw: pd.DataFrame,
        y: pd.Series,
        project_name: str = "optuna_study",
    ) -> TrainingResult:
        """Run full training + optimization"""
        logger.info(
            f"Starting TrainingPipeline | {self.model_class.__name__} | {self.n_trials} trials"
        )

        # Apply features
        X = self.feature_pipeline.fit_transform(X_raw)

        # Align X and y
        X, y = X.align(y, join="inner", axis=0)

        study = optuna.create_study(
            direction="maximize" if self.is_higher_better else "minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(),
        )

        mlflow_callback = MLflowCallback(
            metric_name="cv_score",
            mlflow_experiment_name=mlflow.get_experiment_by_name(
                mlflow.active_run().info.experiment_id if mlflow.active_run() else "Default"
            ).name,
        )

        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        best_params = study.best_params
        logger.info(f"Best CV Score: {study.best_value:.6f} | Params: {best_params}")

        # Retrain on full data with best params
        best_model = self.model_class(**best_params)
        best_model.fit(X, y)
        best_model.metadata.performance_metrics["cv_best"] = study.best_value
        best_model.log_to_mlflow(run_name=f"{project_name}_best")

        # Feature importance (if available)
        importance = None
        if hasattr(best_model, "feature_importance"):
            importance = best_model.feature_importance(top_n=30)

        result = TrainingResult(
            best_model=best_model,
            best_params=best_params,
            best_score=study.best_value,
            trial_history=study.trials_dataframe(),
            feature_importance=importance,
        )

        logger.info("TrainingPipeline completed — model ready for live deployment")
        return result


# Pre-built elite training setups
def train_xgboost_direction(X_raw: pd.DataFrame, y: pd.Series):
    from ..models.xgboost import XGBoostModel
    from .feature_pipeline import get_default_pipeline

    pipeline = TrainingPipeline(
        model_class=XGBoostModel,
        feature_pipeline=get_default_pipeline(),
        task="classification",
        metric="auc",
        n_trials=200,
    )
    return pipeline.run(X_raw, y, project_name="xgboost_direction")


def train_lstm_returns(X_raw: pd.DataFrame, y: pd.Series):
    from ..models.lstm import LSTMModel
    from .feature_pipeline import get_default_pipeline

    pipeline = TrainingPipeline(
        model_class=LSTMModel,
        feature_pipeline=get_default_pipeline(),
        task="regression",
        metric="rmse",
        n_trials=100,
    )
    return pipeline.run(X_raw, y, project_name="lstm_returns")
