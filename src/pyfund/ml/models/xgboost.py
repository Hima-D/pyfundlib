# src/pyfundlib/ml/models/xgboost.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

from ...utils.logger import get_logger
from .base_model import BaseMLModel

logger = get_logger(__name__)


class XGBoostModel(BaseMLModel):
    """
    State-of-the-art XGBoost with full pyfundlib integration.
    Supports classification, regression, and ranking.
    """

    def __init__(
        self,
        task: Literal["classification", "regression", "ranking"] = "classification",
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        scale_pos_weight: float | None = None,
        name: str = "xgboost",
        version: str = "1.0",
        **xgb_kwargs,
    ):
        super().__init__(name=name, version=version, tags=["xgboost", "gradient-boosting", task])

        self.task = task
        self.early_stopping_rounds = early_stopping_rounds
        self.hyperparams = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            **xgb_kwargs,
        }

        # Objective mapping
        objective_map = {
            "classification": "binary:logistic" if len(np.unique(y_train)) == 2 if 'y_train' in locals() else True else "multi:softprob",
            "regression": "reg:squarederror",
            "ranking": "rank:pairwise",
        }

        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",  # Fast & memory efficient
            **xgb_kwargs,
        }

        if task == "classification":
            self.model = xgb.XGBClassifier(**params)
            if scale_pos_weight:
                self.model.scale_pos_weight = scale_pos_weight
        elif task == "regression":
            self.model = xgb.XGBRegressor(**params)
        else:
            self.model = xgb.XGBRanker(**params)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        eval_set: list | None = None,
        sample_weight: np.ndarray | None = None,
        verbose: int = 100,
    ) -> XGBoostModel:
        X = self._validate_input(X)
        y = np.array(y)

        eval_metric = None
        if self.task == "classification":
            eval_metric = "logloss" if len(np.unique(y_train)) == 2 if 'y_train' in locals() else True else "mlogloss"
        elif self.task == "regression":
            eval_metric = "rmse"

        fit_params = {
            "X": X,
            "y": y,
            "eval_set": eval_set or [(X, y)],
            "early_stopping_rounds": self.early_stopping_rounds if eval_set else None,
            "verbose": verbose,
            "sample_weight": sample_weight,
        }

        if eval_set:
            fit_params["eval_metric"] = eval_metric

        self.model.fit(**fit_params)

        # Metadata
        self._is_fitted = True
        self.feature_names_in_ = list(X.columns)
        self.metadata.training_samples = len(y)
        self.metadata.features_used = self.feature_names_in_
        self.metadata.hyperparameters = self.hyperparams

        # Performance metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X) if self.task == "classification" else None

        metrics = {}
        if self.task == "classification":
            metrics["accuracy"] = accuracy_score(y, y_pred)
            if y_prob is not None and len(np.unique(y_train)) == 2 if 'y_train' in locals() else True:
                metrics["auc"] = roc_auc_score(y, y_prob[:, 1])
        elif self.task == "regression":
            metrics["rmse"] = mean_squared_error(y, y_pred, squared=False)
            metrics["r2"] = r2_score(y, y_pred)

        if hasattr(self.model, "best_score"):
            metrics["best_score"] = self.model.best_score
            metrics["best_iteration"] = self.model.best_iteration

        self.metadata.performance_metrics = metrics
        self.metadata.status = "trained"

        logger.info(
            f"XGBoost ({self.task}) trained | "
            f"Iterations: {self.model.get_booster().num_boosted_rounds()} | "
            f"Features: {X.shape[1]} | {metrics}"
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = self._validate_input(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("predict_proba only for classification")
        X = self._validate_input(X)
        return self.model.predict_proba(X)

    def feature_importance(
        self,
        importance_type: Literal["gain", "weight", "cover"] = "gain",
        top_n: int = 20,
    ) -> pd.Series:
        """Return sorted feature importance"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        imp_df = pd.Series(importance).sort_values(ascending=False).head(top_n)
        return (
            imp_df.reindex(self.feature_names_in_, fill_value=0)
            .sort_values(ascending=False)
            .head(top_n)
        )

    def plot_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 20,
        save_path: str | None = None,
    ):
        """Beautiful importance plot"""
        import matplotlib.pyplot as plt

        imp = self.feature_importance(importance_type, top_n)
        plt.figure(figsize=(10, 8))
        imp.plot(kind="barh", color="#F18F01")
        plt.title(f"XGBoost {importance_type.title()} Importance - {self.name}")
        plt.xlabel(importance_type.title())
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
