# src/pyfundlib/ml/models/base_model.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cloudpickle
import numpy as np
import pandas as pd

from ...utils.logger import get_logger

logger = get_logger(__name__)

ModelStatus = Literal["trained", "untrained", "failed"]


@dataclass
class ModelMetadata:
    """Rich metadata stored with every model"""

    model_name: str
    version: str
    trained_at: str | None = None
    author: str = "pyfundlib"
    description: str | None = None
    training_samples: int | None = None
    features_used: list[str] | None = None
    target_column: str | None = None
    performance_metrics: dict[str | float] | None = None
    hyperparameters: dict[str, Any] | None = None
    status: ModelStatus = "untrained"
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(**data)


class BaseMLModel(ABC):
    """
    The ultimate extensible base class for ALL ML models in pyfundlib.
    Used by LSTM, XGBoost, RandomForest, LinearRegression, etc.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self.name = name
        self.version = version
        self.description = description or f"{name} model v{version}"
        self.tags = tags or []

        self.metadata = ModelMetadata(
            model_name=name,
            version=version,
            description=description,
            author="pyfundlib-user",
            tags=self.tags,
        )

        self._is_fitted = False
        self.feature_names_in_: list[str] | None = None

    # ======================= Core Abstract Methods =======================
    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BaseMLModel:
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return raw predictions (regression) or probabilities (classification)"""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities (for classifiers only)"""

    # ======================= Serialization (Best-in-Class) =======================
    def save(self, path: str | Path) -> None:
        """Save model + metadata + feature names with cloudpickle (handles anything)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata.trained_at = datetime.utcnow().isoformat()
        self.metadata.status = "trained" if self._is_fitted else "untrained"

        save_data = {
            "model": self,
            "metadata": self.metadata.to_dict(),
            "feature_names": self.feature_names_in_,
        }

        with open(path, "wb") as f:
            cloudpickle.dump(save_data, f)

        logger.info(f"Model '{self.name}' v{self.version} saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> BaseMLModel:
        """Load and return a fully restored model"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, "rb") as f:
            data = cloudpickle.load(f)

        model: BaseMLModel = data["model"]
        model.metadata = ModelMetadata.from_dict(data["metadata"])
        model.feature_names_in_ = data.get("feature_names")

        logger.info(f"Model '{model.name}' v{model.version} loaded from {path}")
        return model

    # ======================= Utilities & Safety =======================
    def _validate_input(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Internal: convert to DataFrame and validate features"""
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                raise ValueError("Model was not fitted yet or feature names unknown")
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be DataFrame or numpy array")

        if self._is_fitted and self.feature_names_in_ is not None:
            missing = set(self.feature_names_in_) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

        return X

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get hyperparameters (scikit-learn compatible)"""
        params = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                val = self.__dict__[key]
                if deep and hasattr(val, "get_params"):
                    val = val.get_params(deep)
                params[key] = val
        return params

    def set_params(self, **params) -> BaseMLModel:
        """Set hyperparameters"""
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def summary(self) -> None:
        """Pretty print model info"""
        print(f"\n=== {self.name} v{self.version} ===")
        print(f"Status       : {self.metadata.status.upper()}")
        print(f"Trained at   : {self.metadata.trained_at or 'Never'}")
        print(f"Features     : {len(self.feature_names_in_) if self.feature_names_in_ else 0}")
        print(f"Tags         : {', '.join(self.tags) if self.tags else 'None'}")
        if self.metadata.performance_metrics:
            print("Metrics      :")
            for k, v in self.metadata.performance_metrics.items():
                print(f"  {k:12}: {v:.6f}")
        print("=" * 50)

    # ======================= Optional: MLflow Integration Hook =======================
    def log_to_mlflow(self, run_name: str | None = None) -> None:
        """One-click MLflow logging (optional but elite)"""
        try:
            import mlflow
            from mlflow.models import infer_signature

            with mlflow.start_run(
                run_name=run_name or f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ):
                mlflow.log_params(self.get_params())
                if self.metadata.performance_metrics:
                    mlflow.log_metrics(self.metadata.performance_metrics)
                mlflow.set_tags(
                    {
                        "model_name": self.name,
                        "version": self.version,
                        "author": self.metadata.author,
                        "tags": ",".join(self.tags),
                    }
                )

                # Auto-infer signature if possible
                if hasattr(self, "_last_X") and hasattr(self, "_last_y"):
                    signature = infer_signature(self._last_X, self.predict(self._last_X))
                    mlflow.sklearn.log_model(self, artifact_path="model", signature=signature)
                else:
                    mlflow.sklearn.log_model(self, artifact_path="model")

                logger.info(f"Model logged to MLflow → run: {mlflow.active_run().info.run_id}")
        except ImportError:
            logger.warning("MLflow not installed. Run `pip install mlflow` to enable logging.")
