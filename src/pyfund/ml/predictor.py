# src/pyfundlib/ml/predictor.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from ..data.storage import DataStorage
from ..utils.logger import get_logger
from .models.base_model import BaseMLModel
from .pipelines.feature_pipeline import FeaturePipeline
from .pipelines.training_pipeline import TrainingPipeline

logger = get_logger(__name__)


class MLPredictor:
    """
    Central ML orchestrator for pyfundlib.
    Handles training, prediction, versioning, and model registry via MLflow.
    """

    def __init__(
        self,
        model_dir: str = "models",
        mlflow_tracking_uri: str | None = None,
        experiment_name: str = "pyfundlib",
        registry_name: str = "pyfundlib_models",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect local tracking if not provided
        tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Create registry if not exists
        client = mlflow.tracking.MlflowClient()
        try:
            client.create_registered_model(registry_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        self.client = client
        self.storage = DataStorage()

        logger.info(f"MLPredictor initialized | Tracking: {tracking_uri} | Models: {self.model_dir}")

    def train(
        self,
        ticker: str,
        raw_data: pd.DataFrame,
        target: pd.Series,
        model_class: type[BaseMLModel],
        feature_pipeline: FeaturePipeline,
        pipeline_config: dict[str, Any] | None = None,
    ) -> BaseMLModel:
        """Train a model for a ticker with full MLflow tracking"""
        logger.info(f"Training {ticker} → {model_class.__name__}")

        pipeline_config = pipeline_config or {}
        training_pipeline = TrainingPipeline(
            model_class=model_class,
            feature_pipeline=feature_pipeline,
            **pipeline_config,
        )

        run_name = f"{ticker}_{datetime.now():%Y%m%d_%H%M}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model_type", model_class.__name__)
            mlflow.log_param("data_points", len(raw_data))

            result = training_pipeline.run(raw_data, target, project_name=f"{ticker}")

            best_model = result.best_model
            best_model.metadata.tags.extend([ticker, "automated", "retrain-candidate"])

            # Save locally
            local_path = self.model_dir / f"{ticker}_{best_model.name}_v{best_model.version}.pkl"
            best_model.save(local_path)
            mlflow.log_artifact(str(local_path))

            # Log metrics & params
            if result.best_model.metadata.performance_metrics:
                mlflow.log_metrics(result.best_model.metadata.performance_metrics)
            if result.best_params:
                mlflow.log_params(result.best_params)

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts/model"
            model_name = f"{ticker}_{best_model.name}"
            registered = mlflow.register_model(model_uri, model_name)

            # Move to Staging
            self.client.transition_model_version_stage(
                name=registered.name,
                version=registered.version,
                stage="Staging",
                archive_existing_versions=False,
            )

            logger.info(f"Model registered: {model_name} v{registered.version} → Staging")
            return best_model

    def predict(
        self,
        ticker: str,
        raw_data: pd.DataFrame,
        model_name: str | None = None,
        stage: str = "Production",
    ) -> np.ndarray:
        model = self.load_latest(ticker, model_name or "xgboost", stage=stage)
        if model is None:
            logger.warning(f"No {stage} model for {ticker}. Returning zeros.")
            return np.zeros(len(raw_data))

        latest_row = raw_data.iloc[-1:]
        try:
            pred = model.predict(latest_row)
            logger.info(f"{ticker} prediction: {pred[0]:.4f}")
            return pred
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return np.zeros(len(raw_data))

    def load_latest(
        self,
        ticker: str,
        model_name: str = "xgboost",
        stage: str = "Production",
    ) -> BaseMLModel | None:
        try:
            versions = self.client.get_latest_versions(f"{ticker}_{model_name}", stages=[stage])
            if not versions:
                logger.warning(f"No {stage} version found for {ticker}_{model_name}")
                return None

            version = versions[0]
            model_uri = f"models:/{version.name}/{version.version}"
            loaded = mlflow.pyfunc.load_model(model_uri)

            if isinstance(loaded, BaseMLModel):
                logger.info(f"Loaded {stage} model: {version.name} v{version.version}")
                return loaded
            else:
                logger.warning("Loaded model is not BaseMLModel subclass. Check wrapper.")
                return None
        except Exception as e:
            logger.error(f"Failed to load model {ticker}_{model_name}: {e}")
            return None

    def promote_to_production(self, ticker: str, model_name: str, version: int) -> None:
        full_name = f"{ticker}_{model_name}"
        self.client.transition_model_version_stage(
            name=full_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Promoted {full_name} v{version} → Production")

    def list_models(self, ticker: str | None = None) -> pd.DataFrame:
        models = self.client.search_registered_models()
        rows = []
        for m in models:
            for v in m.latest_versions:
                rows.append({
                    "ticker": m.name.split("_")[0] if "_" in m.name else "unknown",
                    "model_name": m.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                })
        df = pd.DataFrame(rows)
        if ticker:
            df = df[df["ticker"] == ticker]
        return df.sort_values(["ticker", "stage", "version"], ascending=[True, False, False])

    def train_all(self, tickers: list[str] | None = None) -> None:
        """
        Retrain all (or specified) tickers.
        This is the method called by the automation job.
        """
        # You'll need to adjust these imports based on your actual project structure
        try:
            from ..data.fetcher import DataFetcher
            from ml.pipelines.feature_pipeline import FeaturePipeline
            from .models.xgboost import XGBoostModel  # or your preferred default
        except ImportError as e:
            logger.error(f"Required modules not found for train_all(): {e}")
            raise

        if tickers is None:
            # Example: get from storage or hardcode
            tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]

        logger.info(f"Starting batch retraining for {len(tickers)} tickers: {tickers}")

        fetcher = DataFetcher()
        model_class = XGBoostModel
        pipeline = FeaturePipeline()

        for ticker in tickers:
            try:
                logger.info(f"Fetching & training {ticker}...")
                df = fetcher.get_historical(ticker, period="max")
                if df is None or len(df) < 252:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue

                # Simple next-day direction target
                returns = df["close"].pct_change().shift(-1)
                target = (returns > 0).astype(int)

                valid = ~target.isna()
                data = df[valid].copy()
                target = target[valid]

                self.train(
                    ticker=ticker,
                    raw_data=data,
                    target=target,
                    model_class=model_class,
                    feature_pipeline=pipeline,
                )
            except Exception as e:
                logger.error(f"Failed to retrain {ticker}: {e}", exc_info=True)

        logger.info("Batch retraining completed successfully!")


# ——————————————————————————————————————————————————————————————
# Automation entrypoint — works when run directly or imported
# ——————————————————————————————————————————————————————————————

def retrain_job() -> None:
    """Standalone function for cron, GitHub Actions, Airflow, etc."""
    print("\n[Automation] Starting pyfundlib ML retraining job...")
    print(f"[Automation] Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    try:
        predictor = MLPredictor()
        predictor.train_all()
        print("\n[Automation] All models retrained and registered successfully!\n")
    except Exception as e:
        print(f"\n[Automation] Retraining FAILED: {e}\n")
        raise


# Allow direct execution
if __name__ == "__main__":
    retrain_job()