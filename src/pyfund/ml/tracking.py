# src/pyfundlib/ml/tracking.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from ..utils.logger import get_logger
from .models.base_model import BaseMLModel

logger = get_logger(__name__)


class ModelTracker:
    """
    Full MLflow model registry & tracking interface for pyfundlib.
    Manage versions, stages, aliases, comparisons, and deployments.
    """

    def __init__(
        self,
        tracking_uri: str = "file://./mlruns",
        registry_uri: str | None = None,
        experiment_name: str = "pyfundlib",
    ):
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        self.client = MlflowClient(tracking_uri=tracking_uri)
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
        except:
            self.experiment_id = mlflow.create_experiment(experiment_name)

        logger.info(f"ModelTracker ready | Experiment: {experiment_name} | URI: {tracking_uri}")

    def log_model(
        self,
        model: BaseMLModel,
        artifact_path: str = "model",
        run_name: str | None = None,
        tags: dict[str | str] | None = None,
    ) -> str:
        """Log a pyfundlib model to MLflow with full metadata"""
        with mlflow.start_run(
            run_name=run_name or f"{model.name}_{model.version}", experiment_id=self.experiment_id
        ):
            # Log parameters & metrics from model
            mlflow.log_params(model.get_params())
            if model.metadata.performance_metrics:
                mlflow.log_metrics(model.metadata.performance_metrics)

            # Tags
            base_tags = {
                "model_name": model.name,
                "version": model.version,
                "author": model.metadata.author,
                "ticker": ",".join([t for t in model.tags if t.upper() in ["AAPL", "SPY", "BTC"]])
                or "multi",
                "task": model.tags[0] if model.tags else "unknown",
            }
            if tags:
                base_tags.update(tags)
            mlflow.set_tags(base_tags)

            # Save model temporarily and log
            temp_path = Path(f"/tmp/{model.name}_v{model.version}.pkl")
            model.save(temp_path)
            mlflow.log_artifact(str(temp_path), artifact_path)
            temp_path.unlink(missing_ok=True)

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            registered = mlflow.register_model(model_uri, f"{model.name}")

            logger.info(f"Model logged & registered: {model.name} v{registered.version}")
            return registered.name

    def get_latest_version(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any | None:
        """Get latest model version in given stage"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0]
            return None
        except:
            return None

    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
        version: int | None = None,
    ) -> BaseMLModel | None:
        """Load a model from registry"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"

            loaded = mlflow.pyfunc.load_model(model_uri)
            if isinstance(loaded, BaseMLModel):
                logger.info(f"Loaded {model_name} from {stage} (v{loaded.version})")
                return loaded
            else:
                logger.warning("Loaded model is not a BaseMLModel instance")
                return loaded
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def list_models(self, filter_string: str = "") -> pd.DataFrame:
        """List all registered models with latest versions"""
        models = self.client.search_registered_models(filter_string=filter_string)
        rows = []
        for m in models:
            for v in m.latest_versions:
                rows.append(
                    {
                        "name": m.name,
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "tags": v.tags,
                        "description": v.description,
                    }
                )
        return pd.DataFrame(rows).sort_values(["name", "version"], ascending=[True, False])

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> None:
        """Promote model version to Production/Staging"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Promoted {model_name} v{version} → {stage}")

    def compare_models(
        self,
        model_names: list[str],
        metric: str = "auc",
    ) -> pd.DataFrame:
        """Compare performance across models"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=" OR ".join(f"tags.model_name = '{n}'" for n in model_names),
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        if runs.empty:
            logger.warning("No runs found for comparison")
            return pd.DataFrame()

        return runs[
            ["tags.model_name", "tags.version", f"metrics.{metric}", "start_time"]
        ].sort_values(f"metrics.{metric}", ascending=False)

    def delete_model_version(self, model_name: str, version: int) -> None:
        """Delete a model version (use with care)"""
        self.client.delete_model_version(name=model_name, version=version)
        logger.info(f"Deleted {model_name} v{version}")

    def set_alias(self, model_name: str, version: int, alias: str = "champion") -> None:
        """Set alias like 'champion' or 'candidate'"""
        self.client.set_registered_model_alias(model_name, alias, str(version))
        logger.info(f"Set alias '{alias}' → {model_name} v{version}")


# Global tracker instance
tracker = ModelTracker()
