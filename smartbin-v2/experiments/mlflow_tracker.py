"""
MLflow Experiment Tracking and Model Registry Wrapper for SmartBin AI v2.
Logs training hyperparameters, validation metrics (mAP, F1, Loss), confusion matrices,
and registered model artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from smartbin_v2.utils.logger import get_logger

logger = get_logger("mlflow_tracker")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None


class MLflowTracker:
    """
    MLflow logger managing experiment lifecycles, runs, metrics, and artifact registration.
    """

    def __init__(
        self,
        tracking_uri: str = "file:./smartbin-v2/experiments/mlruns",
        experiment_name: str = "smartbin-v2-training",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.active_run = None

        if HAS_MLFLOW:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
        else:
            logger.warning("MLflow not installed in environment; metrics will be logged to stdout only.")

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Start a new tracked MLflow run."""
        if HAS_MLFLOW:
            self.active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
            return self.active_run
        return None

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log model architecture and training hyperparameters."""
        if HAS_MLFLOW and self.active_run:
            mlflow.log_params(params)
        logger.debug(f"Parameters logged: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training or validation metrics (loss, mAP, precision, recall)."""
        if HAS_MLFLOW and self.active_run:
            mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Metrics logged at step {step}: {metrics}")

    def log_artifact(self, local_path: str | Path) -> None:
        """Upload model weights, confusion matrix, or report artifact to MLflow."""
        p = Path(local_path)
        if HAS_MLFLOW and self.active_run and p.exists():
            mlflow.log_artifact(str(p))
            logger.info(f"Uploaded artifact to MLflow: {p.name}")

    def end_run(self) -> None:
        """Close active MLflow run."""
        if HAS_MLFLOW and self.active_run:
            mlflow.end_run()
            self.active_run = None
            logger.info("MLflow run concluded.")
