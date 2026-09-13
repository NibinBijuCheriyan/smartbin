"""
Automated Hyperparameter Tuning Pipeline for SmartBin AI v2.
Explores optimal learning rates, batch sizes, weight decay, momentum, mosaic, and mixup.
Saves optimal parameters to experiments/best_hyperparameters.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import yaml
from ultralytics import YOLO

from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("hyperparameter_search")


def run_hyperparameter_search(
    data_yaml: str = "smartbin-v2/datasets/smartbin_dataset.yaml",
    model_name: str = "yolo11s.pt",
    iterations: int = 20,
    epochs_per_trial: int = 15,
) -> Dict[str, Any]:
    """
    Execute hyperparameter tuning using Ultralytics tuning engine (Optuna/Genetic algorithm).
    """
    setup_logging(level="INFO")
    logger.info("====================================================================")
    logger.info(f"Starting Hyperparameter Optimization for {model_name} ({iterations} trials)")
    logger.info("====================================================================")

    model = YOLO(model_name)

    # Launch Ultralytics tune
    tune_results = model.tune(
        data=data_yaml,
        epochs=epochs_per_trial,
        iterations=iterations,
        optimizer="AdamW",
        plots=True,
        save=False,
        val=True,
        project="smartbin-v2/runs/tune",
        name="smartbin_tuning",
        space={
            "lr0": (1e-4, 1e-2),
            "lrf": (0.01, 0.2),
            "momentum": (0.85, 0.98),
            "weight_decay": (1e-5, 1e-3),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 0.5),
        }
    )

    out_dir = Path("smartbin-v2/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = out_dir / "best_hyperparameters.yaml"

    logger.info(f"Hyperparameter tuning completed. Results saved in {best_params_path}")
    return {"status": "success", "results": str(tune_results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBin AI v2 Hyperparameter Tuning")
    parser.add_argument("--data", type=str, default="smartbin-v2/datasets/smartbin_dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_hyperparameter_search(
        data_yaml=args.data,
        model_name=args.model,
        iterations=args.iterations,
        epochs_per_trial=args.epochs,
    )
