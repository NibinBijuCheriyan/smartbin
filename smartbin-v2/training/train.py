"""
Production YOLO11 Training Pipeline for SmartBin AI v2.
Supports YOLO11n, YOLO11s (default), YOLO11m with Cosine LR, EMA, AMP, Early Stopping,
Multi-GPU training, deterministic seed, and optional MLflow tracking.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from smartbin_v2.utils.config import load_yaml_config
from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("training")


def train_model(
    config_path: str = "smartbin-v2/configs/training_config.yaml",
    model_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Any:
    """
    Train YOLO11 model according to specified training parameters.
    """
    setup_logging(level="INFO")
    logger.info("Initializing SmartBin AI v2 YOLO11 Training Pipeline...")

    cfg = load_yaml_config(config_path)
    train_cfg = cfg.get("training", {})

    model_name = model_override or train_cfg.get("model", "yolo11s.pt")
    data_yaml = train_cfg.get("data", "smartbin-v2/datasets/smartbin_dataset.yaml")
    epochs = epochs_override or train_cfg.get("epochs", 150)
    batch_size = batch_override or train_cfg.get("batch_size", 16)
    imgsz = train_cfg.get("imgsz", 640)
    device = device_override or train_cfg.get("device", "auto")

    logger.info(f"Target Model: {model_name}")
    logger.info(f"Dataset YAML: {data_yaml}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {imgsz}")
    logger.info(f"Device: {device}")

    # Optional MLflow tracking
    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./smartbin-v2/experiments/mlruns"))
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "smartbin-v2-training"))
            logger.info("MLflow tracking successfully initialized.")
        except ImportError:
            logger.warning("MLflow package not installed; proceeding with standard Ultralytics logging.")

    # Instantiate Ultralytics YOLO model
    model = YOLO(model_name)

    # Launch training with industrial hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=train_cfg.get("workers", 4),
        optimizer=train_cfg.get("optimizer", "AdamW"),
        lr0=train_cfg.get("lr0", 0.001),
        lrf=train_cfg.get("lrf", 0.01),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3.0),
        cos_lr=train_cfg.get("cos_lr", True),
        patience=train_cfg.get("patience", 25),
        amp=train_cfg.get("amp", True),
        deterministic=train_cfg.get("deterministic", True),
        seed=train_cfg.get("seed", 42),
        save=train_cfg.get("save", True),
        save_period=train_cfg.get("save_period", 10),
        val=train_cfg.get("val", True),
        plots=train_cfg.get("plots", True),
        project="smartbin-v2/runs/train",
        name=train_cfg.get("experiment_name", "smartbin_yolo11s"),
        exist_ok=True,
    )

    logger.info("Training complete. Best checkpoint saved to runs/train/smartbin_yolo11s/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBin AI v2 YOLO11 Training CLI")
    parser.add_argument("--config", type=str, default="smartbin-v2/configs/training_config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override base model (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device ('0', 'cpu', 'auto')")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        model_override=args.model,
        epochs_override=args.epochs,
        batch_override=args.batch,
        device_override=args.device,
    )
