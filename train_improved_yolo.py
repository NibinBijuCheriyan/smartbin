"""
Cashcrow Smartbin — High-Performance YOLO Waste Detection Fine-Tuning.

Trains YOLOv11 on the 5-class real waste dataset with optimal domain-specific
augmentations, transfer learning from COCO pretrained checkpoint, and cosine LR scheduling.
"""

import os
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainYOLO")


def train_model(
    model_name: str = "yolo11n.pt",
    dataset_yaml: str = "data/dataset.yaml",
    epochs: int = 20,
    imgsz: int = 416,
    batch: int = 32,
    device: str = "cpu",
    project: str = "runs/detect",
    name: str = "improved_yolo",
):
    logger.info("Initializing YOLO model with pretrained weights: %s", model_name)
    model = YOLO(model_name)

    logger.info("Starting training run '%s' for %d epochs at %dpx (device=%s)...", name, epochs, imgsz, device)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=8,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        # Optimizer and LR schedule
        optimizer="auto",
        lr0=0.003,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.0,
        # Waste-domain specific augmentations
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.3,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=5,
        plots=True,
        save=True,
        val=True,
        workers=0,
    )

    # Copy fine-tuned best weights to root best.pt
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, "best.pt")
        shutil.copy(best_pt, "best_improved.pt")
        logger.info("Fine-tuned weights successfully saved to best.pt and best_improved.pt")

    return results


if __name__ == "__main__":
    train_model()
