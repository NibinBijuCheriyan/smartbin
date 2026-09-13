"""
Master Dataset Pipeline Runner for SmartBin AI v2.
Orchestrates downloading, format conversion, deduplication, class balancing,
stratified train/val/test splitting, and automated quality auditing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

from smartbin_v2.dataset_builder.balancer import DatasetBalancer
from smartbin_v2.dataset_builder.deduplicator import ImageDeduplicator
from smartbin_v2.dataset_builder.downloaders import DatasetDownloader
from smartbin_v2.dataset_builder.quality_checks import DatasetQualityAuditor
from smartbin_v2.utils.config import load_yaml_config
from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("build_dataset")


def run_pipeline(config_path: str = "smartbin-v2/configs/dataset_config.yaml") -> int:
    """Execute complete dataset engineering workflow."""
    setup_logging(level="INFO")
    logger.info("====================================================================")
    logger.info("Starting SmartBin AI v2 Industrial Dataset Engineering Pipeline")
    logger.info("====================================================================")

    cfg = load_yaml_config(config_path)
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "smartbin-v2/datasets/raw"))
    processed_dir = Path(paths.get("processed_dir", "smartbin-v2/datasets/processed"))
    splits_dir = Path(paths.get("splits_dir", "smartbin-v2/datasets/splits"))
    reports_dir = Path(paths.get("reports_dir", "smartbin-v2/datasets/reports"))
    yaml_output = Path(paths.get("yaml_output", "smartbin-v2/datasets/smartbin_dataset.yaml"))

    # 1. Download datasets
    logger.info("--- Phase 1: Ingesting Public and Custom Datasets ---")
    downloader = DatasetDownloader(raw_dir=raw_dir)
    # Trigger download checks
    downloaded_sources = downloader.download_all()
    logger.info(f"Verified {len(downloaded_sources)} source directories in {raw_dir}")

    # 2. Deduplication
    logger.info("--- Phase 2: Perceptual Deduplication (dHash) ---")
    dedup_cfg = cfg.get("deduplication", {})
    if dedup_cfg.get("enabled", True):
        deduplicator = ImageDeduplicator(
            hash_size=dedup_cfg.get("hash_size", 8),
            threshold=dedup_cfg.get("hamming_threshold", 4),
        )
        duplicates, unique_images = deduplicator.scan_directory(processed_dir if processed_dir.exists() else raw_dir)
        deduplicator.generate_report(duplicates, reports_dir / "duplicate_report.json")

    # 3. Stratified Train / Val / Test Partitioning
    logger.info("--- Phase 3: Class Balancing & Stratified Partitioning (70/15/15) ---")
    split_ratios = cfg.get("split_ratios", {})
    balancer = DatasetBalancer(
        train_ratio=split_ratios.get("train", 0.70),
        val_ratio=split_ratios.get("val", 0.15),
        test_ratio=split_ratios.get("test", 0.15),
        seed=split_ratios.get("seed", 42),
    )
    
    # Check if processed images and labels exist
    proc_images = list(processed_dir.glob("*.jpg")) + list(processed_dir.glob("*.png"))
    proc_labels = list(processed_dir.glob("*.txt"))
    
    if proc_images and proc_labels:
        label_info = balancer.parse_yolo_labels(processed_dir)
        pairs = [(p, processed_dir / f"{p.stem}.txt") for p in proc_images]
        splits = balancer.create_stratified_splits(pairs, label_info)
        balancer.export_splits(splits, splits_dir)
    else:
        logger.info(f"No existing images in {processed_dir}; ensuring split directory structure...")
        for s in ["train", "val", "test"]:
            (splits_dir / s / "images").mkdir(parents=True, exist_ok=True)
            (splits_dir / s / "labels").mkdir(parents=True, exist_ok=True)

    # 4. Quality Audit & Distribution Heatmaps
    logger.info("--- Phase 4: Automated Quality Audit & Heatmap Generation ---")
    model_cfg = load_yaml_config("smartbin-v2/configs/model_config.yaml")
    class_names = model_cfg.get("class_names", {})

    auditor = DatasetQualityAuditor(class_names=class_names)
    train_img_dir = splits_dir / "train" / "images"
    train_lbl_dir = splits_dir / "train" / "labels"
    audit_report = auditor.audit_dataset(train_img_dir, train_lbl_dir, reports_dir=reports_dir)

    # 5. Output Final Dataset YAML
    logger.info("--- Phase 5: Generating Ultralytics Dataset Descriptor ---")
    dataset_yaml_data = {
        "path": "../datasets",
        "train": "splits/train/images",
        "val": "splits/val/images",
        "test": "splits/test/images",
        "nc": len(class_names) if class_names else 32,
        "names": class_names,
    }
    yaml_output.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_output, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Successfully generated YOLO dataset descriptor: {yaml_output}")
    logger.info("Dataset pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBin AI v2 Dataset Builder CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="smartbin-v2/configs/dataset_config.yaml",
        help="Path to dataset configuration YAML",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args.config))
