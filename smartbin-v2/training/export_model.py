"""
Industrial Model Export Pipeline for Edge Hardware Deployment.
Exports YOLO11 checkpoints to ONNX, TensorRT, OpenVINO, and TFLite (FP32, FP16, INT8).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from ultralytics import YOLO

from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("model_export")


def export_edge_model(
    weights_path: str = "best.pt",
    formats: Optional[List[str]] = None,
    output_dir: str = "smartbin-v2/models",
    imgsz: int = 640,
    int8: bool = False,
    data_yaml: str = "smartbin-v2/datasets/smartbin_dataset.yaml",
) -> Dict[str, Path]:
    """
    Export PyTorch model weights to edge-optimized execution runtimes.

    Args:
        weights_path: Path to PyTorch checkpoint (.pt).
        formats: Target formats ['onnx', 'openvino', 'engine', 'tflite'].
        output_dir: Directory to save exported weights.
        imgsz: Target square input size.
        int8: Enable INT8 post-training quantization.
        data_yaml: Dataset YAML for INT8 calibration.
    """
    setup_logging(level="INFO")
    target_formats = formats or ["onnx", "openvino", "tflite"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        logger.warning(f"Weights file {weights_path} not found. Using pretrained yolo11s.pt for export verification.")
        model = YOLO("yolo11s.pt")
    else:
        logger.info(f"Loading weights from {weights_path}...")
        model = YOLO(str(weights_file))

    exported_files: Dict[str, Path] = {}

    for fmt in target_formats:
        logger.info(f"--- Exporting to {fmt.upper()} (INT8={int8}) ---")
        try:
            export_kwargs = {
                "format": fmt,
                "imgsz": imgsz,
                "half": (fmt in ["engine", "openvino"] and not int8),
                "int8": int8,
            }
            if int8:
                export_kwargs["data"] = data_yaml

            result_path = model.export(**export_kwargs)
            res_p = Path(result_path)
            
            # Copy or move to output_dir
            dest = out_dir / res_p.name
            if res_p.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(res_p, dest)
            else:
                shutil.copy2(res_p, dest)

            exported_files[fmt] = dest
            logger.info(f"Successfully exported {fmt.upper()} -> {dest}")

        except Exception as e:
            logger.error(f"Failed to export to {fmt}: {e}")

    logger.info("Export pipeline completed.")
    return exported_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBin AI v2 Edge Model Exporter")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to .pt weights")
    parser.add_argument("--formats", nargs="+", default=["onnx", "openvino", "tflite"], help="Target formats")
    parser.add_argument("--output", type=str, default="smartbin-v2/models", help="Export output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--data", type=str, default="smartbin-v2/datasets/smartbin_dataset.yaml")
    args = parser.parse_args()

    export_edge_model(
        weights_path=args.weights,
        formats=args.formats,
        output_dir=args.output,
        imgsz=args.imgsz,
        int8=args.int8,
        data_yaml=args.data,
    )
