"""
Dataset format converters.
Transforms COCO JSON annotations and Pascal VOC XML files into normalized YOLO format:
<class_id> <x_center> <y_center> <width> <height>
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from smartbin_v2.utils.logger import get_logger

logger = get_logger("dataset_converters")


def convert_coco_to_yolo(
    coco_json_path: str | Path,
    output_labels_dir: str | Path,
    class_mapping: Optional[Dict[str, int]] = None,
) -> int:
    """
    Convert a COCO annotations JSON file into per-image YOLO .txt annotation files.

    Args:
        coco_json_path: Path to COCO instances JSON.
        output_labels_dir: Directory where YOLO .txt files will be saved.
        class_mapping: Optional dictionary mapping COCO category name to target class ID.

    Returns:
        Total number of annotation files generated.
    """
    json_path = Path(coco_json_path)
    labels_dir = Path(output_labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"COCO file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    annotations = coco_data.get("annotations", [])

    # Group annotations by image_id
    img_annotations: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        img_annotations.setdefault(img_id, []).append(ann)

    count = 0
    for img_id, anns in img_annotations.items():
        if img_id not in images:
            continue
        
        img_info = images[img_id]
        img_w = float(img_info.get("width", 640))
        img_h = float(img_info.get("height", 640))
        file_stem = Path(img_info["file_name"]).stem
        label_file = labels_dir / f"{file_stem}.txt"

        lines: List[str] = []
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, "unknown")
            
            # Map category name or use default ID
            if class_mapping and cat_name in class_mapping:
                target_cls = class_mapping[cat_name]
            elif class_mapping and "*" in class_mapping:
                target_cls = class_mapping["*"]
            else:
                target_cls = cat_id

            # COCO bbox: [x_min, y_min, width, height] in pixels
            x_min, y_min, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Convert to normalized YOLO coordinates
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clip values to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        count += 1

    logger.info(f"Converted {count} COCO images to YOLO labels in {labels_dir}")
    return count


def convert_voc_to_yolo(
    voc_xml_path: str | Path,
    output_label_path: str | Path,
    class_mapping: Optional[Dict[str, int]] = None,
) -> bool:
    """
    Convert a single Pascal VOC XML annotation file to a YOLO .txt file.
    """
    xml_file = Path(voc_xml_path)
    if not xml_file.exists():
        return False

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return False

    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    if img_w <= 0 or img_h <= 0:
        return False

    lines: List[str] = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if class_mapping and cls_name in class_mapping:
            cls_id = class_mapping[cls_name]
        else:
            cls_id = 0

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue

        x_center = (xmin + w / 2.0) / img_w
        y_center = (ymin + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    out_path = Path(output_label_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return True
