"""
Perceptual Image Deduplication module for SmartBin AI v2.
Computes dHash/pHash to identify exact and near-duplicate images across merged datasets.
Generates an audit report and purges redundant duplicates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("dataset_deduplicator")


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Compute difference hash (dHash) for an image.
    Efficient O(N) calculation detecting visual similarity regardless of resolution.
    """
    # Resize to (hash_size + 1, hash_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
    # Compute horizontal differences
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert bool matrix to integer hash
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance (number of differing bits) between two integers."""
    return bin(h1 ^ h2).count("1")


@dataclass
class DuplicatePair:
    original_path: str
    duplicate_path: str
    hamming_distance: int
    similarity_score: float


class ImageDeduplicator:
    """Finds and filters duplicate images using difference hashing."""

    def __init__(self, hash_size: int = 8, threshold: int = 4) -> None:
        self.hash_size = hash_size
        self.threshold = threshold

    def scan_directory(
        self,
        images_dir: str | Path,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> Tuple[List[DuplicatePair], List[Path]]:
        """
        Scan a directory of images, identifying duplicate pairs and unique paths.
        """
        img_dir = Path(images_dir)
        image_paths = [
            p for p in img_dir.rglob("*") if p.suffix.lower() in extensions
        ]
        logger.info(f"Scanning {len(image_paths)} images for duplicates in {img_dir}...")

        hashes: Dict[Path, int] = {}
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                hashes[p] = compute_dhash(img, self.hash_size)

        duplicates: List[DuplicatePair] = []
        discard_paths: Set[Path] = set()
        kept_paths: List[Path] = []

        path_list = list(hashes.keys())
        for i in range(len(path_list)):
            p1 = path_list[i]
            if p1 in discard_paths:
                continue

            h1 = hashes[p1]
            kept_paths.append(p1)

            for j in range(i + 1, len(path_list)):
                p2 = path_list[j]
                if p2 in discard_paths:
                    continue

                h2 = hashes[p2]
                dist = hamming_distance(h1, h2)
                if dist <= self.threshold:
                    sim = 1.0 - (dist / (self.hash_size * self.hash_size))
                    duplicates.append(
                        DuplicatePair(
                            original_path=str(p1),
                            duplicate_path=str(p2),
                            hamming_distance=dist,
                            similarity_score=round(sim, 4),
                        )
                    )
                    discard_paths.add(p2)

        logger.info(
            f"Deduplication complete: {len(duplicates)} duplicates found. "
            f"{len(kept_paths)} unique images retained."
        )
        return duplicates, kept_paths

    def generate_report(
        self, duplicates: List[DuplicatePair], report_path: str | Path
    ) -> Path:
        """Save a JSON duplicate audit report."""
        out_file = Path(report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "total_duplicates_found": len(duplicates),
            "threshold_hamming_distance": self.threshold,
            "pairs": [asdict(d) for d in duplicates],
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Duplicate audit report saved to {out_file}")
        return out_file
