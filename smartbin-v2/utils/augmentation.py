"""
Production Data Augmentation Pipeline using Albumentations and OpenCV.
Supports Lighting, Geometry, Environmental Weather, Sensor Noise, Occlusion, and Mixing.
Fully configurable via YAML.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None


# ---------------------------------------------------------------------------
# Custom Sensor & Environmental Transforms
# ---------------------------------------------------------------------------


class RaspberryPiSensorNoise:
    """
    Simulates CMOS sensor noise typical of Raspberry Pi Camera Module 2/3
    in sub-optimal industrial bin illumination.
    """

    def __init__(self, p: float = 0.3, noise_factor: float = 0.04) -> None:
        self.p = p
        self.noise_factor = noise_factor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        img_float = image.astype(np.float32) / 255.0
        # Shot noise (Poisson-like) + Read noise (Gaussian)
        shot_noise = np.random.poisson(img_float * 255.0) / 255.0 - img_float
        read_noise = np.random.normal(0, self.noise_factor, img_float.shape)
        
        noisy = img_float + 0.5 * shot_noise + read_noise
        noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
        return noisy


class PartialOcclusion:
    """Simulates waste items being partially covered by cardboard flaps or chute edges."""

    def __init__(self, p: float = 0.25, coverage_ratio: float = 0.2) -> None:
        self.p = p
        self.coverage_ratio = coverage_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        h, w, _ = image.shape
        occ_w = int(w * self.coverage_ratio)
        occ_h = int(h * self.coverage_ratio)
        
        x = random.randint(0, max(1, w - occ_w))
        y = random.randint(0, max(1, h - occ_h))
        
        color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
        res = image.copy()
        cv2.rectangle(res, (x, y), (x + occ_w, y + occ_h), color, -1)
        return res


# ---------------------------------------------------------------------------
# Master Augmentation Pipeline Builder
# ---------------------------------------------------------------------------


class WasteAugmentationPipeline:
    """
    Modular Albumentations-based data augmentation pipeline.
    Controls all augmentations via configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get("augmentation", config)
        self.enabled = self.config.get("enabled", True)
        self.rpi_noise = RaspberryPiSensorNoise(
            p=self.config.get("camera_noise", {}).get("rpi_sensor_noise", {}).get("p", 0.3),
            noise_factor=self.config.get("camera_noise", {}).get("rpi_sensor_noise", {}).get("noise_factor", 0.04),
        )
        self.partial_occ = PartialOcclusion(
            p=self.config.get("occlusion", {}).get("partial_covering", {}).get("p", 0.25),
            coverage_ratio=self.config.get("occlusion", {}).get("partial_covering", {}).get("coverage_ratio", 0.2),
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Optional[Any]:
        if not HAS_ALBUMENTATIONS or not self.enabled:
            return None

        transforms: List[Any] = []
        cfg = self.config

        # 1. Lighting Robustness
        lighting = cfg.get("lighting", {})
        if lighting.get("clahe", {}).get("enabled", False):
            transforms.append(A.CLAHE(
                clip_limit=tuple(lighting["clahe"].get("clip_limit", [1.0, 4.0])),
                tile_grid_size=tuple(lighting["clahe"].get("tile_grid_size", [8, 8])),
                p=lighting["clahe"].get("p", 0.4)
            ))
        if lighting.get("random_brightness_contrast", {}).get("enabled", False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=tuple(lighting["random_brightness_contrast"].get("brightness_limit", [-0.2, 0.2])),
                contrast_limit=tuple(lighting["random_brightness_contrast"].get("contrast_limit", [-0.2, 0.2])),
                p=lighting["random_brightness_contrast"].get("p", 0.5)
            ))
        if lighting.get("random_gamma", {}).get("enabled", False):
            transforms.append(A.RandomGamma(
                gamma_limit=tuple(lighting["random_gamma"].get("gamma_limit", [80, 130])),
                p=lighting["random_gamma"].get("p", 0.3)
            ))

        # 2. Geometry Invariance
        geo = cfg.get("geometry", {})
        if geo.get("horizontal_flip", {}).get("enabled", False):
            transforms.append(A.HorizontalFlip(p=geo["horizontal_flip"].get("p", 0.5)))
        if geo.get("vertical_flip", {}).get("enabled", False):
            transforms.append(A.VerticalFlip(p=geo["vertical_flip"].get("p", 0.2)))
        if geo.get("shift_scale_rotate", {}).get("enabled", False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=geo["shift_scale_rotate"].get("shift_limit", 0.0625),
                scale_limit=geo["shift_scale_rotate"].get("scale_limit", 0.15),
                rotate_limit=geo["shift_scale_rotate"].get("rotate_limit", 30),
                border_mode=cv2.BORDER_CONSTANT,
                p=geo["shift_scale_rotate"].get("p", 0.5)
            ))
        if geo.get("perspective", {}).get("enabled", False):
            transforms.append(A.Perspective(
                scale=tuple(geo["perspective"].get("scale", [0.05, 0.10])),
                p=geo["perspective"].get("p", 0.3)
            ))

        # 3. Weather & Environmental Artifacts
        weather = cfg.get("weather", {})
        if weather.get("random_shadow", {}).get("enabled", False):
            transforms.append(A.RandomShadow(
                num_shadows_limit=tuple(weather["random_shadow"].get("num_shadows_limit", [1, 3])),
                p=weather["random_shadow"].get("p", 0.35)
            ))
        if weather.get("random_sun_flare", {}).get("enabled", False):
            transforms.append(A.RandomSunFlare(
                flare_roi=tuple(weather["random_sun_flare"].get("flare_roi", [0, 0, 1, 0.5])),
                angle_range=tuple(weather["random_sun_flare"].get("angle_range", [0, 1])),
                p=weather["random_sun_flare"].get("p", 0.25)
            ))
        if weather.get("random_fog", {}).get("enabled", False):
            transforms.append(A.RandomFog(
                fog_coef_lower=weather["random_fog"].get("fog_coef_lower", 0.1),
                fog_coef_upper=weather["random_fog"].get("fog_coef_upper", 0.4),
                p=weather["random_fog"].get("p", 0.2)
            ))
        if weather.get("random_rain", {}).get("enabled", False):
            transforms.append(A.RandomRain(
                drop_length=weather["random_rain"].get("drop_length", 20),
                drop_width=weather["random_rain"].get("drop_width", 1),
                p=weather["random_rain"].get("p", 0.15)
            ))

        # 4. Camera Noise
        cam_noise = cfg.get("camera_noise", {})
        if cam_noise.get("gaussian_noise", {}).get("enabled", False):
            transforms.append(A.GaussNoise(
                var_limit=tuple(cam_noise["gaussian_noise"].get("var_limit", [10.0, 50.0])),
                p=cam_noise["gaussian_noise"].get("p", 0.35)
            ))
        if cam_noise.get("motion_blur", {}).get("enabled", False):
            transforms.append(A.MotionBlur(
                blur_limit=tuple(cam_noise["motion_blur"].get("blur_limit", [3, 7])),
                p=cam_noise["motion_blur"].get("p", 0.25)
            ))
        if cam_noise.get("jpeg_compression", {}).get("enabled", False):
            transforms.append(A.ImageCompression(
                quality_range=tuple(cam_noise["jpeg_compression"].get("quality_range", [60, 95])),
                p=cam_noise["jpeg_compression"].get("p", 0.4)
            ))

        # 5. Occlusion (Cutout)
        occlusion = cfg.get("occlusion", {})
        if occlusion.get("cutout", {}).get("enabled", False):
            transforms.append(A.CoarseDropout(
                num_holes_range=tuple(occlusion["cutout"].get("num_holes_range", [2, 6])),
                hole_height_range=tuple(occlusion["cutout"].get("hole_height_range", [16, 64])),
                hole_width_range=tuple(occlusion["cutout"].get("hole_width_range", [16, 64])),
                fill_value=occlusion["cutout"].get("fill_value", 0),
                p=occlusion["cutout"].get("p", 0.4)
            ))

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.25
            )
        )

    def apply(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        class_labels: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply data augmentation pipeline to image and YOLO bounding boxes.
        Bboxes format: [x_center, y_center, width, height] normalized (0 to 1).
        """
        if not self.enabled:
            return image, bboxes or [], class_labels or []

        bboxes = bboxes or []
        class_labels = class_labels or []

        # Apply custom hardware noise/occlusion transforms
        res_image = self.rpi_noise(image)
        res_image = self.partial_occ(res_image)

        if self.pipeline is not None and bboxes:
            augmented = self.pipeline(
                image=res_image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return augmented["image"], augmented["bboxes"], augmented["class_labels"]
        elif self.pipeline is not None and not bboxes:
            # Background / hard negative image without annotations
            augmented = self.pipeline(image=res_image, bboxes=[], class_labels=[])
            return augmented["image"], [], []

        return res_image, bboxes, class_labels


# ---------------------------------------------------------------------------
# Mixing: Mosaic, Mixup, Copy-Paste
# ---------------------------------------------------------------------------


def apply_mixup(
    img1: np.ndarray,
    img2: np.ndarray,
    bboxes1: List[List[float]],
    bboxes2: List[List[float]],
    labels1: List[int],
    labels2: List[int],
    alpha: float = 0.5,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Blend two images and concatenate their bounding boxes."""
    h, w, _ = img1.shape
    img2_resized = cv2.resize(img2, (w, h))
    mixed_img = cv2.addWeighted(img1, alpha, img2_resized, 1.0 - alpha, 0)
    
    combined_bboxes = bboxes1 + bboxes2
    combined_labels = labels1 + labels2
    return mixed_img, combined_bboxes, combined_labels
