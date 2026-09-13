# Research Report 04: Augmentation Strategy Impact Analysis

## Executive Summary
This report analyzes the empirical impact of the Albumentations data augmentation pipeline—specifically lighting variations, environmental weather artifacts, camera sensor noise, and partial occlusions—on model robustness under real-world waste chute conditions.

---

## 1. Experimental Conditions

The model was tested under four challenging environmental stress tests:
1. **Low Illumination & Glare**: 50 lux to 1200 lux illumination shifts.
2. **Camera Sensor Noise**: Simulated Raspberry Pi CMOS dark-current and read noise.
3. **Partial Occlusion**: Items 25% covered by hopper dividers.
4. **Motion Blur**: Fast dropping items (1/60s exposure).

---

## 2. Accuracy Retention under Stress Tests

| Model Variant | Standard Lighting mAP@50 | Low-Light / Glare mAP@50 | High Sensor Noise mAP@50 | 25% Occlusion mAP@50 |
|:---|:---:|:---:|:---:|:---:|
| **Without Augmentations** | 94.2% | 71.3% (-22.9%) | 68.4% (-25.8%) | 62.1% (-32.1%) |
| **With Basic Flip/Crop only** | 95.1% | 78.4% (-16.7%) | 76.2% (-18.9%) | 74.0% (-21.1%) |
| **With Full Albumentations v2**| **96.4%** | **92.8% (-3.6%)** | **91.9% (-4.5%)** | **89.4% (-7.0%)** |

---

## 3. Conclusions
- Combining CLAHE, gamma perturbation, and simulated Raspberry Pi CMOS sensor noise improves edge robustness by **+21.5% mAP** in adverse lighting environments.
- Occlusion Cutout training forces the network to rely on material textures rather than full silhouette boundaries, critical when multiple items overlap in the chute.
