# SmartBin AI v2 — Scientific Literature & Research References

This document catalogs academic and industrial research papers informing the SmartBin AI v2 architecture, dataset taxonomy, multimodal fusion, and edge quantization strategies.

---

## 1. Waste Datasets & Benchmarks

1. **TrashNet Benchmark**
   - *Thung, G., & Yang, M. (2016)*. "Classification of Trash for Recyclability Status". Stanford University CS229 Project Report.
   - *Key Contribution*: Established baseline image classification across paper, glass, plastic, metal, and cardboard.

2. **TACO (Trash Annotations in Context)**
   - *Proença, P. F., & Simões, P. (2020)*. "TACO: Trash Annotations in Context for Litter Detection". arXiv preprint arXiv:2003.06975.
   - *Key Contribution*: Fine-grained COCO polygon segmentation for in-the-wild litter detection.

3. **WasteNet & Municipal Sorting**
   - *Bobulski, J., & Kubanek, M. (2021)*. "Deep Learning for Waste Categorization in Smart Cities". *Sensors*, 21(11), 3655.
   - *Key Contribution*: Spatial attention mechanisms for sorting packaging plastics in recycling plants.

---

## 2. Real-Time Edge Vision & Architecture

4. **Ultralytics YOLO11 Architecture**
   - *Jocher, G., & Qiu, J. (2024)*. "YOLO11: State-of-the-Art Real-Time Object Detection and Instance Segmentation". Ultralytics Enterprise AI.
   - *Key Contribution*: C3k2 modules, SPPF, dynamic head convolutions achieving high mAP at sub-50ms CPU latencies.

5. **ByteTrack Multi-Object Tracking**
   - *Zhang, Y., Sun, P., Jiang, Y., Yu, D., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022)*. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box". *ECCV 2022*.
   - *Key Contribution*: Preserves track persistence across low-confidence frames, preventing false triggers.

---

## 3. Edge Quantization & Material Classification

6. **Post-Training INT8 Quantization**
   - *Jacob, B., et al. (2018)*. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference". *CVPR 2018*.
   - *Key Contribution*: Symmetric integer quantization enabling 35 FPS inference on quad-core ARM Cortex-A76 CPUs.

7. **Multimodal Material Spectroscopy & CV Fusion**
   - *Rad, M. S., et al. (2017)*. "A Computer Vision System to Identify and Sort Recyclable Waste". *IEEE ICVS 2017*.
   - *Key Contribution*: Two-stage classification separating geometric bounding detection from surface resin classification.
