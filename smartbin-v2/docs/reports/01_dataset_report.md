# Research Report 01: Dataset Engineering & Composition Analysis

## Executive Summary
This report documents the synthesis, cleaning, and distribution of the 60,000-image SmartBin AI v2 benchmark. Combining seven public datasets with a customized 28-class Indian municipal waste collection, the pipeline achieves balanced representation across Recyclables, Organic Compost, and Landfill streams.

---

## 1. Source Composition & Aggregation Statistics

```
+-------------------------------------------------------------+
| Dataset Source               Images Ingested    Contribution |
+-------------------------------------------------------------+
| TrashNet (Stanford)                 2,527             4.2%  |
| TACO (In-the-wild litter)           1,500             2.5%  |
| WasteNet (Industrial Bboxes)        5,000             8.3%  |
| DeepWaste                           3,200             5.3%  |
| Open Images V7 (Filtered Waste)    18,000            30.0%  |
| Kaggle Garbage Classification       2,467             4.1%  |
| SmartBin Indian Municipal Custom   28,000            46.6%  |
+-------------------------------------------------------------+
| TOTAL INGESTED                     60,694           100.0%  |
+-------------------------------------------------------------+
```

---

## 2. Perceptual Deduplication Analysis

Using 64-bit difference hashing (`dHash`) with a Hamming threshold of $\le 4$:
- **Candidate Duplicates Identified**: 3,842 near-duplicate frames.
- **Root Cause**: Burst mode field captures of identical items under minor lighting shifts.
- **Remediation**: Purged near-duplicates to eliminate data leakage between train and test partitions.
- **Post-Deduplication Unique Corpus**: 56,852 verified images.

---

## 3. Stratified Partitioning & Class Imbalance

- **Splits**: 70% Train (39,796 images), 15% Val (8,528 images), 15% Test (8,528 images).
- **Gini Inequality Index**: Reduced from $0.482$ (raw) to $0.184$ via stratified sampling and controlled synthetic minority oversampling.
- **Bounding Box Aspect Ratios**: Normalized median aspect ratio of $1.35$, preventing bias toward purely elongated or square items.
