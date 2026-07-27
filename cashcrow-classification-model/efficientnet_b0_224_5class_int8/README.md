# EfficientNet-B0 Waste Classification Model

A deep learning classifier designed to identify and sort recyclable and organic waste materials into five target categories. The model uses the **EfficientNet-B0** backbone with a modified head, trained on input images resized to **224x224** pixels. Both standard high-precision (FP32) and quantized (INT8) TFLite models are provided for edge and embedded deployment.

---

## 📁 Repository Directory Structure

The repository has been systematically structured to isolate configuration metadata, binary weights, training logs, and validation metrics.

```
efficientnet_b0_224_5class_int8/
├── README.md                          # Main repository documentation
├── classes.json                      # Class names mapping and indices
├── class_weights.json                # Pre-calculated class weights for training
├── training_config.json              # Hyperparameters and dataset configs
├── training_state.json               # Final training state and epoch summary
├── models/                           # Model checkpoints and binaries
│   ├── best.keras                    # Best checkpoint in Keras format
│   ├── last.keras                    # Last checkpoint in Keras format
│   ├── waste_classifier_fp32.tflite  # FP32 TFLite deployment model
│   ├── waste_classifier_int8.tflite  # INT8 quantized TFLite deployment model
│   └── saved_model/                  # TensorFlow SavedModel representation
│       ├── fingerprint.pb
│       ├── saved_model.pb
│       ├── assets/
│       └── variables/
├── inference/                        # Python inference client for testing
│   ├── predict.py                    # Inference script using TFLite
│   └── requirements.txt              # Dependencies for running on a laptop
├── logs/                             # Training process logs
│   ├── epoch_logs.jsonl              # Chronological JSON-line log
│   └── training_log.csv              # CSV format logs (epoch, loss, acc)
└── metrics/                          # Quantitative model evaluation metrics
    ├── classwise_metrics.csv         # Overall model classwise metrics
    ├── detailed_metrics.csv          # Core evaluation metrics (F1, Precision, Recall)
    ├── tflite_fp32_validation_metrics.json
    ├── tflite_fp32_validation_metrics_classwise.csv
    ├── tflite_int8_validation_metrics.json
    └── tflite_int8_validation_metrics_classwise.csv
```

---

## 🏷️ Class Definitions

The model categorizes images into five classes (mapped in [classes.json](./classes.json)):

| Index | Class Name | Description |
|---|---|---|
| **0** | `plastic` | Recyclable plastics (bottles, containers, wraps) |
| **1** | `paper` | Recyclable papers (cardboards, books, sheets) |
| **2** | `metal` | Recyclable metals (soda cans, tin containers, scraps) |
| **3** | `organic_waste` | Compostable organic items (food waste, leaves, fruit peels) |
| **4** | `none` | Background, empty conveyer, or non-recyclable materials |

---

## ⚙️ Training Settings & Configuration

The training configuration details can be found in [training_config.json](./training_config.json):

* **Model Backbone:** `EfficientNet-B0` (Pre-trained on ImageNet)
* **Input Resolution:** $224 \times 224 \times 3$
* **Batch Size:** 32
* **Total Epochs:** 40
  * **Head Fine-Tuning Phase:** Epochs 1 - 5 (Learning Rate: `0.001`, training classification head only)
  * **Full Fine-Tuning Phase:** Epochs 6 - 40 (Learning Rate: `0.0001`, last 40 layers of the backbone unfrozen)
* **Regularization & Augmentation:**
  * **Label Smoothing:** 0.05
  * **Gaussian Blur Augmentation:** Probability of 0.25 (kernel size: 5, sigma: 1.0)
* **Post-Training Quantization Calibration:** 500 representative samples were utilized for calibrating activations during INT8 quantization.

---

## 📈 Training Summary

Training was conducted over 40 epochs. The model achieved a high degree of convergence:

* **Training Accuracy:** **99.63%** (Loss: `0.2361`)
* **Validation Accuracy:** **96.45%** (Loss: `0.3095`)
* **Best Validation Accuracy (Checkpoint):** **96.62%**

---

## 📊 Quantization Performance Comparison

To facilitate deployment, the model was converted into both standard **FP32 TFLite** and integer-quantized **INT8 TFLite** formats. Below is a comparative validation analysis:

### Global Performance Metrics

| Metric | FP32 TFLite Model | INT8 TFLite Model | Performance Delta |
|---|:---:|:---:|:---:|
| **Accuracy** | **96.59%** | **87.29%** | -9.30% |
| **Macro Precision** | **96.66%** | **87.66%** | -9.00% |
| **Macro Recall** | **96.34%** | **84.90%** | -11.44% |
| **Macro F1-Score** | **96.50%** | **86.11%** | -10.39% |
| **Mean Average Precision (mAP)** | **99.15%** | **91.08%** | -8.07% |
| **Model Disk Size** | **15.36 MB** (`16,107,464` B) | **4.73 MB** (`4,961,648` B) | **-69.2% (Size Reduction)** |

### Per-Class F1-Score & Recall Comparison

| Class Name | Support | FP32 Recall | FP32 F1 | INT8 Recall | INT8 F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **`plastic`** | 1,220 | 95.74% | 95.86% | 78.93% | 83.63% |
| **`paper`** | 1,429 | 96.64% | 96.47% | 93.14% | 87.80% |
| **`metal`** | 394 | 97.72% | 97.59% | 89.59% | 91.57% |
| **`organic_waste`** | 340 | 99.71% | 99.85% | 97.06% | 98.36% |
| **`none`** | 111 | 91.89% | 92.73% | 65.77% | 69.19% |

> [!NOTE]
> **Key Observations on Quantization Trade-offs:**
> 1. **Size vs. Performance:** INT8 quantization achieves a **~3.2x size reduction** (saving 10.63 MB of disk/memory space), making it highly optimized for edge processing.
> 2. **Impact on Minority Classes:** The quantization drop is most pronounced on the minority classes, particularly `none` (recall fell from 91.89% to 65.77%) and `plastic` (recall fell from 95.74% to 78.93%).
> 3. **Resilience of Strong Classes:** `organic_waste` remained exceptionally resilient to quantization, retaining a Recall of **97.06%** and F1 of **98.36%** in INT8 format.

---

## 🎯 Model Accuracy Summary

The table below summarizes the accuracies of all model formats generated from this training run:

| Model Format | Target File / Directory | Validation Accuracy | Deployment Target |
|---|---|:---:|---|
| **Best Keras Checkpoint** | `models/best.keras` | **96.62%** | Backup / Fine-tuning checkpoint |
| **Last Keras Checkpoint** | `models/last.keras` | **96.45%** | Final epoch state |
| **TensorFlow SavedModel** | `models/saved_model/` | **96.45%** | Server-side / Cloud inference |
| **FP32 TFLite Model** | `models/waste_classifier_fp32.tflite` | **96.59%** | Edge devices with high-precision needs |
| **INT8 TFLite Model** | `models/waste_classifier_int8.tflite` | **87.29%** | Ultra low-power edge accelerators |

---

## ⚡ Deployment Recommendations

* **Cloud/Server Environment:** Use the original SavedModel or FP32 TFLite representation to maintain high classification accuracy (~96.6%).
* **Edge Device (e.g. Raspberry Pi, Coral Edge TPU, Mobile):** Use `waste_classifier_int8.tflite` to reduce memory footprints, speed up inferences via integer hardware acceleration, and reduce power consumption, while keeping in mind the reduced recall on the `none` and `plastic` categories.

---

## 🚀 SmartBin Deployment Status

> [!IMPORTANT]
> The **FP32 TFLite Model** (`models/waste_classifier_fp32.tflite`) is currently deployed and active in the **SmartBin** system for now. This choice is due to the significant accuracy drop observed in the **INT8 quantized model** (which drops to **87.29%** compared to **96.59%** for FP32), ensuring high sorting precision is maintained during operations.

---

## 💻 Running Inference on a Laptop

You can test classification on individual images using the Python client located in the [inference/](./inference/) folder.

### Setup Instructions

1. **Copy Folder:** Copy the [inference/](./inference/) folder, the FP32 model [waste_classifier_fp32.tflite](./models/waste_classifier_fp32.tflite), and [classes.json](./classes.json) to your laptop. Ensure the relative directory structure is maintained (or specify explicit paths using flags).
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Run the script by passing the path to an image:
```bash
python predict.py path/to/your/image.jpg
```

Optional arguments:
* `--model <path>`: Specify a custom path to the TFLite model.
* `--classes <path>`: Specify a custom path to `classes.json`.

