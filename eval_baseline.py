"""
Evaluate Baseline YOLO model on the new 5-class Real Validation Dataset.
"""
from pathlib import Path
from ultralytics import YOLO
import json

def eval_baseline():
    print("\n=======================================================")
    print("       EVALUATING BASELINE (OLD) MODEL ON REAL DATA    ")
    print("=======================================================")
    model = YOLO("best.pt")
    results = model.val(data="data/dataset.yaml", split="val", imgsz=640, verbose=True)

    print("\nBaseline Model Validation Summary:")
    print(f"mAP@0.5      : {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {results.box.map:.4f}")
    print(f"Precision    : {results.box.mp:.4f}")
    print(f"Recall       : {results.box.mr:.4f}")
    
    # Per class breakdown
    names = results.names
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<12}")
    print("-" * 65)
    for idx, name in names.items():
        p = results.box.p[idx] if idx < len(results.box.p) else 0.0
        r = results.box.r[idx] if idx < len(results.box.r) else 0.0
        m50 = results.box.map50  # class specific maps
        m = results.box.maps[idx] if idx < len(results.box.maps) else 0.0
        print(f"{name:<15} | {p:>9.4f}  | {r:>9.4f}  | {m:>9.4f}   | {m:>11.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    eval_baseline()
