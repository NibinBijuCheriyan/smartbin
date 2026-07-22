import os
import sys
import zipfile
import json
import random
import argparse
import shutil
import urllib.request
import numpy as np
import cv2
from pathlib import Path

# Mapping of TACO category names to Cashcrow target classes
TACO_TO_CASHCROW = {
    "Aluminium foil": "metal",
    "Battery": "e-waste",
    "Aluminium blister pack": "metal",
    "Carded blister pack": "paper",
    "Clear plastic bottle": "plastic",
    "Glass bottle": "glass",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Metal bottle cap": "metal",
    "Broken glass": "glass",
    "Drink can": "metal",
    "Food Can": "metal",
    "Food can": "metal",
    "Corrugated carton": "paper",
    "Drink carton": "paper",
    "Egg carton": "paper",
    "Meal carton": "paper",
    "Other carton": "paper",
    "Paper cup": "paper",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Glass cup": "glass",
    "Other plastic cup": "plastic",
    "Food waste": "organic",
    "Plastic lid": "plastic",
    "Metal lid": "metal",
    "Magazine paper": "paper",
    "Wrapping paper": "paper",
    "Pizza box": "paper",
    "Paper bag": "paper",
    "Plastic bag - wrapper": "plastic",
    "Cigarette": "other",
    "Cigarette box": "other",
    "Unlabeled litter": "other",
    "Garbage bag": "plastic",
    "Single-use carrier bag": "plastic",
    "Polyethylene bag": "plastic",
    "Straw": "other",
    "Paper straw": "paper",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Plastic glooves": "plastic",
    "Plastic gloves": "plastic",
    "Paper glooves": "paper",
    "Paper gloves": "paper",
    "Metal glooves": "metal",
    "Metal gloves": "metal",
    "Plastic film": "plastic",
    "Squeezable tube": "plastic",
    "Toothbrush": "other",
    "Shoe": "other",
    "Polypropylene bag": "plastic",
    "Toilet paper": "paper",
    "Plaster": "other",
    "Glass jar": "glass",
    "Foil paper": "metal",
    "Foil wrapper": "metal",
    "Bubble wrap": "plastic",
    "Plastic bottle pack": "plastic",
    "Carded pack": "paper",
    "Cardboard box": "paper",
    "Other paper": "paper",
    "Tetra pack": "paper",
}

# Mapping of TrashNet categories to Cashcrow target classes
TRASHNET_TO_CASHCROW = {
    "glass": "glass",
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "plastic",
    "metal": "metal",
    "trash": "other",
}

CLASSES = ["plastic", "paper", "metal", "glass", "e-waste", "organic", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

def generate_mock_dataset(dataset_dir: Path, num_images: int = 140):
    """Generates a synthetic dataset for testing the training and benchmark flow."""
    print("Generating synthetic mock dataset...")
    
    # Create directory structure
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = ["train"] * int(num_images * 0.8) + ["val"] * (num_images - int(num_images * 0.8))
    random.shuffle(splits)

    # Visual features of each class to make synthetic images slightly realistic
    class_visuals = {
        "plastic": {"color": (255, 0, 0), "shape": "rect"},       # Blue rectangle
        "paper": {"color": (19, 136, 219), "shape": "rect"},     # Brown rectangle
        "metal": {"color": (192, 192, 192), "shape": "circle"},  # Grey circle
        "glass": {"color": (0, 255, 0), "shape": "ellipse"},     # Green ellipse
        "e-waste": {"color": (0, 0, 255), "shape": "poly"},       # Red polygon (battery)
        "organic": {"color": (0, 255, 255), "shape": "star"},     # Yellow shape
        "other": {"color": (128, 128, 128), "shape": "line"},    # Grey line/dust
    }

    for i, split in enumerate(splits):
        # Choose a class
        class_name = CLASSES[i % len(CLASSES)]
        class_idx = CLASS_TO_IDX[class_name]
        
        # Create a blank image with varied background
        bg_val = random.randint(30, 80)
        img = np.full((320, 320, 3), (bg_val, bg_val, bg_val), dtype=np.uint8)
        
        # Add noise
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a synthetic object
        vis = class_visuals[class_name]
        cx = random.randint(100, 220)
        cy = random.randint(100, 220)
        w = random.randint(60, 140)
        h = random.randint(60, 140)
        
        x1 = max(10, cx - w // 2)
        y1 = max(10, cy - h // 2)
        x2 = min(310, cx + w // 2)
        y2 = min(310, cy + h // 2)
        
        # Draw on image
        color = vis["color"]
        if vis["shape"] == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif vis["shape"] == "circle":
            cv2.circle(img, (cx, cy), w // 2, color, -1)
        elif vis["shape"] == "ellipse":
            cv2.ellipse(img, (cx, cy), (w // 2, h // 2), random.randint(0, 360), 0, 360, color, -1)
        else:
            # Draw a polygon/shape
            pts = np.array([
                [cx, y1], [x2, cy], [cx, y2], [x1, cy]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        # Save image
        img_name = f"{class_name}_{i:04d}.jpg"
        img_path = dataset_dir / "images" / split / img_name
        cv2.imwrite(str(img_path), img)

        # Normalize bounding box for YOLO
        nx = cx / 320.0
        ny = cy / 320.0
        nw = (x2 - x1) / 320.0
        nh = (y2 - y1) / 320.0

        # Save label
        label_path = dataset_dir / "labels" / split / f"{class_name}_{i:04d}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_idx} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Mock dataset generated successfully at {dataset_dir}")

def download_and_extract(url: str, output_path: Path):
    """Downloads a file with basic progress indication."""
    print(f"Downloading {url} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom User-Agent to avoid issues with some servers
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download completed.")

def setup_real_dataset(data_dir: Path):
    """Attempts to download and construct dataset from TrashNet and TACO."""
    dataset_dir = data_dir / "dataset"
    trashnet_zip = data_dir / "trashnet.zip"
    taco_json = data_dir / "taco_annotations.json"

    # 1. Download TrashNet
    try:
        if not trashnet_zip.exists():
            download_and_extract(
                "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
                trashnet_zip
            )
        
        trashnet_extract_dir = data_dir / "trashnet_extracted"
        if not trashnet_extract_dir.exists():
            print("Extracting TrashNet...")
            with zipfile.ZipFile(trashnet_zip, 'r') as zip_ref:
                zip_ref.extractall(trashnet_extract_dir)
        
        # 2. Download TACO Annotations metadata
        if not taco_json.exists():
            download_and_extract(
                "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
                taco_json
            )
        
        print("Parsing annotations...")
        with open(taco_json, 'r') as f:
            taco_data = json.load(f)

        # Setup target directory
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Remap TrashNet classification into object detection (bounding boxes covering full frame)
        print("Processing TrashNet images...")
        trashnet_orig_dir = trashnet_extract_dir / "dataset-resized"
        
        all_trashnet_images = []
        for class_dir_name in os.listdir(trashnet_orig_dir):
            class_path = trashnet_orig_dir / class_dir_name
            if not class_path.is_dir():
                continue
            
            mapped_class = TRASHNET_TO_CASHCROW.get(class_dir_name)
            if not mapped_class:
                continue
            
            class_idx = CLASS_TO_IDX[mapped_class]
            
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_trashnet_images.append((class_path / file_name, class_idx))

        random.shuffle(all_trashnet_images)
        # Use a subset of 300 TrashNet images to keep it fast
        trashnet_subset = all_trashnet_images[:300]
        
        for idx, (img_path, class_idx) in enumerate(trashnet_subset):
            split = "train" if idx < len(trashnet_subset) * 0.8 else "val"
            dest_img = dataset_dir / "images" / split / f"trashnet_{idx:04d}.jpg"
            shutil.copy(img_path, dest_img)
            
            # Label: full frame bounding box
            dest_label = dataset_dir / "labels" / split / f"trashnet_{idx:04d}.txt"
            with open(dest_label, "w") as lf:
                lf.write(f"{class_idx} 0.500000 0.500000 1.000000 1.000000\n")

        # Process a small subset of TACO images
        print("Processing TACO annotations...")
        taco_categories = {cat["id"]: cat["name"] for cat in taco_data["categories"]}
        
        # Group annotations by image
        img_annotations = {}
        for ann in taco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

        taco_images = taco_data["images"]
        random.shuffle(taco_images)
        
        # Download and label 50 TACO images
        taco_count = 0
        for img_entry in taco_images:
            if taco_count >= 50:
                break
            
            img_id = img_entry["id"]
            if img_id not in img_annotations:
                continue

            file_name = img_entry["file_name"]
            # Get the flickr or coco URL to download
            url = img_entry.get("flickr_640_url") or img_entry.get("flickr_url")
            if not url:
                continue

            # Remap classes for this image
            labels_lines = []
            img_w = img_entry["width"]
            img_h = img_entry["height"]
            
            for ann in img_annotations[img_id]:
                cat_name = taco_categories.get(ann["category_id"])
                mapped_class = TACO_TO_CASHCROW.get(cat_name)
                if not mapped_class:
                    continue
                class_idx = CLASS_TO_IDX[mapped_class]
                
                # Bbox is [x, y, width, height] in absolute coordinates
                bbox = ann["bbox"]
                bx = (bbox[0] + bbox[2] / 2) / img_w
                by = (bbox[1] + bbox[3] / 2) / img_h
                bw = bbox[2] / img_w
                bh = bbox[3] / img_h
                labels_lines.append(f"{class_idx} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

            if not labels_lines:
                continue

            # Download image
            split = "train" if taco_count < 40 else "val"
            dest_img = dataset_dir / "images" / split / f"taco_{taco_count:04d}.jpg"
            
            try:
                download_and_extract(url, dest_img)
                dest_label = dataset_dir / "labels" / split / f"taco_{taco_count:04d}.txt"
                with open(dest_label, "w") as lf:
                    lf.write("\n".join(labels_lines) + "\n")
                taco_count += 1
            except Exception as e:
                print(f"Skipping image due to download issue: {e}")
                
        print(f"Successfully processed {taco_count} TACO images.")
        
    except Exception as e:
        print(f"Error occurred while preparing real dataset: {e}")
        print("Falling back to generating synthetic mock dataset...")
        generate_mock_dataset(dataset_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock dataset generation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dataset_dir = data_dir / "dataset"

    # Clean previous dataset if any
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if args.mock:
        generate_mock_dataset(dataset_dir)
    else:
        setup_real_dataset(data_dir)

    # Create dataset.yaml
    dataset_yaml = data_dir / "dataset.yaml"
    yaml_content = f"""path: {dataset_dir.absolute().as_posix()}
train: images/train
val: images/val

names:
  0: plastic
  1: paper
  2: metal
  3: glass
  4: e-waste
  5: organic
  6: other
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Dataset configurations written to {dataset_yaml}")
    
    # Train YOLO Model
    print("Initializing YOLO training...")
    from ultralytics import YOLO
    
    # Load pretrained nano model
    model = YOLO("yolo11n.pt")
    
    # Run training with specified augmentations
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=320,
        device="cpu",
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=2,
    )
    
    # Save weight file
    best_weights_dst = Path("best.pt")

    search_dirs = [
        Path("runs/detect"),
        Path.home() / "runs" / "detect",
    ]

    found_weights = None
    for sdir in search_dirs:
        if sdir.exists():
            train_folders = sorted(list(sdir.glob("train*")), key=os.path.getmtime, reverse=True)
            for tf in train_folders:
                candidate = tf / "weights" / "best.pt"
                if candidate.exists():
                    found_weights = candidate
                    break
        if found_weights:
            break

    if found_weights and found_weights.exists():
        shutil.copy(found_weights, best_weights_dst)
        print(f"Fine-tuned weights copied to {best_weights_dst.absolute()} from {found_weights}")
    else:
        print("Error: Could not locate best.pt weights file.")

if __name__ == "__main__":
    main()
