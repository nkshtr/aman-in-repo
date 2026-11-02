"""
train.py
--------
Beginner-friendly training script for fine-tuning YOLOv8 Nano on a road damage dataset.

This script demonstrates how to train YOLOv8 locally on a laptop using transfer learning.
Transfer learning starts with weights pretrained on general objects (COCO) and adapts them
to recognize specific road distress types: potholes, longitudinal cracks, and alligator cracks.

HOW TO USE THIS SCRIPT:
1. Prepare your dataset in YOLO format with a `data.yaml` configuration file:
      train: dataset/images/train
      val: dataset/images/val
      nc: 4
      names: ['alligator crack', 'transverse crack', 'longitudinal crack', 'pothole']

2. Run training in your terminal:
      python train.py

3. Once training finishes, your best model weights will be saved at:
      runs/detect/road_damage_model/weights/best.pt
"""

import os
from ultralytics import YOLO


def train_road_damage_model():
    """
    Fine-tunes the YOLOv8 Nano model on road damage data.
    """
    print("==================================================")
    print(" YOLOv8 Nano Road Damage Training Script")
    print("==================================================\n")

    # Step 1: Select the base architecture
    # 'yolov8n.pt' is YOLOv8 Nano, the lightest model (around 6 MB).
    # It is fast, memory-efficient, and runs comfortably on standard laptop CPUs.
    base_model = "yolov8n.pt"
    print(f"[Step 1] Loading base model weights: {base_model}...")
    model = YOLO(base_model)

    # Step 2: Configure dataset path
    # Replace this with the path to your data.yaml file if training on a custom dataset
    data_yaml_path = "dataset.yaml"

    # For demonstration/instructional purposes: check if dataset exists
    if not os.path.exists(data_yaml_path):
        print(f"[Notice] Dataset configuration '{data_yaml_path}' not found.")
        print("To train on your own road photos:")
        print("  1. Label your images with Roboflow or LabelImg.")
        print("  2. Export in YOLOv8 format.")
        print("  3. Point data_yaml_path to your data.yaml file.")
        print("\nBelow is an example training command executed by this script:")
        print("""
            model.train(
                data='data.yaml',   # Path to dataset yaml
                epochs=25,          # Number of complete passes over the dataset
                imgsz=640,          # Standard input resolution
                batch=8,            # Batch size suitable for laptop CPU/RAM
                name='road_damage', # Output folder name
                device='cpu'        # Use 'cpu' or 0 for GPU
            )
        """)
        return

    # Step 3: Run the training process
    print("[Step 2] Commencing model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=30,             # 25-30 epochs is sufficient for small fine-tuning
        imgsz=640,             # 640x640 standard resolution
        batch=8,               # 8 images per batch (safe for 8GB RAM laptops)
        name="road_damage_run",# Output directory under runs/detect/
        device="cpu",          # Use 'cpu' or 0 (NVIDIA GPU if available)
        save=True,             # Save checkpoints
        patience=10            # Early stopping if no improvement for 10 epochs
    )

    print("\n[Step 3] Training Complete!")
    print("Best model weights saved to: runs/detect/road_damage_run/weights/best.pt")
    print("Copy 'best.pt' into 'models/road_damage_yolov8n.pt' to use it in the web app!")


if __name__ == "__main__":
    train_road_damage_model()
