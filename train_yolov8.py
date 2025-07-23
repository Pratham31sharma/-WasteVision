import torch
from ultralytics import YOLO
import os
import sys

# Disable W&B to avoid ValueError
os.environ["WANDB_MODE"] = "disabled"

# Patch torch.load to use weights_only=False by default
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch globally
torch.load = patched_torch_load

def train_yolo():
    # Use pre-trained model
    model = YOLO("yolov8n.pt")
    
    # Train the model and let Ultralytics handle saving
    results = model.train(
        data="datasets/data/taco/taco_data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        save_period=-1,  # Only save the last/best model
        save=True,       # Let YOLO handle saving
        project="saved_models",  # Save under this directory
        name="yolov8n_custom"    # Subdirectory for this run
    )
    print("Training complete. Model saved by Ultralytics.")
    print(f"Best model path: {os.path.join('saved_models', 'yolov8n_custom', 'weights', 'best.pt')}")


if __name__ == "__main__":
    train_yolo()