import os
import torch
from torch.serialization import safe_globals
from ultralytics.nn.tasks import DetectionModel

# Path to your trained model
f = "saved_models/best.pt"  # <-- change this to the correct path if needed

# Safe model load with Unpickling workaround
with safe_globals([DetectionModel]):
    x = torch.load(f, map_location=torch.device("cpu"))

# Function to create required folder structure
def create_folder_structure(base_dir):
    os.makedirs(f"{base_dir}/train/images", exist_ok=True)
    os.makedirs(f"{base_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{base_dir}/val/images", exist_ok=True)
    os.makedirs(f"{base_dir}/val/labels", exist_ok=True)

# Example usage
create_folder_structure("datasets/data/taco")
