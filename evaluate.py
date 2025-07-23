import torch

# Patch torch.load to use weights_only=False by default
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO

def evaluate_model():
    weights = "saved_models/yolov8n_custom/weights/best.pt"  # or your model path
    model = YOLO(weights)
    metrics = model.val()
    print("Mean Precision:", metrics.box.mp)
    print("Mean Recall:", metrics.box.mr)
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)

if __name__ == "__main__":
    evaluate_model()
