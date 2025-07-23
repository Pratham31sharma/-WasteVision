import torch
import cv2
from ultralytics import YOLO
import os
import glob

# Load class names from classes.txt
CLASSES_PATH = "datasets/data/taco/classes.txt"
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = [str(i) for i in range(60)]  # fallback

# Patch torch.load to use weights_only=False by default
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Load the trained model
try:
    model = YOLO("saved_models/yolov8n_custom/weights/best.pt")
    print("Loaded trained model")
except Exception as e:
    print(f"Could not load trained model: {e}")
    print("Using pre-trained model instead")
    model = YOLO("yolov8n.pt")

def detect_image(img_path):
    results = model(img_path, conf=0.1)  # Lowered confidence threshold
    boxes = results[0].boxes
    annotated_img = results[0].plot()

    # Print detected classes and confidence
    if boxes is not None and boxes.cls.numel() > 0:
        print("Detections:")
        for c, conf in zip(boxes.cls, boxes.conf):
            class_idx = int(c.item())
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            print(f"  {class_name}: {conf.item():.2f}")
    else:
        print("No detections.")

    # Overlay class names and confidence on the image
    if boxes is not None and boxes.cls.numel() > 0:
        for box, c, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_idx = int(c.item())
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            label = f"{class_name} {conf.item():.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Resize the image to make it smaller for display
    height, width = annotated_img.shape[:2]
    scale_percent = 50
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_img = cv2.resize(annotated_img, (new_width, new_height))

    cv2.imshow("Plastic Waste Detection", resized_img)
    print("Press 'q' to quit, 'n' for next image, or any other key to continue...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key == ord('q')

def detect_all_images(base_folder):
    """Process all images from all batch folders"""
    # Get all image files from all batch folders
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    # Get all batch folders
    batch_folders = glob.glob(os.path.join(base_folder, "batch_*"))
    print(f"Found {len(batch_folders)} batch folders: {[os.path.basename(f) for f in batch_folders]}")
    
    for batch_folder in batch_folders:
        print(f"Processing folder: {os.path.basename(batch_folder)}")
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(batch_folder, ext)))
    
    print(f"Found {len(image_files)} total images to process")
    
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        try:
            # Check if user wants to quit
            if detect_image(img_path):
                print("User requested to quit. Exiting...")
                break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.1)  # Lowered confidence threshold
        annotated_frame = results[0].plot()
        
        # Resize the video frame to make it smaller for display
        height, width = annotated_frame.shape[:2]
        # Scale down to 50% of original size
        scale_percent = 50
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
        
        cv2.imshow("Plastic Waste Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Process all images from all batch folders
    base_folder = "datasets/data/taco/images"
    detect_all_images(base_folder)
    detect_video("video.mp4")
