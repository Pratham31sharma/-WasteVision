import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

def convert_annotations(taco_json, taco_images_dir, output_images_dir, output_labels_dir, classes_txt_path=None):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(taco_json, 'r') as f:
        data = json.load(f)

    # Build category_id to YOLO class index mapping and class names list
    categories = data["categories"]
    category_id_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    # Optionally write classes.txt
    if classes_txt_path:
        with open(classes_txt_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

    images = {img["id"]: img["file_name"] for img in data["images"]}

    for ann in tqdm(data["annotations"]):
        img_id = ann["image_id"]
        file_name = images[img_id]
        bbox = ann["bbox"]
        # Use correct class index
        class_id = category_id_to_index[ann["category_id"]]

        # Convert bbox to YOLO format
        img_path = os.path.join(taco_images_dir, file_name)
        img = Image.open(img_path)
        w, h = img.size
        x_center = (bbox[0] + bbox[2] / 2) / w
        y_center = (bbox[1] + bbox[3] / 2) / h
        width = bbox[2] / w
        height = bbox[3] / h

        # Copy image
        dst_path = os.path.join(output_images_dir, file_name)
        if os.path.abspath(img_path) != os.path.abspath(dst_path):
            shutil.copy(img_path, dst_path)

        # Write label
        label_path = os.path.join(output_labels_dir, file_name.replace('.jpg', '.txt').replace('.JPG', '.txt'))
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    convert_annotations(
        taco_json="datasets/data/taco/annotations.json",
        taco_images_dir="datasets/data/taco/train/images",
        output_images_dir="datasets/data/taco/train/images",
        output_labels_dir="datasets/data/taco/train/labels",
        classes_txt_path="datasets/data/taco/classes.txt"
    )
