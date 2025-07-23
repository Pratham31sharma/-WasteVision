# 🧴 WasteVision

**AI-Driven Multi-Class Waste Detection & Classification**

_Protecting the environment with intelligent monitoring_

---

## 🚀 Overview

WasteVision is an advanced system that leverages YOLOv8 and the TACO dataset to detect and classify various types of waste in images. The project provides end-to-end tools for data preprocessing, model training, evaluation, and a user-friendly Gradio web app for real-time inference.

---

## ✨ Key Features

- 🧠 **Deep Learning Detection**: Accurate variou waste detection using YOLOv8
- 🗂️ **Dataset Preprocessing**: Utilities for TACO dataset preparation and annotation
- 📈 **Training & Evaluation**: Scripts for model training and performance analysis
- 🖼️ **Results Visualization**: Confusion matrices, F1 curves, and more
- 🌐 **Web Interface**: Gradio app for easy, interactive inference
- 💾 **Model Saving**: Automatic saving of best and last model weights

---

## 🛠️ Technology Stack

- **Deep Learning**: YOLOv8 (Ultralytics), PyTorch
- **Web App**: Gradio
- **Data**: TACO Dataset
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Python, NumPy, OpenCV

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wastevision.git
   cd wastevision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the TACO dataset**  
   - Download from [TACO Dataset](https://tacodataset.org/)
   - Place images and annotations in the `datasets/data/taco/` directory as described in the project structure.

---

## 🎯 Usage

### 1. Data Preprocessing

Prepare the TACO dataset for training:
```bash
python data/src/preprocess_taco.py
```

### 2. Model Training

Train the YOLOv8 model:
```bash
python src/train_yolov8.py
```
- Models and results are saved in `saved_models/yolov8n_custom/`.

### 3. Inference

#### a. Script-based Inference
```bash
python src/inference.py --weights saved_models/yolov8n_custom/weights/best.pt --source path/to/image_or_folder
```

#### b. Gradio Web App
```bash
python app/gradio_app.py
```
- Open the provided local URL in your browser.

---

## 📊 Results & Examples

- Training and validation results (confusion matrix, F1 curve, etc.) are saved in `runs/detect/` and `saved_models/yolov8n_custom/`.
- Example output images and metrics are available in these folders.

---

## 🧩 Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

- ultralytics
- torch
- gradio
- opencv-python
- numpy
- matplotlib
- (others as listed in requirements.txt)

Install with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- TACO dataset and open-source community
- Ultralytics YOLOv8
- Contributors and environmental researchers

---

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**🧴 WasteVision** – _Protecting the environment with intelligent monitoring_ 
