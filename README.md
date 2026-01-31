# Automatic Number-Plate Recognition (ANPR) System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLO](https://img.shields.io/badge/YOLO-v12-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Extraction-yellow)

An end-to-end Computer Vision system engineered to detect license plates in real-time video streams and extract alphanumeric text with high accuracy. Built using **YOLOv12** for object detection and **EasyOCR** for character recognition, optimized for GPU performance.

## 🚀 Features
* **Real-Time Detection:** Utilizes the state-of-the-art **YOLOv12** architecture for millisecond-latency plate detection.
* **Robust OCR Pipeline:** Integrates **EasyOCR** with custom image preprocessing (upscaling, binarization, and noise reduction) to handle low-light and blurry conditions.
* **GPU Acceleration:** Fully optimized for NVIDIA CUDA (RTX 3050 support enabled).
* **Dynamic Visuals:** Draws bounding boxes and text overlays on live video feeds.
* **Video Processing:** Capable of processing raw video files and saving analyzed outputs automatically.

## 🛠️ Tech Stack
* **Core Logic:** Python 3.11
* **Detection Model:** YOLOv12 (via Ultralytics)
* **Optical Character Recognition:** EasyOCR (PyTorch backend)
* **Image Processing:** OpenCV (cv2)
* **Hardware Acceleration:** CUDA 12.1 / PyTorch GPU

## 📂 Project Structure
```text
ANPR_System/
├── dataset/
├── runs/
├── video/
│   ├── test/
│   └── Output/
├── main.py
├── train_anpr.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ANPR-System.git
cd ANPR-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> For GPU support, ensure CUDA toolkit is installed and use GPU-enabled PyTorch.

### 3. Download / Train Model
Train using `train_anpr.ipynb` or place `best.pt` in the `weights/` directory.

## 🏃 Usage
```bash
python main.py
```

Processed videos are saved in `video/Output/`.

## 🧠 How It Works
1. Frame capture from video stream
2. YOLOv12 detects license plates
3. Image preprocessing (crop, upscale, threshold)
4. EasyOCR extracts text
5. Regex-based filtering and visualization

## 📊 Dataset
Kaggle License Plate Detection Dataset (10,000+ images)

## 📜 License
Distributed under the **AGPL-3.0 License**.
