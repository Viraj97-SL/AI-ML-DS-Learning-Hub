# Computer Vision

> A comprehensive guide to Computer Vision for ML/AI practitioners — from OpenCV fundamentals through modern deep learning architectures used in production.

---

## Overview

Computer Vision (CV) is the field of AI that enables machines to interpret and understand visual information from the world. It combines image processing, deep learning, and domain knowledge to tackle tasks ranging from simple image classification to real-time object tracking and 3D scene reconstruction.

CV is the backbone of self-driving cars, medical imaging diagnostics, facial recognition, quality control in manufacturing, augmented reality, and satellite imagery analysis. For ML engineers and AI engineers, CV skills are in extremely high demand — with dedicated roles like Computer Vision Engineer and Perception Engineer commanding top salaries.

The field has been transformed by deep learning. CNNs, Vision Transformers, and foundation models (SAM, CLIP, DINO) have replaced hand-crafted feature engineering and now achieve superhuman performance on many benchmarks.

---

## Key Concepts

### Image Fundamentals
- **Pixel representation**: Images as tensors of shape `(H, W, C)` — height, width, channels (RGB)
- **Color spaces**: RGB, BGR (OpenCV default), HSV, LAB — each useful for different tasks
- **Image preprocessing**: Normalization, resizing, padding, augmentation (flips, rotations, color jitter)
- **Convolution**: Sliding filter operation that extracts local features (edges, textures, patterns)

### CNN Architecture Components
- **Convolutional layers**: Learn spatial feature detectors via learned filters
- **Pooling**: MaxPool / AvgPool reduce spatial dimensions, add translation invariance
- **Batch normalization**: Stabilizes training, allows higher learning rates
- **Residual connections (skip connections)**: Allow gradients to flow directly, enabling very deep networks (ResNet)
- **Depthwise separable convolutions**: Efficient factored convolution used in MobileNet, EfficientNet

### Core Task Types

| Task | Input | Output | Example Models |
|------|-------|--------|----------------|
| Classification | Image | Class label | ResNet, EfficientNet, ViT |
| Object Detection | Image | Bounding boxes + labels | YOLO, Faster R-CNN, DETR |
| Semantic Segmentation | Image | Per-pixel class mask | FCN, DeepLab, SegFormer |
| Instance Segmentation | Image | Per-object masks | Mask R-CNN, SAM |
| Pose Estimation | Image | Keypoint coordinates | OpenPose, MediaPipe |
| Depth Estimation | RGB image | Depth map | MiDaS, Depth Anything |

---

## Learning Path

### Beginner
1. Python image I/O with OpenCV (`cv2.imread`, `cv2.imshow`, color space conversions)
2. Basic image transformations (resize, rotate, crop, blur, edge detection with Canny)
3. Histograms, thresholding, morphological operations
4. Building a simple CNN from scratch in PyTorch — MNIST or CIFAR-10

### Intermediate
5. Transfer learning: fine-tuning ResNet/EfficientNet on custom datasets
6. Data augmentation pipelines with Albumentations
7. Object detection with YOLOv8 (training, inference, custom datasets)
8. Semantic segmentation with U-Net
9. Building a CV inference API with FastAPI + ONNX

### Advanced
10. Vision Transformers (ViT) and hybrid architectures (Swin Transformer)
11. Self-supervised learning (SimCLR, DINO, MAE)
12. Foundation models: SAM (Segment Anything), CLIP (zero-shot classification)
13. Model optimization: quantization, pruning, TensorRT deployment
14. Multi-camera systems, stereo vision, 3D reconstruction

---

## Code Examples

### Transfer Learning with ResNet (PyTorch)

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Load pretrained ResNet50 — feature extractor trained on ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all layers except the final classifier
for param in model.parameters():
    param.requires_grad = False

# Replace the head with a new classifier for your num_classes
num_classes = 10
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

# Only train the new head
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Training loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
```

### Object Detection with YOLOv8

```python
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 nano — smallest/fastest variant
model = YOLO("yolov8n.pt")

# Inference on an image
results = model("path/to/image.jpg", conf=0.5)

# Draw results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]
        print(f"{cls_name}: {conf:.2f} @ ({x1},{y1})-({x2},{y2})")

# Fine-tune on custom dataset (YOLO format)
# model = YOLO("yolov8n.pt")
# model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=16)
```

### OpenCV Image Processing Pipeline

```python
import cv2
import numpy as np

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """Classic CV pipeline: denoise → threshold → morphology."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding handles varying illumination
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Morphological closing fills small holes in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned
```

---

## Tools & Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| **OpenCV** (`cv2`) | Classical CV, image I/O, video processing | Industry standard, C++ speed |
| **PyTorch + torchvision** | Deep learning, model training | Most used in research |
| **Ultralytics YOLOv8** | Object detection/segmentation | Easiest YOLO to use |
| **Albumentations** | Data augmentation | Fastest augmentation library |
| **Hugging Face Transformers** | Vision Transformers, CLIP, SAM | Access to latest models |
| **ONNX + ONNXRuntime** | Model export & cross-platform inference | Production deployment |
| **TensorRT** | NVIDIA GPU inference optimization | 2-10x faster inference |
| **Roboflow** | Dataset management, labeling, augmentation | Manage CV datasets |
| **LabelImg / CVAT** | Image annotation tools | Build custom datasets |
| **Supervision** | CV postprocessing utilities | Bounding box tracking, etc. |

---

## Resources

### Courses & Tutorials
- [Stanford CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) — The definitive CV course; free lecture notes and assignments
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) — Top-down approach, gets you training models on day 1
- [PyImageSearch](https://pyimagesearch.com/) — Practical OpenCV + deep learning tutorials with code
- [Roboflow Learn](https://roboflow.com/learn) — Object detection end-to-end with YOLOv8

### Books
- *Deep Learning for Vision Systems* — Mohamed Elgendy (Manning, 2020) — Best practical CV book
- *Programming Computer Vision with Python* — Jan Erik Solem — Free PDF available online
- *Computer Vision: Algorithms and Applications* — Richard Szeliski — Free PDF, comprehensive reference

### Key Papers
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al. 2015 — Skip connections
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) — Redmon & Farhadi — Real-time detection
- [An Image is Worth 16x16 Words: ViT](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. 2020 — Vision Transformers
- [Segment Anything](https://arxiv.org/abs/2304.02643) — Kirillov et al. 2023 — Foundation model for segmentation

---

## Projects & Exercises

**Project 1 — Custom Image Classifier**
Train a fine-tuned EfficientNet-B0 on a dataset of your choice (e.g., plant diseases, food categories, or fashion items). Deploy it as a FastAPI endpoint that accepts image uploads and returns class probabilities. Target: >90% accuracy.

**Project 2 — Real-Time Object Counter**
Use YOLOv8 to count specific objects (people, cars, or products) in a video stream. Add a tracking algorithm (ByteTrack or DeepSORT) to track unique objects across frames without double-counting. Output: per-frame count + annotated video.

**Project 3 — Document Scanner (Classical CV)**
Build an OpenCV pipeline that: detects a document's corners in a photo, applies a perspective warp to produce a flat top-down view, applies adaptive thresholding to create a clean scanned look. No deep learning — pure classical CV.

---

## Related Topics
- [ML Engineer Track →](../../02_ML_Engineer/README.md) — Model serving and deployment pipelines
- [Deep Learning Notebook →](../../01_Data_Scientist/advanced/01_deep_learning_intro.ipynb) — Neural network fundamentals
- [AI Engineer Track →](../../03_AI_Engineer/README.md) — Multimodal AI (vision + language)
