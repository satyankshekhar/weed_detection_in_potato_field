# Weed Detection and Segmentation in Potato Field Environments

This project implements and evaluates deep learning architectures for the automated identification of weeds in potato crops. It focuses on solving the challenges of plant detection and pixel-level segmentation in complex, highly occluded field environments.

## 📌 Project Overview
Precision agriculture requires robust computer vision to differentiate between crops and weeds, especially during the post-emergence stage where foliage overlaps. This project utilizes two complementary approaches:
- **Object Detection (YOLOv8):** Fast and efficient bounding-box detection.
- **Instance Segmentation (Mask R-CNN):** Precise, pixel-level masking for detailed spatial analysis.

## 📂 Project Structure
- `YOLOv8WeedDetection (1).ipynb`: Pipeline for training the YOLOv8s (small) model, including data integration via Roboflow.
- `maskedRcnnWeedDetection.ipynb`: Implementation of the Mask R-CNN framework using a ResNet-50 backbone for instance segmentation.

## 🛠️ Implementation Details

### YOLOv8 Setup
- **Model:** YOLOv8s (Small variant for improved accuracy over Nano).
- **Hardware:** Trained on a Tesla T4 GPU.
- **Parameters:** 50 epochs, 640px image size, and AdamW optimizer.
- **Data:** 1,014 training images with specialized augmentations (Blur, MedianBlur, CLAHE).

### Mask R-CNN Setup
- **Backbone:** ResNet-50 with Feature Pyramid Network (FPN).
- **Environment:** Google Colab with COCO-formatted annotations.

## 📊 Results and Performance

The following results were achieved during the final validation phases of each respective model:

### 1. YOLOv8 Detection Results (Best Weights)
The model was validated on 183 images containing 483 plant instances.

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | **483** | **0.646** | **0.662** | **0.651** | **0.523** |
| Weed | 320 | 0.456 | 0.416 | 0.373 | 0.241 |
| Crop (Potato) | 163 | 0.835 | 0.908 | 0.929 | 0.804 |

- **Inference Speed:** 4.2ms per image.
- **Training Time:** ~0.622 hours for 50 epochs.

### 2. Mask R-CNN Segmentation Results
The segmentation model provided high-resolution masks for individual plants.

| Metric | Value |
| :--- | :--- |
| **Average Precision (AP) @[IoU=0.50:0.95]** | **0.436** |
| **Average Precision (AP) @[IoU=0.50]** | **0.575** |
| **Average Precision (AP) @[IoU=0.75]** | **0.460** |
| **Average Recall (AR) @[IoU=0.50:0.95]** | **0.506** |

## 🚀 Usage
1. **Requirements:** Install dependencies:
   ```bash
   pip install ultralytics roboflow torch torchvision opencv-python
