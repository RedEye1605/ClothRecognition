---
title: FashionAI - Cloth Recognition
emoji: 👕
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# 👕 FashionAI - Cloth Recognition

AI-powered clothing detection with **color classification** using YOLOv8.

## Features

- 🎯 **8 Clothing Classes**: T-Shirt, Dress, Jacket, Pants, Shirt, Shorts, Skirt, Sweater
- 🎨 **8 Color Classes**: Beige, Black, Blue, Gray, Green, Pattern, Red, White  
- ⚡ **Real-time Detection**: Fast inference with YOLOv8
- 📹 **Webcam Support**: Live detection mode

## How It Works

**Two-Stage Detection Pipeline:**
1. **Stage 1**: Detect clothing items with bounding boxes
2. **Stage 2**: Classify color for each detected item

## Models

This Space uses two YOLOv8 models:
- `best.pt` - Trained on clothing dataset for type detection
- `color_classifier.pt` - Trained on clothing colors for color classification

## Usage

1. Upload an image or use webcam
2. Adjust confidence threshold
3. View detected items with colors (e.g., "Blue T-Shirt 95%")

## Tech Stack

- YOLOv8 (Ultralytics)
- Gradio
- PyTorch
