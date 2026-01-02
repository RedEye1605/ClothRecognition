# 👕 FashionAI - Cloth Recognition System

AI-powered clothing detection using **YOLOv8**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 🎯 **8 Clothing Classes**: Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
- 📸 **Image Upload**: Drag & drop or click to upload
- 📹 **Live Webcam**: Real-time detection from camera
- ⚡ **Fast Inference**: YOLOv8n optimized for speed
- 🎨 **Modern UI**: Dark theme with premium design

## 🏷️ Detectable Classes

| Class | Color |
|-------|-------|
| Tshirt | 🔴 Red |
| Dress | 🔵 Blue |
| Jacket | 🟢 Green |
| Pants | 🩵 Teal |
| Shirt | 🟡 Yellow |
| Short | 🟩 Mint |
| Skirt | 🟣 Purple |
| Sweater | 🟠 Gold |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Model
Place your trained `best.pt` in the `models/` folder.

### 3. Run Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open Frontend
Open `frontend/index.html` in your browser.

## 📁 Project Structure

```
cloth-recognition-yolo/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Web UI
├── models/
│   └── best.pt              # Trained model (add after training)
├── notebooks/
│   └── cloth_detection_training.ipynb  # Colab training
├── deployment/
│   └── huggingface/         # HuggingFace Spaces files
└── render.yaml              # Render deployment config
```

## 🎓 Training

1. Open `notebooks/cloth_detection_training.ipynb` in Google Colab
2. Run all cells (uses T4 GPU)
3. Download `best.pt` after training
4. Place in `models/` folder

## 🌐 Deployment

### Render (Backend)
```bash
# Push to GitHub, then:
# 1. Connect repo to render.com
# 2. Use render.yaml for auto-config
```

### Vercel (Frontend)
```bash
# 1. Update API_URL in index.html
# 2. Deploy frontend/ to Vercel
```

### Hugging Face (All-in-One)
```bash
# Upload to HuggingFace Spaces:
# - deployment/huggingface/app.py
# - deployment/huggingface/requirements.txt
# - best.pt
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| GET | `/classes` | List classes |
| POST | `/detect` | Detect clothing |
| POST | `/detect/batch` | Batch detection |

## 🛠️ Tech Stack

- **ML**: YOLOv8 (Ultralytics), PyTorch
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render, Vercel, HuggingFace

## 📄 License

MIT License - Free to use and modify.
