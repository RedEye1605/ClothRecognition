# 👕 FashionAI - Cloth Recognition

AI-powered clothing detection with **color classification** using **YOLOv8**.

## ✨ Features

- 🎯 **8 Clothing Classes**: Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
- 🎨 **8 Color Classes**: Beige, Black, Blue, Gray, Green, Pattern, Red, White
- 📸 **Image Upload**: Drag & drop detection
- 📹 **Live Webcam**: Real-time detection
- ⚡ **Fast**: YOLOv8n optimized for speed
- 🌐 **Web UI**: Modern dark theme

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Models
Place models in `models/` folder:
- `cloth_classifier.pt` - Clothing detection model
- `color_classifier.pt` - Color classification model

### 3. Run Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Open Frontend
```bash
cd frontend
python -m http.server 5500
```
Then open http://127.0.0.1:5500

## 📁 Project Structure

```
cloth-recognition-yolo/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py             # App entry point
│   │   ├── config.py           # Configuration
│   │   ├── schemas.py          # Pydantic models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── detector.py     # Detection service
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── detection.py    # API endpoints
│   ├── run.py                  # Alternative entry point
│   └── requirements.txt
├── frontend/
│   └── index.html              # Web UI
├── models/
│   ├── cloth_classifier.pt     # Clothing detection model
│   └── color_classifier.pt     # Color classification model
├── notebooks/
│   └── cloth_detection_training.ipynb
└── deployment/
    └── huggingface/            # HuggingFace deployment
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/detect` | POST | Detect clothing with color |
| `/detect/batch` | POST | Batch detection |
| `/classes` | GET | List supported classes |

### Example Response
```json
{
  "success": true,
  "detections": [
    {
      "className": "Tshirt",
      "confidence": 0.95,
      "bbox": [100, 100, 200, 200],
      "color": "blue",
      "colorConfidence": 0.87,
      "colorHex": "#3B82F6",
      "label": "Blue Tshirt"
    }
  ]
}
```

## 🎓 Training

1. Open `notebooks/cloth_detection_training.ipynb` in Google Colab
2. Run all cells (uses T4 GPU)
3. Download `cloth_classifier.pt` and `color_classifier.pt`
4. Place in `models/` folder

## 🌐 HuggingFace Deployment

Upload files from `deployment/huggingface/` to HuggingFace Spaces:
1. Create new Space (Gradio SDK)
2. Upload all files including both models
3. Your app will be live!

## 🛠️ Tech Stack

- **ML**: YOLOv8, PyTorch
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gradio (HuggingFace)

## 📄 License

MIT
