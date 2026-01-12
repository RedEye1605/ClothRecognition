---
title: Cloth Recognition
emoji: 👕
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# FashionAI - AI-Powered Clothing Detection

<div align="center">
  <img src="frontend/Logo.png" alt="FashionAI Logo" width="150">
  
  **Intelligent Clothing Detection & Color Classification using YOLOv8**
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://docs.ultralytics.com)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  
  [Live Demo](https://github.com/RedEye1605/ClothRecognition) • [Documentation](#-documentation) • [API Reference](#-api-endpoints)
</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **8 Clothing Classes** | T-Shirt, Dress, Jacket, Pants, Shirt, Shorts, Skirt, Sweater |
| 🎨 **8 Color Classes** | Beige, Black, Blue, Gray, Green, Pattern, Red, White |
| 📸 **Image Upload** | Drag & drop or click to upload images |
| 📹 **Live Webcam** | Real-time detection with adjustable confidence |
| ⚡ **Fast Detection** | YOLOv8n optimized for speed |
| 🌐 **Modern Web UI** | Premium dark theme with glass-morphism design |
| 📱 **Responsive** | Works on desktop, tablet, and mobile |
| 🔌 **REST API** | Full-featured FastAPI backend |

---

## 🖼️ Screenshots

### Landing Page
Modern landing page with feature showcase and quick navigation.

### Detection App
Real-time clothing detection with bounding boxes, labels, and confidence scores.

### Documentation
Comprehensive documentation with API reference and deployment guides.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Clone Repository
```bash
git clone https://github.com/RedEye1605/ClothRecognition.git
cd ClothRecognition
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Add Models
Place trained models in `models/` folder:
- `cloth_classifier.pt` - Clothing detection model (YOLOv8)
- `color_classifier.pt` - Color classification model (YOLOv8-cls)

> 💡 **Tip**: Train your own models using the Jupyter notebook in `notebooks/`

### 5. Start Backend Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

You should see:
```
🔄 Loading detection model from: .../models/cloth_classifier.pt
✅ Detection model loaded: cloth_classifier.pt
🔄 Loading color model from: .../models/color_classifier.pt
✅ Color model loaded: color_classifier.pt
🚀 FashionAI API v3.0.0 starting...
```

### 6. Start Frontend Server
Open a new terminal:
```bash
cd frontend
python -m http.server 5500
```

### 7. Open in Browser
Navigate to: **http://127.0.0.1:5500**

---

## 📁 Project Structure

```
ClothRecognition/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── schemas.py           # Pydantic models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── detector.py      # YOLO detection service
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── detection.py     # API endpoints
│   ├── run.py                   # Alternative entry point
│   └── requirements.txt
│
├── frontend/
│   ├── index.html               # Landing page
│   ├── app.html                 # Detection application
│   ├── docs.html                # Documentation page
│   ├── styles.css               # Shared styles
│   └── Logo.png                 # Project logo
│
├── models/
│   ├── cloth_classifier.pt      # Clothing detection model
│   └── color_classifier.pt      # Color classification model
│
├── notebooks/
│   └── cloth_detection_training.ipynb  # Training notebook
│
├── deployment/
│   ├── Dockerfile               # Docker configuration
│   ├── docker-compose.yml       # Docker Compose
│   └── huggingface/             # HuggingFace Spaces deployment
│
├── requirements.txt             # Root dependencies
├── QUICKSTART.md               # Quick start guide
└── README.md                   # This file
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/detect` | POST | Detect clothing with color classification |
| `/detect/batch` | POST | Batch detection for multiple images |
| `/classes` | GET | List all supported classes |

### Detection Request
```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence=0.25"
```

### Response Format
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
  ],
  "processingTime": 0.045,
  "imageSize": [640, 480]
}
```

---

## 🎓 Training Custom Models

1. Open `notebooks/cloth_detection_training.ipynb` in Google Colab
2. Upload your dataset or use the default dataset
3. Run all cells (recommended: T4 GPU runtime)
4. Download trained models:
   - `cloth_classifier.pt`
   - `color_classifier.pt`
5. Place models in the `models/` folder

---

## 🐳 Docker Deployment

### Using Docker Compose
```bash
cd deployment
docker-compose up -d
```

### Using Docker
```bash
docker build -t fashionai .
docker run -p 8000:8000 fashionai
```

---

## 🌐 HuggingFace Spaces Deployment

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Select **Gradio** SDK
3. Upload all files from `deployment/huggingface/`:
   - `app.py`
   - `requirements.txt`
   - Model files (`.pt`)
4. Your app will be live!

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Machine Learning** | YOLOv8, PyTorch, Ultralytics |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Design** | Glass-morphism, Dark Theme |
| **Deployment** | Docker, HuggingFace Spaces |

---

## 📖 Documentation

The project includes comprehensive documentation:

- **Landing Page** (`frontend/index.html`) - Overview and features
- **Application** (`frontend/app.html`) - Detection interface
- **Documentation** (`frontend/docs.html`) - Full API docs and guides

Access the documentation by running the frontend server and navigating to the Docs page.

---

## 🔧 Configuration

Environment variables can be set in `.env` or passed directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../models` | Path to model files |
| `API_PORT` | `8000` | Backend API port |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default detection confidence |

---

## 🐛 Troubleshooting

### Models not loading?
- Verify `models/` folder contains both `.pt` files
- Check file names match expected names

### API not connecting?
- Ensure backend is running on port 8000
- Check CORS settings if using different ports

### Slow first detection?
- First request loads models into memory
- Subsequent requests will be faster

### Camera not working?
- Camera requires HTTPS or localhost
- Check browser permissions

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing framework
- [PyTorch](https://pytorch.org/) for deep learning capabilities

---

<div align="center">
  <p>Made with ❤️ for AI enthusiasts</p>
  
  [⬆ Back to Top](#fashionai---ai-powered-clothing-detection)
</div>
