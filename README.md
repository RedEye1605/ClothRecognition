# 👕 FashionAI - Cloth Recognition

AI-powered clothing detection using **YOLOv8**.

## ✨ Features

- 🎯 **8 Classes**: Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
- 📸 **Image Upload**: Drag & drop detection
- ⚡ **Fast**: YOLOv8n optimized for speed
- 🌐 **Web UI**: Modern dark theme

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Model
Place trained `best.pt` in `models/` folder.

### 3. Run Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open Frontend
Open `frontend/index.html` in browser.

## 📁 Structure

```
cloth-recognition-yolo/
├── backend/
│   ├── main.py           # FastAPI server
│   └── requirements.txt
├── frontend/
│   └── index.html        # Web UI
├── models/
│   └── best.pt           # Trained model
├── notebooks/
│   └── cloth_detection_training.ipynb
└── deployment/
    └── huggingface/      # HuggingFace files
```

## 🎓 Training

1. Open `notebooks/cloth_detection_training.ipynb` in Google Colab
2. Run all cells (uses T4 GPU)
3. Download `best.pt`
4. Place in `models/` folder

## 🌐 Deployment

### Hugging Face Spaces
Upload files from `deployment/huggingface/` to HuggingFace Spaces:
1. Create new Space (Gradio SDK)
2. Upload `app.py`, `requirements.txt`, `best.pt`, `README.md` from `deployment/huggingface/`
3. Your app will be live at `https://huggingface.co/spaces/your-username/your-space`

## 📡 API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Detect clothing |
| `/classes` | GET | List classes |

## 🛠️ Tech Stack

- YOLOv8, PyTorch
- FastAPI, Uvicorn
- HTML, CSS, JavaScript
- Gradio (HuggingFace)

## 📄 License

MIT
