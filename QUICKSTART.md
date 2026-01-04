# 🚀 FashionAI Quick Start

Get the FashionAI cloth recognition system up and running in 5 minutes.

## Prerequisites

- Python 3.9+
- pip

## Step 1: Clone and Setup

```bash
# Navigate to project
cd cloth-recognition-yolo

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
cd backend
pip install -r requirements.txt
```

## Step 2: Add Models

Download or copy models to `models/` folder:
- `cloth_classifier.pt` - Clothing detection (YOLOv8)
- `color_classifier.pt` - Color classification (YOLOv8-cls)

## Step 3: Start Backend

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

## Step 4: Start Frontend

Open new terminal:
```bash
cd frontend
python -m http.server 5500
```

## Step 5: Open in Browser

Go to: http://127.0.0.1:5500

## Test the API

```bash
# Health check
curl http://127.0.0.1:8000/health

# List classes
curl http://127.0.0.1:8000/classes
```

## Troubleshooting

**Models not loading?**
- Check `models/` folder has both `.pt` files
- Verify file names are correct

**API not connecting?**
- Ensure backend is running on port 8000
- Check firewall settings

**Slow detection?**
- First request may be slow (model loading)
- Subsequent requests will be fast

## Next Steps

- 📖 Read full [README.md](README.md)
- 🎓 Train custom models with notebooks
- 🌐 Deploy to HuggingFace Spaces
