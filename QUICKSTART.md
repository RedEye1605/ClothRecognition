# 🚀 FashionAI Quick Start

<div align="center">
  <img src="frontend/Logo.png" alt="FashionAI Logo" width="120">
  
  **Get FashionAI running in 5 minutes**
</div>

---

## Prerequisites

- ✅ Python 3.9+
- ✅ pip
- ✅ Trained models (`.pt` files)

---

## Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/RedEye1605/ClothRecognition.git
cd ClothRecognition

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
# source .venv/bin/activate
```

---

## Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

## Step 3: Add Models

Place trained models in `models/` folder:

| Model | Description |
|-------|-------------|
| `cloth_classifier.pt` | Clothing detection (YOLOv8) |
| `color_classifier.pt` | Color classification (YOLOv8-cls) |

> 💡 Train your own models using `notebooks/cloth_detection_training.ipynb`

---

## Step 4: Start Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Expected output:**
```
🔄 Loading detection model from: .../models/cloth_classifier.pt
✅ Detection model loaded: cloth_classifier.pt
🔄 Loading color model from: .../models/color_classifier.pt
✅ Color model loaded: color_classifier.pt
🚀 FashionAI API v3.0.0 starting...
```

---

## Step 5: Start Frontend

Open a **new terminal**:

```bash
cd frontend
python -m http.server 5500
```

---

## Step 6: Open Browser

🌐 Navigate to: **http://127.0.0.1:5500**

You'll see:
- **Home** - Landing page with features
- **Try App** - Detection interface
- **Docs** - Full documentation

---

## 🧪 Test the API

```bash
# Health check
curl http://127.0.0.1:8000/health

# List classes
curl http://127.0.0.1:8000/classes
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Models not loading** | Check `models/` folder has both `.pt` files |
| **API not connecting** | Ensure backend runs on port 8000 |
| **Camera not working** | Requires HTTPS or localhost |
| **Slow first request** | Normal - models load on first use |

---

## 📚 Next Steps

- 📖 Read full [README.md](README.md)
- 🎓 Train custom models with notebooks
- 🐳 Deploy with Docker
- 🌐 Deploy to HuggingFace Spaces

---

<div align="center">
  <p>Need help? Check the <a href="#-troubleshooting">troubleshooting section</a></p>
</div>
