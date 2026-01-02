# 🚀 Quick Start Guide - Cloth Recognition

## ⚡ Langkah Cepat (5 Menit Setup)

### 1️⃣ Install Dependencies

```bash
cd cloth-recognition-yolo
pip install -r requirements.txt
```

### 2️⃣ Training di Google Colab

1. **Upload notebook** ke Google Colab:
   - File: `notebooks/cloth_detection_training.ipynb`

2. **Aktifkan GPU**:
   - `Runtime` → `Change runtime type` → `GPU (T4)`

3. **Jalankan semua cell** dan tunggu training selesai

4. **Download model** (`best.pt`) yang sudah trained

### 3️⃣ Test Model Lokal

```bash
# Copy model ke folder models/
mkdir models
# Copy best.pt ke models/

# Jalankan prediction
python src/inference/predict.py --image path/to/image.jpg --model models/best.pt
```

### 4️⃣ Jalankan Backend API

```bash
cd backend
uvicorn main:app --reload --port 8000

# API tersedia di: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5️⃣ Buka Frontend

```bash
# Buka di browser
start frontend/index.html
```

---

## 🎯 Workflow Lengkap

```
┌──────────────────────────────────────────────────────────┐
│  1. TRAINING (Google Colab)                               │
│     └── Dataset → YOLOv8 → best.pt                       │
├──────────────────────────────────────────────────────────┤
│  2. BACKEND (FastAPI)                                     │
│     └── best.pt → API Endpoint → JSON Response           │
├──────────────────────────────────────────────────────────┤
│  3. FRONTEND (HTML/JS)                                    │
│     └── Upload Image → API Call → Display Results        │
├──────────────────────────────────────────────────────────┤
│  4. DEPLOYMENT (Hugging Face)                             │
│     └── Gradio App → Free Hosting                        │
└──────────────────────────────────────────────────────────┘
```

---

## 📂 Struktur File Penting

```
cloth-recognition-yolo/
├── notebooks/
│   └── cloth_detection_training.ipynb  ← Training notebook
├── backend/
│   └── main.py                         ← FastAPI server
├── frontend/
│   └── index.html                      ← Web interface
├── deployment/
│   └── huggingface/
│       └── app.py                      ← Gradio deploy
├── models/
│   └── best.pt                         ← Model Anda (setelah training)
└── requirements.txt
```

---

## ❓ FAQ

**Q: Tidak punya GPU?**
A: Gunakan Google Colab (gratis, GPU T4)

**Q: Berapa lama training?**
A: ~30 menit - 1 jam di Colab T4

**Q: Dataset dari mana?**
A: Roboflow Universe (gratis, format YOLO)

**Q: Deploy gratis dimana?**
A: Hugging Face Spaces (gratis, GPU support)

---

## 🆘 Troubleshooting

### Error: CUDA not available
```
# Pastikan di Colab GPU aktif
Runtime > Change runtime type > GPU
```

### Error: Model not found
```bash
# Pastikan model ada di folder models/
ls models/
# Harus ada: best.pt
```

### API Error 503
```bash
# Model belum di-load, cek path model
# Edit backend/main.py: MODEL_PATH
```

---

**🎉 Happy Coding!**
