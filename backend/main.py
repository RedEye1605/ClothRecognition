"""
👕 FashionAI - Cloth Recognition API
=====================================
FastAPI backend for clothing detection using YOLOv8.

Classes (8):
- Tshirt, Dress, Jacket, Pants
- Shirt, Short, Skirt, Sweater

Author: FashionAI Team
Version: 2.0.0
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

# ============================================
# Configuration
# ============================================
# Model path - relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Fallback to environment variable or default
if not MODEL_PATH.exists():
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

# Class configuration for new dataset
CLASS_NAMES = ["Tshirt", "dress", "jacket", "pants", "shirt", "short", "skirt", "sweater"]

CLASS_COLORS = {
    "tshirt": "#FF6B6B",
    "t-shirt": "#FF6B6B",
    "dress": "#45B7D1",
    "jacket": "#96CEB4",
    "pants": "#4ECDC4",
    "shirt": "#FFEAA7",
    "short": "#98D8C8",
    "shorts": "#98D8C8",
    "skirt": "#DDA0DD",
    "sweater": "#F7DC6F"
}

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="👕 FashionAI API",
    description="""
    AI-powered clothing detection API using YOLOv8.
    
    ## Features
    - Single image detection
    - Batch processing
    - Adjustable confidence threshold
    
    ## Supported Classes (8)
    Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Models
# ============================================
class Detection(BaseModel):
    """Single detection result"""
    className: str = "Tshirt"
    confidence: float = 0.95
    bbox: List[float] = [100, 100, 200, 200]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    """Detection API response"""
    success: bool = True
    detections: List[Detection] = []
    image_size: List[int] = [640, 480]
    inference_time: float = 0.05
    model_name: str = "YOLOv8n"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    model_loaded: bool = True
    model_path: str = "models/best.pt"
    classes: List[str] = CLASS_NAMES
    version: str = "2.0.0"

# ============================================
# Load Model
# ============================================
model = None

def load_model():
    """Load YOLO model at startup"""
    global model
    model_path_str = str(MODEL_PATH)
    
    print(f"🔄 Loading model from: {model_path_str}")
    
    try:
        if Path(model_path_str).exists():
            model = YOLO(model_path_str)
            print(f"✅ Custom model loaded: {model_path_str}")
        else:
            # Fallback to pretrained
            print(f"⚠️ Model not found at {model_path_str}, using pretrained YOLOv8n")
            model = YOLO("yolov8n.pt")
        
        # Print model info
        class_names = list(model.names.values())
        print(f"📋 Classes ({len(class_names)}): {class_names}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = YOLO("yolov8n.pt")

# Load on startup
load_model()

# ============================================
# Endpoints
# ============================================
@app.get("/", tags=["Info"])
async def root():
    """API root - welcome message"""
    return {
        "message": "👕 FashionAI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API and model health"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        classes=list(model.names.values()) if model else CLASS_NAMES,
        version="2.0.0"
    )

@app.get("/classes", tags=["Info"])
async def get_classes():
    """Get list of detectable classes"""
    if model:
        return {
            "classes": list(model.names.values()),
            "count": len(model.names),
            "colors": CLASS_COLORS
        }
    return {
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES),
        "colors": CLASS_COLORS
    }

@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_clothing(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    confidence: float = Form(0.25, ge=0.1, le=0.9, description="Confidence threshold")
):
    """
    Detect clothing items in an uploaded image.
    
    - **file**: Image file to analyze
    - **confidence**: Minimum confidence threshold (0.1 - 0.9)
    
    Returns bounding boxes, class names, and confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file contents
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Try to open as image
        try:
            image = Image.open(io.BytesIO(contents))
            image = image.convert("RGB")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(img_error)}")
        
        image_np = np.array(image)
        
        # Run inference
        import time
        start_time = time.time()
        results = model.predict(image_np, conf=confidence, iou=0.45, verbose=False)
        inference_time = time.time() - start_time
        
        # Parse results
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                detections.append({
                    "className": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in bbox]
                })
        
        return {
            "success": True,
            "detections": detections,
            "image_size": [image.width, image.height],
            "inference_time": round(inference_time, 4),
            "model_name": "YOLOv8"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/detect/batch", tags=["Detection"])
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    confidence: float = Form(0.25, ge=0.1, le=0.9)
):
    """
    Process multiple images in batch.
    
    - **files**: List of image files
    - **confidence**: Confidence threshold
    
    Returns detection results for each image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(image)
            
            predictions = model.predict(image_np, conf=confidence, iou=0.45, verbose=False)
            
            detections = []
            for box in predictions[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = predictions[0].names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    "className": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in bbox]
                })
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results, "total": len(results)}

# ============================================
# Startup / Shutdown
# ============================================
@app.on_event("startup")
async def startup_event():
    print("🚀 FashionAI API starting...")
    print(f"📍 Model path: {MODEL_PATH}")
    print(f"📋 Classes: {CLASS_NAMES}")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 FashionAI API shutting down...")

# ============================================
# Run with uvicorn
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
