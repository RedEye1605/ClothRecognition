"""
👕 FashionAI - Detection Router
=================================
API endpoints for clothing detection.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
from PIL import Image
import io

from ..schemas import DetectionResponse, HealthResponse, ClassesResponse
from ..services.detector import get_detection_service
from ..config import (
    API_VERSION, 
    CLASS_COLORS, 
    COLOR_HEX,
    CLOTH_CLASSIFIER_PATH,
    COLOR_CLASSIFIER_PATH
)

router = APIRouter()


@router.get("/", tags=["Info"])
async def root():
    """API root - welcome message"""
    return {
        "message": "👕 FashionAI API",
        "version": API_VERSION,
        "features": ["Clothing Detection", "Color Classification"],
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API and model health"""
    service = get_detection_service()
    
    return HealthResponse(
        status="healthy",
        detection_model_loaded=service.is_ready,
        color_model_loaded=service.has_color_model,
        detection_model_path=str(CLOTH_CLASSIFIER_PATH),
        color_model_path=str(COLOR_CLASSIFIER_PATH),
        clothing_classes=service.detection_classes,
        color_classes=service.color_classes,
        version=API_VERSION
    )


@router.get("/classes", response_model=ClassesResponse, tags=["Info"])
async def get_classes():
    """Get list of detectable classes and colors"""
    service = get_detection_service()
    
    return ClassesResponse(
        clothing_classes=service.detection_classes,
        clothing_count=len(service.detection_classes),
        clothing_colors=CLASS_COLORS,
        color_classes=service.color_classes,
        color_count=len(service.color_classes),
        color_hex=COLOR_HEX
    )


@router.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_clothing(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    confidence: float = Form(0.25, ge=0.1, le=0.9, description="Confidence threshold")
):
    """
    Detect clothing items in an uploaded image with COLOR classification.
    
    **Two-Stage Pipeline:**
    1. Detect clothing with bounding boxes
    2. Classify color for each detected item
    
    Returns bounding boxes, class names, colors, and confidence scores.
    """
    service = get_detection_service()
    
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Detection model not loaded")
    
    try:
        # Read file contents
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Open image
        try:
            image = Image.open(io.BytesIO(contents))
            image = image.convert("RGB")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(img_error)}")
        
        # Detect
        detections, inference_time = service.detect(image, confidence=confidence)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            image_size=[image.width, image.height],
            inference_time=round(inference_time, 4),
            model_name="YOLOv8",
            color_model="YOLOv8-cls" if service.has_color_model else "None"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.post("/detect/batch", tags=["Detection"])
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    confidence: float = Form(0.25, ge=0.1, le=0.9)
):
    """
    Process multiple images in batch with color detection.
    """
    service = get_detection_service()
    
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            detections, _ = service.detect(image, confidence=confidence)
            
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
