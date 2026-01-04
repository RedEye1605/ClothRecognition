"""
👕 FashionAI - Pydantic Schemas
================================
Request/Response models for API endpoints.
"""

from pydantic import BaseModel
from typing import List, Optional


class Detection(BaseModel):
    """Single detection result with color"""
    className: str
    confidence: float
    bbox: List[float]
    color: Optional[str] = None
    colorConfidence: Optional[float] = None
    colorHex: Optional[str] = None
    label: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "className": "Tshirt",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "color": "blue",
                "colorConfidence": 0.87,
                "colorHex": "#3B82F6",
                "label": "Blue Tshirt"
            }
        }


class DetectionResponse(BaseModel):
    """Detection API response"""
    success: bool = True
    detections: List[Detection] = []
    image_size: List[int] = [640, 480]
    inference_time: float = 0.0
    model_name: str = "YOLOv8"
    color_model: str = "YOLOv8-cls"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    detection_model_loaded: bool = True
    color_model_loaded: bool = True
    detection_model_path: str = ""
    color_model_path: str = ""
    clothing_classes: List[str] = []
    color_classes: List[str] = []
    version: str = "3.0.0"


class ClassesResponse(BaseModel):
    """Available classes response"""
    clothing_classes: List[str] = []
    clothing_count: int = 0
    clothing_colors: dict = {}
    color_classes: List[str] = []
    color_count: int = 0
    color_hex: dict = {}
