"""
👕 FashionAI - Detection Service
==================================
Handles clothing detection and color classification using YOLOv8.
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image

from ..core.config import (
    CLOTH_CLASSIFIER_PATH, 
    COLOR_CLASSIFIER_PATH,
    CLOTHING_CLASSES,
    COLOR_CLASSES,
    COLOR_HEX,
    CLASS_MAPPING
)


# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")


class DetectionService:
    """
    Service for clothing detection and color classification.
    Implements two-stage detection pipeline.
    """
    
    def __init__(self):
        self.detection_model: Optional[YOLO] = None
        self.color_model: Optional[YOLO] = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load detection and color classification models."""
        # Use absolute path for reliability
        model_path = str(CLOTH_CLASSIFIER_PATH.resolve())
        print(f"🔄 Loading detection model from: {model_path}")
        try:
            if Path(model_path).exists():
                self.detection_model = YOLO(model_path)
                print(f"✅ Detection model loaded: {CLOTH_CLASSIFIER_PATH.name}")
            else:
                print(f"⚠️ Model not found at {model_path}, using pretrained YOLOv8n")
                self.detection_model = YOLO("yolov8n.pt")
            
            class_names = list(self.detection_model.names.values())
            print(f"📋 Detection model classes ({len(class_names)}): {class_names}")
            
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            self.detection_model = YOLO("yolov8n.pt")
        
        # Load color classification model
        color_path = str(COLOR_CLASSIFIER_PATH.resolve())
        print(f"🔄 Loading color model from: {color_path}")
        try:
            if Path(color_path).exists():
                self.color_model = YOLO(color_path)
                print(f"✅ Color model loaded: {COLOR_CLASSIFIER_PATH.name}")
                color_names = list(self.color_model.names.values())
                print(f"🎨 Color model classes ({len(color_names)}): {color_names}")
            else:
                print(f"⚠️ Color model not found at {color_path}")
                self.color_model = None
                
        except Exception as e:
            print(f"❌ Error loading color model: {e}")
            self.color_model = None
    
    @property
    def is_ready(self) -> bool:
        """Check if detection model is loaded."""
        return self.detection_model is not None
    
    @property
    def has_color_model(self) -> bool:
        """Check if color model is loaded."""
        return self.color_model is not None
    
    @property
    def detection_classes(self) -> List[str]:
        """Get list of detection classes."""
        return CLOTHING_CLASSES
    
    @property
    def color_classes(self) -> List[str]:
        """Get list of color classes."""
        return COLOR_CLASSES
    
    def classify_color(self, image_crop: Image.Image) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Classify color of a clothing crop.
        
        Args:
            image_crop: PIL Image of the clothing item
            
        Returns:
            Tuple of (color_name, confidence, hex_code)
        """
        if self.color_model is None:
            return None, 0.0, None
        
        try:
            results = self.color_model.predict(image_crop, verbose=False)
            probs = results[0].probs
            
            color_name = results[0].names[probs.top1]
            color_conf = float(probs.top1conf)
            color_hex = COLOR_HEX.get(color_name.lower(), "#808080")
            
            return color_name, color_conf, color_hex
        except Exception as e:
            print(f"⚠️ Color classification error: {e}")
            return None, 0.0, None
    
    def format_label(self, color: Optional[str], class_name: str) -> str:
        """Format label as 'Color ClassName' with proper capitalization."""
        # Map class name to display name
        display_class = CLASS_MAPPING.get(class_name.lower(), class_name.title())
        
        if color:
            color_display = "Pattern" if color.lower() == "pattren" else color.capitalize()
            return f"{color_display} {display_class}"
        return display_class
    
    def detect(
        self, 
        image: Image.Image, 
        confidence: float = 0.25,
        iou: float = 0.45
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect clothing items with color classification.
        
        Args:
            image: PIL Image to process
            confidence: Minimum confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            Tuple of (detections list, inference time)
        """
        import time
        
        if not self.is_ready:
            return [], 0.0
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Stage 1: Detect clothing
        start_time = time.time()
        results = self.detection_model.predict(
            image_np, 
            conf=confidence, 
            iou=iou, 
            verbose=False
        )
        
        # Stage 2: Classify color for each detection
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                # Crop and classify color
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                
                color_name = None
                color_conf = 0.0
                color_hex = None
                
                if self.has_color_model and (x2 - x1) > 10 and (y2 - y1) > 10:
                    crop = image.crop((x1, y1, x2, y2))
                    color_name, color_conf, color_hex = self.classify_color(crop)
                
                label = self.format_label(color_name, cls_name)
                
                detections.append({
                    "className": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in bbox],
                    "color": color_name,
                    "colorConfidence": round(color_conf, 4) if color_conf else None,
                    "colorHex": color_hex,
                    "label": label
                })
        
        inference_time = time.time() - start_time
        return detections, inference_time


# Singleton instance
_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Get or create detection service singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
