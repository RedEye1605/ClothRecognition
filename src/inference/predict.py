"""
Cloth Recognition - Inference Script
=====================================
Script untuk menjalankan inference pada gambar menggunakan trained model.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# ============================================
# Configuration
# ============================================
DEFAULT_MODEL = "models/best.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45

CLASS_COLORS = {
    "shirt": (255, 107, 107),
    "pants": (78, 205, 196),
    "dress": (69, 183, 209),
    "jacket": (150, 206, 180),
    "skirt": (255, 234, 167),
    "sweater": (221, 160, 221),
    "shorts": (152, 216, 200),
    "coat": (247, 220, 111),
    "hat": (187, 143, 206),
    "shoes": (133, 193, 233),
}

# ============================================
# Main Functions
# ============================================
def load_model(model_path: str) -> YOLO:
    """Load YOLO model from path."""
    print(f"📦 Loading model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"⚠️ Model not found at {model_path}")
        print("   Using pretrained YOLOv8n for demo...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)
    
    return model


def detect_clothing(
    model: YOLO,
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    save: bool = True,
    show: bool = False
) -> dict:
    """
    Detect clothing items in an image.
    
    Args:
        model: YOLO model instance
        image_path: Path to the image file
        confidence: Minimum confidence threshold
        iou: IOU threshold for NMS
        save: Save annotated image
        show: Display result in window
    
    Returns:
        dict: Detection results
    """
    print(f"\n🔍 Processing: {image_path}")
    
    # Run inference
    results = model.predict(
        image_path,
        conf=confidence,
        iou=iou,
        verbose=False
    )
    
    # Parse results
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "bbox": [round(x, 2) for x in bbox]
        })
        
        print(f"   ✓ {cls_name}: {conf:.1%}")
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Save results
    if save:
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\n💾 Saved: {output_path}")
    
    # Show results
    if show:
        cv2.imshow("Cloth Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        "image": image_path,
        "total_detections": len(detections),
        "detections": detections
    }


def detect_batch(
    model: YOLO,
    image_dir: str,
    confidence: float = DEFAULT_CONFIDENCE,
    output_dir: Optional[str] = None
) -> list:
    """
    Detect clothing in all images in a directory.
    
    Args:
        model: YOLO model instance
        image_dir: Directory containing images
        confidence: Minimum confidence threshold
        output_dir: Directory to save results
    
    Returns:
        list: All detection results
    """
    image_dir = Path(image_dir)
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    
    print(f"\n📂 Found {len(images)} images in {image_dir}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    results = []
    for img_path in images:
        result = detect_clothing(
            model,
            str(img_path),
            confidence=confidence,
            save=output_dir is not None,
            show=False
        )
        results.append(result)
    
    # Summary
    total_images = len(results)
    total_detections = sum(r["total_detections"] for r in results)
    
    print(f"\n📊 Summary:")
    print(f"   - Images processed: {total_images}")
    print(f"   - Total detections: {total_detections}")
    print(f"   - Average per image: {total_detections/total_images:.1f}")
    
    return results


# ============================================
# CLI Interface
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="🎯 Cloth Recognition - Detect clothing in images using YOLOv8"
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to single image file"
    )
    
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Path to directory with images"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to YOLO model (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display results in window"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image and not args.dir:
        parser.error("Please provide --image or --dir")
    
    # Load model
    model = load_model(args.model)
    
    # Run detection
    if args.image:
        result = detect_clothing(
            model,
            args.image,
            confidence=args.confidence,
            save=args.output is not None,
            show=args.show
        )
        results = [result]
    else:
        results = detect_batch(
            model,
            args.dir,
            confidence=args.confidence,
            output_dir=args.output
        )
    
    # JSON output
    if args.json:
        print("\n📋 JSON Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
