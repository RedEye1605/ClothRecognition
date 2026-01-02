"""
Image Processing Utilities
==========================
Helper functions for image processing in the cloth detection pipeline.
"""

import io
import base64
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image
import cv2

# ============================================
# Image Loading
# ============================================

def load_image(
    source: Union[str, bytes, np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        source: Image path, bytes, numpy array, or PIL Image
    
    Returns:
        numpy array (RGB)
    """
    if isinstance(source, str):
        # File path
        image = cv2.imread(source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(source, bytes):
        # Bytes
        image = Image.open(io.BytesIO(source))
        image = np.array(image)
    elif isinstance(source, Image.Image):
        # PIL Image
        image = np.array(source)
    elif isinstance(source, np.ndarray):
        # Already numpy
        image = source
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image


def save_image(
    image: np.ndarray,
    path: str,
    quality: int = 95
) -> str:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array (RGB)
        path: Output path
        quality: JPEG quality (1-100)
    
    Returns:
        Saved file path
    """
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Determine format
    ext = Path(path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == '.png':
        cv2.imwrite(path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(path, image_bgr)
    
    return path


# ============================================
# Image Transformations
# ============================================

def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image.
    
    Args:
        image: Input image
        size: Target size (width, height)
        keep_aspect: Maintain aspect ratio
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    if keep_aspect:
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to exact size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
    else:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    return resized


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize (YOLO standard preprocessing).
    
    Args:
        image: Input image
        new_shape: Target shape
        color: Padding color
    
    Returns:
        Tuple of (resized image, scale, padding)
    """
    shape = image.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return image, r, (dw, dh)


# ============================================
# Encoding/Decoding
# ============================================

def image_to_base64(
    image: np.ndarray,
    format: str = 'jpeg',
    quality: int = 90
) -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: Image as numpy array
        format: Output format ('jpeg' or 'png')
        quality: JPEG quality
    
    Returns:
        Base64 encoded string
    """
    # Convert to PIL
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR from OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image)
    
    # Encode
    buffer = io.BytesIO()
    if format.lower() == 'png':
        pil_image.save(buffer, format='PNG')
    else:
        pil_image.save(buffer, format='JPEG', quality=quality)
    
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_image(b64_string: str) -> np.ndarray:
    """
    Convert base64 string to image.
    
    Args:
        b64_string: Base64 encoded image
    
    Returns:
        Image as numpy array
    """
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


# ============================================
# Bounding Box Utilities
# ============================================

def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        color: Box color (RGB)
        thickness: Line thickness
        font_scale: Font size
    
    Returns:
        Image with drawn bbox
    """
    image = image.copy()
    x1, y1, x2, y2 = [int(x) for x in bbox]
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        image,
        (x1, y1 - text_h - baseline - 5),
        (x1 + text_w + 5, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    return image


def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    class_colors: Optional[dict] = None
) -> np.ndarray:
    """
    Draw all detections on image.
    
    Args:
        image: Input image
        detections: List of detection dicts with 'bbox', 'class', 'confidence'
        class_colors: Dict mapping class names to colors
    
    Returns:
        Annotated image
    """
    default_colors = {
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
    
    colors = class_colors or default_colors
    
    for det in detections:
        class_name = det.get('class', det.get('class_name', 'unknown'))
        confidence = det.get('confidence', 0)
        bbox = det['bbox']
        color = colors.get(class_name.lower(), (255, 255, 255))
        
        label = f"{class_name} {confidence:.0%}"
        image = draw_bbox(image, bbox, label, color)
    
    return image


# ============================================
# Validation
# ============================================

def validate_image(
    image: Union[bytes, np.ndarray],
    max_size: int = 10 * 1024 * 1024,  # 10MB
    min_dim: int = 32,
    max_dim: int = 4096
) -> Tuple[bool, str]:
    """
    Validate image for processing.
    
    Args:
        image: Image bytes or array
        max_size: Maximum file size in bytes
        min_dim: Minimum dimension
        max_dim: Maximum dimension
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check size if bytes
        if isinstance(image, bytes):
            if len(image) > max_size:
                return False, f"Image too large: {len(image)} bytes (max: {max_size})"
            image = load_image(image)
        
        # Check dimensions
        h, w = image.shape[:2]
        if w < min_dim or h < min_dim:
            return False, f"Image too small: {w}x{h} (min: {min_dim})"
        if w > max_dim or h > max_dim:
            return False, f"Image too large: {w}x{h} (max: {max_dim})"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Invalid image: {str(e)}"
