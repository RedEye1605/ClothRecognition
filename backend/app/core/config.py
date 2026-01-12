"""
👕 FashionAI - Configuration
=============================
Centralized configuration for the FashionAI backend.
"""

from pathlib import Path
from typing import Dict, List

# ============================================
# Paths
# ============================================
# Absolute path to backend/app/core/config.py
CONFIG_PATH = Path(__file__).resolve()
CORE_DIR = CONFIG_PATH.parent
APP_DIR = CORE_DIR.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent # This might be the container root
# In the new structure with Docker, models are in /code/models or relative to backend
# If running locally: backend/models

# Let's make it robust for both local and Docker
# If we are in backend/app/core, Models are in ../../models (relative to backend dir)
MODELS_DIR = BACKEND_DIR / "models"

# Model paths
CLOTH_CLASSIFIER_PATH = (MODELS_DIR / "cloth_classifier.pt").absolute()
COLOR_CLASSIFIER_PATH = (MODELS_DIR / "color_classifier.pt").absolute()


# ============================================
# Class Configuration
# ============================================
CLOTHING_CLASSES: List[str] = [
    "T-Shirt", "Dress", "Jacket", "Pants", 
    "Shirt", "Shorts", "Skirt", "Sweater"
]

# Mapping from model output (lowercase) to display labels
CLASS_MAPPING: Dict[str, str] = {
    "tshirt": "T-Shirt",
    "t-shirt": "T-Shirt",
    "dress": "Dress",
    "jacket": "Jacket",
    "pants": "Pants",
    "shirt": "Shirt",
    "short": "Shorts",
    "shorts": "Shorts",
    "skirt": "Skirt",
    "sweater": "Sweater"
}

COLOR_CLASSES: List[str] = [
    "Beige", "Black", "Blue", "Gray", 
    "Green", "Pattern", "Red", "White"
]

# UI Colors for clothing types
CLASS_COLORS: Dict[str, str] = {
    "t-shirt": "#FF6B6B",
    "tshirt": "#FF6B6B",
    "dress": "#45B7D1",
    "jacket": "#96CEB4",
    "pants": "#4ECDC4",
    "shirt": "#FFEAA7",
    "shorts": "#98D8C8",
    "short": "#98D8C8",
    "skirt": "#DDA0DD",
    "sweater": "#F7DC6F"
}

# Hex codes for detected colors
COLOR_HEX: Dict[str, str] = {
    "beige": "#D4A574",
    "black": "#2D2D2D",
    "blue": "#3B82F6",
    "gray": "#6B7280",
    "grey": "#6B7280",
    "green": "#22C55E",
    "pattren": "#8B5CF6",
    "pattern": "#8B5CF6",
    "red": "#EF4444",
    "white": "#F9FAFB"
}

# ============================================
# API Configuration
# ============================================
API_VERSION = "3.0.0"
API_TITLE = "👕 FashionAI API"
API_DESCRIPTION = """
AI-powered clothing detection API using YOLOv8.
Now with **COLOR DETECTION** using two-stage pipeline!

## Features
- 🎯 Clothing Type Detection (8 classes)
- 🎨 Color Classification (8 colors)
- ⚡ Real-time inference
- 📦 Batch processing

## Supported Clothing Classes
T-Shirt, Dress, Jacket, Pants, Shirt, Shorts, Skirt, Sweater

## Supported Colors
Beige, Black, Blue, Gray, Green, Pattern, Red, White
"""
