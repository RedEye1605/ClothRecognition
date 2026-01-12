"""
👕 FashionAI - Cloth Recognition API
=====================================
FastAPI backend for clothing detection using YOLOv8.
Now with COLOR DETECTION using two-stage pipeline!

Classes (8):
- Tshirt, Dress, Jacket, Pants
- Shirt, Short, Skirt, Sweater

Colors (8):
- Beige, Black, Blue, Gray
- Green, Pattern, Red, White

Author: FashionAI Team
Version: 3.0.0
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import API_TITLE, API_DESCRIPTION, API_VERSION
from .api.v1.endpoints import detection
from .services.detector import get_detection_service


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
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

# Include routers
app.include_router(detection.router)

# ============================================
# Startup / Shutdown Events
# ============================================
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print(f"🚀 FashionAI API v{API_VERSION} starting...")
    
    # Pre-load detection service
    service = get_detection_service()
    
    print(f"👕 Clothing classes: {service.detection_classes}")
    print(f"🎨 Color classes: {service.color_classes}")
    print("✅ API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 FashionAI API shutting down...")
