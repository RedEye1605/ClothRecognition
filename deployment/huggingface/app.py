"""
👕 FashionAI - Cloth Recognition
================================
Hugging Face Spaces Deployment
YOLOv8 clothing detection with Premium Gradio UI
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "ultralytics", "-q"])
    from ultralytics import YOLO

# ============================================
# Configuration
# ============================================
MODEL_PATH = "best.pt"

CLASS_CONFIG = {
    "tshirt": {"color": "#f87171", "label": "T-Shirt", "icon": "👕"},
    "t-shirt": {"color": "#f87171", "label": "T-Shirt", "icon": "👕"},
    "dress": {"color": "#34d399", "label": "Dress", "icon": "👗"},
    "jacket": {"color": "#60a5fa", "label": "Jacket", "icon": "🧥"},
    "pants": {"color": "#a78bfa", "label": "Pants", "icon": "👖"},
    "shirt": {"color": "#fbbf24", "label": "Shirt", "icon": "👔"},
    "short": {"color": "#f472b6", "label": "Shorts", "icon": "🩳"},
    "shorts": {"color": "#f472b6", "label": "Shorts", "icon": "🩳"},
    "skirt": {"color": "#38bdf8", "label": "Skirt", "icon": "🎽"},
    "sweater": {"color": "#fb923c", "label": "Sweater", "icon": "🧶"}
}

CLASSES = ["T-Shirt", "Dress", "Jacket", "Pants", "Shirt", "Shorts", "Skirt", "Sweater"]

# ============================================
# Load Model
# ============================================
print("🔄 Loading model...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    model_status = "✅ Custom Model Loaded"
except Exception as e:
    print(f"⚠️ Using pretrained model: {e}")
    model = YOLO("yolov8n.pt")
    model_status = "⚠️ Using Pretrained Model"

print(f"📋 Classes: {list(model.names.values())}")

# ============================================
# Detection Functions
# ============================================
def hex_to_bgr(hex_color):
    """Convert hex color to BGR for OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def draw_custom_boxes(image, results):
    """Draw custom styled bounding boxes"""
    annotated = image.copy()
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        config = CLASS_CONFIG.get(cls_name, {"color": "#888888", "label": cls_name})
        color = hex_to_bgr(config["color"])
        label = f"{config['label']} {conf:.0%}"
        
        # Draw bounding box with rounded corners effect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 8), font, font_scale, (0, 0, 0), thickness)
    
    return annotated

def detect_image(image, confidence):
    """Detect clothing in uploaded image"""
    if image is None:
        return None, create_empty_results()
    
    results = model.predict(image, conf=confidence, iou=0.45, verbose=False)
    annotated = draw_custom_boxes(image, results)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if len(annotated.shape) == 3 else annotated
    
    # Build results HTML
    results_html = create_results_html(results)
    
    return annotated_rgb, results_html

def detect_webcam(frame, confidence):
    """Process webcam frame"""
    if frame is None:
        return None
    
    results = model.predict(frame, conf=confidence, iou=0.45, verbose=False)
    annotated = draw_custom_boxes(frame, results)
    return annotated

def create_empty_results():
    """Create empty results placeholder"""
    return """
    <div style="text-align: center; padding: 60px 20px; color: rgba(255,255,255,0.5);">
        <div style="font-size: 48px; margin-bottom: 16px;">📸</div>
        <div style="font-size: 18px; font-weight: 600;">Upload an image to start</div>
        <div style="font-size: 14px; margin-top: 8px;">Drag & drop or click to select</div>
    </div>
    """

def create_results_html(results):
    """Create styled results HTML"""
    detections = []
    total_conf = 0
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id].lower()
        conf = float(box.conf[0])
        total_conf += conf
        
        config = CLASS_CONFIG.get(cls_name, {"color": "#888", "label": cls_name, "icon": "👕"})
        
        detections.append({
            "label": config["label"],
            "color": config["color"],
            "icon": config.get("icon", "👕"),
            "confidence": conf
        })
    
    if not detections:
        return """
        <div style="text-align: center; padding: 40px 20px; color: rgba(255,255,255,0.6);">
            <div style="font-size: 40px; margin-bottom: 12px;">🔍</div>
            <div style="font-size: 16px; font-weight: 600;">No clothing detected</div>
            <div style="font-size: 13px; margin-top: 8px; color: rgba(255,255,255,0.4);">Try lowering the confidence threshold</div>
        </div>
        """
    
    avg_conf = total_conf / len(detections) if detections else 0
    
    # Build HTML
    items_html = ""
    for det in detections:
        items_html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; 
                    padding: 12px 16px; background: rgba(255,255,255,0.05); 
                    border-radius: 10px; margin-bottom: 8px;
                    border-left: 3px solid {det['color']};">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">{det['icon']}</span>
                <span style="font-weight: 600; color: white;">{det['label']}</span>
            </div>
            <span style="background: {det['color']}; color: #000; padding: 4px 10px; 
                         border-radius: 20px; font-size: 13px; font-weight: 700;">
                {det['confidence']:.0%}
            </span>
        </div>
        """
    
    return f"""
    <div style="padding: 16px;">
        <div style="display: flex; gap: 12px; margin-bottom: 20px;">
            <div style="flex: 1; background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2)); 
                        padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 28px; font-weight: 800; color: #a78bfa;">{len(detections)}</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px;">Items Found</div>
            </div>
            <div style="flex: 1; background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(16,185,129,0.2)); 
                        padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 28px; font-weight: 800; color: #34d399;">{avg_conf:.0%}</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px;">Avg Confidence</div>
            </div>
        </div>
        
        <div style="font-size: 14px; font-weight: 600; color: rgba(255,255,255,0.7); 
                    margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;">
            Detected Items
        </div>
        
        {items_html}
    </div>
    """

# ============================================
# Premium CSS Theme
# ============================================
custom_css = """
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(180deg, #0a0a0f 0%, #12121a 50%, #0a0a0f 100%) !important;
    min-height: 100vh;
}

/* Header */
.header-container {
    text-align: center;
    padding: 40px 20px 30px;
    background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
}

.header-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    border-radius: 20px;
    font-size: 32px;
    margin-bottom: 16px;
    box-shadow: 0 0 40px rgba(99, 102, 241, 0.4);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.8) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
}

/* Feature Badges */
.feature-badges {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 20px;
}

.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #818cf8;
}

/* Tab Styling */
.tabs {
    background: transparent !important;
    border: none !important;
}

button.selected {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
}

/* Cards */
.card {
    background: rgba(26, 26, 40, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px);
}

/* Image containers */
.image-container {
    border-radius: 12px;
    overflow: hidden;
    background: #1a1a28;
}

/* Slider */
input[type="range"] {
    accent-color: #6366f1 !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.5) !important;
}

/* Results Panel */
.results-panel {
    background: rgba(26, 26, 40, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    min-height: 400px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px 20px;
    margin-top: 40px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(18, 18, 26, 0.8);
}

.footer-brand {
    font-size: 18px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.footer-text {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.5);
}

.tech-badges {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-top: 16px;
}

.tech-badge {
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
}

/* Class Tags */
.class-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
}

.class-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.8);
}

.class-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
"""

# ============================================
# Gradio Interface
# ============================================
with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="FashionAI - Cloth Recognition") as demo:
    
    # Header
    gr.HTML("""
    <div class="header-container">
        <div class="header-logo">👕</div>
        <div class="header-title">FashionAI</div>
        <div class="header-subtitle">AI-powered clothing detection using YOLOv8 deep learning</div>
        
        <div class="feature-badges">
            <span class="feature-badge">🏷️ 8 Classes</span>
            <span class="feature-badge">⚡ Real-time</span>
            <span class="feature-badge">📹 Webcam Support</span>
        </div>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        # Upload Tab
        with gr.Tab("🖼️ Upload Image", id="upload"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 16px; font-weight: 600; color: white; margin-bottom: 12px;'>📷 Input Image</div>")
                    img_input = gr.Image(
                        type="numpy", 
                        label="",
                        height=400,
                        sources=["upload", "clipboard"]
                    )
                    
                    with gr.Row():
                        conf_slider = gr.Slider(
                            minimum=0.1, 
                            maximum=0.9, 
                            value=0.25, 
                            step=0.05,
                            label="🎯 Confidence Threshold"
                        )
                    
                    detect_btn = gr.Button("✨ Detect Clothing", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 16px; font-weight: 600; color: white; margin-bottom: 12px;'>🎯 Detection Results</div>")
                    img_output = gr.Image(
                        type="numpy", 
                        label="",
                        height=400,
                        interactive=False
                    )
                    results_html = gr.HTML(value=create_empty_results())
            
            detect_btn.click(detect_image, [img_input, conf_slider], [img_output, results_html])
            img_input.change(detect_image, [img_input, conf_slider], [img_output, results_html])
        
        # Webcam Tab
        with gr.Tab("📹 Live Camera", id="webcam"):
            gr.HTML("""
            <div style="text-align: center; padding: 16px; color: rgba(255,255,255,0.7);">
                Point your camera at clothing items for real-time detection
            </div>
            """)
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 16px; font-weight: 600; color: white; margin-bottom: 12px;'>📷 Camera Input</div>")
                    cam_input = gr.Image(
                        sources=["webcam"], 
                        type="numpy", 
                        label="",
                        height=400,
                        streaming=True
                    )
                    cam_conf = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.25, 
                        step=0.05,
                        label="🎯 Confidence Threshold"
                    )
                
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 16px; font-weight: 600; color: white; margin-bottom: 12px;'>🎯 Live Detection</div>")
                    cam_output = gr.Image(
                        type="numpy", 
                        label="",
                        height=400,
                        interactive=False
                    )
            
            cam_input.stream(detect_webcam, [cam_input, cam_conf], [cam_output], stream_every=0.1, time_limit=300)
    
    # Class Tags
    gr.HTML("""
    <div class="class-tags">
        <span class="class-tag"><span class="class-dot" style="background: #f87171;"></span> T-Shirt</span>
        <span class="class-tag"><span class="class-dot" style="background: #34d399;"></span> Dress</span>
        <span class="class-tag"><span class="class-dot" style="background: #60a5fa;"></span> Jacket</span>
        <span class="class-tag"><span class="class-dot" style="background: #a78bfa;"></span> Pants</span>
        <span class="class-tag"><span class="class-dot" style="background: #fbbf24;"></span> Shirt</span>
        <span class="class-tag"><span class="class-dot" style="background: #f472b6;"></span> Shorts</span>
        <span class="class-tag"><span class="class-dot" style="background: #38bdf8;"></span> Skirt</span>
        <span class="class-tag"><span class="class-dot" style="background: #fb923c;"></span> Sweater</span>
    </div>
    """)
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <div class="footer-brand">👕 FashionAI</div>
        <div class="footer-text">AI-powered clothing detection for modern applications</div>
        <div class="tech-badges">
            <span class="tech-badge">🤖 YOLOv8</span>
            <span class="tech-badge">🔥 PyTorch</span>
            <span class="tech-badge">🚀 Gradio</span>
        </div>
        <div style="margin-top: 16px; font-size: 12px; color: rgba(255,255,255,0.3);">
            © 2024 FashionAI. Built with ❤️ for AI enthusiasts.
        </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
