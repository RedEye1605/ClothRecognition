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
DETECTION_MODEL_PATH = "cloth_classifier.pt"
COLOR_MODEL_PATH = "color_classifier.pt"

CLASS_CONFIG = {
    "tshirt": {"color": "#FF6B6B", "label": "T-Shirt", "icon": "👕"},
    "t-shirt": {"color": "#FF6B6B", "label": "T-Shirt", "icon": "👕"},
    "dress": {"color": "#45B7D1", "label": "Dress", "icon": "👗"},
    "jacket": {"color": "#96CEB4", "label": "Jacket", "icon": "🧥"},
    "pants": {"color": "#4ECDC4", "label": "Pants", "icon": "👖"},
    "shirt": {"color": "#FFEAA7", "label": "Shirt", "icon": "👔"},
    "short": {"color": "#98D8C8", "label": "Shorts", "icon": "🩳"},
    "shorts": {"color": "#98D8C8", "label": "Shorts", "icon": "🩳"},
    "skirt": {"color": "#DDA0DD", "label": "Skirt", "icon": "🎽"},
    "sweater": {"color": "#F7DC6F", "label": "Sweater", "icon": "🧶"}
}

COLOR_CONFIG = {
    "beige": {"hex": "#D4A574", "label": "Beige"},
    "black": {"hex": "#2D2D2D", "label": "Black"},
    "blue": {"hex": "#3B82F6", "label": "Blue"},
    "gray": {"hex": "#6B7280", "label": "Gray"},
    "grey": {"hex": "#6B7280", "label": "Gray"},
    "green": {"hex": "#22C55E", "label": "Green"},
    "pattren": {"hex": "#8B5CF6", "label": "Pattern"},
    "pattern": {"hex": "#8B5CF6", "label": "Pattern"},
    "red": {"hex": "#EF4444", "label": "Red"},
    "white": {"hex": "#F9FAFB", "label": "White"}
}

CLASSES = ["T-Shirt", "Dress", "Jacket", "Pants", "Shirt", "Shorts", "Skirt", "Sweater"]
COLORS = ["Beige", "Black", "Blue", "Gray", "Green", "Pattern", "Red", "White"]

# ============================================
# Load Models
# ============================================
print("🔄 Loading detection model...")
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
    print(f"✅ Detection model loaded: {DETECTION_MODEL_PATH}")
    detection_status = "✅ Detection Model Loaded"
except Exception as e:
    print(f"⚠️ Using pretrained detection model: {e}")
    detection_model = YOLO("yolov8n.pt")
    detection_status = "⚠️ Using Pretrained Detection Model"

print(f"📋 Detection Classes: {list(detection_model.names.values())}")

print("🔄 Loading color model...")
try:
    color_model = YOLO(COLOR_MODEL_PATH)
    print(f"✅ Color model loaded: {COLOR_MODEL_PATH}")
    color_status = "✅ Color Model Loaded"
    print(f"🎨 Color Classes: {list(color_model.names.values())}")
except Exception as e:
    print(f"⚠️ Color model not available: {e}")
    color_model = None
    color_status = "⚠️ Color Model Not Available"

def classify_color(image_crop):
    """Classify color of a clothing crop using PIL Image"""
    if color_model is None:
        return None, 0.0, None
    
    try:
        from PIL import Image
        # Convert numpy array to PIL Image if needed
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        
        results = color_model.predict(image_crop, verbose=False)
        probs = results[0].probs
        
        color_name = results[0].names[probs.top1]
        color_conf = float(probs.top1conf)
        color_hex = COLOR_CONFIG.get(color_name.lower(), {}).get("hex", "#808080")
        
        return color_name, color_conf, color_hex
    except Exception as e:
        print(f"⚠️ Color classification error: {e}")
        return None, 0.0, None

def format_label(color, class_name):
    """Format label as 'Color ClassName' with proper display names."""
    # Mapping model output to display names
    class_mapping = {
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
    
    display_class = class_mapping.get(class_name.lower(), class_name.title())
    
    if color:
        color_display = "Pattern" if color.lower() == "pattren" else color.capitalize()
        return f"{color_display} {display_class}"
    return display_class
# ============================================
# Detection Functions
# ============================================
def hex_to_bgr(hex_color):
    """Convert hex color to BGR for OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def draw_custom_boxes(image, detections):
    """Draw custom styled bounding boxes with color labels"""
    annotated = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color_hex = det.get('colorHex') or det['classColor']
        color = hex_to_bgr(color_hex)
        label = det['label']
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
        
        # Text color - white for dark backgrounds, black for light
        text_color = (0, 0, 0) if det.get('color') in ['white', 'beige'] else (255, 255, 255)
        cv2.putText(annotated, label, (x1 + 5, y1 - 8), font, font_scale, text_color, thickness)
    
    return annotated

def detect_image(image, confidence):
    """Detect clothing in uploaded image with color classification"""
    if image is None:
        return None, create_empty_results()
    
    # Stage 1: Detect clothing
    results = detection_model.predict(image, conf=confidence, iou=0.45, verbose=False)
    
    # Stage 2: Classify color for each detection
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        config = CLASS_CONFIG.get(cls_name, {"color": "#888888", "label": cls_name.title(), "icon": "👕"})
        
        # Classify color
        color_name, color_conf, color_hex = None, 0.0, None
        if color_model and (x2 - x1) > 10 and (y2 - y1) > 10:
            # Ensure bounds
            h, w = image.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            crop = image[cy1:cy2, cx1:cx2]
            color_name, color_conf, color_hex = classify_color(crop)
        
        display_label = format_label(color_name, config['label']) if color_name else config['label']
        
        detections.append({
            'className': cls_name,
            'classLabel': config['label'],
            'classColor': config['color'],
            'icon': config.get('icon', '👕'),
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'color': color_name,
            'colorConfidence': color_conf,
            'colorHex': color_hex,
            'label': f"{display_label} {conf:.0%}"
        })
    
    # Draw boxes with color info
    annotated = draw_custom_boxes(image, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if len(annotated.shape) == 3 else annotated
    
    # Build results HTML
    results_html = create_results_html_v2(detections)
    
    return annotated_rgb, results_html

def detect_webcam(frame, confidence):
    """Process webcam frame with color classification"""
    if frame is None:
        return None
    
    # Stage 1: Detect clothing
    results = detection_model.predict(frame, conf=confidence, iou=0.45, verbose=False)
    
    # Stage 2: Classify color for each detection  
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        config = CLASS_CONFIG.get(cls_name, {"color": "#888888", "label": cls_name.title()})
        
        # Classify color
        color_name, color_conf, color_hex = None, 0.0, None
        if color_model and (x2 - x1) > 10 and (y2 - y1) > 10:
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            crop = frame[cy1:cy2, cx1:cx2]
            color_name, color_conf, color_hex = classify_color(crop)
        
        display_label = format_label(color_name, config['label']) if color_name else config['label']
        
        detections.append({
            'className': cls_name,
            'classLabel': config['label'],
            'classColor': config['color'],
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'color': color_name,
            'colorHex': color_hex,
            'label': f"{display_label} {conf:.0%}"
        })
    
    annotated = draw_custom_boxes(frame, detections)
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

def create_results_html_v2(detections):
    """Create styled results HTML with color info"""
    if not detections:
        return """
        <div style="text-align: center; padding: 40px 20px; color: rgba(255,255,255,0.6);">
            <div style="font-size: 40px; margin-bottom: 12px;">🔍</div>
            <div style="font-size: 16px; font-weight: 600;">No clothing detected</div>
            <div style="font-size: 13px; margin-top: 8px; color: rgba(255,255,255,0.4);">Try lowering the confidence threshold</div>
        </div>
        """
    
    total_conf = sum(d['confidence'] for d in detections)
    avg_conf = total_conf / len(detections) if detections else 0
    
    # Build HTML
    items_html = ""
    for det in detections:
        color_hex = det.get('colorHex') or det['classColor']
        color_name = det.get('color', '')
        display_label = det['label'].rsplit(' ', 1)[0]  # Remove the percentage from label
        icon = det.get('icon', '👕')
        
        # Color swatch HTML
        color_swatch = f'<span style="display: inline-block; width: 16px; height: 16px; background: {color_hex}; border-radius: 4px; box-shadow: 0 0 0 1px rgba(255,255,255,0.2);"></span>' if color_name else ''
        
        items_html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; 
                    padding: 12px 16px; background: rgba(255,255,255,0.05); 
                    border-radius: 10px; margin-bottom: 8px;
                    border-left: 3px solid {color_hex};">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">{icon}</span>
                {color_swatch}
                <span style="font-weight: 600; color: white;">{display_label}</span>
            </div>
            <span style="background: {color_hex}; color: {'#000' if color_name in ['white', 'beige'] else '#fff'}; padding: 4px 10px; 
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
        <div class="header-subtitle">AI-powered clothing detection with color classification using YOLOv8</div>
        
        <div class="feature-badges">
            <span class="feature-badge">🏷️ 8 Classes</span>
            <span class="feature-badge">🎨 8 Colors</span>
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
    <div style="text-align: center; margin-top: 20px;">
        <div style="font-size: 12px; color: rgba(255,255,255,0.5); margin-bottom: 8px;">CLOTHING TYPES</div>
        <div class="class-tags">
            <span class="class-tag">T-Shirt</span>
            <span class="class-tag">Dress</span>
            <span class="class-tag">Jacket</span>
            <span class="class-tag">Pants</span>
            <span class="class-tag">Shirt</span>
            <span class="class-tag">Shorts</span>
            <span class="class-tag">Skirt</span>
            <span class="class-tag">Sweater</span>
        </div>
        <div style="font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 16px; margin-bottom: 8px;">DETECTED COLORS</div>
        <div class="class-tags">
            <span class="class-tag"><span class="class-dot" style="background: #D4A574;"></span> Beige</span>
            <span class="class-tag"><span class="class-dot" style="background: #2D2D2D;"></span> Black</span>
            <span class="class-tag"><span class="class-dot" style="background: #3B82F6;"></span> Blue</span>
            <span class="class-tag"><span class="class-dot" style="background: #6B7280;"></span> Gray</span>
            <span class="class-tag"><span class="class-dot" style="background: #22C55E;"></span> Green</span>
            <span class="class-tag"><span class="class-dot" style="background: #8B5CF6;"></span> Pattern</span>
            <span class="class-tag"><span class="class-dot" style="background: #EF4444;"></span> Red</span>
            <span class="class-tag"><span class="class-dot" style="background: #F9FAFB; border: 1px solid rgba(255,255,255,0.3);"></span> White</span>
        </div>
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
