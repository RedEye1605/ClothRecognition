"""
👕 FashionAI - Gradio App for Hugging Face
==========================================
AI-powered clothing detection using YOLOv8.

Classes (8): Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
"""

import gradio as gr
import numpy as np
import cv2

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

CLASS_COLORS = {
    "tshirt": "#FF6B6B", "t-shirt": "#FF6B6B",
    "dress": "#45B7D1", "jacket": "#96CEB4",
    "pants": "#4ECDC4", "shirt": "#FFEAA7",
    "short": "#98D8C8", "shorts": "#98D8C8",
    "skirt": "#DDA0DD", "sweater": "#F7DC6F"
}

CLASS_NAMES = ["Tshirt", "Dress", "Jacket", "Pants", "Shirt", "Short", "Skirt", "Sweater"]

# ============================================
# Load Model
# ============================================
print("🔄 Loading model...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except:
    print("⚠️ Custom model not found, using pretrained")
    model = YOLO("yolov8n.pt")

print(f"📋 Classes: {list(model.names.values())}")

# ============================================
# Detection Functions
# ============================================
def detect(image, conf):
    """Detect clothing in image"""
    if image is None:
        return None, ""
    
    results = model.predict(image, conf=conf, iou=0.45, verbose=False)
    annotated = results[0].plot(linewidth=3, font_size=12)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Build results HTML
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detections.append((name, confidence))
    
    if not detections:
        return annotated_rgb, "<div style='text-align:center;color:#888;padding:20px;'>No clothing detected</div>"
    
    html = ""
    for name, conf in detections:
        color = CLASS_COLORS.get(name.lower(), "#888")
        html += f"""
        <div style='display:flex;align-items:center;gap:12px;padding:10px;margin:6px 0;background:rgba(255,255,255,0.05);border-radius:10px;'>
            <div style='width:12px;height:12px;border-radius:50%;background:{color};'></div>
            <span style='flex:1;font-weight:600;'>{name}</span>
            <span style='background:rgba(255,255,255,0.1);padding:4px 10px;border-radius:12px;'>{conf:.0%}</span>
        </div>
        """
    
    return annotated_rgb, html

def detect_webcam(frame, conf):
    """Process webcam frame"""
    if frame is None:
        return None
    
    results = model.predict(frame, conf=conf, iou=0.45, verbose=False)
    annotated = results[0].plot(linewidth=2, font_size=10)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ============================================
# CSS Theme
# ============================================
css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
body, .gradio-container { font-family: 'Plus Jakarta Sans', sans-serif !important; background: #0b0f19 !important; }
.title { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
.subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.btn-primary { background: linear-gradient(135deg, #6366f1, #4f46e5) !important; border: none !important; }
"""

# ============================================
# Interface
# ============================================
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    
    gr.HTML("<h1 class='title'>FashionAI</h1><p class='subtitle'>AI Clothing Detection • 8 Classes</p>")
    
    with gr.Tabs():
        # Upload Tab
        with gr.Tab("🖼️ Upload"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="numpy", label="Input", height=400)
                    conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                    btn = gr.Button("✨ Detect", variant="primary")
                with gr.Column():
                    img_out = gr.Image(type="numpy", label="Result", height=400)
                    results = gr.HTML()
            
            btn.click(detect, [img_in, conf], [img_out, results])
        
        # Webcam Tab
        with gr.Tab("📹 Live Cam"):
            with gr.Row():
                with gr.Column():
                    cam_in = gr.Image(sources=["webcam"], type="numpy", label="Camera", height=400)
                    cam_conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                with gr.Column():
                    cam_out = gr.Image(type="numpy", label="Output", height=400)
            
            cam_in.stream(detect_webcam, [cam_in, cam_conf], [cam_out], stream_every=0.1, time_limit=60)
    
    gr.HTML(f"<div style='text-align:center;padding:20px;color:#666;'>Classes: {', '.join(CLASS_NAMES)}</div>")

if __name__ == "__main__":
    demo.launch()
