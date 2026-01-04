"""
👕 FashionAI - Cloth Recognition
================================
Hugging Face Spaces Deployment
YOLOv8 clothing detection with Gradio UI
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

COLORS = {
    "tshirt": "#FF6B6B", "t-shirt": "#FF6B6B",
    "dress": "#4ECDC4", "jacket": "#45B7D1",
    "pants": "#96CEB4", "shirt": "#FFEAA7",
    "short": "#DDA0DD", "shorts": "#DDA0DD",
    "skirt": "#74B9FF", "sweater": "#FDCB6E"
}

CLASSES = ["Tshirt", "Dress", "Jacket", "Pants", "Shirt", "Short", "Skirt", "Sweater"]

# ============================================
# Load Model
# ============================================
print("🔄 Loading model...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Using pretrained model: {e}")
    model = YOLO("yolov8n.pt")

print(f"📋 Classes: {list(model.names.values())}")

# ============================================
# Detection Functions
# ============================================
def detect_image(image, confidence):
    """Detect clothing in uploaded image"""
    if image is None:
        return None, "Upload an image to start"
    
    results = model.predict(image, conf=confidence, iou=0.45, verbose=False)
    annotated = results[0].plot(linewidth=2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Build results text
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = results[0].names[cls_id]
        conf = float(box.conf[0])
        detections.append(f"• **{name}**: {conf:.0%}")
    
    if detections:
        text = f"## 🎯 Found {len(detections)} items\n\n" + "\n".join(detections)
    else:
        text = "## 🔍 No clothing detected\n\nTry lowering the confidence threshold."
    
    return annotated_rgb, text

def detect_webcam(frame, confidence):
    """Process webcam frame"""
    if frame is None:
        return None
    
    results = model.predict(frame, conf=confidence, iou=0.45, verbose=False)
    annotated = results[0].plot(linewidth=2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ============================================
# Gradio Interface
# ============================================
css = """
.gradio-container { font-family: 'Inter', sans-serif; }
h1 { text-align: center; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    
    gr.Markdown("# 👕 FashionAI - Cloth Recognition")
    gr.Markdown("AI-powered clothing detection • 8 Classes")
    
    with gr.Tabs():
        # Upload Tab
        with gr.Tab("🖼️ Upload Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Input", height=400)
                    conf_slider = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                    btn = gr.Button("✨ Detect", variant="primary")
                
                with gr.Column():
                    img_output = gr.Image(type="numpy", label="Result", height=400)
                    results_text = gr.Markdown("Upload an image to start")
            
            btn.click(detect_image, [img_input, conf_slider], [img_output, results_text])
            img_input.change(detect_image, [img_input, conf_slider], [img_output, results_text])
        
        # Webcam Tab
        with gr.Tab("📹 Webcam"):
            gr.Markdown("Point camera at clothing for real-time detection")
            
            with gr.Row():
                with gr.Column():
                    cam_input = gr.Image(sources=["webcam"], type="numpy", label="Camera", height=400)
                    cam_conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                
                with gr.Column():
                    cam_output = gr.Image(type="numpy", label="Output", height=400)
            
            cam_input.stream(detect_webcam, [cam_input, cam_conf], [cam_output], stream_every=0.1, time_limit=60)
    
    gr.Markdown(f"**Classes:** {', '.join(CLASSES)}")

if __name__ == "__main__":
    demo.launch()
