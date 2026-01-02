# Cloth Recognition - YOLOv8 Training Script
# ==========================================
# Run this script locally if you have a GPU

from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Please install ultralytics")
    print("Run: pip install ultralytics")
    exit(1)

# ============================================
# Configuration
# ============================================
CONFIG = {
    # Model
    'model': 'yolov8n.pt',  # nano (fastest) | yolov8s.pt | yolov8m.pt
    
    # Data
    'data': 'src/training/config.yaml',
    
    # Training
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'patience': 10,
    
    # Device
    'device': 0,  # GPU index or 'cpu'
    
    # Optimization
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.01,
    
    # Augmentation
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.1,
    
    # Output
    'project': 'runs/train',
    'name': 'cloth_detection',
    'exist_ok': True,
    'save': True,
    'plots': True
}

# ============================================
# Training Functions
# ============================================

def validate_config():
    """Validate training configuration."""
    # Check data config exists
    if not Path(CONFIG['data']).exists():
        print(f"❌ Data config not found: {CONFIG['data']}")
        print("   Please create the config file or download dataset first.")
        return False
    
    # Load and validate data config
    with open(CONFIG['data'], 'r') as f:
        data = yaml.safe_load(f)
    
    if 'nc' not in data or 'names' not in data:
        print("❌ Invalid data config: missing 'nc' or 'names'")
        return False
    
    print(f"✅ Data config valid: {data['nc']} classes")
    return True


def train():
    """Run training."""
    print("=" * 50)
    print("👕 Cloth Recognition - YOLOv8 Training")
    print("=" * 50)
    
    # Validate
    if not validate_config():
        return
    
    # Load model
    print(f"\n📦 Loading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    # Training info
    print(f"\n🎯 Training Configuration:")
    print(f"   - Epochs: {CONFIG['epochs']}")
    print(f"   - Image Size: {CONFIG['imgsz']}")
    print(f"   - Batch Size: {CONFIG['batch']}")
    print(f"   - Device: {CONFIG['device']}")
    
    # Train
    print("\n🚀 Starting training...\n")
    
    results = model.train(
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        patience=CONFIG['patience'],
        device=CONFIG['device'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        augment=CONFIG['augment'],
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=CONFIG['exist_ok'],
        save=CONFIG['save'],
        plots=CONFIG['plots'],
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("🎉 Training Complete!")
    print("=" * 50)
    
    # Results
    weights_dir = Path(CONFIG['project']) / CONFIG['name'] / 'weights'
    print(f"\n📁 Model saved to: {weights_dir}")
    print(f"   - best.pt: Best model weights")
    print(f"   - last.pt: Final model weights")
    
    return results


def evaluate(model_path: str = None):
    """Evaluate trained model."""
    if model_path is None:
        model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📊 Evaluating: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=CONFIG['data'])
    
    print("\n📈 Results:")
    print(f"   - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   - Precision: {metrics.box.mp:.4f}")
    print(f"   - Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export(model_path: str = None, format: str = 'onnx'):
    """Export model to different formats."""
    if model_path is None:
        model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Exporting: {model_path}")
    print(f"   Format: {format}")
    
    model = YOLO(model_path)
    exported = model.export(format=format)
    
    print(f"\n✅ Exported to: {exported}")
    return exported


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Cloth Detection")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Evaluate model')
    parser.add_argument('--export', type=str, help='Export format (onnx, torchscript, etc.)')
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    
    args = parser.parse_args()
    
    # Override config
    if args.epochs:
        CONFIG['epochs'] = args.epochs
    if args.batch:
        CONFIG['batch'] = args.batch
    
    if args.train:
        train()
    elif args.eval:
        evaluate(args.model)
    elif args.export:
        export(args.model, args.export)
    else:
        # Default: run training
        train()
