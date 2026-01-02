"""
Model Evaluation Script
=======================
Evaluate trained YOLOv8 model performance with detailed metrics.
"""

import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# ============================================
# Configuration
# ============================================
DEFAULT_MODEL = "models/best.pt"
DATA_CONFIG = "src/training/config.yaml"

# ============================================
# Evaluation Functions
# ============================================

def evaluate_model(
    model_path: str = DEFAULT_MODEL,
    data_config: str = DATA_CONFIG,
    save_results: bool = True
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model
        data_config: Path to data.yaml
        save_results: Save results to JSON
    
    Returns:
        dict: Evaluation metrics
    """
    print("=" * 50)
    print("📊 Model Evaluation - Cloth Recognition")
    print("=" * 50)
    
    # Load model
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print("\n🔄 Running evaluation...")
    metrics = model.val(data=data_config, verbose=False)
    
    # Extract metrics
    results = {
        "model": model_path,
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": {}
    }
    
    # Per-class metrics
    class_names = model.names
    for i, (p, r, ap50, ap) in enumerate(zip(
        metrics.box.p, metrics.box.r, 
        metrics.box.ap50, metrics.box.ap
    )):
        class_name = class_names.get(i, f"class_{i}")
        results["per_class"][class_name] = {
            "precision": float(p),
            "recall": float(r),
            "AP50": float(ap50),
            "AP50-95": float(ap)
        }
    
    # Print results
    print("\n" + "=" * 50)
    print("📈 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   mAP@0.5:      {results['metrics']['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {results['metrics']['mAP50-95']:.4f}")
    print(f"   Precision:    {results['metrics']['precision']:.4f}")
    print(f"   Recall:       {results['metrics']['recall']:.4f}")
    
    print(f"\n📋 Per-Class Performance:")
    print("-" * 45)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'AP50':<12}")
    print("-" * 45)
    
    for class_name, class_metrics in results["per_class"].items():
        print(f"{class_name:<15} {class_metrics['precision']:.4f}       "
              f"{class_metrics['recall']:.4f}       {class_metrics['AP50']:.4f}")
    
    # Save results
    if save_results:
        output_path = Path("evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return results


def plot_metrics(results: dict, save_path: str = "evaluation_plot.png"):
    """
    Plot evaluation metrics as bar chart.
    
    Args:
        results: Evaluation results dict
        save_path: Path to save plot
    """
    if not results or "per_class" not in results:
        print("❌ Invalid results for plotting")
        return
    
    # Prepare data
    classes = list(results["per_class"].keys())
    ap50_values = [results["per_class"][c]["AP50"] for c in classes]
    precision_values = [results["per_class"][c]["precision"] for c in classes]
    recall_values = [results["per_class"][c]["recall"] for c in classes]
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AP50 bar chart
    axes[0].barh(classes, ap50_values, color='#667eea')
    axes[0].set_xlabel('AP@0.5')
    axes[0].set_title('Average Precision (AP@0.5)')
    axes[0].set_xlim(0, 1)
    
    # Precision bar chart
    axes[1].barh(classes, precision_values, color='#4ECDC4')
    axes[1].set_xlabel('Precision')
    axes[1].set_title('Precision per Class')
    axes[1].set_xlim(0, 1)
    
    # Recall bar chart
    axes[2].barh(classes, recall_values, color='#FF6B6B')
    axes[2].set_xlabel('Recall')
    axes[2].set_title('Recall per Class')
    axes[2].set_xlim(0, 1)
    
    plt.suptitle('🎯 Cloth Recognition Model Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")
    plt.close()


def compare_models(model_paths: list, data_config: str = DATA_CONFIG):
    """
    Compare multiple models.
    
    Args:
        model_paths: List of model paths to compare
        data_config: Path to data.yaml
    """
    print("=" * 50)
    print("📊 Model Comparison")
    print("=" * 50)
    
    results = []
    for model_path in model_paths:
        if Path(model_path).exists():
            model = YOLO(model_path)
            metrics = model.val(data=data_config, verbose=False)
            results.append({
                "model": model_path,
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
            })
    
    # Print comparison
    print("\n📋 Comparison Results:")
    print("-" * 60)
    print(f"{'Model':<30} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15}")
    print("-" * 60)
    
    for r in results:
        model_name = Path(r['model']).name
        print(f"{model_name:<30} {r['mAP50']:.4f}          {r['mAP50-95']:.4f}")
    
    return results


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Cloth Detection Model")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, 
                       help="Path to model")
    parser.add_argument("--data", "-d", type=str, default=DATA_CONFIG,
                       help="Path to data config")
    parser.add_argument("--plot", "-p", action="store_true",
                       help="Generate evaluation plots")
    parser.add_argument("--compare", "-c", nargs="+",
                       help="Compare multiple models")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.data)
    else:
        results = evaluate_model(args.model, args.data)
        if results and args.plot:
            plot_metrics(results)
