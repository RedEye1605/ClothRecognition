"""
Dataset Preparation Script
==========================
Download and prepare dataset from Roboflow for YOLOv8 training.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import yaml

# ============================================
# Configuration
# ============================================
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================
# Roboflow Dataset Download
# ============================================

def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int = 1,
    output_dir: Optional[str] = None
) -> str:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
        output_dir: Output directory
    
    Returns:
        Path to downloaded dataset
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Please install roboflow: pip install roboflow")
        return None
    
    print(f"📥 Downloading from Roboflow...")
    print(f"   Workspace: {workspace}")
    print(f"   Project: {project}")
    print(f"   Version: {version}")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=output_dir)
    
    print(f"✅ Downloaded to: {dataset.location}")
    return dataset.location


# ============================================
# Dataset Utilities
# ============================================

def verify_dataset_structure(dataset_path: str) -> dict:
    """
    Verify YOLO dataset structure.
    
    Args:
        dataset_path: Path to dataset
    
    Returns:
        dict with dataset info
    """
    dataset_path = Path(dataset_path)
    
    # Check for data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml not found in {dataset_path}")
        return None
    
    # Load config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Count images
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split / 'images'
        if split_dir.exists():
            images = list(split_dir.glob('*'))
            splits[split] = len(images)
        else:
            splits[split] = 0
    
    info = {
        'path': str(dataset_path),
        'classes': config.get('names', []),
        'num_classes': config.get('nc', len(config.get('names', []))),
        'splits': splits
    }
    
    print(f"\n📊 Dataset Info:")
    print(f"   Path: {info['path']}")
    print(f"   Classes: {info['num_classes']}")
    print(f"   Train images: {splits.get('train', 0)}")
    print(f"   Validation images: {splits.get('valid', 0)}")
    print(f"   Test images: {splits.get('test', 0)}")
    
    return info


def create_dataset_yaml(
    output_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    class_names: list
) -> str:
    """
    Create data.yaml for YOLO training.
    
    Args:
        output_path: Path to save data.yaml
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images
        class_names: List of class names
    
    Returns:
        Path to created file
    """
    config = {
        'path': str(Path(output_path).parent),
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created: {output_path}")
    return str(output_path)


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
):
    """
    Split dataset into train/val/test.
    
    Args:
        images_dir: Directory with images
        labels_dir: Directory with labels
        output_dir: Output directory
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
    """
    import random
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Get all images
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in extensions]
    
    # Shuffle
    random.shuffle(images)
    
    # Calculate split indices
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': images[:train_end],
        'valid': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"\n📂 Splitting {n} images:")
    
    for split_name, split_images in splits.items():
        # Create directories
        img_dir = output_dir / split_name / 'images'
        lbl_dir = output_dir / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_path in split_images:
            # Copy image
            shutil.copy(img_path, img_dir / img_path.name)
            
            # Copy label
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy(label_path, lbl_dir / label_path.name)
        
        print(f"   {split_name}: {len(split_images)} images")
    
    print(f"\n✅ Split complete: {output_dir}")


# ============================================
# Quick Setup Functions
# ============================================

def setup_clothing_dataset(api_key: str = None):
    """
    Quick setup for clothing detection dataset.
    
    Args:
        api_key: Roboflow API key (optional)
    """
    print("👕 Setting up Clothing Detection Dataset\n")
    
    # Default clothing classes
    CLOTHING_CLASSES = [
        "shirt", "pants", "dress", "jacket", "skirt",
        "sweater", "shorts", "coat", "hat", "shoes"
    ]
    
    if api_key:
        # Download from Roboflow
        print("📥 Option 1: Download from Roboflow")
        dataset_path = download_from_roboflow(
            api_key=api_key,
            workspace="roboflow-100",
            project="apparel-detection",
            version=1,
            output_dir=str(PROCESSED_DIR)
        )
        
        if dataset_path:
            verify_dataset_structure(dataset_path)
            return dataset_path
    
    else:
        print("📋 Option 2: Manual Setup (Recommended)")
        print("\nTo set up the dataset manually:")
        print("1. Go to https://universe.roboflow.com")
        print("2. Search for 'clothing detection'")
        print("3. Download in YOLOv8 format")
        print("4. Extract to data/processed/")
        return None
    
    return None


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for Cloth Detection")
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Roboflow API key for dataset download"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="roboflow-100",
        help="Roboflow workspace name"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="apparel-detection",
        help="Roboflow project name"
    )
    
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Path to existing dataset to verify"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset_structure(args.verify)
    elif args.api_key:
        download_from_roboflow(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=str(PROCESSED_DIR)
        )
    else:
        setup_clothing_dataset()
