"""
Data Augmentation Pipeline for Cloth Detection
===============================================
Additional augmentation using Albumentations library.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import random

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️ Albumentations not installed. Run: pip install albumentations")


# ============================================
# Augmentation Transforms
# ============================================

def get_train_transforms(img_size: int = 640) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.7),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.3),
        
        # Quality
        A.OneOf([
            A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
        ], p=0.2),
        
        # Resize to target size
        A.Resize(img_size, img_size),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """
    Get validation transforms (resize only).
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


# ============================================
# Augmentation Functions
# ============================================

def augment_image(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    transform: A.Compose
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Apply augmentation to image and bounding boxes.
    
    Args:
        image: Input image (HxWxC)
        bboxes: List of bboxes in YOLO format [x_center, y_center, width, height]
        class_labels: List of class IDs
        transform: Albumentations transform
    
    Returns:
        Augmented image, bboxes, and labels
    """
    result = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    
    return (
        result['image'],
        result['bboxes'],
        result['class_labels']
    )


def load_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Load YOLO format labels from file.
    
    Args:
        label_path: Path to label file (.txt)
    
    Returns:
        Tuple of (bboxes, class_labels)
    """
    bboxes = []
    class_labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
    
    return bboxes, class_labels


def save_yolo_labels(
    label_path: str,
    bboxes: List[List[float]],
    class_labels: List[int]
):
    """
    Save labels in YOLO format.
    
    Args:
        label_path: Path to save label file
        bboxes: List of bboxes in YOLO format
        class_labels: List of class IDs
    """
    with open(label_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            line = f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


# ============================================
# Batch Augmentation
# ============================================

def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    num_augmentations: int = 3,
    img_size: int = 640
):
    """
    Augment entire dataset directory.
    
    Args:
        images_dir: Input images directory
        labels_dir: Input labels directory
        output_images_dir: Output images directory
        output_labels_dir: Output labels directory
        num_augmentations: Number of augmented copies per image
        img_size: Target image size
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("❌ Albumentations required for augmentation")
        return
    
    # Create output directories
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
    
    # Get transform
    transform = get_train_transforms(img_size)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = [f for f in Path(images_dir).iterdir() 
              if f.suffix.lower() in image_extensions]
    
    print(f"📂 Found {len(images)} images")
    print(f"🔄 Creating {num_augmentations} augmented copies each...")
    
    total_created = 0
    
    for img_path in images:
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        bboxes, class_labels = load_yolo_labels(str(label_path))
        
        # Copy original
        output_img_path = Path(output_images_dir) / img_path.name
        output_label_path = Path(output_labels_dir) / (img_path.stem + '.txt')
        
        cv2.imwrite(str(output_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        save_yolo_labels(str(output_label_path), bboxes, class_labels)
        
        # Create augmentations
        for i in range(num_augmentations):
            try:
                aug_image, aug_bboxes, aug_labels = augment_image(
                    image, bboxes, class_labels, transform
                )
                
                # Save augmented
                aug_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                aug_img_path = Path(output_images_dir) / aug_name
                aug_label_path = Path(output_labels_dir) / f"{img_path.stem}_aug{i+1}.txt"
                
                cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                save_yolo_labels(str(aug_label_path), aug_bboxes, aug_labels)
                
                total_created += 1
                
            except Exception as e:
                print(f"⚠️ Error augmenting {img_path.name}: {e}")
    
    print(f"\n✅ Created {total_created} augmented images")
    print(f"📁 Output: {output_images_dir}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Augmentation for Cloth Detection")
    parser.add_argument("--images", required=True, help="Input images directory")
    parser.add_argument("--labels", required=True, help="Input labels directory")
    parser.add_argument("--output-images", required=True, help="Output images directory")
    parser.add_argument("--output-labels", required=True, help="Output labels directory")
    parser.add_argument("--num-aug", type=int, default=3, help="Number of augmentations per image")
    parser.add_argument("--img-size", type=int, default=640, help="Target image size")
    
    args = parser.parse_args()
    
    augment_dataset(
        args.images,
        args.labels,
        args.output_images,
        args.output_labels,
        args.num_aug,
        args.img_size
    )
