#!/usr/bin/env python3
"""
Download and Prepare CIFAR-10 Dataset from Hugging Face for Federated Learning.

This script:
1. Downloads CIFAR-10 using Hugging Face `datasets` (uoft-cs/cifar10).
2. Exports images into labeled class directories (airplane, automobile, etc.).
3. Splits data into Client 1 (for Raspberry Pi) and Client 2 (for Mac).
"""

import os
import argparse
from pathlib import Path
from PIL import Image

def download_and_export_cifar10(output_dir: str, num_per_class: int = 500, client1_ratio: float = 0.5):
    """
    Downloads CIFAR-10 from Hugging Face and saves images into class folders.
    
    Args:
        output_dir: Path where client1_data and client2_data will be created.
        num_per_class: Number of images per class to export (default: 500, or -1 for all 5,000 per class).
        client1_ratio: Ratio of images allocated to Client 1 (default: 0.5 for 50-50 split).
    """
    print("⏳ Loading CIFAR-10 from Hugging Face ('uoft-cs/cifar10')...")
    from datasets import load_dataset
    
    # Load dataset from Hugging Face
    ds = load_dataset("uoft-cs/cifar10")
    train_data = ds["train"]
    
    # Get class names from Hugging Face features
    class_names = train_data.features["label"].names
    print(f"✅ Loaded CIFAR-10 successfully! Found {len(class_names)} classes:")
    for idx, c in enumerate(class_names):
        print(f"   {idx}: {c}")
    
    base_path = Path(output_dir).expanduser().resolve()
    client1_dir = base_path / "client1_data"
    client2_dir = base_path / "client2_data"
    
    # Create directories for all 10 classes in both clients
    for c in class_names:
        (client1_dir / c).mkdir(parents=True, exist_ok=True)
        (client2_dir / c).mkdir(parents=True, exist_ok=True)
    
    # Organize dataset by class
    class_indices = {c: [] for c in class_names}
    for idx, item in enumerate(train_data):
        c_name = class_names[item["label"]]
        class_indices[c_name].append(idx)
    
    print(f"\n📂 Exporting images to {base_path}...")
    total_c1, total_c2 = 0, 0
    
    for c_name in class_names:
        indices = class_indices[c_name]
        if num_per_class > 0:
            indices = indices[:num_per_class]
            
        n_total = len(indices)
        n_c1 = int(n_total * client1_ratio)
        c1_indices = indices[:n_c1]
        c2_indices = indices[n_c1:]
        
        # Save Client 1 images
        for i, data_idx in enumerate(c1_indices):
            img: Image.Image = train_data[data_idx]["img"]
            img.save(client1_dir / c_name / f"{c_name}_{i:05d}.png")
            
        # Save Client 2 images
        for i, data_idx in enumerate(c2_indices):
            img: Image.Image = train_data[data_idx]["img"]
            img.save(client2_dir / c_name / f"{c_name}_{i:05d}.png")
            
        total_c1 += len(c1_indices)
        total_c2 += len(c2_indices)
        print(f"   Class '{c_name}': {len(c1_indices)} -> Client 1, {len(c2_indices)} -> Client 2")
        
    print("\n" + "=" * 60)
    print("🎉 CIFAR-10 DATASET PREPARATION COMPLETED!")
    print("=" * 60)
    print(f"Client 1 (Raspberry Pi): {total_c1} images -> {client1_dir}")
    print(f"Client 2 (Mac):          {total_c2} images -> {client2_dir}")
    print(f"Total exported:          {total_c1 + total_c2} images across 10 classes")
    print("=" * 60)
    print("\nTo run federated learning with CIFAR-10, pass:")
    print(f"  --data_dir {client1_dir} --num_classes 10\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CIFAR-10 from Hugging Face for FL")
    parser.add_argument("--output_dir", default="./data/cifar10",
                        help="Target output directory for client datasets (default: ./data/cifar10)")
    parser.add_argument("--num_per_class", type=int, default=500,
                        help="Number of images per class (default: 500 = 5,000 total; use -1 for all 50,000)")
    parser.add_argument("--client1_ratio", type=float, default=0.5,
                        help="Ratio of images for Client 1 (default: 0.5)")
    args = parser.parse_args()
    
    download_and_export_cifar10(args.output_dir, args.num_per_class, args.client1_ratio)

