import os
import shutil
import random
from pathlib import Path
import argparse


def split_flower_dataset(source_dir, output_dir, client1_ratio=0.5, seed=42):
    """
    Split flower dataset between two clients while maintaining class distribution.
    
    Args:
        source_dir: Path to flower_photos directory
        output_dir: Path where client datasets will be created
        client1_ratio: Ratio of data for client 1 (default: 0.5 for 50-50 split)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)
    
    # Create output directories
    client1_dir = Path(output_dir).expanduser().resolve() / "client1_data"
    client2_dir = Path(output_dir).expanduser().resolve() / "client2_data"
    
    # Remove existing directories if they exist
    if client1_dir.exists():
        shutil.rmtree(client1_dir)
    if client2_dir.exists():
        shutil.rmtree(client2_dir)
    
    client1_dir.mkdir(parents=True, exist_ok=True)
    client2_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir).expanduser().resolve()
    
    # Get all flower classes (subdirectories)
    flower_classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(flower_classes)} flower classes:")
    for flower_class in flower_classes:
        print(f"  - {flower_class.name}")
    
    total_client1_images = 0
    total_client2_images = 0
    
    # Split each class between clients
    for flower_class in flower_classes:
        class_name = flower_class.name
        
        # Create class directories for both clients
        client1_class_dir = client1_dir / class_name
        client2_class_dir = client2_dir / class_name
        client1_class_dir.mkdir(exist_ok=True)
        client2_class_dir.mkdir(exist_ok=True)
        
        # Get all images in this class
        image_files = list(flower_class.glob("*.jpg"))
        random.shuffle(image_files)  # Shuffle for random distribution
        
        # Split images between clients
        num_client1 = int(len(image_files) * client1_ratio)
        client1_images = image_files[:num_client1]
        client2_images = image_files[num_client1:]
        
        # Copy images to client directories
        for img in client1_images:
            shutil.copy2(img, client1_class_dir / img.name)
        
        for img in client2_images:
            shutil.copy2(img, client2_class_dir / img.name)
        
        total_client1_images += len(client1_images)
        total_client2_images += len(client2_images)
        
        print(f"Class '{class_name}': {len(client1_images)} → Client1, {len(client2_images)} → Client2")
    
    print(f"\nDataset split completed:")
    print(f"  Client 1 (Raspberry Pi): {total_client1_images} images")
    print(f"  Client 2 (Mac): {total_client2_images} images")
    print(f"  Total: {total_client1_images + total_client2_images} images")
    
    print(f"\nClient datasets created at:")
    print(f"  {client1_dir}")
    print(f"  {client2_dir}")
    
    return client1_dir, client2_dir


def create_iid_split(source_dir, output_dir, seed=42):
    """Create IID (Independent and Identically Distributed) split - balanced across clients."""
    print("Creating IID split (balanced distribution)...")
    return split_flower_dataset(source_dir, output_dir, client1_ratio=0.5, seed=seed)


def create_non_iid_split(source_dir, output_dir, seed=42):
    """Create Non-IID split - imbalanced distribution to simulate real-world scenario."""
    print("Creating Non-IID split (imbalanced distribution)...")
    
    source_path = Path(source_dir)
    client1_dir = Path(output_dir) / "client1_data"
    client2_dir = Path(output_dir) / "client2_data"
    
    # Remove existing directories if they exist
    if client1_dir.exists():
        shutil.rmtree(client1_dir)
    if client2_dir.exists():
        shutil.rmtree(client2_dir)
    
    client1_dir.mkdir(parents=True, exist_ok=True)
    client2_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all flower classes
    flower_classes = [d for d in source_path.iterdir() if d.is_dir()]
    flower_classes.sort()  # Sort for consistent ordering
    
    random.seed(seed)
    
    # Assign different ratios to each class to create non-IID distribution
    class_ratios = [0.8, 0.2, 0.7, 0.3, 0.6]  # Client1 ratios for each class
    
    total_client1_images = 0
    total_client2_images = 0
    
    for i, flower_class in enumerate(flower_classes):
        class_name = flower_class.name
        client1_ratio = class_ratios[i % len(class_ratios)]
        
        # Create class directories
        client1_class_dir = client1_dir / class_name
        client2_class_dir = client2_dir / class_name
        client1_class_dir.mkdir(exist_ok=True)
        client2_class_dir.mkdir(exist_ok=True)
        
        # Get all images in this class
        image_files = list(flower_class.glob("*.jpg"))
        random.shuffle(image_files)
        
        # Split images with different ratios
        num_client1 = int(len(image_files) * client1_ratio)
        client1_images = image_files[:num_client1]
        client2_images = image_files[num_client1:]
        
        # Copy images
        for img in client1_images:
            shutil.copy2(img, client1_class_dir / img.name)
        
        for img in client2_images:
            shutil.copy2(img, client2_class_dir / img.name)
        
        total_client1_images += len(client1_images)
        total_client2_images += len(client2_images)
        
        print(f"Class '{class_name}': {len(client1_images)} → Client1 ({client1_ratio:.1%}), {len(client2_images)} → Client2")
    
    print(f"\nNon-IID split completed:")
    print(f"  Client 1 (Raspberry Pi): {total_client1_images} images")
    print(f"  Client 2 (Mac): {total_client2_images} images")
    
    return client1_dir, client2_dir


def verify_split(client1_dir, client2_dir):
    """Verify the dataset split and show statistics."""
    print("\n" + "="*50)
    print("DATASET SPLIT VERIFICATION")
    print("="*50)
    
    for client_name, client_dir in [("Client 1 (Raspberry Pi)", client1_dir), ("Client 2 (Mac)", client2_dir)]:
        print(f"\n{client_name}:")
        print(f"  Location: {client_dir}")
        
        total_images = 0
        for class_dir in Path(client_dir).iterdir():
            if class_dir.is_dir():
                num_images = len(list(class_dir.glob("*.jpg")))
                total_images += num_images
                print(f"    {class_dir.name}: {num_images} images")
        
        print(f"  Total: {total_images} images")


def main():
    parser = argparse.ArgumentParser(description="Split flower dataset for federated learning")
    parser.add_argument("--source_dir", required=True, 
                       help="Path to flower_photos directory")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for client datasets")
    parser.add_argument("--split_type", choices=["iid", "non_iid"], default="iid",
                       help="Type of split: iid (balanced) or non_iid (imbalanced)")
    parser.add_argument("--client1_ratio", type=float, default=0.5,
                       help="Ratio of data for client 1 (only for iid split)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Verify source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split dataset
    if args.split_type == "iid":
        client1_dir, client2_dir = create_iid_split(args.source_dir, args.output_dir, args.seed)
    else:
        client1_dir, client2_dir = create_non_iid_split(args.source_dir, args.output_dir, args.seed)
    
    # Verify split
    verify_split(client1_dir, client2_dir)
    
    print(f"\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print(f"1. Copy {client1_dir} to your Raspberry Pi")
    print(f"2. Use {client2_dir} on your Mac")
    print(f"3. In federated learning:")
    print(f"   - Raspberry Pi: --data_dir {client1_dir}")
    print(f"   - Mac: --data_dir {client2_dir}")


if __name__ == "__main__":
    main()