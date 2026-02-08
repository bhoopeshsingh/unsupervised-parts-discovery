import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import shutil

def prepare_imagenet_subset(
    output_dir: str,
    num_images_per_class: int = 5000,
    seed: int = 42,
    target_classes: list = None
):
    """
    Download and prepare ImageNet subset from Hugging Face.
    Dataset: benjamin-paine/imagenet-1k-128x128
    """
    print(f"Preparing data in {output_dir}...")
    output_path = Path(output_dir)
    
    # Define class mapping (ImageNet index -> our class name)
    full_class_mapping = {
        404: 'airplane',
        281: 'cat',
        282: 'cat',  # Combine tabby and tiger cat
        283: 'cat',
        284: 'cat',
        285: 'cat',
        # Dogs
        162: 'dog',  # Beagle
        207: 'dog',  # Golden retriever
        208: 'dog',  # Labrador retriever
        235: 'dog',  # German shepherd
        254: 'dog',  # Pug
        # Birds (New)
        11: 'bird', # Goldfinch
        12: 'bird', # House finch
        13: 'bird', # Junco
        14: 'bird', # Indigo bunting
        15: 'bird'  # Robin
    }

    # Filter classes if specified
    if target_classes:
        print(f"Filtering for classes: {target_classes}")
        class_mapping = {k: v for k, v in full_class_mapping.items() if v in target_classes}
        if not class_mapping:
            print(f"Warning: No valid indices found for classes {target_classes}")
            return
    else:
        class_mapping = full_class_mapping
    
    target_indices = set(class_mapping.keys())
    
    # Create directories
    for class_name in set(class_mapping.values()):
        (output_path / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (output_path / 'val' / class_name).mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset stream from Hugging Face...")
    # Stream dataset to avoid downloading everything
    dataset = load_dataset(
        "benjamin-paine/imagenet-1k-128x128", 
        split="train", 
        streaming=True
    )
    
    # Counters
    counts = {name: 0 for name in set(class_mapping.values())}
    # Calculate needed based on filtered mapping
    total_needed = num_images_per_class * len(set(class_mapping.values()))
    
    print(f"Target: {num_images_per_class} images per class")
    
    pbar = tqdm(total=total_needed)
    
    for item in dataset:
        label = item['label']
        
        if label in target_indices:
            class_name = class_mapping[label]
            
            if counts[class_name] < num_images_per_class:
                # Save image
                image = item['image']
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Determine split (simple 80/20 split based on count)
                is_val = (counts[class_name] % 5) == 0
                split_dir = 'val' if is_val else 'train'
                
                # Save
                img_path = output_path / split_dir / class_name / f"{counts[class_name]}.png"
                image.save(img_path)
                
                counts[class_name] += 1
                pbar.update(1)
                
                # Check if we're done
                if all(c >= num_images_per_class for c in counts.values()):
                    break
    
    pbar.close()
    print("\nDownload complete!")
    print("Counts:")
    for name, count in counts.items():
        print(f"  - {name}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/custom", help="Output directory")
    parser.add_argument("--num_images", type=int, default=500, help="Images per class")
    parser.add_argument("--classes", nargs='+', default=None, help="Specific classes to download (e.g. cat dog)")
    args = parser.parse_args()
    
    prepare_imagenet_subset(args.output_dir, args.num_images, target_classes=args.classes)
