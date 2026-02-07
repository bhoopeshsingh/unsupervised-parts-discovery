import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os

# Define class mapping from our dataset to DeepLabV3 (VOC) classes
# VOC classes: 
# 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 
# 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable, 
# 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 
# 17=sheep, 18=sofa, 19=train, 20=tv/monitor

VOC_CLASS_MAP = {
    'airplane': 1,  # aeroplane
    'bird': 3,
    'cat': 8,
    'dog': 12
}

def load_segmentation_model(device='auto'):
    """Load pre-trained DeepLabV3 model."""
    if device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading DeepLabV3 model on {device}...")
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
    model.to(device)
    model.eval()
    return model, device

def remove_background(image, class_name, model, device):
    """
    Remove background from image using DeepLabV3.
    
    Args:
        image: PIL Image
        class_name: Target class name (e.g., 'bird')
        model: Loaded DeepLabV3 model
        device: Torch device
        
    Returns:
        PIL Image with background removed (black)
    """
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get prediction (argmax)
    output_predictions = output.argmax(0)
    
    # Create mask for target class
    target_idx = VOC_CLASS_MAP.get(class_name)
    if target_idx is None:
        print(f"Warning: Class '{class_name}' not found in VOC mapping. Returning original.")
        return image
        
    # Mask: 1 where class matches, 0 otherwise
    mask = (output_predictions == target_idx).byte().cpu().numpy()
    
    # If no object detected, maybe fallback or return black? 
    # Let's return original if nothing detected (or maybe black?)
    # User wants to remove noise. If we detect nothing, maybe the image is bad?
    # Let's keep original but print warning if debug.
    if mask.sum() == 0:
        # Try finding *any* relevant object? No, stick to target.
        # Actually, let's return the original image if segmentation fails to find the object,
        # otherwise we might end up with a blank image which is useless for training.
        return image

    # Apply mask
    img_array = np.array(image)
    # Ensure mask matches image size (it should, but DeepLab output is same size)
    
    # Mask is (H, W), Image is (H, W, 3)
    # Expand mask to 3 channels
    mask_3d = np.stack([mask]*3, axis=-1)
    
    # Apply mask: keep pixels where mask is 1, else 0
    masked_img_array = img_array * mask_3d
    
    return Image.fromarray(masked_img_array)

def process_dataset(input_dir, output_dir, model, device):
    """Process entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    splits = ['train', 'val']
    
    for split in splits:
        split_dir = input_path / split
        if not split_dir.exists():
            continue
            
        print(f"Processing {split} split...")
        
        # Iterate over classes
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in VOC_CLASS_MAP:
                print(f"Skipping unknown class: {class_name}")
                continue
                
            # Create output directory
            out_class_dir = output_path / split / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Process images
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            
            for img_path in tqdm(images, desc=f"{class_name}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    masked_img = remove_background(img, class_name, model, device)
                    
                    # Save
                    masked_img.save(out_class_dir / img_path.name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Remove backgrounds from dataset')
    parser.add_argument('--input-dir', type=str, default='data/custom',
                        help='Input dataset directory')
    parser.add_argument('--output-dir', type=str, default='data/custom_masked',
                        help='Output directory for masked dataset')
    args = parser.parse_args()
    
    model, device = load_segmentation_model()
    
    process_dataset(args.input_dir, args.output_dir, model, device)
    
    print("\nBackground removal complete!")
    print(f"Masked dataset saved to {args.output_dir}")

if __name__ == '__main__':
    main()
