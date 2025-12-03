"""Dataset classes for CIFAR-10 subset and other datasets"""

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
import os
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional


class CIFAR10Subset(Dataset):
    """
    CIFAR-10 dataset filtered to specific classes.
    
    Args:
        root: Root directory for dataset
        train: If True, use training set, else test set
        transform: Transformations to apply
        download: If True, download dataset
        classes: List of class names to include (e.g., ['airplane', 'cat'])
    """
    
    # CIFAR-10 class names in order
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[object] = None,
        download: bool = True,
        classes: Optional[List[str]] = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Load full CIFAR-10 dataset
        self.full_dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=None,  # We'll apply transforms manually
            download=download
        )
        
        # Filter to specific classes
        if classes is not None:
            self.class_names = classes
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
            
            # Get original CIFAR-10 indices for these classes
            self.original_indices = [
                self.CIFAR10_CLASSES.index(name) for name in classes
            ]
            
            # Filter dataset
            self.indices = self._filter_by_classes()
        else:
            # Use all classes
            self.class_names = self.CIFAR10_CLASSES
            self.class_to_idx = {name: idx for idx, name in enumerate(self.CIFAR10_CLASSES)}
            self.original_indices = list(range(10))
            self.indices = list(range(len(self.full_dataset)))
        
        print(f"{'Train' if train else 'Test'} dataset: {len(self.indices)} images across {len(self.class_names)} classes")
        for class_name in self.class_names:
            # Count by iterating through dataset indices (0 to len(self.indices)-1)
            count = sum(1 for dataset_idx in range(len(self.indices)) 
                       if self._get_label(dataset_idx) == self.class_to_idx[class_name])
            print(f"  - {class_name}: {count} images")
    
    def _filter_by_classes(self) -> List[int]:
        """Filter dataset to only include specified classes"""
        indices = []
        for idx in range(len(self.full_dataset)):
            _, label = self.full_dataset[idx]
            if label in self.original_indices:
                indices.append(idx)
        return indices
    
    def _get_label(self, dataset_idx: int) -> int:
        """Get the remapped label for a dataset index"""
        _, original_label = self.full_dataset[self.indices[dataset_idx]]
        # Remap to new class indices (0, 1, 2, 3 for 4 classes)
        new_label = self.original_indices.index(original_label)
        return new_label
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Transformed image tensor
            label: Class label (remapped to 0, 1, ..., num_classes-1)
        """
        # Get original image and label
        image, _ = self.full_dataset[self.indices[idx]]
        label = self._get_label(idx)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_image_with_metadata(self, idx: int) -> dict:
        """
        Get image with additional metadata for visualization
        
        Returns:
            dict with keys: 'image', 'label', 'class_name', 'original_idx'
        """
        image, label = self[idx]
        return {
            'image': image,
            'label': label,
            'class_name': self.class_names[label],
            'original_idx': self.indices[idx]
        }


class CustomImageFolder(Dataset):
    """
    Custom dataset for loading images from a folder structure.
    Expected structure: root/class_name/image.png
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[object] = None,
        classes: Optional[List[str]] = None
    ):
        self.root = root
        self.transform = transform
        self.classes = classes
        
        # Find all images
        self.samples = []
        self.class_to_idx = {}
        
        if classes is not None:
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for class_name in classes:
                class_dir = os.path.join(root, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: Class directory not found: {class_dir}")
                    continue
                    
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, fname)
                        self.samples.append((path, self.class_to_idx[class_name]))
        else:
            # Auto-discover classes
            class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            class_dirs.sort()
            self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
            self.classes = class_dirs
            
            for class_name in class_dirs:
                class_dir = os.path.join(root, class_name)
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, fname)
                        self.samples.append((path, self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images across {len(self.class_to_idx)} classes")
        for class_name in self.class_to_idx:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  - {class_name}: {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        # Load image
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

    def get_image_with_metadata(self, idx: int) -> dict:
        path, label = self.samples[idx]
        image, _ = self[idx]
        return {
            'image': image,
            'label': label,
            'class_name': self.classes[label] if self.classes else str(label),
            'original_idx': idx,
            'path': path
        }
