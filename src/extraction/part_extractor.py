"""
Part Extractor with Rich Descriptors

Extracts parts from Slot Attention outputs with rich descriptors combining:
- Slot features (learned representations)
- Visual features (pretrained ResNet on cropped regions)
- Spatial features (normalized position, size, coverage)
- Shape features (aspect ratio, compactness, edge density)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


@dataclass
class PartDescriptor:
    """Rich part descriptor with all feature types."""
    slot_features: np.ndarray    # [slot_dim] - learned from Slot Attention
    visual_features: np.ndarray  # [2048] - pretrained ResNet features
    spatial_features: np.ndarray # [5] - x, y, width, height, coverage
    shape_features: np.ndarray   # [3] - aspect_ratio, compactness, edge_density
    
    # Metadata
    image_idx: int
    slot_idx: int
    class_label: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # attention mask
    crop: np.ndarray  # cropped image region
    
    def to_combined_vector(self) -> np.ndarray:
        """Combine all features into single vector."""
        return np.concatenate([
            self.slot_features,
            self.visual_features,
            self.spatial_features,
            self.shape_features
        ])
    
    @property
    def total_dim(self) -> int:
        """Total dimension of combined features."""
        return (len(self.slot_features) + len(self.visual_features) + 
                len(self.spatial_features) + len(self.shape_features))


class PartExtractor:
    """
    Extract rich part descriptors from Slot Attention outputs.
    
    Combines slot features with visual, spatial, and shape features
    to produce more discriminative part representations for clustering.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        visual_feature_dim: int = 2048
    ):
        self.device = device or torch.device('cpu')
        self.visual_feature_dim = visual_feature_dim
        
        # Load pretrained ResNet for visual features
        self._init_visual_encoder()
        
        # Image preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _init_visual_encoder(self):
        """Initialize pretrained ResNet for visual feature extraction."""
        self.visual_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove classification head, keep feature extractor
        self.visual_encoder = nn.Sequential(*list(self.visual_encoder.children())[:-1])
        self.visual_encoder = self.visual_encoder.to(self.device)
        self.visual_encoder.eval()
        
        # Freeze weights
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
    
    def get_bounding_box(
        self, 
        attn_mask: np.ndarray, 
        threshold: float = 0.2
    ) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from attention mask.
        
        Args:
            attn_mask: Attention mask [H, W]
            threshold: Threshold for binary mask
            
        Returns:
            (x1, y1, x2, y2) bounding box coordinates
        """
        binary_mask = (attn_mask > threshold).astype(np.uint8)
        
        # Find non-zero regions
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not rows.any() or not cols.any():
            # No valid region, return full image
            H, W = attn_mask.shape
            return (0, 0, W, H)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add small padding
        H, W = attn_mask.shape
        pad = 2
        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(W, x_max + pad + 1)
        y2 = min(H, y_max + pad + 1)
        
        return (x1, y1, x2, y2)
    
    def crop_region(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop image region using bounding box.
        
        Args:
            image: Image [H, W, 3] or [3, H, W]
            bbox: (x1, y1, x2, y2)
            
        Returns:
            Cropped image region
        """
        # Handle different image formats
        if image.shape[0] == 3 and len(image.shape) == 3:
            # [C, H, W] -> [H, W, C]
            image = np.transpose(image, (1, 2, 0))
        
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2].copy()
        
        # Ensure minimum size
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            # Return original image if crop is too small
            return image
        
        return crop
    
    @torch.no_grad()
    def extract_visual_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract visual features from cropped region using pretrained ResNet.
        
        Args:
            crop: Cropped image [H, W, 3], values in [0, 1]
            
        Returns:
            Visual features [2048]
        """
        # Ensure proper format
        if crop.max() <= 1.0:
            crop = (crop * 255).astype(np.uint8)
        else:
            crop = crop.astype(np.uint8)
        
        # Preprocess
        tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.visual_encoder(tensor)
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def compute_spatial_features(
        self, 
        bbox: Tuple[int, int, int, int],
        attn_mask: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute spatial features: normalized position, size, coverage.
        
        Args:
            bbox: (x1, y1, x2, y2)
            attn_mask: Attention mask [H, W]
            image_size: (height, width)
            
        Returns:
            Spatial features [5]: x_center, y_center, width, height, coverage
        """
        H, W = image_size
        x1, y1, x2, y2 = bbox
        
        # Normalized center position
        x_center = (x1 + x2) / 2 / W
        y_center = (y1 + y2) / 2 / H
        
        # Normalized size
        width = (x2 - x1) / W
        height = (y2 - y1) / H
        
        # Coverage (fraction of image covered by attention)
        coverage = attn_mask.sum() / (H * W)
        
        return np.array([x_center, y_center, width, height, coverage], dtype=np.float32)
    
    def compute_shape_features(
        self,
        bbox: Tuple[int, int, int, int],
        attn_mask: np.ndarray,
        crop: np.ndarray
    ) -> np.ndarray:
        """
        Compute shape features: aspect ratio, compactness, edge density.
        
        Args:
            bbox: (x1, y1, x2, y2)
            attn_mask: Attention mask [H, W]
            crop: Cropped image region
            
        Returns:
            Shape features [3]
        """
        x1, y1, x2, y2 = bbox
        
        # Aspect ratio
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = width / height
        
        # Compactness (how circular/compact the mask is)
        # 4π * area / perimeter² = 1 for circle
        binary_mask = (attn_mask > 0.2).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0.0
        else:
            compactness = 0.0
        
        # Edge density (measure of structure)
        if crop.ndim == 3:
            gray = cv2.cvtColor(
                (crop * 255).astype(np.uint8) if crop.max() <= 1.0 else crop.astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = (crop * 255).astype(np.uint8) if crop.max() <= 1.0 else crop.astype(np.uint8)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.size * 255)  # Normalize to [0, 1]
        
        return np.array([aspect_ratio, compactness, edge_density], dtype=np.float32)
    
    def extract_part_with_context(
        self,
        image: np.ndarray,
        slot_features: np.ndarray,
        attn_mask: np.ndarray,
        image_idx: int,
        slot_idx: int,
        class_label: int
    ) -> PartDescriptor:
        """
        Extract a rich part descriptor with spatial and visual context.
        
        Args:
            image: Original image [H, W, 3] or [3, H, W]
            slot_features: Slot features from Slot Attention [slot_dim]
            attn_mask: Attention mask [H, W]
            image_idx: Index of source image
            slot_idx: Index of slot
            class_label: Class label of image
            
        Returns:
            PartDescriptor with all feature types
        """
        # Get image dimensions
        if image.shape[0] == 3:
            H, W = image.shape[1], image.shape[2]
            image_hwc = np.transpose(image, (1, 2, 0))
        else:
            H, W = image.shape[0], image.shape[1]
            image_hwc = image
        
        # 1. Get bounding box
        bbox = self.get_bounding_box(attn_mask)
        
        # 2. Crop region
        crop = self.crop_region(image_hwc, bbox)
        
        # 3. Extract visual features
        visual_features = self.extract_visual_features(crop)
        
        # 4. Compute spatial features
        spatial_features = self.compute_spatial_features(bbox, attn_mask, (H, W))
        
        # 5. Compute shape features  
        shape_features = self.compute_shape_features(bbox, attn_mask, crop)
        
        return PartDescriptor(
            slot_features=np.asarray(slot_features, dtype=np.float32),
            visual_features=visual_features,
            spatial_features=spatial_features,
            shape_features=shape_features,
            image_idx=image_idx,
            slot_idx=slot_idx,
            class_label=class_label,
            bbox=bbox,
            mask=attn_mask,
            crop=crop
        )
    
    def extract_parts_from_batch(
        self,
        images: torch.Tensor,
        slots: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        image_offset: int = 0
    ) -> List[PartDescriptor]:
        """
        Extract rich part descriptors from a batch.
        
        Args:
            images: Batch of images [B, 3, H, W]
            slots: Slot representations [B, num_slots, slot_dim]
            masks: Attention masks [B, num_slots, H, W]
            labels: Class labels [B]
            image_offset: Offset for image indices (for batched processing)
            
        Returns:
            List of PartDescriptor objects
        """
        B, num_slots = slots.shape[:2]
        parts = []
        
        # Convert to numpy
        images_np = images.cpu().numpy()
        slots_np = slots.cpu().numpy()
        masks_np = masks.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        for b in range(B):
            for s in range(num_slots):
                part = self.extract_part_with_context(
                    image=images_np[b],
                    slot_features=slots_np[b, s],
                    attn_mask=masks_np[b, s],
                    image_idx=image_offset + b,
                    slot_idx=s,
                    class_label=int(labels_np[b])
                )
                parts.append(part)
        
        return parts
