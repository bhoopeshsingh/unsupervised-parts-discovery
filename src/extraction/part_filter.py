"""
Part Filter Module

Filters out low-quality parts before clustering to improve cluster quality.
Uses multiple criteria:
- Coverage: Rejects parts that are too small (<2%) or too large (>80%)
- Spatial coherence: Rejects parts with scattered attention (many components)
- Edge density: Rejects mostly blank/uniform regions
- Variance: Rejects single-color regions
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for part filtering."""
    # Coverage thresholds
    min_coverage: float = 0.02  # Minimum 2% coverage
    max_coverage: float = 0.80  # Maximum 80% coverage
    
    # Spatial coherence
    max_connected_components: int = 2  # Max components before rejection
    mask_threshold: float = 0.2  # Threshold for binary mask
    
    # Edge density
    min_edge_density: float = 0.05  # Minimum edge density (reject blank)
    
    # Variance check
    min_std: float = 0.1  # Minimum std dev (reject single-color)
    
    # Size constraints
    min_bbox_size: int = 4  # Minimum bbox dimension


class PartFilter:
    """
    Filter low-quality parts before clustering.
    
    Removes parts that are:
    - Too small or too large (background)
    - Scattered (multiple disconnected regions)
    - Blank/uniform (no structure)
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
    
    def check_coverage(self, mask: np.ndarray) -> bool:
        """
        Check if part coverage is within acceptable range.
        
        Args:
            mask: Attention mask [H, W]
            
        Returns:
            True if coverage is acceptable
        """
        coverage = mask.sum() / mask.size
        return self.config.min_coverage <= coverage <= self.config.max_coverage
    
    def check_spatial_coherence(self, mask: np.ndarray) -> bool:
        """
        Check if part is spatially coherent (single connected region).
        
        Args:
            mask: Attention mask [H, W]
            
        Returns:
            True if spatially coherent
        """
        # Create binary mask
        binary_mask = (mask > self.config.mask_threshold).astype(np.uint8)
        
        # Count connected components
        num_labels, _ = cv2.connectedComponents(binary_mask)
        # Note: num_labels includes background (label 0)
        num_components = num_labels - 1
        
        return num_components <= self.config.max_connected_components
    
    def check_edge_density(self, crop: np.ndarray) -> bool:
        """
        Check if cropped region has sufficient edge structure.
        
        Args:
            crop: Cropped image [H, W, 3] or [H, W]
            
        Returns:
            True if edge density is sufficient
        """
        # Convert to uint8 if needed
        if crop.max() <= 1.0:
            crop_uint8 = (crop * 255).astype(np.uint8)
        else:
            crop_uint8 = crop.astype(np.uint8)
        
        # Convert to grayscale if color
        if len(crop_uint8.shape) == 3:
            gray = cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_uint8
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.size * 255)
        
        return edge_density >= self.config.min_edge_density
    
    def check_variance(self, crop: np.ndarray) -> bool:
        """
        Check if cropped region has sufficient variance (not single-color).
        
        Args:
            crop: Cropped image
            
        Returns:
            True if variance is sufficient
        """
        return crop.std() >= self.config.min_std
    
    def check_bbox_size(self, bbox: tuple) -> bool:
        """
        Check if bounding box is large enough.
        
        Args:
            bbox: (x1, y1, x2, y2)
            
        Returns:
            True if bbox is large enough
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return (width >= self.config.min_bbox_size and 
                height >= self.config.min_bbox_size)
    
    def is_valid(self, part: Any) -> bool:
        """
        Check if a part passes all quality filters.
        
        Args:
            part: PartDescriptor or dict with 'mask', 'crop', 'bbox' keys
            
        Returns:
            True if part is valid
        """
        # Extract data from part (handle both PartDescriptor and dict)
        if hasattr(part, 'mask'):
            mask = part.mask
            crop = part.crop
            bbox = part.bbox
        else:
            mask = part['mask']
            crop = part['crop']
            bbox = part['bbox']
        
        # Check all criteria
        checks = [
            ('coverage', self.check_coverage(mask)),
            ('spatial_coherence', self.check_spatial_coherence(mask)),
            ('edge_density', self.check_edge_density(crop)),
            ('variance', self.check_variance(crop)),
            ('bbox_size', self.check_bbox_size(bbox)),
        ]
        
        return all(passed for _, passed in checks)
    
    def get_rejection_reason(self, part: Any) -> Optional[str]:
        """
        Get reason for rejecting a part.
        
        Args:
            part: PartDescriptor or dict
            
        Returns:
            Rejection reason or None if valid
        """
        if hasattr(part, 'mask'):
            mask = part.mask
            crop = part.crop
            bbox = part.bbox
        else:
            mask = part['mask']
            crop = part['crop']
            bbox = part['bbox']
        
        checks = [
            ('too_small_or_large', not self.check_coverage(mask)),
            ('scattered_attention', not self.check_spatial_coherence(mask)),
            ('blank_region', not self.check_edge_density(crop)),
            ('single_color', not self.check_variance(crop)),
            ('bbox_too_small', not self.check_bbox_size(bbox)),
        ]
        
        for reason, failed in checks:
            if failed:
                return reason
        return None
    
    def filter_valid_parts(
        self, 
        parts: List[Any],
        return_stats: bool = False
    ) -> List[Any]:
        """
        Filter parts, keeping only valid ones.
        
        Args:
            parts: List of PartDescriptor or dict objects
            return_stats: If True, return (filtered_parts, stats) tuple
            
        Returns:
            List of valid parts (and optionally stats dict)
        """
        valid_parts = []
        rejection_counts = {
            'too_small_or_large': 0,
            'scattered_attention': 0,
            'blank_region': 0,
            'single_color': 0,
            'bbox_too_small': 0,
        }
        
        for part in parts:
            if self.is_valid(part):
                valid_parts.append(part)
            else:
                reason = self.get_rejection_reason(part)
                if reason:
                    rejection_counts[reason] += 1
        
        if return_stats:
            stats = {
                'total_input': len(parts),
                'total_valid': len(valid_parts),
                'total_rejected': len(parts) - len(valid_parts),
                'rejection_rate': (len(parts) - len(valid_parts)) / max(1, len(parts)),
                'rejection_reasons': rejection_counts
            }
            return valid_parts, stats
        
        return valid_parts
    
    def print_filter_stats(self, parts: List[Any]):
        """Print filtering statistics for debugging."""
        _, stats = self.filter_valid_parts(parts, return_stats=True)
        
        print(f"\n=== Part Filtering Statistics ===")
        print(f"Total input parts: {stats['total_input']}")
        print(f"Valid parts: {stats['total_valid']}")
        print(f"Rejected parts: {stats['total_rejected']} ({stats['rejection_rate']:.1%})")
        print(f"\nRejection reasons:")
        for reason, count in stats['rejection_reasons'].items():
            if count > 0:
                print(f"  - {reason}: {count}")
