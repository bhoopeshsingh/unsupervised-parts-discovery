
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional

def get_affine_params(
    config: Dict[str, Any], 
    batch_size: int, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random affine transformation parameters efficiently on GPU.
    
    Args:
        config: Dictionary containing transform constraints:
            - degrees: max rotation degrees (float)
            - translate: max translation (float or [x, y])
            - scale: range [min, max] or None
            - hflip: probability of horizontal flip (bool or float)
        batch_size: Number of samples
        device: Torch device
        
    Returns:
        theta: Affine matrices [B, 2, 3]
        params: Dictionary of generated params for logging (optional)
    """
    # 1. Rotation (radians)
    max_deg = config.get('degrees', 0.0)
    if max_deg > 0:
        angle = torch.rand(batch_size, device=device) * 2 * max_deg - max_deg
        angle_rad = angle * math.pi / 180.0
    else:
        angle_rad = torch.zeros(batch_size, device=device)
        
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    # 2. Translation (fraction of image size)
    trans_conf = config.get('translate', None)
    if trans_conf:
        if isinstance(trans_conf, (list, tuple)):
            max_dx, max_dy = trans_conf
        else:
            max_dx = max_dy = trans_conf
        
        tx = torch.rand(batch_size, device=device) * 2 * max_dx - max_dx
        ty = torch.rand(batch_size, device=device) * 2 * max_dy - max_dy
    else:
        tx = torch.zeros(batch_size, device=device)
        ty = torch.zeros(batch_size, device=device)
        
    # 3. Scaling
    scale_conf = config.get('scale', None)
    if scale_conf:
        min_s, max_s = scale_conf
        sx = torch.rand(batch_size, device=device) * (max_s - min_s) + min_s
        sy = sx.clone() # Keep aspect ratio fixed for now
    else:
        sx = torch.ones(batch_size, device=device)
        sy = torch.ones(batch_size, device=device)
        
    # 4. Horizontal Flip
    if config.get('hflip', False):
        # 50% chance
        flip = torch.rand(batch_size, device=device) < 0.5
        sx = torch.where(flip, -sx, sx)
        
    # Construct Affine Matrix [B, 2, 3]
    # Matrix maps output coordinates to input coordinates (inverse transform needed for grid_sample)
    # But PyTorch grid_sample expects theta mapping output -> input.
    # The standard affine matrix is:
    # [ sx*cos(a)   -sy*sin(a)   tx ]
    # [ sx*sin(a)    sy*cos(a)   ty ]
    #
    # Wait, PyTorch F.affine_grid(theta) expects theta to map TARGET coordinates to SOURCE coordinates.
    # If we want to rotate image by +a, we need to sample from -a.
    # So we should use -angle and -translation?
    # Let's verify standard behavior.
    
    # Actually, constructing the forward matrix and inverting it is safer.
    # Forward Matrix M: [x_new] = M * [x_old]
    # M = [ [sx*cos, -sin, tx], [sin, sy*cos, ty], [0, 0, 1] ]
    # We need M_inv for grid_sample.
    
    # Simplified approach: direct construction of sampling matrix.
    # To rotate by angle `a` (counter-clockwise) and translate by `t`:
    # The sampling grid needs to look at location R(-a) * (x - t).
    
    # Let's stick to simple individual matrices for clarity
    
    # Rotation matrix (inverse)
    # Cos(a)  Sin(a)   0
    # -Sin(a) Cos(a)   0
    # 0       0        1
    
    # Translation (inverse)
    # 1 0 -tx
    # 0 1 -ty
    # 0 0  1
    
    # Scaling (inverse)
    # 1/sx 0    0
    # 0    1/sy 0
    # 0    0    1
    
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)
    
    # Rotation + Scale
    # We want to perform R * S * x + t
    # For inverse: x_old = S^-1 * R^-1 * (x_new - t)
    
    # Let's keep it simple: small perturbations.
    # theta = [ [cos(a)/sx, sin(a)/sy, -tx], [-sin(a)/sx, cos(a)/sy, -ty] ]
    
    # Adjust for aspect ratio? Assuming square images.
    
    r00 = cos_a / sx
    r01 = sin_a / sy # Check sign? sin(-a) = -sin(a). If we want +a rot, sample at -a?
    # Standard: positive angle in grid_sample rotates image Counter-Clockwise? No, it rotates the GRID.
    # If we rotate grid by +a, we sample from +a relative to center.
    # This effectively rotates image by -a.
    # So to get random rotation, sign doesn't matter much as we sample from uniform.
    
    r02 = -tx 
    
    r10 = -sin_a / sx
    r11 = cos_a / sy
    r12 = -ty
    
    theta = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1)
    ], dim=1) # [B, 2, 3]
    
    # Return params for logging (empty for now)
    params = {}
    return theta, params

def apply_transform(
    tensor: torch.Tensor, 
    theta: torch.Tensor, 
    mode: str = 'bilinear',
    padding_mode: str = 'zeros'
) -> torch.Tensor:
    """
    Apply affine transformation to a tensor.
    
    Args:
        tensor: [B, C, H, W]
        theta: [B, 2, 3]
        mode: 'bilinear' or 'nearest'
        padding_mode: 'zeros', 'border', or 'reflection'
        
    Returns:
        Transformed tensor [B, C, H, W]
    """
    B, C, H, W = tensor.shape
    grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
    transformed = F.grid_sample(tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return transformed
