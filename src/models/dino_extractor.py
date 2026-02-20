# src/models/dino_extractor.py
"""
DINO ViT-based patch feature extractor for concept-grounded parts discovery.
Extracts patch-level features: shape [784, 384] per 224x224 image.
784 = 28x28 spatial grid of 8x8 patches. 384 = embedding dimension for ViT-S.
"""
import warnings

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

warnings.filterwarnings("ignore")


class DinoExtractor:
    """
    Wraps a frozen DINO ViT-S/8 model.
    Extracts patch-level features: shape [784, 384] per 224x224 image.
    784 = 28x28 spatial grid of 8x8 patches. 384 = embedding dimension for ViT-S.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str = "dino_vits8",
        device: str = "mps",
        image_size: int = 224,
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.patch_size = 8
        self.grid_size = image_size // self.patch_size  # 28
        self.feat_dim = 384

        print(f"Loading {model_name} ...")
        self.model = torch.hub.load(
            "facebookresearch/dino:main", model_name, pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        print("DINO loaded and frozen.")

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

    def load_image(self, path) -> torch.Tensor:
        """Load and transform a single image to tensor [1, 3, H, W]."""
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor [B, 3, 224, 224]
        Returns:
            patch_features [B, 784, 384]
        """
        feats = self.model.get_intermediate_layers(image_tensor, n=1)[0]
        # feats shape: [B, 785, 384] — index 0 is CLS token
        return feats[:, 1:, :]  # drop CLS → [B, 784, 384]

    @torch.no_grad()
    def extract_attention(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns attention maps from last layer heads.
        Shape: [B, num_heads, 785, 785]
        Useful for visualisation and validation.
        """
        return self.model.get_last_selfattention(image_tensor)

    def extract_from_path(self, image_path) -> torch.Tensor:
        """Convenience: load image from disk and extract patches."""
        tensor = self.load_image(image_path)
        return self.extract_patches(tensor).squeeze(0).cpu()  # [784, 384]

    def extract_foreground_patches(
        self, image_path, fg_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Load image, extract patches, then mask to foreground using DINO attention.
        Matches the foreground masking applied during batch feature extraction.

        Args:
            image_path: path to image file
            fg_threshold: quantile threshold — keep patches with attention > this quantile.
                          0.5 = keep top 50% (matches extract_dino_features.py default).
        Returns:
            foreground patch features [K, 384] where K <= 784
        """
        tensor = self.load_image(image_path)
        feats = self.extract_patches(tensor).squeeze(0).cpu()   # [784, 384]
        attn = self.extract_attention(tensor)                    # [1, heads, 785, 785]
        cls_attn = attn[0, :, 0, 1:].mean(dim=0).cpu()         # [784] mean across heads
        threshold = cls_attn.quantile(fg_threshold)
        fg_indices = (cls_attn > threshold).nonzero(as_tuple=True)[0]
        if len(fg_indices) < 10:
            fg_indices = cls_attn.topk(max(10, feats.shape[0] // 4)).indices
        return feats[fg_indices]                                 # [K, 384]

    def get_spatial_grid(self):
        """Returns (grid_size, grid_size) = (28, 28)."""
        return self.grid_size, self.grid_size
