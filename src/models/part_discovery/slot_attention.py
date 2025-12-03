"""
Slot Attention for Unsupervised Part Discovery

Based on: "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
Paper: https://arxiv.org/abs/2006.15055

Adapted for part discovery in CIFAR-10 images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math


class SlotAttention(nn.Module):
    """
    Slot Attention mechanism for discovering parts
    
    Args:
        num_slots: Number of slots (parts) to discover
        slot_dim: Dimension of each slot
        num_iterations: Number of attention iterations
        mlp_hidden_dim: Hidden dimension for MLP
    """
    
    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 128,
        num_iterations: int = 3,
        mlp_hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = 1e-8
        
        # Slot initialization parameters (learnable)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        # Layer norms
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        # Attention
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        
        # Slot update MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )
        
        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply slot attention
        
        Args:
            inputs: Input features [B, N, D] where N is number of input elements
        
        Returns:
            slots: Discovered slot representations [B, num_slots, slot_dim]
            attn: Attention weights [B, num_slots, N]
        """
        B, N, D = inputs.shape
        
        # Initialize slots
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, N, D]
        v = self.project_v(inputs)  # [B, N, D]
        
        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            q = self.project_q(slots)  # [B, num_slots, D]
            
            # Compute attention weights
            # [B, num_slots, D] @ [B, D, N] -> [B, num_slots, N]
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.slot_dim)
            attn = F.softmax(attn_logits, dim=-1)  # [B, num_slots, N]
            
            # Normalize attention over slots (competition)
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=1, keepdim=True)
            
            # Weighted mean
            updates = torch.bmm(attn, v)  # [B, num_slots, D]
            
            # Update slots with GRU
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            
            # MLP update
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn


class SpatialBroadcastDecoder(nn.Module):
    """
    Decoder for reconstructing images from slots
    Uses spatial broadcast decoder
    """
    
    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dims: list = [256, 512],
        output_size: int = 32,
        output_channels: int = 3
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.output_size = output_size
        self.output_channels = output_channels
        
        # Position encoding grid
        self.register_buffer(
            "pos_grid",
            self._get_position_grid(output_size)
        )
        
        # Decoder network
        layers = []
        in_dim = slot_dim + 2  # slot features + 2D position
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        # Output layer: RGB + alpha (mask)
        layers.append(nn.Linear(in_dim, output_channels + 1))
        
        self.decoder = nn.Sequential(*layers)
    
    def _get_position_grid(self, size: int) -> torch.Tensor:
        """Create normalized position grid [-1, 1]"""
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        return grid
    
    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode slots to image
        
        Args:
            slots: Slot representations [B, num_slots, slot_dim]
        
        Returns:
            recon: Reconstructed image [B, C, H, W]
            masks: Per-slot masks [B, num_slots, H, W]
        """
        B, num_slots, _ = slots.shape
        H, W = self.output_size, self.output_size
        
        # Broadcast slots to all spatial positions
        slots = slots.unsqueeze(2).unsqueeze(3)  # [B, num_slots, 1, 1, D]
        slots = slots.expand(-1, -1, H, W, -1)  # [B, num_slots, H, W, D]
        
        # Broadcast position grid
        pos = self.pos_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        pos = pos.expand(B, num_slots, -1, -1, -1)  # [B, num_slots, H, W, 2]
        
        # Concatenate slots and positions
        decoder_input = torch.cat([slots, pos], dim=-1)  # [B, num_slots, H, W, D+2]
        
        # Decode
        # decoder_input = decoder_input.reshape(B * num_slots * H * W, -1)
        # decoder_output = self.decoder(decoder_input)
        # decoder_output = decoder_output.reshape(B, num_slots, H, W, -1)
        
        # Chunked decoding to avoid OOM
        # Total elements: B * num_slots * H * W
        # With B=32, S=6, H=128, W=128 -> 3.1M vectors
        # Each vector is ~512 floats (hidden dim) -> 6GB memory
        
        flat_input = decoder_input.reshape(-1, decoder_input.shape[-1])
        chunk_size = 500000  # Process 500k vectors at a time (~1GB output)
        output_chunks = []
        
        for i in range(0, flat_input.shape[0], chunk_size):
            chunk = flat_input[i:i + chunk_size]
            output_chunks.append(self.decoder(chunk))
            
        decoder_output = torch.cat(output_chunks, dim=0)
        decoder_output = decoder_output.reshape(B, num_slots, H, W, -1)
        
        # Split into RGB and masks
        recons = decoder_output[..., :-1]  # [B, num_slots, H, W, C]
        masks = decoder_output[..., -1:]   # [B, num_slots, H, W, 1]
        
        # Normalize masks with softmax
        masks = F.softmax(masks, dim=1)  # [B, num_slots, H, W, 1]
        
        # Combine slots with masks
        recons = recons.permute(0, 1, 4, 2, 3)  # [B, num_slots, C, H, W]
        masks = masks.permute(0, 1, 4, 2, 3)    # [B, num_slots, 1, H, W]
        
        recon = (recons * masks).sum(dim=1)  # [B, C, H, W]
        masks = masks.squeeze(2)  # [B, num_slots, H, W]
        
        return recon, masks


class FeatureEncoder(nn.Module):
    """Encode backbone features to slot-compatible dimension"""
    
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 128,
        hidden_dims: list = [512, 256]
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features
        
        Args:
            features: Backbone features [B, C, H, W] or [B, N, C]
        
        Returns:
            encoded: Encoded features [B, N, output_dim]
        """
        if features.dim() == 4:
            # Spatial features [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        return self.encoder(features)


class SlotAttentionModel(nn.Module):
    """
    Complete Slot Attention model for part discovery
    
    Combines: Encoder -> Slot Attention -> Decoder
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        slot_config = config['slot_attention']
        
        # Feature encoder
        self.encoder = FeatureEncoder(
            input_dim=slot_config['encoder']['input_dim'],
            output_dim=slot_config['encoder']['output_dim'],
            hidden_dims=slot_config['encoder']['hidden_dims']
        )
        
        # Slot attention
        self.slot_attention = SlotAttention(
            num_slots=slot_config['num_slots'],
            slot_dim=slot_config['slot_dim'],
            num_iterations=slot_config['num_iterations'],
            mlp_hidden_dim=slot_config['hidden_dim']
        )
        
        # Decoder
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_config['slot_dim'],
            hidden_dims=slot_config['decoder']['hidden_dims'],
            output_size=slot_config['decoder']['output_size'],
            output_channels=slot_config['decoder']['output_channels']
        )
        
        self.num_slots = slot_config['num_slots']
        print(f"SlotAttentionModel initialized with {self.num_slots} slots")
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: Backbone features [B, C, H, W]
        
        Returns:
            recon: Reconstructed images [B, 3, 32, 32]
            masks: Per-slot attention masks [B, num_slots, 32, 32]
            slots: Slot representations [B, num_slots, slot_dim]
            attn: Attention weights [B, num_slots, N]
        """
        # Encode features
        encoded = self.encoder(features)  # [B, N, D]
        
        # Apply slot attention
        slots, attn = self.slot_attention(encoded)  # [B, num_slots, D], [B, num_slots, N]
        
        # Decode slots
        recon, masks = self.decoder(slots)  # [B, 3, H, W], [B, num_slots, H, W]
        
        return recon, masks, slots, attn
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlotAttentionModel":
        """Create model from config dict"""
        return cls(config)
