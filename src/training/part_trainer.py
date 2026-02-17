
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import mlflow
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..models.backbone import ResNetBackbone
from ..models.part_discovery.slot_attention import SlotAttentionModel
from ..utils import save_checkpoint, count_parameters
from ..utils.transforms import get_affine_params, apply_transform


class PartDiscoveryTrainer:
    """
    Trainer for unsupervised part discovery with Slot Attention
    """
    
    def __init__(
        self,
        backbone: ResNetBackbone,
        slot_model: SlotAttentionModel,
        device: torch.device,
        config: Dict[str, Any],
        use_tracking: bool = True
    ):
        self.backbone = backbone.to(device)
        self.slot_model = slot_model.to(device)
        self.device = device
        self.config = config
        self.use_tracking = use_tracking
        
        # Loss weights from config
        loss_config = config['slot_attention']['loss']
        self.recon_weight = loss_config.get('reconstruction_weight', 1.0)
        self.diversity_weight = loss_config.get('diversity_weight', 1.0)
        self.spatial_weight = loss_config.get('spatial_coherence_weight', 0.1)
        self.size_reg_weight = loss_config.get('size_regularization_weight', 1.0)
        self.edge_weight = loss_config.get('edge_weight', 0.0)
        self.compactness_weight = loss_config.get('compactness_weight', 0.0)
        
        # New loss weights
        self.equivariance_weight = loss_config.get('equivariance_weight', 0.0)
        self.concentration_weight = loss_config.get('concentration_weight', 0.0)
        self.spatial_entropy_weight = loss_config.get('spatial_entropy_weight', 0.0)
        self.mask_overlap_weight = loss_config.get('mask_overlap_weight', 0.0)

        # Warmup epochs
        self.equivariance_warmup = loss_config.get('equivariance_warmup_epochs', 0)
        self.concentration_warmup = loss_config.get('concentration_warmup_epochs', 0)

        # Transforms config
        self.transforms_config = loss_config.get('transforms', {})

        # Coverage constraints from config
        self.min_coverage = loss_config.get('min_coverage', 0.0)
        self.max_coverage = loss_config.get('max_coverage', 1.0)

        # Training config
        train_config = config['part_discovery']
        self.epochs = train_config['epochs']
        self.log_every = train_config['log_every']
        self.visualize_every = train_config.get('visualize_every', 100)
        self.checkpoint_dir = Path(train_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_config['save_every']
        
        # Combine parameters from both models
        params = list(self.backbone.parameters()) + list(self.slot_model.parameters())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            params,
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # Scheduler
        scheduler_config = train_config['scheduler']
        if scheduler_config['type'] == 'cosine_warmup':
            # Custom warmup + cosine schedule
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            max_epochs = self.epochs
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return float(epoch + 1) / float(warmup_epochs)
                else:
                    # Cosine decay
                    progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config['eta_min']
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.best_loss = float('inf')
        self.global_step = 0
        
        print(f"\nPartDiscoveryTrainer initialized:")
        print(f"  Backbone parameters: {count_parameters(self.backbone):,}")
        print(f"  Slot model parameters: {count_parameters(self.slot_model):,}")
        print(f"  Total parameters: {count_parameters(self.backbone) + count_parameters(self.slot_model):,}")
        print(f"  Loss weights: recon={self.recon_weight}, diversity={self.diversity_weight}, spatial={self.spatial_weight}, size_reg={self.size_reg_weight}")
        print(f"  New weights: equiv={self.equivariance_weight}, conc={self.concentration_weight}")

    def _compute_spatial_coherence_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Original spatial coherence logic retained"""
        B, num_slots, H, W = masks.shape
        y_coords = torch.linspace(0, 1, H, device=masks.device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W)
        masks_norm = masks / (masks.sum(dim=(2, 3), keepdim=True) + 1e-8)
        center_y = (masks_norm * y_coords).sum(dim=(2, 3)) 
        center_x = (masks_norm * x_coords).sum(dim=(2, 3)) 
        var_y = (masks_norm * (y_coords - center_y.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(2, 3))
        var_x = (masks_norm * (x_coords - center_x.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(2, 3))
        spatial_spread = (var_y + var_x).mean()
        
        tv_h = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :]).mean()
        tv_w = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1]).mean()
        boundary_smoothness = (tv_h + tv_w) * 0.1 

        eps = 1e-8
        entropy = -(masks_norm * torch.log(masks_norm + eps)).sum(dim=(2, 3)).mean()
        max_entropy = torch.log(torch.tensor(H * W, dtype=torch.float32, device=masks.device))
        entropy_penalty = entropy / max_entropy * 0.2

        return spatial_spread + boundary_smoothness + entropy_penalty

    def _compute_size_regularization_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Original size reg logic"""
        B, num_slots, H, W = masks.shape
        total_pixels = H * W
        coverage = masks.sum(dim=(2, 3)) / total_pixels
        too_small = F.relu(self.min_coverage - coverage)
        too_large = F.relu(coverage - self.max_coverage)
        return (too_small + too_large).mean()

    def _compute_edge_loss(self, masks: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Original edge loss logic"""
        B, num_slots, H, W = masks.shape
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
        gray = images.mean(dim=1, keepdim=True)
        img_edge_x = F.conv2d(gray, sobel_x, padding=1)
        img_edge_y = F.conv2d(gray, sobel_y, padding=1)
        img_edges = torch.sqrt(img_edge_x ** 2 + img_edge_y ** 2 + 1e-8)
        img_edges = img_edges / (img_edges.max() + 1e-8)
        masks_flat = masks.view(B * num_slots, 1, H, W)
        mask_edge_x = F.conv2d(masks_flat, sobel_x, padding=1)
        mask_edge_y = F.conv2d(masks_flat, sobel_y, padding=1)
        mask_edges = torch.sqrt(mask_edge_x ** 2 + mask_edge_y ** 2 + 1e-8)
        mask_edges = mask_edges.view(B, num_slots, H, W)
        img_edges_expanded = img_edges.expand(-1, num_slots, -1, -1)
        edge_misalignment = mask_edges * (1 - img_edges_expanded)
        return edge_misalignment.mean()

    def _compute_compactness_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Original compactness logic"""
        import math
        B, num_slots, H, W = masks.shape
        eps = 1e-8
        area = masks.sum(dim=(2, 3))
        tv_h = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :]).sum(dim=(2, 3))
        tv_w = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1]).sum(dim=(2, 3))
        perimeter = tv_h + tv_w
        quotient = (4 * math.pi * area) / (perimeter ** 2 + eps)
        quotient = torch.clamp(quotient, 0, 1)
        return (1 - quotient).mean()

    def _compute_concentration_loss(self, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute concentration penalty to reduce red-wash.
        1. Binary sharpness: mean(m * (1-m))
        2. Spatial entropy: penalize large spread
        3. Mask overlap: sum(m_i * m_j)
        """
        B, num_slots, H, W = masks.shape
        
        # 1. Binary Sharpness: Push values to 0 or 1
        sharpness_loss = torch.mean(masks * (1 - masks))
        
        # 2. Spatial Entropy: Penalize spatially large masks
        # Normalize each mask to a probability distribution
        masks_sum = masks.sum(dim=(2, 3), keepdim=True) + 1e-8
        masks_prob = masks / masks_sum
        entropy = -(masks_prob * torch.log(masks_prob + 1e-8)).sum(dim=(2, 3)) # [B, num_slots]
        # Normalize by max entropy log(H*W)
        max_entropy = torch.log(torch.tensor(H * W, dtype=torch.float32, device=masks.device))
        spatial_entropy_loss = (entropy / max_entropy).mean()
        
        # 3. Mask Overlap: Encourage slots to attend to distinct regions
        # masks: [B, S, H, W]
        masks_flat = masks.view(B, num_slots, -1) # [B, S, N]
        # Compute pairwise overlap matrix: [B, S, S]
        overlap = torch.bmm(masks_flat, masks_flat.transpose(1, 2))
        # Zero out diagonal (self-overlap is allowed/expected)
        mask_diag = 1 - torch.eye(num_slots, device=masks.device).unsqueeze(0)
        overlap_loss = (overlap * mask_diag).sum() / (B * num_slots * (num_slots - 1))
        
        return {
            'sharpness_loss': sharpness_loss,
            'spatial_entropy_loss': spatial_entropy_loss,
            'mask_overlap_loss': overlap_loss
        }

    def _compute_equivariance_loss(
        self, 
        images: torch.Tensor, 
        masks_orig: torch.Tensor,
        current_epoch: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute equivariance loss.
        1. Transform images -> images_trans
        2. Forward pass -> masks_trans
        3. Transform masks_orig -> masks_orig_trans
        4. Match slots (Hungarian) between masks_orig_trans and masks_trans
        5. Loss = MSE(matched_masks)
        
        Returns:
            loss: Scaler loss
            matching_cost: For logging
        """
        B, num_slots, H, W = masks_orig.shape
        
        # 1. Generate transforms
        # For memory efficiency, keep gradients? 
        # Standard equivariance training usually flows gradients through the transformed path too.
        
        theta, _ = get_affine_params(self.transforms_config, B, self.device)
        
        # 2. Transform images and forward pass
        images_trans = apply_transform(images, theta, mode='bilinear')
        
        # Forward pass on transformed images
        # We can either share gradients or stop gradient on target.
        # Usually we want the model to output consistent masks for transformed input.
        # So we want gradients for both.
        features_trans = self.backbone.get_feature_maps(images_trans)
        _, masks_trans, _, _ = self.slot_model(features_trans)
        
        # 3. Transform original masks to match the transformed view
        masks_orig_trans = apply_transform(masks_orig, theta, mode='bilinear')
        
        # 4. Hungarian Matching
        # Flatten spatial dims: [B, S, N]
        m1 = masks_trans.view(B, num_slots, -1)
        m2 = masks_orig_trans.view(B, num_slots, -1)
        
        # Cost C[i, j] = -(m1[i] . m2[j]) (maximize overlap)
        scores = torch.bmm(m1, m2.transpose(1, 2))  # [B, S, S]
        costs_np = -scores.detach().cpu().numpy()
        
        indices = []
        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(costs_np[b])
            indices.append(col_ind)
            
        # Reorder masks_orig_trans to match masks_trans
        masks_orig_trans_ordered = torch.zeros_like(masks_orig_trans)
        for b in range(B):
            order = indices[b]
            masks_orig_trans_ordered[b] = masks_orig_trans[b, order]
            
        # 5. Compute MSE Loss
        loss = F.mse_loss(masks_trans, masks_orig_trans_ordered)
        
        # Calculate matching cost
        avg_cost = -torch.mean(torch.stack([scores[b, i, indices[b][i]] for b in range(B) for i in range(num_slots)]))
        
        return loss, avg_cost.item()

    def compute_loss(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses with warmups
        """
        loss_dict = {}
        
        # 1. Base Reconstruction Loss
        recon_loss = F.mse_loss(recon, images)
        loss_dict['recon_loss'] = recon_loss
        
        total_loss = self.recon_weight * recon_loss
        
        # 2. Original Regularizers
        slots_norm = F.normalize(slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        B, num_slots = slots.shape[:2]
        mask_mat = 1 - torch.eye(num_slots, device=self.device)
        diversity_loss = (similarity.abs() * mask_mat.unsqueeze(0)).sum() / (B * num_slots * (num_slots - 1))
        loss_dict['diversity_loss'] = diversity_loss
        total_loss += self.diversity_weight * diversity_loss
        
        spatial_loss = self._compute_spatial_coherence_loss(masks)
        loss_dict['spatial_coherence_loss'] = spatial_loss
        total_loss += self.spatial_weight * spatial_loss
        
        size_loss = self._compute_size_regularization_loss(masks)
        loss_dict['size_reg_loss'] = size_loss
        total_loss += self.size_reg_weight * size_loss
        
        if self.edge_weight > 0:
            edge_loss = self._compute_edge_loss(masks, images)
            loss_dict['edge_loss'] = edge_loss
            total_loss += self.edge_weight * edge_loss
            
        if self.compactness_weight > 0:
            compact_loss = self._compute_compactness_loss(masks)
            loss_dict['compactness_loss'] = compact_loss
            total_loss += self.compactness_weight * compact_loss
            
        # 3. New: Concentration & Sparsity (with Warmup)
        if hasattr(self, 'concentration_warmup') and epoch > self.concentration_warmup:
            conc_metrics = self._compute_concentration_loss(masks)
            
            w_sharp = self.concentration_weight
            w_spat = self.spatial_entropy_weight
            w_over = self.mask_overlap_weight
            
            conc_loss_total = (
                w_sharp * conc_metrics['sharpness_loss'] + 
                w_spat * conc_metrics['spatial_entropy_loss'] + 
                w_over * conc_metrics['mask_overlap_loss']
            )
            
            total_loss += conc_loss_total
            loss_dict['concentration_loss'] = conc_loss_total
            loss_dict['sharpness_loss'] = conc_metrics['sharpness_loss']
            loss_dict['spatial_entropy_loss'] = conc_metrics['spatial_entropy_loss']
            loss_dict['mask_overlap_loss'] = conc_metrics['mask_overlap_loss']
        else:
            loss_dict['concentration_loss'] = torch.tensor(0.0, device=self.device)
            
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.backbone.train()
        self.slot_model.train()
        
        epoch_losses = {
            'total_loss': 0.0, 
            'recon_loss': 0.0, 
            'diversity_loss': 0.0,
            'spatial_coherence_loss': 0.0,
            'size_reg_loss': 0.0,
            'edge_loss': 0.0,
            'compactness_loss': 0.0,
            'concentration_loss': 0.0,
            'equivariance_loss': 0.0,
            'matching_cost': 0.0
        }
        
        # Track active slots
        total_active_slots = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            features = self.backbone.get_feature_maps(images)
            recon, masks, slots, attn = self.slot_model(features)
            
            # Compute base losses
            losses = self.compute_loss(images, recon, masks, slots, epoch)
            total_loss = losses['total_loss']
            
            # Compute Equivariance Loss (with Warmup)
            eq_loss = torch.tensor(0.0, device=self.device)
            match_cost = 0.0
            
            if hasattr(self, 'equivariance_warmup') and epoch > self.equivariance_warmup:
                eq_loss, match_cost = self._compute_equivariance_loss(images, masks, epoch)
                total_loss += self.equivariance_weight * eq_loss
                losses['equivariance_loss'] = eq_loss
            else:
                losses['equivariance_loss'] = eq_loss
                
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.slot_model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            slot_activity = masks.mean(dim=(2, 3)) # [B, S]
            active_slots = (slot_activity > 0.01).float().sum(dim=1).mean().item()
            total_active_slots += active_slots
            num_batches += 1
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    val = losses[key]
                    epoch_losses[key] += val.item() if torch.is_tensor(val) else val
            epoch_losses['matching_cost'] += match_cost

            # Logging
            if batch_idx % self.log_every == 0:
                if self.use_tracking:
                    metrics = {
                        f'train_{k}': v.item() if torch.is_tensor(v) else v 
                        for k, v in losses.items()
                    }
                    metrics['train_matching_cost'] = match_cost
                    metrics['train_active_slots'] = active_slots
                    metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                    if batch_idx % self.visualize_every == 0:
                        try:
                            import torchvision.utils as vutils
                            with torch.no_grad():
                                # Take first sample
                                img = images[0].cpu().detach()
                                rec = recon[0].cpu().detach()
                                msk = masks[0].cpu().detach() # [S, H, W]
                                
                                # Denormalize image if necessary (assuming standard mean/std)
                                # For visualization, just clip to 0-1
                                img = torch.clamp(img, 0, 1)
                                rec = torch.clamp(rec, 0, 1)
                                
                                # Log original and recon
                                mlflow.log_image(img.permute(1, 2, 0).numpy(), f"epoch_{epoch}_step_{self.global_step}_orig.png")
                                mlflow.log_image(rec.permute(1, 2, 0).numpy(), f"epoch_{epoch}_step_{self.global_step}_recon.png")
                                
                                # Create a grid of masks (unsqueeze to make them [S, 1, H, W])
                                # Use separate grid for masks
                                masks_grid = vutils.make_grid(msk.unsqueeze(1), normalize=True, padding=2, nrow=5)
                                mlflow.log_image(masks_grid.permute(1, 2, 0).numpy(), f"epoch_{epoch}_step_{self.global_step}_masks.png")
                        except Exception as e:
                            print(f"Visualization failed: {e}")
                    mlflow.log_metrics(metrics, step=self.global_step)
                
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'eq': f"{eq_loss.item():.4f}",
                    'conc': f"{losses.get('concentration_loss', 0):.4f}"
                })
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            
        epoch_losses['avg_active_slots'] = total_active_slots / num_batches if num_batches > 0 else 0
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set"""
        self.backbone.eval()
        self.slot_model.eval()
        
        val_losses = {k: 0.0 for k in [
            'total_loss', 'recon_loss', 'diversity_loss', 'spatial_coherence_loss',
            'size_reg_loss', 'edge_loss', 'compactness_loss', 
            'concentration_loss', 'sharpness_loss', 'spatial_entropy_loss', 'mask_overlap_loss'
        ]}
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                
                features = self.backbone.get_feature_maps(images)
                recon, masks, slots, attn = self.slot_model(features)
                
                # For validation, we can assume epoch=999 to compute all losses for monitoring
                losses = self.compute_loss(images, recon, masks, slots, epoch=999)
                
                for key in val_losses:
                    if key in losses:
                        val = losses[key]
                        val_losses[key] += val.item() if torch.is_tensor(val) else val
                        
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop"""
        print(f"\nStarting Part Discovery Training...")
        print(f"Epochs: {self.epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            train_losses = self.train_epoch(train_loader, epoch)
            
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                
                if self.use_tracking:
                    val_metrics = {f'val_{k}': v for k, v in val_losses.items()}
                    mlflow.log_metrics(val_metrics, step=epoch)
                
                print(f"\nEpoch {epoch}/{self.epochs}:")
                # Format to avoid errors if keys missing
                eq_log = train_losses.get('equivariance_loss', 0)
                conc_log = train_losses.get('concentration_loss', 0)
                print(f"  Train: {train_losses['total_loss']:.4f} (Eq: {eq_log:.4f}, Conc: {conc_log:.4f})")
                print(f"  Val: {val_losses['total_loss']:.4f}")
                
                if val_losses['total_loss'] < self.best_loss:
                    self.best_loss = val_losses['total_loss']
                    self.save_checkpoint(epoch, val_losses['total_loss'], is_best=True)
                    print(f"  New best model saved! Loss: {self.best_loss:.4f}")
            else:
                print(f"\nEpoch {epoch}/{self.epochs}: Train {train_losses['total_loss']:.4f}")
            
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, train_losses['total_loss'], is_best=False)
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("\nTraining completed!")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save checkpoint"""
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        save_checkpoint(
            model=nn.ModuleDict({
                'backbone': self.backbone,
                'slot_model': self.slot_model
            }),
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            save_path=str(path),
            additional_info={
                'global_step': self.global_step,
                'best_loss': self.best_loss
            }
        )
