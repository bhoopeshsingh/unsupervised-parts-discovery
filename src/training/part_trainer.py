"""Part Discovery Training Pipeline"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import wandb

from ..models.backbone import ResNetBackbone
from ..models.part_discovery.slot_attention import SlotAttentionModel
from ..utils import save_checkpoint, count_parameters


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
        use_wandb: bool = True
    ):
        self.backbone = backbone.to(device)
        self.slot_model = slot_model.to(device)
        self.device = device
        self.config = config
        self.use_wandb = use_wandb
        
        # Loss weights
        loss_config = config['slot_attention']['loss']
        self.recon_weight = loss_config['reconstruction_weight']
        self.diversity_weight = loss_config.get('diversity_weight', 0.1)
        
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
    
    def compute_loss(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses
        
        Args:
            images: Original images [B, 3, H, W]
            recon: Reconstructed images [B, 3, H, W]
            masks: Slot masks [B, num_slots, H, W]
            slots: Slot representations [B, num_slots, D]
        
        Returns:
            Dict of losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, images)
        
        # Diversity loss: encourage different slots to be different
        # Compute pairwise cosine similarity between slots
        slots_norm = F.normalize(slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, num_slots, num_slots]
        
        # Penalize high similarity (exclude diagonal)
        B, num_slots = slots.shape[:2]
        mask_mat = 1 - torch.eye(num_slots, device=self.device)
        diversity_loss = (similarity.abs() * mask_mat.unsqueeze(0)).sum() / (B * num_slots * (num_slots - 1))
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + self.diversity_weight * diversity_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'diversity_loss': diversity_loss
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.backbone.train()
        self.slot_model.train()
        
        epoch_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'diversity_loss': 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get backbone features (spatial)
            features = self.backbone.get_feature_maps(images)  # [B, 2048, H', W']
            
            # Slot attention
            recon, masks, slots, attn = self.slot_model(features)
            
            # Compute losses
            losses = self.compute_loss(images, recon, masks, slots)
            
            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Logging
            if batch_idx % self.log_every == 0:
                if self.use_wandb:
                    wandb.log({
                        'train/total_loss': losses['total_loss'].item(),
                        'train/recon_loss': losses['recon_loss'].item(),
                        'train/diversity_loss': losses['diversity_loss'].item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
                
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'recon': f"{losses['recon_loss'].item():.4f}"
                })
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set"""
        self.backbone.eval()
        self.slot_model.eval()
        
        val_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'diversity_loss': 0.0}
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                
                features = self.backbone.get_feature_maps(images)
                recon, masks, slots, attn = self.slot_model(features)
                
                losses = self.compute_loss(images, recon, masks, slots)
                
                for key in val_losses:
                    val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        return val_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """Full training loop"""
        print(f"\nStarting Part Discovery Training...")
        print(f"Epochs: {self.epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                
                if self.use_wandb:
                    wandb.log({
                        'val/total_loss': val_losses['total_loss'],
                        'val/recon_loss': val_losses['recon_loss'],
                        'val/diversity_loss': val_losses['diversity_loss'],
                        'epoch': epoch
                    })
                
                print(f"\nEpoch {epoch}/{self.epochs}:")
                print(f"  Train Loss: {train_losses['total_loss']:.4f}")
                print(f"  Val Loss: {val_losses['total_loss']:.4f}")
                
                # Save best model
                if val_losses['total_loss'] < self.best_loss:
                    self.best_loss = val_losses['total_loss']
                    self.save_checkpoint(epoch, val_losses['total_loss'], is_best=True)
                    print(f"  New best model saved! Loss: {self.best_loss:.4f}")
            else:
                print(f"\nEpoch {epoch}/{self.epochs}:")
                print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            
            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, train_losses['total_loss'], is_best=False)
            
            # Update scheduler
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
