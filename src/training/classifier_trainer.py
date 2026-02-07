"""Classification Training Pipeline"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import mlflow

from ..models.backbone import ResNetBackbone
from ..models.classifier import Classifier
from ..utils import save_checkpoint, count_parameters


class ClassifierTrainer:
    """
    Trainer for supervised classification
    """
    
    def __init__(
        self,
        backbone: ResNetBackbone,
        classifier: Classifier,
        device: torch.device,
        config: Dict[str, Any],
        use_tracking: bool = True
    ):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.config = config
        self.use_tracking = use_tracking
        
        # Training config
        train_config = config['classification']
        self.epochs = train_config['epochs']
        self.log_every = train_config['log_every']
        self.checkpoint_dir = Path(train_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_config['save_every']
        
        # Combine parameters
        params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            params,
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # Scheduler
        scheduler_config = train_config['scheduler']
        if scheduler_config['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.epochs)
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.best_acc = 0.0
        self.global_step = 0
        
        print(f"\nClassifierTrainer initialized:")
        print(f"  Backbone parameters: {count_parameters(self.backbone):,}")
        print(f"  Classifier parameters: {count_parameters(self.classifier):,}")
        print(f"  Total parameters: {count_parameters(self.backbone) + count_parameters(self.classifier):,}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.backbone.train()
        self.classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            features = self.backbone(images)
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Logging
            if batch_idx % self.log_every == 0:
                acc = 100.0 * correct / total
                if self.use_tracking:
                    mlflow.log_metrics({
                        'train_loss': loss.item(),
                        'train_accuracy': acc,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc:.2f}%"
                })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set"""
        self.backbone.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = [0] * self.classifier.num_classes
        class_total = [0] * self.classifier.num_classes
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.backbone(images)
                logits = self.classifier(features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class stats
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100.0 * correct / total
        
        # Per-class accuracy
        class_acc = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(self.classifier.num_classes)]
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'class_accuracy': class_acc
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """Full training loop"""
        print(f"\nStarting Classification Training...")
        print(f"Epochs: {self.epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                if self.use_tracking:
                    log_dict = {
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy']
                    }
                    # Log per-class accuracy
                    for i, acc in enumerate(val_metrics['class_accuracy']):
                        log_dict[f'val_class_{i}_accuracy'] = acc
                    mlflow.log_metrics(log_dict, step=epoch)
                
                print(f"\nEpoch {epoch}/{self.epochs}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  Per-class Val Acc: {[f'{acc:.1f}%' for acc in val_metrics['class_accuracy']]}")
                
                # Save best model
                if val_metrics['accuracy'] > self.best_acc:
                    self.best_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                    print(f"  New best model saved! Acc: {self.best_acc:.2f}%")
            else:
                print(f"\nEpoch {epoch}/{self.epochs}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, train_metrics['loss'], is_best=False)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save checkpoint"""
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        save_checkpoint(
            model=nn.ModuleDict({
                'backbone': self.backbone,
                'classifier': self.classifier
            }),
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            save_path=str(path),
            additional_info={
                'global_step': self.global_step,
                'best_acc': self.best_acc
            }
        )
