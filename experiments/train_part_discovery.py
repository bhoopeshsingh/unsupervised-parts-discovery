"""Train Part Discovery Model with Slot Attention"""

import sys
sys.path.append('.')

import torch
import wandb
from pathlib import Path

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.training.part_trainer import PartDiscoveryTrainer
from src.utils import load_config, get_device, set_seed


def main():
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    
    # Set seed
    set_seed(data_config.get('seed', 42))
    
    # Device
    device = get_device(model_config.get('device', 'auto'))
    
    # Prepare configuration for training
    full_config = {
        **model_config,
        **training_config
    }
    
    # Initialize W&B
    wandb_config = training_config['wandb']
    if wandb_config['mode'] != 'disabled':
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            config=full_config,
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            mode=wandb_config['mode']
        )
        use_wandb = True
    else:
        use_wandb = False
    
    # Create dataloaders
    print("\nPreparing datasets...")
    train_loader, val_loader = create_dataloaders(
        data_config['dataset'],
        data_config['augmentation'],
        data_config['dataloader']
    )
    
    # Create models
    print("\nInitializing models...")
    backbone = ResNetBackbone.from_config(model_config['backbone'])
    slot_model = SlotAttentionModel.from_config(model_config)
    
    # Create trainer
    trainer = PartDiscoveryTrainer(
        backbone=backbone,
        slot_model=slot_model,
        device=device,
        config=full_config,
        use_wandb=use_wandb
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
    
    print("\nPart discovery training completed!")
    print(f"Best model saved at: {trainer.checkpoint_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
