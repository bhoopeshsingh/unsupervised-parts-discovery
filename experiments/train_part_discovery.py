"""Train Part Discovery Model with Slot Attention"""

import sys
sys.path.append('.')

import torch
import mlflow
import time
from pathlib import Path

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.training.part_trainer import PartDiscoveryTrainer
from src.utils import load_config, get_device, set_seed


def main():
    # Load configurations
    # Load configurations
    unified_config = load_config('configs/unified_config.yaml')
    
    # Extract sections for compatibility
    data_config = unified_config
    model_config = unified_config
    training_config = unified_config
    
    # Set seed
    set_seed(data_config.get('seed', 42))
    
    # Device
    device = get_device(model_config.get('device', 'auto'))
    
    # Prepare configuration for training
    full_config = {
        **model_config,
        **training_config
    }
    
    # Initialize MLFlow
    mlflow_config = training_config['mlflow']
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    if mlflow_config.get('enabled', True):
        run_name = f"{mlflow_config.get('run_name_prefix', 'run')}_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Log all hyperparameters (flatten nested dicts)
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float, str, bool)):
                    items.append((new_key, v))
            return dict(items)
        
        mlflow.log_params(flatten_dict(model_config))
        mlflow.log_params({f"train.{k}": v for k, v in flatten_dict(training_config['part_discovery']).items()})
        
        # Set tags
        for key, value in mlflow_config.get('tags', {}).items():
            mlflow.set_tag(key, value)
        if 'notes' in mlflow_config:
            mlflow.set_tag('notes', mlflow_config['notes'])
        
        use_tracking = True
    else:
        use_tracking = False
    
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
        use_tracking=use_tracking
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Finish MLFlow run
    if use_tracking:
        mlflow.end_run()
    
    print("\nPart discovery training completed!")
    print(f"Best model saved at: {trainer.checkpoint_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
