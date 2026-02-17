"""Train Classification Model"""

import sys
sys.path.append('.')

import torch
import mlflow
import time
from pathlib import Path

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.classifier import Classifier
from src.training.classifier_trainer import ClassifierTrainer
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
        run_name = f"classifier_{time.strftime('%Y%m%d_%H%M%S')}"
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
        mlflow.log_params({f"train.{k}": v for k, v in flatten_dict(training_config['classification']).items()})
        
        # Set tags
        for key, value in mlflow_config.get('tags', {}).items():
            mlflow.set_tag(key, value)
        mlflow.set_tag('task', 'classification')
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
    classifier = Classifier.from_config(model_config['classifier'])
    
    # Create trainer
    trainer = ClassifierTrainer(
        backbone=backbone,
        classifier=classifier,
        device=device,
        config=full_config,
        use_tracking=use_tracking
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Finish MLFlow run
    if use_tracking:
        mlflow.end_run()
    
    print("\nClassification training completed!")
    print(f"Best model saved at: {trainer.checkpoint_dir / 'best_model.pt'}")
    print(f"Best accuracy: {trainer.best_acc:.2f}%")


if __name__ == '__main__':
    main()
