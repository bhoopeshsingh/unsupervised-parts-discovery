# Unsupervised Parts Discovery Pipeline

## Quick Start (4-Day Plan)

### Setup

```bash
cd /Users/dev/ml-projects/unsupervised-parts-discovery

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### Day 1: Test Data & Start Part Discovery Training

```bash
# Test data loading (verify CIFAR-10 subset loads correctly)
python experiments/test_data_loading.py

# Start part discovery training (will take several hours)
python experiments/train_part_discovery.py
```

### Day 2: Continue Part Discovery (if needed) + Start Classification

```bash
# Train classification model
python experiments/train_classifier.py
```

### Day 3: Extract Parts & Cluster

```bash
# Extract parts from trained model
python experiments/extract_parts.py --visualize --num-samples 10

# Cluster parts
python experiments/cluster_parts.py

# Launch Streamlit labeling interface
streamlit run src/clustering/streamlit_labeler.py
```

### Day 4: Analysis & Linking

```bash
# Link parts to classes and generate interpretability report
python experiments/link_parts_to_classes.py
```

## Project Status

### ✅ Completed (Phase 1)
- [x] Project structure
- [x] Configuration files (data, model, training)
- [x] CIFAR-10 subset data pipeline
- [x] ResNet50 backbone
- [x] Slot Attention model
- [x] Classifier model
- [x] Training pipelines (part discovery + classification)
- [x] W&B integration
- [x] Part discovery model trained (30 epochs)

### ✅ Completed (Phase 2)
- [x] Part extraction from trained Slot Attention model
- [x] Clustering module with optimal K selection
- [x] Quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- [x] t-SNE visualization
- [x] Streamlit labeling interface
- [x] Cluster-class correlation analysis
- [x] Linking mechanisms and interpretability reports

### 🔄 Next Steps
- [ ] Train classification model (if not done)
- [ ] Run full pipeline on extracted parts
- [ ] Complete semantic labeling of all clusters
- [ ] Generate final interpretability analysis

## Configuration

Edit `configs/*.yaml` files to modify:
- **data_config.yaml**: Dataset classes, augmentation, batch size
- **model_config.yaml**: Model architectures, num_slots, dimensions
- **training_config.yaml**: Training hyperparameters, W&B settings

## Hardware Note

Models are configured for **M4 Mac (MPS)**. Training times:
- Part Discovery: ~2-3 hours (30 epochs)
- Classification: ~1-2 hours (50 epochs)

## Monitoring

View training progress at: https://wandb.ai

Tracks:
- Reconstruction loss
- Diversity loss  
- Classification accuracy
- Per-class metrics

---

**Target**: Complete Phase 1-4 in 4 days (Nov 26-29, 2024)