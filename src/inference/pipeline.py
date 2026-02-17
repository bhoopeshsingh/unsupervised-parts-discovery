import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.utils import load_config, get_device, load_checkpoint

class InferencePipeline:
    def __init__(self, 
                 model_config_path='configs/model_config.yaml',
                 data_config_path='configs/data_config.yaml',
                 checkpoint_path=None,
                 clusters_dir=None):
        
        self.device = get_device('auto')
        self.model_config = load_config(model_config_path)
        
        # Load data config for paths
        self.data_config = load_config(data_config_path)
        paths = self.data_config.get('paths', {})
        
        if checkpoint_path is None:
            checkpoint_path = paths.get('best_model', 'checkpoints/cat_parts_improved/best_model.pt')
            
        if clusters_dir is None:
            clusters_dir = paths.get('clusters', 'parts/clusters_cat_improved')
            
        self.clusters_dir = Path(clusters_dir)
        
        # Load Models
        print(f"Loading checkpoint from {checkpoint_path}...")
        # Use config resolution (128x128)
        # self.model_config['slot_attention']['decoder']['output_size'] = 32
        
        self.backbone = ResNetBackbone.from_config(self.model_config['backbone'])
        self.slot_model = SlotAttentionModel.from_config(self.model_config)
        
        # Checkpoint is likely 32x32, but config is 128x128
        # We need to handle this mismatch if we want to run the old checkpoint
        # But we can't easily check the checkpoint before loading...
        # Actually we can catch the error, but better to just override if we know we are using the old checkpoint
        
        # Override for compatibility with old checkpoint if needed
        # self.model_config['slot_attention']['decoder']['output_size'] = 32
        
        # Load Checkpoint
        # Handle the shared state dict logic we discovered earlier
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        slot_dict = {k.replace('slot_model.', ''): v for k, v in state_dict.items() if k.startswith('slot_model.')}
        
        self.backbone.load_state_dict(backbone_dict)
        
        # Handle potential resolution mismatch in checkpoint (e.g. loading 32x32 checkpoint for 128x128 model)
        if 'decoder.pos_grid' in slot_dict:
            if slot_dict['decoder.pos_grid'].shape != self.slot_model.decoder.pos_grid.shape:
                print(f"Ignoring checkpoint pos_grid due to shape mismatch: {slot_dict['decoder.pos_grid'].shape} vs {self.slot_model.decoder.pos_grid.shape}")
                del slot_dict['decoder.pos_grid']
        
        # Use strict=False to allow missing pos_grid if we deleted it
        self.slot_model.load_state_dict(slot_dict, strict=False)
        
        self.backbone.to(self.device).eval()
        self.slot_model.to(self.device).eval()
        
        # Load Cluster Metadata (Centroids & Labels)
        self.load_cluster_data()
        
        # Transforms
        img_size = self.model_config['slot_attention']['decoder']['output_size']
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
    def load_cluster_data(self):
        """Load centroids and semantic labels"""
        print(f"Loading cluster data from {self.clusters_dir}")
        
        # Load metadata with centroids
        with open(self.clusters_dir / 'cluster_metadata.json', 'r') as f:
            self.cluster_metadata = json.load(f)
            
        # Load user labels if they exist
        labels_path = Path('parts/labels/cluster_labels.json')
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                self.user_labels = json.load(f)
        else:
            self.user_labels = {}
            
        # Prepare centroids matrix for fast matching
        # Map index -> global_cluster_id
        self.centroid_map = [] 
        centroids = []
        
        # Handle if cluster_metadata is list or dict
        if isinstance(self.cluster_metadata, list):
            # It's a list of dicts, likely from the new per-class clustering
            # We need to index it by cluster ID
            temp_metadata = {}
            for i, data in enumerate(self.cluster_metadata):
                # Use 'cluster_id' if available, else index
                cid = data.get('cluster_id', i)
                temp_metadata[str(cid)] = data
            self.cluster_metadata = temp_metadata
            
        for global_id, data in self.cluster_metadata.items():
            # Skip summary stats (which are not dicts)
            if not isinstance(data, dict):
                continue
                
            if 'centroid' in data:
                centroids.append(data['centroid'])
                self.centroid_map.append(int(global_id))
                
        if centroids:
            self.centroids = np.array(centroids)
            # Normalize centroids for cosine similarity
            from sklearn.preprocessing import normalize
            self.centroids = normalize(self.centroids, norm='l2', axis=1)
        else:
            print("WARNING: No centroids found in metadata!")
            self.centroids = None

    def load_labels(self):
        """Reload user labels from disk"""
        labels_path = Path('parts/labels/cluster_labels.json')
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                self.user_labels = json.load(f)
            print(f"Loaded {len(self.user_labels)} labels from {labels_path}")
        else:
            self.user_labels = {}
            print(f"Labels file not found at {labels_path}")

    def process_image(self, image):
        """
        Run inference on a single image
        
        Args:
            image: PIL Image or path
            
        Returns:
            dict with results
        """
        # Reload labels to ensure we have the latest updates
        self.load_labels()
        
        # Handle different input types
        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                print(f"Error opening image: {e}")
                raise ValueError(f"Could not open image. Input type: {type(image)}")
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.backbone.get_feature_maps(img_tensor)
            recon, masks, slots, attn = self.slot_model(features)
            
        # Post-process
        slots = slots.cpu().numpy()[0] # [num_slots, dim]
        masks = masks.cpu().numpy()[0] # [num_slots, H, W]
        recon = recon.cpu().numpy()[0].transpose(1, 2, 0) # [H, W, 3]
        
        # Denormalize recon for display
        recon = (recon - recon.min()) / (recon.max() - recon.min())
        
        # Match slots to clusters
        results = []
        if self.centroids is not None:
            # Normalize slots
            from sklearn.preprocessing import normalize
            slots_norm = normalize(slots, norm='l2', axis=1)
            
            # Compute similarity
            similarities = cosine_similarity(slots_norm, self.centroids) # [num_slots, num_clusters]
            
            for i in range(len(slots)):
                best_idx = np.argmax(similarities[i])
                global_id = self.centroid_map[best_idx]
                score = similarities[i, best_idx]
                
                # Get semantic label
                label_info = self.user_labels.get(str(global_id), {})
                semantic_label = label_info.get('label', f"Cluster {global_id}")
                
                # Debug print
                print(f"Slot {i}: Cluster {global_id}, Label: {semantic_label}, Found in dict: {str(global_id) in self.user_labels}")
                
                results.append({
                    'slot_idx': i,
                    'cluster_id': global_id,
                    'score': float(score),
                    'label': semantic_label,
                    'mask': masks[i]
                })
                
        # Predict Class
        predicted_class = "Unknown"
        class_probs = {}
        
        if results:
            # Simple voting: Sum scores for each class associated with the detected clusters
            # We need the cluster -> class mapping from metadata
            cluster_classes = {}
            for gid, meta in self.cluster_metadata.items():
                # Skip summary stats
                if not isinstance(meta, dict):
                    continue
                    
                if 'class_name' in meta: # Assuming we saved this
                    cluster_classes[int(gid)] = meta['class_name']
                # Fallback: check dominant class from distribution if available
                elif 'class_distribution' in meta:
                    dist = meta['class_distribution']
                    # Find max class
                    max_cls = max(dist.items(), key=lambda x: x[1])[0]
                    cluster_classes[int(gid)] = max_cls
            
            scores = {}
            for res in results:
                cid = res['cluster_id']
                if cid in cluster_classes:
                    cls = cluster_classes[cid]
                    scores[cls] = scores.get(cls, 0) + res['score']
            
            if scores:
                predicted_class = max(scores.items(), key=lambda x: x[1])[0]
                total_score = sum(scores.values())
                class_probs = {k: v/total_score for k, v in scores.items()}

        return {
            'original': image,
            'reconstruction': recon,
            'parts': results,
            'predicted_class': predicted_class,
            'class_probs': class_probs
        }
