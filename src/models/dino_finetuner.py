# src/models/dino_finetuner.py
"""
Fine-tunes last 2 DINO transformer blocks using cluster pseudo-labels.
Semantic consistency loss: same cluster → similar embeddings,
different cluster → push apart.

Usage (via run_pipeline.py):
  python experiments/run_pipeline.py --stage finetune
  python experiments/run_pipeline.py --stage extract   # re-extract with fine-tuned weights
"""
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Simple image dataset for fine-tuning
# ---------------------------------------------------------------------------

class ImagePathDataset(Dataset):
    """Loads images from a list of paths — no labels needed."""
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, image_paths: list, image_size: int = 224):
        self.paths = image_paths
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class DinoSemanticFinetuner:
    """
    Fine-tunes last 2 DINO blocks using cluster assignments as pseudo-labels.
    Domain knowledge = the semantic cluster structure you discovered.
    """

    def __init__(self, dino_model, device, lr=1e-5):
        self.model = dino_model
        self.device = device

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only last 2 transformer blocks
        for name, param in self.model.named_parameters():
            if "blocks.10" in name or "blocks.11" in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}  (last 2 blocks only)")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

    def semantic_consistency_loss(self, embeddings, pseudo_labels,
                                  margin=0.5, n_pairs=512):
        """
        Contrastive loss:
          same cluster  → pull embeddings together  (sim → 1)
          diff cluster  → push apart                (sim < margin)

        embeddings:    (N, D) patch features — L2-normalised
        pseudo_labels: (N,)  cluster ids
        """
        embeddings = F.normalize(embeddings, dim=1)
        N = embeddings.shape[0]
        idx_a = torch.randint(0, N, (n_pairs,), device=self.device)
        idx_b = torch.randint(0, N, (n_pairs,), device=self.device)

        sim  = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1)
        same = (pseudo_labels[idx_a] == pseudo_labels[idx_b]).float()

        loss = same * (1 - sim) + (1 - same) * F.relu(sim - margin)
        return loss.mean()

    def train_epoch(self, dataloader, clusterer, n_pairs: int = 512):
        """
        One epoch of semantic consistency fine-tuning.

        Args:
            dataloader: yields (img_tensor [B,3,H,W], paths)
            clusterer:  fitted PatchClusterer — used to assign pseudo-labels
        Returns:
            mean loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for batch_imgs, _ in dataloader:
            batch_imgs = batch_imgs.to(self.device)
            B = batch_imgs.shape[0]

            # Extract patch features from current model state
            feats = self.model.get_intermediate_layers(batch_imgs, n=1)[0]
            # feats: [B, 785, 384] — index 0 is CLS
            feats = feats[:, 1:, :]          # [B, 784, 384]
            flat_feats = feats.reshape(B * 784, -1)

            # Get pseudo-labels from clusterer (no_grad — clusterer is fixed)
            with torch.no_grad():
                pseudo_labels = torch.tensor(
                    clusterer.predict(flat_feats.detach().cpu()),
                    device=self.device,
                )

            loss = self.semantic_consistency_loss(flat_feats, pseudo_labels,
                                                   n_pairs=n_pairs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def save(self, path: str):
        """Save fine-tuned model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Fine-tuned DINO weights saved → {path}")

    @staticmethod
    def load_weights_into_extractor(extractor, weights_path: str):
        """Load saved fine-tuned weights back into a DinoExtractor."""
        state = torch.load(weights_path, map_location=extractor.device,
                           weights_only=True)
        extractor.model.load_state_dict(state)
        extractor.model.eval()
        print(f"Fine-tuned weights loaded from {weights_path}")
