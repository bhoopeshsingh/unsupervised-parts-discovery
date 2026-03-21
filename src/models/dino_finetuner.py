# src/models/dino_finetuner.py
"""
Fine-tunes DINO transformer blocks using cluster pseudo-labels.
Semantic consistency loss: same cluster → similar embeddings, different cluster → push apart.

Single-layer mode (use_multilayer=False): unfreezes blocks 10, 11 (layer 12 only).
Multi-layer mode  (use_multilayer=True):  unfreezes blocks 7, 9, 11 (layers 8, 10, 12) —
  the same blocks whose outputs are concatenated during feature extraction.

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
    Fine-tunes DINO transformer blocks using cluster assignments as pseudo-labels.
    Semantic consistency loss: same cluster → similar embeddings,
    different cluster → push apart.

    use_multilayer=True  → extracts layers 8, 10, 12. Unfreezes blocks 9 and 11 only.
                           Block 7 (layer 8 = texture) stays frozen — its pretrained
                           texture features are too valuable to disturb with pseudo-labels.
    use_multilayer=False → extracts layer 12 only. Unfreezes last 2 blocks (10, 11).
    """

    # Mirror DinoExtractor: get_intermediate_layers(n=5) index → block index
    _MULTILAYER_INDICES = (0, 2, 4)   # → blocks 7, 9, 11 (layers 8, 10, 12)

    # Block 7 (layer 8) produces pretrained texture features crucial for part separation.
    # Fine-tuning it with a weak pseudo-label signal degrades these representations.
    # Only unfreeze blocks 9 and 11 — adapt structure+semantics, preserve texture.
    _MULTILAYER_BLOCKS  = ("blocks.9", "blocks.11")
    _SINGLELAYER_BLOCKS = ("blocks.10", "blocks.11")

    def __init__(self, dino_model, device, lr=1e-5, use_multilayer=False):
        self.model = dino_model
        self.device = device
        self.use_multilayer = use_multilayer

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Bug 3 fix: unfreeze the blocks that actually contribute to the extracted features
        unfreeze = self._MULTILAYER_BLOCKS if use_multilayer else self._SINGLELAYER_BLOCKS
        for name, param in self.model.named_parameters():
            if any(b in name for b in unfreeze):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        block_desc = "blocks 9,11 (layers 10,12 — block 7/layer 8 frozen)" if use_multilayer else "blocks 10,11"
        print(f"Trainable params: {trainable:,}  ({block_desc})")

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
        n_patches = 784  # 28×28 spatial grid

        for batch_imgs, _ in dataloader:
            batch_imgs = batch_imgs.to(self.device)
            B = batch_imgs.shape[0]

            # Bug 1 fix: match the same feature extraction used during inference
            if self.use_multilayer:
                all_layers = self.model.get_intermediate_layers(batch_imgs, n=5)
                feats = torch.cat(
                    [all_layers[i][:, 1:, :] for i in self._MULTILAYER_INDICES], dim=-1
                )  # [B, 784, 1152]
            else:
                feats = self.model.get_intermediate_layers(batch_imgs, n=1)[0]
                feats = feats[:, 1:, :]      # [B, 784, 384]

            flat_feats = feats.reshape(B * n_patches, -1)   # [B*784, D]

            # Bug 2 fix: generate patch_ids so the clusterer can append spatial dims
            # patch_ids[i] = 0..783, repeating for each image in the batch
            patch_ids = torch.arange(n_patches).repeat(B)   # [B*784]

            # Get pseudo-labels from clusterer (no_grad — clusterer is fixed)
            with torch.no_grad():
                pseudo_labels = torch.tensor(
                    clusterer.predict(flat_feats.detach().cpu(), patch_ids=patch_ids),
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
