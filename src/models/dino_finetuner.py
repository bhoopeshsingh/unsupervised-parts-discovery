# src/models/dino_finetuner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoSemanticFinetuner:
    """
    Fine-tunes last 2 DINO blocks using cluster assignments as pseudo-labels.
    Domain knowledge = the semantic cluster structure you discovered.
    """

    def __init__(self, dino_model, device, lr=1e-5):
        self.model = dino_model
        self.device = device

        # Freeze everything except last 2 blocks
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Unfreeze only last 2 transformer blocks
        for name, param in self.model.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        print(f"Trainable params: {trainable:,} (last 2 blocks only)")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.01
        )

    def semantic_consistency_loss(self, embeddings, pseudo_labels,
                                  margin=0.5, n_pairs=512):
        """
        Contrastive loss: same concept → similar embeddings,
                          different concept → push apart.
        embeddings:    (N, D) patch features
        pseudo_labels: (N,)   cluster ids
        """
        embeddings = F.normalize(embeddings, dim=1)

        # Sample random pairs
        N = embeddings.shape[0]
        idx_a = torch.randint(0, N, (n_pairs,), device=self.device)
        idx_b = torch.randint(0, N, (n_pairs,), device=self.device)

        sim = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1)  # cosine sim
        same = (pseudo_labels[idx_a] == pseudo_labels[idx_b]).float()

        # Pull same-concept together, push different apart
        loss = same * (1 - sim) + (1 - same) * F.relu(sim - margin)
        return loss.mean()

    def train_epoch(self, dataloader, kmeans, cfg):
        """One epoch of semantic consistency fine-tuning."""
        self.model.train()
        total_loss = 0

        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(self.device)

            # Get patch features from current model state
            with torch.cuda.amp.autocast(enabled=False):
                feats = self.model.get_patch_features(batch_imgs)
                # feats: (B, num_patches, D)
                B, P, D = feats.shape
                flat_feats = feats.reshape(B * P, D)

                # Get pseudo-labels from existing kmeans
                with torch.no_grad():
                    labels_np = kmeans.predict(
                        flat_feats.detach().cpu().numpy()
                    )
                pseudo_labels = torch.tensor(labels_np, device=self.device)

                # Semantic consistency loss
                loss = self.semantic_consistency_loss(flat_feats, pseudo_labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)