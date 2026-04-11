# src/models/panel_ft_transformer.py
"""
Panel-Structured Feature Tokenization Transformer (PSFTT)
for multi-granularity clinical lab concept discovery.

Architecture is grounded in:
  - FT-Transformer (Gorishniy et al., NeurIPS 2021): per-feature linear tokenization
  - Tab-PET (arXiv 2024): structured positional encoding on tabular transformers
  - Two-scale SSL: within-panel masking (VIME/SAINT inspired) +
                   cross-panel dropout (novel objective)

The intermediate-layer extraction directly mirrors DINO Phase 1:
  DINO:  patch_j  = concat(L8,  L10, L12)  →  visual concept
  PSFTT: panel_i  = concat(L2,  L4,  L6)   →  clinical concept

Key design decisions:
  - Panel-aware positional encoding: E_panel[panel(i)] added to each test token,
    encoding biological system membership (CBC / BMP / Lipid)
  - Two-scale SSL separates what early vs late layers learn:
      Scale 1 (within-panel masking)  → early layers learn test co-deviations
      Scale 2 (cross-panel dropout)   → late layers learn physiological inter-system deps
  - Panel-patches are the "spatial patches" analog: each panel is a functional region
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FeatureTokenizer(nn.Module):
    """
    FT-Transformer feature tokenizer (Gorishniy et al., NeurIPS 2021).

    Each numerical feature i gets its own linear map:
        T_i = x_i * W_i + b_i   (W_i, b_i ∈ R^d)

    No weight sharing across features — each test has its own learned projection.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N]  deviation values
        Returns:
            [B, N, d_model]
        """
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class PanelFTTransformer(nn.Module):
    """
    Panel-Structured FT-Transformer.

    Token construction:
        T_i = FT_tokenize(x_i)          # per-test linear projection
            + E_panel[panel(i)]           # panel membership PE (new vs. FT-Transformer)

    Intermediate layers extracted at positions `extract_layers` (0-indexed).
    Panel-patch vector = concat of per-panel mean-pooled tokens at each extract layer.
    """

    def __init__(
        self,
        n_features: int,
        panel_ids: List[int],
        n_panels: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        extract_layers: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.n_features = n_features
        self.n_panels = n_panels
        self.d_model = d_model
        self.n_layers = n_layers
        self.extract_layers = extract_layers

        self.register_buffer("panel_ids", torch.tensor(panel_ids, dtype=torch.long))

        self.feature_tokenizer = FeatureTokenizer(n_features, d_model)
        self.panel_embedding = nn.Embedding(n_panels, d_model)

        # Learnable special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Individual transformer layers (needed to extract intermediates)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,    # Pre-LayerNorm — more stable for small datasets
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # SSL reconstruction head: predicts scalar deviation value at masked positions
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Panel dropout prediction head: CLS token → predict mean dev of dropped panel
        self.panel_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _tokenize(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build token sequence with panel PE.

        Args:
            x:    [B, N]      deviation values
            mask: [B, N] bool positions to replace with mask_token
        Returns:
            [B, N+1, d_model]  (CLS prepended at index 0)
        """
        B = x.shape[0]
        tokens = self.feature_tokenizer(x)                          # [B, N, d]
        tokens = tokens + self.panel_embedding(self.panel_ids)      # panel PE broadcast

        if mask is not None:
            m = mask.unsqueeze(-1).float()
            tokens = tokens * (1.0 - m) + self.mask_token * m

        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, tokens], dim=1)                      # [B, N+1, d]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x:    [B, N] deviation values
            mask: [B, N] bool  (optional)
        Returns:
            final:         [B, N+1, d_model]
            intermediates: list of [B, N+1, d_model] at self.extract_layers (detached)
        """
        h = self._tokenize(x, mask)
        intermediates = []
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i in self.extract_layers:
                intermediates.append(h.detach().clone())
        return self.norm(h), intermediates

    # ------------------------------------------------------------------
    # Multi-granularity panel-patch extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_panel_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-panel multi-granularity vectors.

        For each panel p, mean-pools its token representations at each extract_layer,
        then concatenates across layers → panel-patch of size len(extract_layers)*d_model.

        Args:
            x: [B, N] deviation tensor
        Returns:
            patches: [B, n_panels, len(extract_layers) * d_model]
        """
        self.eval()
        _, intermediates = self.forward(x)       # intermediates: list of [B, N+1, d]

        panel_patches = []
        for p in range(self.n_panels):
            # +1 offset: index 0 is CLS
            feat_idx = (self.panel_ids == p).nonzero(as_tuple=True)[0] + 1   # [k_p]
            layer_vecs = []
            for h in intermediates:
                panel_mean = h[:, feat_idx, :].mean(dim=1)    # [B, d]
                layer_vecs.append(panel_mean)
            panel_patches.append(torch.cat(layer_vecs, dim=-1))    # [B, L*d]

        return torch.stack(panel_patches, dim=1)   # [B, n_panels, L*d]

    # ------------------------------------------------------------------
    # Two-scale SSL pre-training
    # ------------------------------------------------------------------

    def ssl_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Two-scale SSL loss:

        Scale 1 — Within-panel masked reconstruction:
            Pick one panel per record, mask 40% of its tests, reconstruct
            original deviation values. Forces early layers to learn
            within-panel test co-deviations (HGB ↔ MCV ↔ MCH correlations).

        Scale 2 — Cross-panel panel dropout:
            Drop one entire panel per record, predict its mean deviation from
            the other panels via the CLS token. Forces late layers to learn
            physiological cross-panel dependencies (renal ↔ hematological).

        Args:
            x: [B, N] deviation values
        Returns:
            loss: scalar
            info: {'scale1': float, 'scale2': float}
        """
        B, N = x.shape
        device = x.device

        # ── Scale 1: within-panel masked reconstruction ──────────────────
        mask1 = torch.zeros(B, N, dtype=torch.bool, device=device)
        chosen = torch.randint(0, self.n_panels, (B,), device=device)

        for b in range(B):
            feat_idx = (self.panel_ids == chosen[b]).nonzero(as_tuple=True)[0]
            n_mask = max(1, int(0.4 * len(feat_idx)))
            perm = torch.randperm(len(feat_idx), device=device)[:n_mask]
            mask1[b, feat_idx[perm]] = True

        final1, _ = self.forward(x, mask=mask1)
        # Predict at masked positions (skip CLS: offset +1)
        feature_out = final1[:, 1:, :]                              # [B, N, d]
        masked_pred = self.reconstruction_head(feature_out[mask1]).squeeze(-1)
        loss1 = F.mse_loss(masked_pred, x[mask1])

        # ── Scale 2: cross-panel dropout ─────────────────────────────────
        mask2 = torch.zeros(B, N, dtype=torch.bool, device=device)
        targets2 = torch.zeros(B, device=device)
        dropped = torch.randint(0, self.n_panels, (B,), device=device)

        for b in range(B):
            feat_idx = (self.panel_ids == dropped[b]).nonzero(as_tuple=True)[0]
            mask2[b, feat_idx] = True
            targets2[b] = x[b, feat_idx].mean()

        final2, _ = self.forward(x, mask=mask2)
        pred2 = self.panel_pred_head(final2[:, 0, :]).squeeze(-1)   # CLS → prediction
        loss2 = F.mse_loss(pred2, targets2)

        loss = loss1 + 0.5 * loss2
        return loss, {"scale1": loss1.item(), "scale2": loss2.item()}

    def pretrain(
        self,
        features: torch.Tensor,
        n_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
        save_path: Optional[str] = None,
    ):
        """
        Run two-scale SSL pre-training.

        Args:
            features:   [N, n_features] clinical deviation tensor
            n_epochs:   training epochs
            batch_size: mini-batch size
            lr:         AdamW learning rate
            device:     'cpu' | 'mps' | 'cuda'
            save_path:  if provided, save weights after training
        """
        self.to(device)
        self.train()

        loader = DataLoader(
            TensorDataset(features),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        print(f"  Pre-training on {len(features):,} records | "
              f"{n_epochs} epochs | batch={batch_size} | device={device}")
        print(f"  Model: {self.n_layers}L d={self.d_model} h=4 | "
              f"extract_layers={self.extract_layers}")

        for epoch in range(n_epochs):
            total, s1_sum, s2_sum = 0.0, 0.0, 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss, info = self.ssl_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
                s1_sum += info["scale1"]
                s2_sum += info["scale2"]
            scheduler.step()
            n = len(loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={total/n:.4f}  "
                      f"scale1={s1_sum/n:.4f}  "
                      f"scale2={s2_sum/n:.4f}")

        if save_path:
            self.save(save_path)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_features":     self.n_features,
                    "n_panels":       self.n_panels,
                    "d_model":        self.d_model,
                    "n_layers":       self.n_layers,
                    "extract_layers": list(self.extract_layers),
                    "panel_ids":      self.panel_ids.tolist(),
                },
            },
            path,
        )
        print(f"  ✓  Transformer saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PanelFTTransformer":
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state["config"]
        model = cls(
            n_features=cfg["n_features"],
            panel_ids=cfg["panel_ids"],
            n_panels=cfg["n_panels"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            extract_layers=tuple(cfg["extract_layers"]),
        )
        model.load_state_dict(state["state_dict"])
        model.to(device)
        return model


# ---------------------------------------------------------------------------
# Helper: build panel_ids list from config
# ---------------------------------------------------------------------------

def build_panel_ids(
    config_path: str,
    shuffle_panels: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[List[int], List[str], List[str]]:
    """
    Read config_lab.yaml and return:
        panel_ids:    List[int]  — panel index per feature (same order as feature_cols)
        feature_cols: List[str]  — ordered feature column names
        panel_names:  List[str]  — panel name per panel index

    Default panel ordering matches config 'panels' dict order:
        0 = cbc, 1 = biochem, 2 = lipid

    Args:
        shuffle_panels:  If True, randomise the panel order before building the
                         feature sequence.  The panel PE indices follow the shuffled
                         order, so panel 0/1/2 are re-assigned.  Useful to verify
                         that the model learns from panel *identity* (via PE) rather
                         than fixed feature-sequence position.
        random_seed:     Seed for reproducible shuffling (None = non-deterministic).
    """
    import random
    import yaml
    cfg = yaml.safe_load(open(config_path))
    panels = cfg["lab_data"]["panels"]

    panel_items = list(panels.items())   # [('cbc', cfg), ('biochem', cfg), ('lipid', cfg)]

    if shuffle_panels:
        rng = random.Random(random_seed)
        rng.shuffle(panel_items)

    feature_cols = []
    panel_ids = []
    panel_names = [k for k, _ in panel_items]

    for p_idx, (_, panel_cfg) in enumerate(panel_items):
        for col in panel_cfg["features"]:
            feature_cols.append(col)
            panel_ids.append(p_idx)

    return panel_ids, feature_cols, panel_names
