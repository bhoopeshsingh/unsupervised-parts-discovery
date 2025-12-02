# backbone.py
"""
Backbone wrapper(s) for the part-discovery pipeline.

This file provides:
 - ViTBackbone: thin wrapper to load a ViT-like backbone and return patch tokens
 - SimCLRBackboneLegacy: loader for legacy SimCLR encoder (optional)

Notes:
 - The aim is to standardize the output as: (B, H_p, W_p, D)
 - Models should be saved/loaded in .keras format by caller if needed.
 - For alignment and inspiration see the uploaded reference:
   /mnt/data/PDiscoFormer-Relaxing Part Discovery.pdf
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ViTBackbone:
    """
    Minimal ViT wrapper that exposes `get_patch_features(images) -> (B, H_p, W_p, D)`.

    Two usage modes:
      1. Load a pre-exported Keras model that already outputs patch tokens.
      2. If you have a full ViT model that outputs class+patch tokens, provide a small
         wrapper to extract patch tokens and reshape them to (B, H_p, W_p, D).

    CONFIG keys used (example):
      config["MODEL"]["BACKBONE_PATH"]  -> optional path to .keras model
      config["MODEL"]["PATCH_SIZE"]
      config["MODEL"]["EMBED_DIM"]
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.backbone_path = None
        if self.config.get("MODEL"):
            self.backbone_path = self.config["MODEL"].get("BACKBONE_PATH", None)
        self._initialized = False

    def load(self):
        """Load the backbone model if path given. Otherwise leave for manual injection."""
        if self.backbone_path and os.path.exists(self.backbone_path):
            # Expect the saved model to produce patch tokens (B, P, D) or (B, H_p, W_p, D)
            self.model = keras.models.load_model(self.backbone_path)
            self._initialized = True
        else:
            # No-op: user must set self.model manually (for example using a DINOv2 wrapper)
            self._initialized = False

    def set_model(self, keras_model):
        """Inject an already-built keras model that returns patch tokens."""
        self.model = keras_model
        self._initialized = True

    def get_patch_features(self, images: tf.Tensor) -> tf.Tensor:
        """
        Args:
            images: Tensor [B, H, W, 3] or [B, 3, H, W] depending on preprocessing
        Returns:
            patch_features: Tensor [B, H_p, W_p, D]
        """
        if self.model is None:
            raise RuntimeError("Backbone model not loaded. Call load() or set_model().")

        # Run forward
        out = self.model(images, training=False)

        # Accept multiple possible shapes:
        # - (B, P, D) -> reshape to (B, H_p, W_p, D) if patch grid info present in config
        # - (B, H_p, W_p, D) -> pass through
        out_shape = tf.shape(out)
        if out.shape.rank == 3:
            # (B, P, D)
            B = out_shape[0]
            P = out_shape[1]
            D = out_shape[2]
            # Try to compute H_p and W_p from config patch grid (if provided)
            H_p = self.config.get("MODEL", {}).get("PATCH_H", None)
            W_p = self.config.get("MODEL", {}).get("PATCH_W", None)
            if H_p is None or W_p is None:
                # attempt square
                s = int(np.sqrt(int(P)))
                H_p = s
                W_p = s
            out = tf.reshape(out, (B, H_p, W_p, D))
        elif out.shape.rank == 4:
            # already (B, H_p, W_p, D)
            pass
        else:
            raise ValueError(f"Unexpected backbone output shape: {out.shape}")

        return out


class SimCLRBackboneLegacy:
    """
    Simple loader for a legacy SimCLR encoder exported as a Keras model.
    This class provides a compatibility shim so the rest of pipeline can call
    .get_patch_features(...) and still work if the legacy model projects to
    per-patch descriptors (or we project features to patches).
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SimCLR model path not found: {model_path}")
        self.model = keras.models.load_model(model_path)

    def get_patch_features(self, images: tf.Tensor) -> tf.Tensor:
        """
        If the SimCLR model outputs global features, this method must be adjusted
        to map/reshape features into a pseudo patch-grid (for legacy comparisons).
        """
        if self.model is None:
            raise RuntimeError("Load the simclr model first with .load(path).")
        out = self.model(images, training=False)
        # Attempt to reshape if possible
        if out.shape.rank == 2:
            # (B, D) -> expand to (B, 1, 1, D)
            out = tf.expand_dims(tf.expand_dims(out, axis=1), axis=1)
        return out