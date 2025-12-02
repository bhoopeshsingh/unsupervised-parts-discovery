# parts_core.py
"""
PDisco-style parts core implementation (minimal, ready-to-extend).

Provides:
 - PDiscoModel: Keras model that accepts patch features (B, H_p, W_p, D)
                 and returns part attention maps, per-part embeddings and logits.
 - utility: gumbel_softmax_sample (for differentiable assignments)

This module focuses on forward computation + saving in .keras format.
Loss computation (TV, entropy, presence, orthogonality) is delegated to losses.py.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def sample_gumbel(shape, eps=1e-20):
    """Sample Gumbel(0,1)"""
    u = tf.random.uniform(shape, minval=0.0, maxval=1.0)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


def gumbel_softmax_sample(logits, tau=1.0, hard=False):
    """
    logits: [..., K] (unnormalized log-probs)
    returns relaxed one-hot vectors same shape as logits
    """
    g = sample_gumbel(tf.shape(logits))
    y = logits + g
    y = tf.nn.softmax(y / tau, axis=-1)

    if hard:
        # Straight-through trick: replace y with one-hot in forward pass but keep soft grad
        y_hard = tf.one_hot(tf.argmax(y, axis=-1), tf.shape(y)[-1])
        y = tf.stop_gradient(y_hard - y) + y
    return y


class PDiscoModel(keras.Model):
    """
    Minimal PDisco-like model implemented in Keras.

    Expected inputs:
      patch_features: tf.Tensor (B, H_p, W_p, D)

    Config keys:
      config["MODEL"]["NUM_PARTS"]  -> K
      config["MODEL"]["EMBED_DIM"]  -> D
      config["MODEL"]["NUM_CLASSES"] -> C  (optional; if not present, logits omitted)
      config["MODEL"]["GUMBEL_TAU"] -> initial tau
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config.get("MODEL", {})
        self.K = int(model_cfg.get("NUM_PARTS", 8))
        self.D = int(model_cfg.get("EMBED_DIM", 768))
        self.C = int(model_cfg.get("NUM_CLASSES", 0))  # 0 => no classifier
        self.use_classifier = self.C > 0
        self.tau = float(model_cfg.get("GUMBEL_TAU", 0.5))

        # prototypes: (K+1, D) -> last prototype = background
        # initialize small random values; make them trainable variables
        init_scale = 0.02
        self.prototypes = tf.Variable(
            initial_value=tf.random.normal([self.K + 1, self.D], stddev=init_scale),
            trainable=True,
            name="part_prototypes"
        )

        # per-part modulation (simple LayerNorm + affine weight)
        self.part_layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="part_layernorm")

        # shared classifier applied per-part (if requested)
        if self.use_classifier:
            self.classifier = layers.Dense(self.C, name="shared_classifier")
        else:
            self.classifier = None

    def call(self, patch_features, training=False, gumbel_tau=None, hard_gumbel=False):
        """
        Forward pass.

        Args:
            patch_features: Tensor (B, H_p, W_p, D)
            training: bool
            gumbel_tau: optional override for tau
            hard_gumbel: if True, use hard one-hot during forward (still straight-through)

        Returns:
            outputs dict:
              - A: (B, K+1, H_p, W_p) attention maps (soft allocations)
              - v: (B, K, D) per-part embeddings (foreground parts only)
              - logits_per_part: (B, K, C) optional
              - logits: (B, C) aggregated (mean over parts) optional
              - aux: contains distances or raw assignment logits for diagnostics
        """
        if gumbel_tau is None:
            gumbel_tau = self.tau

        # patch_features -> [B, P, D]
        x = tf.convert_to_tensor(patch_features)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        D_shape = tf.shape(x)[3]

        P = H * W
        x_flat = tf.reshape(x, [B, P, D_shape])  # (B, P, D)

        # compute negative squared euclidean distances: logits = -||z - p||^2
        # z: (B, P, D), prototypes: (K+1, D)
        # expand dims for broadcasting
        z_exp = tf.expand_dims(x_flat, axis=2)  # (B, P, 1, D)
        p_exp = tf.reshape(self.prototypes, [1, 1, self.K + 1, self.D])  # (1, 1, K+1, D)
        dists = tf.reduce_sum(tf.square(z_exp - p_exp), axis=-1)  # (B, P, K+1)
        logits = -dists  # higher similarity -> higher logit

        # Gumbel-Softmax assignment
        # Use relaxed soft assignment for training, straight softmax for eval, or hard straight-through if requested
        if training:
            a = gumbel_softmax_sample(logits, tau=gumbel_tau, hard=hard_gumbel)  # (B, P, K+1)
        else:
            a = tf.nn.softmax(logits, axis=-1)

        # reshape to attention maps: (B, K+1, H, W)
        a_perm = tf.transpose(a, perm=[0, 2, 1])  # (B, K+1, P)
        A = tf.reshape(a_perm, [B, self.K + 1, H, W])

        # compute per-part embeddings: weighted average over patches
        # v_all: (B, K+1, D)
        a_exp = tf.expand_dims(a, axis=-1)  # (B, P, K+1, 1)
        z_exp2 = tf.expand_dims(x_flat, axis=2)  # (B, P, 1, D)
        weighted = a_exp * z_exp2  # (B, P, K+1, D)
        v_all = tf.reduce_sum(weighted, axis=1) / tf.cast(P, tf.float32)  # (B, K+1, D)

        # separate foreground parts (first K) and background (last)
        v_fg = v_all[:, :self.K, :]  # (B, K, D)

        # per-part modulation (simple LN followed by learned affine via trainable gamma/beta handled by LayerNorm)
        # apply layernorm across D
        Bv = tf.reshape(v_fg, [-1, self.D])  # (B*K, D)
        v_norm = self.part_layernorm(Bv)
        v_norm = tf.reshape(v_norm, [B, self.K, self.D])  # (B, K, D)

        outputs = {
            "A": A,               # (B, K+1, H, W)
            "v": v_norm,          # (B, K, D)
            "logits_per_part": None,
            "logits": None,
            "aux": {
                "distances": dists,   # (B, P, K+1)
                "assignment": a       # (B, P, K+1)
            }
        }

        # classifier if present: apply shared classifier to each part embedding
        if self.use_classifier:
            # flatten parts dimension for Dense layer, then reshape
            v_for_cls = tf.reshape(v_norm, [-1, self.D])  # (B*K, D)
            logits_parts_flat = self.classifier(v_for_cls)  # (B*K, C)
            logits_parts = tf.reshape(logits_parts_flat, [B, self.K, self.C])  # (B, K, C)
            # aggregate via mean across parts
            logits_agg = tf.reduce_mean(logits_parts, axis=1)  # (B, C)

            outputs["logits_per_part"] = logits_parts
            outputs["logits"] = logits_agg

        return outputs

    def save_as_keras(self, path: str):
        """
        Save the model in .keras format. Keras requires a built model.
        To ensure the model is buildable, we run a dummy call before saving.
        """
        # Build graph by calling once (if not already built)
        dummy_H = int(self.config.get("MODEL", {}).get("PATCH_H", 8))
        dummy_W = int(self.config.get("MODEL", {}).get("PATCH_W", 8))
        dummy_D = int(self.D)
        dummy_input = tf.zeros([1, dummy_H, dummy_W, dummy_D], dtype=tf.float32)
        _ = self.call(dummy_input, training=False)
        # Save model
        # This saves weights + model config; it may not save custom call signature.
        self.save(path)