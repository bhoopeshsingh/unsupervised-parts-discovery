# losses.py
"""
Shaping losses for unsupervised part discovery.
Implements:
 - Total Variation (TV) loss
 - Entropy loss
 - Presence loss (foreground + background)
 - Orthogonality loss

Based on concepts used in PDiscoFormer paper:
  "/mnt/data/PDiscoFormer-Relaxing Part Discovery.pdf"
"""

import tensorflow as tf


# -----------------------------------------------------------
# Total Variation (TV) Loss
# Encourages spatial smoothness and connectivity of parts.
# A: (B, K+1, H, W)
# -----------------------------------------------------------
def total_variation_loss(A):
    A = tf.convert_to_tensor(A)
    dh = tf.abs(A[:, :, 1:, :] - A[:, :, :-1, :])
    dw = tf.abs(A[:, :, :, 1:] - A[:, :, :, :-1])
    return tf.reduce_mean(dh) + tf.reduce_mean(dw)


# -----------------------------------------------------------
# Entropy Loss
# Minimizes assignment ambiguity: -sum(a * log(a))
# A: (B, K+1, H, W)
# -----------------------------------------------------------
def entropy_loss(A, eps=1e-8):
    A = tf.convert_to_tensor(A)
    # Normalize per pixel → probability across parts
    A_prob = A / tf.reduce_sum(A, axis=1, keepdims=True)
    entropy = -tf.reduce_sum(A_prob * tf.math.log(A_prob + eps), axis=1)
    return tf.reduce_mean(entropy)


# -----------------------------------------------------------
# Presence Loss
# Foreground parts must appear somewhere in batch.
# Background must appear everywhere (higher weight near edges).
# Follows ideas from PDiscoFormer.
# -----------------------------------------------------------
def presence_loss(A):
    A = tf.convert_to_tensor(A)
    B, Kp, H, W = A.shape

    # Foreground parts: first K
    fg = A[:, :-1, :, :]            # (B, K, H, W)
    bg = A[:, -1:, :, :]            # (B, 1, H, W)

    # Each foreground part must appear somewhere in batch
    # max over (i, j) and batch-level average
    fg_max = tf.reduce_max(fg, axis=[0, 2, 3])  # (K,)
    L_fg = 1.0 - tf.reduce_mean(fg_max)

    # Background must appear in all images
    bg_max_per_img = tf.reduce_max(bg, axis=[1, 2, 3])  # (B,)
    L_bg = -tf.reduce_mean(tf.math.log(bg_max_per_img + 1e-8))

    return L_fg + L_bg


# -----------------------------------------------------------
# Orthogonality Loss
# Enforces part embeddings to be distinct.
# v: (B, K, D)
# -----------------------------------------------------------
def orthogonality_loss(v, eps=1e-6):
    v = tf.convert_to_tensor(v)
    B, K, D = v.shape

    # Normalize embeddings
    v_norm = tf.nn.l2_normalize(v, axis=-1)  # (B, K, D)

    # Compute pairwise cosine similarities per batch
    sims = []
    for b in range(B):
        vb = v_norm[b]                       # (K, D)
        sim = tf.matmul(vb, vb, transpose_b=True)  # (K, K)
        mask = 1 - tf.eye(K)
        sims.append(tf.reduce_mean(tf.abs(sim * mask)))
    return tf.reduce_mean(sims)