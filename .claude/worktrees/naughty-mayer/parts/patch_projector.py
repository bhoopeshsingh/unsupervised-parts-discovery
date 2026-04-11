# patch_projector.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def combine_multiscale_features(feature_maps, method='upsample', resize_mode='bilinear'):
    """
    Upsample all feature_maps to the largest spatial resolution and concat along channels.
    feature_maps: list of tensors [F_C2 (B,H2,W2,C2), F_C3, F_C4]
    Returns: concatenated map of shape (B, H_max, W_max, C_total)
    """
    # compute target H, W using shapes (dynamic)
    shapes = [tf.shape(f)[1:3] for f in feature_maps]
    heights = [s[0] for s in shapes]
    widths = [s[1] for s in shapes]
    H = tf.reduce_max(heights)
    W = tf.reduce_max(widths)
    resized = []
    for f in feature_maps:
        resized.append(tf.image.resize(f, size=(H, W), method=resize_mode))
    return tf.concat(resized, axis=-1)

def make_patch_projector(proj_dim=128, norm=True, name="patch_projector"):
    """
    Create a projector that maps a multi-scale feature map (B,H,W,C) -> (B,H,W,proj_dim).
    Implementation: 1x1 conv -> LayerNorm -> small MLP (1x1 convs) -> L2 normalize
    """
    inp = layers.Input(shape=(None, None, None), name="proj_input")
    x = layers.Conv2D(filters=proj_dim, kernel_size=1, padding='same', name="proj_conv")(inp)
    x = layers.LayerNormalization(axis=-1, name="proj_ln")(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=proj_dim, kernel_size=1, padding='same', name="proj_mlp")(x)
    # final projection (no activation)
    x = layers.Conv2D(filters=proj_dim, kernel_size=1, padding='same', name="proj_out")(x)
    if norm:
        # apply L2 normalization across channels per spatial location
        x = tf.nn.l2_normalize(x, axis=-1, name="proj_l2norm")
    return Model(inputs=inp, outputs=x, name=name)
