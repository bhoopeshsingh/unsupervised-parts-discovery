# log_explain_wandb.py
"""
Full W&B-integrated explainability pipeline:
- GradCAM
- Parts discovery
- Reconstruction (if encoder_spatial & decoder exist)
- Logging visuals and metadata to W&B
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import wandb

from utils import find_last_conv_in_model
from visualize_utils import apply_colormap, blend_overlay
from parts_discovery import parts_discovery


# ---------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer=None, pred_index=None):
    if img_array.max() <= 1.0:
        img = img_array * 255.0
    else:
        img = img_array

    img_tensor = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0)

    # Auto-select last conv
    if last_conv_layer is None:
        last_conv_layer = find_last_conv_in_model(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        target = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  # (C,)

    conv_outputs = conv_outputs[0]                       # (Hc,Wc,C)
    cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to input spatial size
    cam = tf.image.resize(cam[..., np.newaxis],
                          (img.shape[0], img.shape[1])).numpy()[..., 0]

    return cam


def overlay_gradcam_on_image(img_array, heatmap, alpha=0.4):
    import matplotlib

    if img_array.max() <= 1.0:
        img = (img_array * 255).astype(np.uint8)
    else:
        img = img_array.astype(np.uint8)

    cmap = matplotlib.cm.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    blended = blend_overlay(img, heatmap_color, alpha=alpha)
    return blended


# ---------------------------------------------------------
# MAIN EXPLAINABILITY FUNCTION
# ---------------------------------------------------------
def explain_main(
    encoder_path,
    decoder_path,
    classifier_path,
    test_image_paths,
    out_dir,
    n_parts=4,
):
    """
    The unified method:
    - Loads classifier, encoder_spatial, decoder
    - For each test image:
       ✓ Grad-CAM + overlay
       ✓ Parts discovery + overlay
       ✓ Reconstruction (if possible)
       ✓ Prediction scores
       ✓ Logs into WandB
    """

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------
    # Load models
    # ------------------------------------
    classifier = tf.keras.models.load_model(classifier_path, compile=False)

    try:
        encoder_spatial = tf.keras.models.load_model(encoder_path, compile=False)
        spatial_enabled = True
    except:
        encoder_spatial = None
        spatial_enabled = False

    try:
        decoder = tf.keras.models.load_model(decoder_path, compile=False)
        decoder_enabled = True
    except:
        decoder = None
        decoder_enabled = False

    # ------------------------------------
    # Start W&B run
    # ------------------------------------
    wandb.init(project="self-supervised-parts-discovery",
               name="explainability-run",
               reinit=True)

    class_names = classifier.output_names if hasattr(classifier, "output_names") else None
    last_conv = find_last_conv_in_model(classifier)

    # ------------------------------------
    # PROCESS EACH TEST IMAGE
    # ------------------------------------
    for img_path in test_image_paths:

        pil = Image.open(img_path).convert("RGB").resize((128,128))
        arr = np.array(pil).astype("float32") / 255.0

        # -----------------------------
        # Forward pass → prediction
        # -----------------------------
        preds = classifier.predict(np.expand_dims(arr, 0), verbose=0)[0]
        pred_label = np.argmax(preds)
        pred_conf = float(np.max(preds))

        # -----------------------------
        # GRAD-CAM
        # -----------------------------
        heatmap = make_gradcam_heatmap(arr, classifier, last_conv)
        grad_overlay = overlay_gradcam_on_image(arr, heatmap)

        # -----------------------------
        # PARTS DISCOVERY
        # -----------------------------
        parts_map = parts_discovery(arr * 255.0, classifier,
                                    last_conv_layer=last_conv,
                                    n_parts=n_parts)

        # colormap & overlay
        parts_color = apply_colormap(parts_map)
        parts_overlay = blend_overlay((arr*255).astype(np.uint8),
                                      parts_color,
                                      alpha=0.5)

        # -----------------------------
        # RECONSTRUCTION
        # -----------------------------
        recon_img = None
        if spatial_enabled and decoder_enabled:
            z = encoder_spatial(np.expand_dims(arr, 0), training=False)
            recon = decoder(z).numpy()[0]
            recon_img = (recon * 255).astype(np.uint8)

        # -----------------------------
        # LOG INTO W&B
        # -----------------------------
        wandb.log({
            "image/original": wandb.Image((arr*255).astype(np.uint8)),
            "gradcam/heatmap": wandb.Image(heatmap),
            "gradcam/overlay": wandb.Image(grad_overlay),
            "parts/mask": wandb.Image(parts_color),
            "parts/overlay": wandb.Image(parts_overlay),
            "prediction/label": int(pred_label),
            "prediction/confidence": pred_conf,
        })

        if recon_img is not None:
            wandb.log({"reconstruction/image": wandb.Image(recon_img)})

    wandb.finish()
    print(f"Explainability logs saved to WandB & PNG saved in: {out_dir}")
