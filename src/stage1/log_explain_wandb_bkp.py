"""
log_explain_wandb.py

Purpose:
- Produce explainability visualizations (Grad-CAM, parts overlays, reconstructions) and
  log them to Weights & Biases. This module ties together the classifier, encoder, and
  decoder to give a multi-view explanation of model decisions.

How it works (brief):
- Load models (classifier, encoder, decoder).
- Select a few validation images per class.
- For each image: get classifier prediction, compute Grad-CAM using the last conv layer,
  run parts_discovery (per-image KMeans on conv features), reconstruct from encoder embedding,
  then log the original image, gradcam overlay, parts overlay, and reconstruction to W&B.
"""

# Helper utilities and orchestration for explainability: Grad-CAM, parts overlays, and reconstructions.
# This script ties together visual explanation methods and logs them to Weights & Biases.

import os, argparse, numpy as np, tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.cm as cm
import wandb

from utils import (
    get_image_datasets,
    load_model as utils_load_model,
    preprocess_image,
    find_last_conv_in_model,
)
from parts_discovery import parts_discovery


# -------------------------
# Grad-CAM helper
# -------------------------
# make_gradcam_heatmap: compute Grad-CAM map from an input image and classifier
def make_gradcam_heatmap(img_array, clf, last_conv_layer, pred_index=None):
    """
    Compute a Grad-CAM heatmap for a single image and class index.

    Returns a float32 array in [0,1] shaped like the input image's spatial dims.
    """
    import tensorflow as tf
    import numpy as np

    #img_tensor = tf.expand_dims(tf.convert_to_tensor(img_array, dtype=tf.float32), 0)
    img_array = img_array.numpy() / 255.0

    grad_model = tf.keras.models.Model(
        inputs=clf.inputs,
        outputs=[last_conv_layer.output, clf.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        # FIX: classifier may output a list → unwrap it
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[-1]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, int(pred_index)]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("Gradients are None in make_gradcam_heatmap.")
        return np.zeros(img_array.shape[:2], dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_arr = conv_outputs[0].numpy()
    pooled_grads_arr = pooled_grads.numpy()

    for i in range(pooled_grads_arr.shape[-1]):
        conv_outputs_arr[..., i] *= pooled_grads_arr[i]

    cam = np.sum(conv_outputs_arr, axis=-1)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    cam = tf.image.resize(cam[..., np.newaxis],
                          img_array.shape[:2],
                          method="bilinear").numpy()[..., 0]
    return cam.astype(np.float32)


# overlay_gradcam: blend a heatmap with the original image for visualization
def overlay_gradcam(img, heatmap, alpha=0.4, cmap="jet"):
    """
    Blend a Grad-CAM heatmap into the original image.
    Inputs:
      img: numpy array or Tensor, shape (H,W,C), values in [0,1]
      heatmap: numpy array, shape (H,W), values in [0,1]
    """
    import numpy as np
    import tensorflow as tf
    from matplotlib import cm

    # Normalize heatmap
    heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))

    # Apply colormap
    colormap = cm.get_cmap(cmap)
    colors = colormap(np.arange(256))[:, :3]
    heatmap_color = colors[heatmap]   # (H,W,3)

    # Resize heatmap to match input image size
    heatmap_resized = tf.image.resize(
        np.expand_dims(heatmap_color, 0),
        (img.shape[0], img.shape[1])
    ).numpy()[0]

    # Blend
    superimposed = heatmap_resized * alpha + img

    # --- FORCE numpy conversion ---
    if not isinstance(superimposed, np.ndarray):
        superimposed = superimposed.numpy()

    # Normalize
    max_val = superimposed.max()
    if max_val > 0:
        superimposed = superimposed / max_val

    superimposed = np.clip(superimposed, 0, 1)
    return (superimposed * 255).astype("uint8")


import matplotlib.cm as cm

# overlay_parts: create a colored overlay for parts discovery output
def overlay_parts(img, parts_map, alpha=0.4, cmap="viridis"):
    import numpy as np
    import tensorflow as tf
    from matplotlib import cm

    # Normalize parts map
    parts_norm = np.uint8(255 * np.clip(parts_map, 0, 1))

    # Apply colormap
    colormap = cm.get_cmap(cmap)
    colors = colormap(np.arange(256))[:, :3]
    color_mask = colors[parts_norm]   # (H, W, 3)

    # Blend (TensorFlow may return Tensor)
    overlay = color_mask * alpha + img

    # --- FORCE numpy conversion ---
    if not isinstance(overlay, np.ndarray):
        overlay = overlay.numpy()

    # Normalize
    max_val = overlay.max()
    if max_val > 0:
        overlay = overlay / max_val

    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype("uint8")


# overlay_heatmap: alternate simple overlay using PIL colorize
def overlay_heatmap(img, heatmap):
    # expect img float [0,1], heatmap [0,1]
    h = (heatmap * 255).astype("uint8")
    from PIL import Image, ImageOps
    hm = Image.fromarray(h).convert("L").resize((img.shape[1], img.shape[0]))
    img_pil = Image.fromarray((img * 255).astype("uint8"))
    hm_colored = ImageOps.colorize(hm, (0,0,0), (255,0,0))
    blended = Image.blend(img_pil.convert("RGBA"), hm_colored.convert("RGBA"), alpha=0.5)
    return np.array(blended.convert("RGB"))

import matplotlib.pyplot as plt

# show_explain_grid: small utility to display original + explainability visualizations side-by-side
def show_explain_grid(img, gradcam_overlay, parts_overlay, recon):
    """
    Creates a nice 4-panel visualization showing all our explainability methods at once.
    Makes it easy to compare what different techniques reveal about the model's behavior.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(parts_overlay)
    axes[2].set_title("Parts")
    axes[2].axis("off")

    axes[3].imshow(recon)
    axes[3].set_title("Reconstruction")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def explain_main_v1(encoder_path, decoder_path, classifier_path, args):
    import numpy as np
    import tensorflow as tf
    import wandb

    print("Explainability with Grad-CAM, Parts Discovery, Reconstructions")
    # init wandb
    if args.wandb_project:
        if wandb.run is None:
            wandb.init(project=args.wandb_project, config=vars(args))

    # Load models using utils.load_model (which registers L2Normalization)
    print("Loading models...")
    clf = utils_load_model(classifier_path, compile=False)
    full_encoder = utils_load_model(encoder_path, compile=False)
    decoder = utils_load_model(decoder_path, compile=False)

    # defensive naming
    clf._name = getattr(clf, "_name", "classifier_model")
    full_encoder._name = getattr(full_encoder, "_name", "encoder_model")
    decoder._name = getattr(decoder, "_name", "decoder_model")

    print("Classifier layers (top-level):", [l.name for l in clf.layers])
    print("Full encoder layers (top-level):", [l.name for l in full_encoder.layers])

    # Find last conv layer to use for Grad-CAM: prefer classifier's convs (so Grad-CAM reflects classifier)
    last_conv_layer = find_last_conv_in_model(clf)
    if last_conv_layer is None:
        # fall back to encoder if classifier has no convs (unlikely)
        last_conv_layer = find_last_conv_in_model(full_encoder)

    if last_conv_layer is None:
        print("No conv layers found — Grad-CAM & parts will be skipped.")
    else:
        print("Using last conv layer for Grad-CAM:", last_conv_layer.name)

    # Build spatial encoder to feed decoder and parts pipeline (extract the same conv layer by name from full_encoder)
    spatial_encoder = None
    if last_conv_layer is not None:
        layer_name = last_conv_layer.name
        try:
            spatial_layer = full_encoder.get_layer(layer_name)
            spatial_encoder = tf.keras.Model(inputs=full_encoder.input, outputs=spatial_layer.output,
                                             name="spatial_encoder_explain")
            print("Spatial encoder built from full_encoder layer:", layer_name)
            print("spatial_encoder.output_shape =", spatial_encoder.output_shape)
            print("decoder.input_shape =", decoder.input_shape)
        except Exception as e:
            print(f"Could not extract layer '{layer_name}' from full_encoder: {e}")
            # fallback: try to find last conv in full_encoder instead
            flc = find_last_conv_in_model(full_encoder)
            if flc is not None:
                spatial_encoder = tf.keras.Model(inputs=full_encoder.input, outputs=flc.output,
                                                 name="spatial_encoder_explain_fallback")
                print("Fallback spatial encoder built from full_encoder:", flc.name)
                print("Spatial_encoder.output_shape =", spatial_encoder.output_shape)
            else:
                print("No spatial encoder available. Reconstruction & parts may fail.")

    # dataset (batch_size=1)
    _, val_ds, class_names = get_image_datasets(
        args.train_dir, args.val_dir,
        (args.img_size, args.img_size),
        batch_size=1, shuffle=False
    )

    # collect 5 per class
    names = args.class_names.split(",")
    samples = {n: [] for n in names}
    for x, y in val_ds.unbatch():
        img = preprocess_image(x, (args.img_size, args.img_size))
        lab = int(y.numpy())
        cls_name = names[lab]
        if len(samples[cls_name]) < 5:
            samples[cls_name].append(img)
        if all(len(v) >= 5 for v in samples.values()):
            break

    table = wandb.Table(columns=["true", "pred", "prob", "orig", "gradcam", "parts", "recon"])

    for true_cls, imgs in samples.items():
        for img in imgs:
            inp = np.expand_dims(img, 0)  # (1,H,W,C)

            # ----- classifier prediction (be defensive if clf.predict returns list) -----
            preds_raw = clf.predict(inp, verbose=0)
            # handle list/tuple outputs (take last)
            if isinstance(preds_raw, (list, tuple)):
                preds = preds_raw[-1]
            else:
                preds = preds_raw

            pred_idx = int(np.argmax(preds[0]))
            prob = float(preds[0][pred_idx])
            pred_cls = names[pred_idx]

            # Grad-CAM: use classifier and last_conv_layer (if present)
            heatmap = None
            gradcam_overlay = np.zeros_like((img * 255).astype("uint8"))
            if last_conv_layer is not None:
                heatmap = make_gradcam_heatmap(img, clf, last_conv_layer, pred_index=pred_idx)
                gradcam_overlay = overlay_gradcam(img, heatmap)

            # Parts discovery (uses classifier and last_conv_layer to get feature maps)
            parts_overlay = np.zeros_like((img * 255).astype("uint8"))
            try:
                parts_map = parts_discovery(img, clf, last_conv_layer, n_parts=args.n_parts)
                parts_overlay = overlay_parts(img, parts_map)
            except Exception as e:
                print("parts_discovery failed:", e)

            # Reconstruction: MUST use spatial_encoder (not the classifier's flattened encoder)
            recon_uint8 = np.zeros_like((img * 255).astype("uint8"))
            if spatial_encoder is not None:
                try:
                    emb = spatial_encoder.predict(inp, verbose=0)   # (1,H',W',C')
                    print("[explain_main] spatial encoded shape:", emb.shape)
                    recon = decoder.predict(emb, verbose=0)[0]
                    recon_uint8 = (recon * 255).astype("uint8")
                except Exception as e:
                    print("Reconstruction failed:", e)
            else:
                print("No spatial encoder available — skipping reconstruction.")

            # W&B logging
            table.add_data(
                true_cls, pred_cls, prob,
                wandb.Image(to_uint8(img), caption="orig"),
                wandb.Image(to_uint8(gradcam_overlay), caption="gradcam"),
                wandb.Image(to_uint8(parts_overlay), caption="parts"),
                wandb.Image(to_uint8(recon_uint8), caption="recon"),
            )

            # Local side-by-side visualization (only show once per class)
            if len(samples[true_cls]) == 1:
                show_explain_grid(
                    (img * 255).astype("uint8"),
                    gradcam_overlay,
                    parts_overlay,
                    recon_uint8,
                )

    wandb.log({"explainability_samples": table})
    if wandb.run:
        wandb.run.summary["explainability_samples"] = table
    print("Logged explainability results to W&B")


# -------------------------
# Main explainability routine
# -------------------------
# explain_main: run Grad-CAM, parts discovery, reconstructions and log to W&B
def explain_main(encoder_path, decoder_path, classifier_path, args):
    """
    The main explainability pipeline - runs all our interpretability methods and logs
    everything to Weights & Biases.

    For each test image, we generate:
    - Grad-CAM heatmap (what the classifier looks at)
    - Parts segmentation (semantic parts found by clustering)
    - Reconstruction (can we recreate the image from its embedding?)

    This gives us multiple angles to understand what the model learned!
    """
    print("Explainability with Grad-CAM, Parts Discovery, Reconstructions")
    # init wandb
    if args.wandb_project:
        if wandb.run is None:
            wandb.init(project=args.wandb_project, config=vars(args))

    # Load models using utils.load_model (which registers L2Normalization)
    print("Loading models...")
    clf = utils_load_model(args.classifier_path, compile=False)
    decoder = utils_load_model(args.decoder_path, compile=False)

    #encoder = utils_load_model(args.encoder_path, compile=False)
    full_encoder = utils_load_model(args.encoder_path, compile=False)

    # Extract the spatial feature output
    spatial_layer = full_encoder.get_layer("conv5_block3_3_conv")

    # Build the true spatial encoder
    encoder = tf.keras.Model(
        inputs=full_encoder.input,
        outputs=spatial_layer.output,
        name="spatial_encoder_explain"
    )

    print(">>> Spatial encoder restored:", spatial_layer.name)
    print(">>> Spatial encoder output shape:", encoder.output_shape)
    print(">>> Decoder expected:", decoder.input_shape)

    # small defensive rename (not strictly required if you use classifier-based graphs)
    clf._name = getattr(clf, "_name", "classifier_model")
    encoder._name = getattr(encoder, "_name", "encoder_model")
    decoder._name = getattr(decoder, "_name", "decoder_model")

    print("Classifier layers (top-level):", [l.name for l in clf.layers])
    print("Encoder layers (top-level):", [l.name for l in encoder.layers])

    # Debug: list all conv layers inside encoder
    conv_names = []
    print("\nConv layers inside encoder (name : output shape):")
    for l in encoder.layers:
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
            conv_names.append((l.name, getattr(l.output, "shape", None)))
            #print(f" - {l.name}  {getattr(l.output, 'shape', None)}")
    print("")

    #last_conv_layer = find_last_conv_in_model(encoder)
    # Important: pick last conv layer inside the classifier, not the standalone encoder
    last_conv_layer = find_last_conv_in_model(clf)

    if last_conv_layer is None:
        print("No conv layers found in encoder — Grad-CAM & parts will be skipped.")
    else:
        print("Using last conv layer for Grad-CAM:", last_conv_layer.name)

    # dataset
    _, val_ds, class_names = get_image_datasets(
        args.train_dir, args.val_dir,
        (args.img_size, args.img_size),
        batch_size=1, shuffle=False
    )

    # collect 5 per class
    names = args.class_names.split(",")
    samples = {n: [] for n in names}
    for x, y in val_ds.unbatch():
        img = preprocess_image(x, (args.img_size, args.img_size))
        lab = int(y.numpy())
        cls_name = names[lab]
        if len(samples[cls_name]) < 5:
            samples[cls_name].append(img)
        if all(len(v) >= 5 for v in samples.values()):
            break

    table = wandb.Table(columns=["true", "pred", "prob", "orig", "gradcam", "parts", "recon"])

    for true_cls, imgs in samples.items():
        for img in imgs:
            inp = np.expand_dims(img, 0)
            preds = clf.predict(inp, verbose=0)
            pred_idx = int(np.argmax(preds[0]))
            prob = float(preds[0][pred_idx])
            pred_cls = names[pred_idx]

            # Grad-CAM
            heatmap = make_gradcam_heatmap(img, clf, last_conv_layer, pred_index=pred_idx)
            gradcam_overlay = overlay_gradcam(img, heatmap)

            # Parts discovery
            parts_map = parts_discovery(img, clf, last_conv_layer, n_parts=args.n_parts)
            parts_overlay = overlay_parts(img, parts_map)

            # Reconstruction
            emb = encoder.predict(inp, verbose=0)
            recon = decoder.predict(emb, verbose=0)[0]
            recon_uint8 = (recon * 255).astype("uint8")

            # W&B logging
            table.add_data(
                true_cls, pred_cls, prob,
                wandb.Image(to_uint8(img), caption="orig"),
                wandb.Image(to_uint8(gradcam_overlay), caption="gradcam"),
                wandb.Image(to_uint8(parts_overlay), caption="parts"),
                wandb.Image(to_uint8(recon_uint8), caption="recon"),
            )

            # Local side-by-side visualization (only show once per class)
            if len(samples[true_cls]) == 1:
                show_explain_grid(
                    (img * 255).astype("uint8"),
                    gradcam_overlay,
                    parts_overlay,
                    recon_uint8,
                )

    wandb.log({"explainability_samples": table})
    # Save table reference in run summary for quick access in the UI
    if wandb.run:
        wandb.run.summary["explainability_samples"] = table
    print("✅ Logged explainability results to W&B")

def to_uint8(x):
    """Convert Tensor or NumPy array in [0,1] to uint8 image."""
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    return (np.clip(x, 0, 1) * 255).astype("uint8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--encoder_path', required=True, help='path to saved encoder (.keras)')
    p.add_argument('--decoder_path', required=True, help='path to saved decoder (.keras)')
    p.add_argument('--classifier_path', required=True, help='path to saved classifier (.keras)')
    p.add_argument('--train_dir', required=True, help='training directory (used to load dataset class names)')
    p.add_argument('--val_dir', required=True, help='validation directory')
    p.add_argument('--img_size', type=int, default=128, help='image size to use for preprocessing')
    p.add_argument('--n_parts', type=int, default=6, help='number of parts for per-image parts discovery')
    p.add_argument('--class_names', default='airplane,cat', help='comma-separated class names in label order')
    p.add_argument('--wandb_project', default=None, help='W&B project name (optional)')
    args = p.parse_args()

    # call the main pipeline
    explain_main_v1(args.encoder_path, args.decoder_path, args.classifier_path, args)
