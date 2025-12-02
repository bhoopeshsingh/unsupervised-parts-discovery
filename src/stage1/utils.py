"""
utils.py

Common utilities used across the project:
- Small Keras helpers (custom layer, model save/load, warmup)
- Dataset helpers and simple preprocessing
- Data augmentation helper used for SimCLR-style pretraining
- NT-Xent contrastive loss implementation
- A small decoder builder for image reconstruction

Keep these helpers simple and well-documented — they are intentionally lightweight so
other modules can import them without pulling in a lot of logic.
"""

import os
import sys
import wandb
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

AUTOTUNE = tf.data.AUTOTUNE


class WandbEpochLogger(tf.keras.callbacks.Callback):
    """
    Tiny callback that logs per-epoch metrics to Weights & Biases.

    Usage: include in `model.fit(..., callbacks=[WandbEpochLogger()])` so metrics are
    visible in your W&B dashboard without adding much ceremony.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # prefix keys so they are organized under `train/` in W&B
            wandb.log({f"train/{k}": v for k, v in logs.items()}, step=epoch)


# ---------------------------
# Serializable L2 Norm Layer
# ---------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class L2Normalization(layers.Layer):
    """
    Simple L2-normalization layer registered for Keras serialization.

    It normalizes vectors along axis=1 which is convenient for embedding outputs
    coming from a dense projection head (shape [batch, dim]).
    """
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()


# ---------------------------
# Dataset helpers
# ---------------------------
def get_image_datasets(train_dir, val_dir, img_size=(128,128), batch_size=32, shuffle=True, seed=42):
    """
    Create tf.data Datasets from directory layout (one folder per class).

    Returns: (train_ds, val_ds, class_names)
    The datasets yield (image, label) tuples where image is uint8 in [0,255] and
    label is an integer index. Callers should cast/scale images as needed.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Capture class names before any mapping/transforms
    class_names = train_ds.class_names

    # Prefetch to improve pipeline throughput; actual normalization is left to the caller
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def preprocess_image(x, img_size=(128,128)):
    """
    Normalize an image tensor/array to float32 in [0,1] and resize.

    Notes:
    - Accepts either a tf.Tensor or a numpy array.
    - Always returns a tf.Tensor (float32) resized to `img_size`.

    This function keeps behavior predictable for callers that expect numeric arrays.
    """
    x = tf.cast(x, tf.float32)
    # If inputs were in 0..255 range, scale down to 0..1. If already floats but >1, clamp and scale.
    x = tf.clip_by_value(x, 0.0, 255.0) / 255.0
    x = tf.image.resize(x, img_size)
    return x


# ---------------------------
# Augmentations (TF-only)
# ---------------------------
def simclr_augment(image, img_size=(128,128)):
    """
    A compact set of augmentations inspired by SimCLR.

    Operations:
    - resize -> random horizontal flip -> color jitter (sometimes) -> blur (sometimes)
    - returns a tf.Tensor float32 in [0,1]
    """
    image = tf.image.resize(image, img_size)
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0) / 255.0
    image = tf.image.random_flip_left_right(image)

    def apply_color(x):
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        x = tf.image.random_saturation(x, 0.8, 1.2)
        x = tf.image.random_hue(x, 0.02)
        return tf.clip_by_value(x, 0.0, 1.0)

    image = tf.cond(tf.random.uniform([], 0, 1) < 0.8, lambda: apply_color(image), lambda: image)

    def apply_blur(x):
        # tiny average-pool blur to simulate mild smoothing
        x = tf.expand_dims(x, 0)
        x = tf.nn.avg_pool(x, ksize=3, strides=1, padding="SAME")
        return tf.squeeze(x, 0)

    image = tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: apply_blur(image), lambda: image)
    return image


# ---------------------------
# SimCLR NT-Xent Loss
# ---------------------------
class NTXentLoss(tf.keras.losses.Loss):
    """
    NT-Xent contrastive loss implementation used for SimCLR pretraining.

    Inputs: two batches of embeddings (zis, zjs) with shape (batch, dim).
    Output: scalar loss value.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def call(self, zis, zjs):
        # Concatenate positive pairs along the batch axis, normalize and compute similarity matrix
        z = tf.concat([zis, zjs], axis=0)
        z = tf.math.l2_normalize(z, axis=1)
        sim = tf.matmul(z, z, transpose_b=True) / self.temperature
        b = tf.shape(zis)[0]
        mask = tf.eye(2 * b)
        # Prevent self-similarity from contributing to the softmax by setting it to a large negative number
        sim = sim * (1 - mask) + mask * -1e9
        positives = tf.concat([tf.range(b, 2*b), tf.range(0, b)], 0)
        labels = tf.one_hot(positives, 2*b)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, sim)
        return tf.reduce_mean(loss)


# ---------------------------
# Model builders
# ---------------------------
def build_encoder(input_shape=(128, 128, 3), projection_dim=128):
    """
    Build a ResNet50-based encoder with a small projection head.

    Returns: a Keras Model mapping images -> L2-normalized projection vectors.
    """
    inputs = layers.Input(shape=input_shape)

    base = ResNet50(
        include_top=False,
        weights="imagenet",
        pooling=None,        # keep conv feature maps
        input_tensor=inputs
    )

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(projection_dim)(x)
    outputs = L2Normalization()(x)
    return models.Model(inputs, outputs, name="simclr_encoder")


def build_classifier_from_encoder(
        encoder, num_classes=2, freeze=False,
        embed_dense=256, dropout=0.3):
    import tensorflow as tf
    from tensorflow.keras import layers

    if freeze:
        encoder.trainable = False

    # ----------------------------
    # Resolve encoder input/output
    # ----------------------------
    if isinstance(encoder, tf.keras.Model):
        inputs = encoder.input
        raw_output = encoder.output
    else:
        inputs = tf.keras.Input(shape=encoder.input_shape[1:])
        raw_output = encoder(inputs)

    # -------------------------------------
    # If encoder outputs a list → pick last
    # -------------------------------------
    if isinstance(raw_output, (list, tuple)):
        print("[DEBUG] Encoder has multiple outputs. Using the LAST one.")
        x = raw_output[-1]  # usually embedding or useful head output
    else:
        x = raw_output

    # -------------------------------
    # Spatial → GAP → Dense pathway
    # -------------------------------
    if len(x.shape) == 4:  # (B,H,W,C)
        print("[DEBUG] Spatial encoder detected → applying GAP.")
        x = layers.GlobalAveragePooling2D(name="spatial_pool")(x)

    # -------------------------------
    # Classification head
    # -------------------------------
    x = layers.Dense(embed_dense, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="head_logits")(x)

    clf = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier_clean")
    return clf


def build_image_reconstructor(encoder_output_dim, target_shape=(128,128,3), base_spatial=8, base_filters=256):
    """
    Build a decoder that progressively upsamples to target_shape.
    - encoder_output_dim: int (vector dim) OR tuple/list (spatial feature map shape HWC).
    - target_shape: desired output image shape (H, W, C).
    - base_spatial: starting spatial size when encoder provides a vector (default 8 -> 8x8).

    This decoder is kept intentionally simple; it's good enough to visualize what information
    the encoder preserves without being an image-quality state-of-the-art model.
    """
    # Accept either a flat vector embedding or a spatial feature map
    if isinstance(encoder_output_dim, (tuple, list)):
        # e.g., encoder_output_dim = (h, w, c)
        inputs = tf.keras.Input(shape=tuple(encoder_output_dim))
        x = inputs
        current_h = int(encoder_output_dim[0])
        filters = int(encoder_output_dim[2]) if len(encoder_output_dim) == 3 else base_filters
    else:
        # flat embedding -> project and reshape to (base_spatial, base_spatial, base_filters)
        inputs = tf.keras.Input(shape=(int(encoder_output_dim),))
        x = layers.Dense(base_spatial * base_spatial * base_filters, activation="relu")(inputs)
        x = layers.Reshape((base_spatial, base_spatial, base_filters))(x)
        current_h = base_spatial
        filters = base_filters

    target_h, target_w = int(target_shape[0]), int(target_shape[1])

    # Progressive upsampling using UpSampling2D + Conv2D (bilinear upsampling reduces checkerboard artifacts)
    while current_h < target_h:
        # halve filters gradually but keep at least 32
        filters = max(32, filters // 2)
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        current_h *= 2

        # safety: break if loop becomes infinite
        if current_h > target_h * 2:
            break

    # If we accidentally exceeded target due to mismatched sizes, adjust by a final Conv2D
    if current_h != target_h:
        # apply a Conv2D to reach the exact size (this keeps it simple; usually sizes will match)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(target_shape[2], 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='decoder')

# ---------------------------
# Recursive Conv Finder
# ---------------------------
def find_last_conv_in_model(model):
    """
    Recursively search for the last Conv2D/Conv2DTranspose layer inside a model.
    Returns a Layer object or None.
    """
    last_conv = None
    for l in model.layers:
        if isinstance(l, (layers.Conv2D, layers.Conv2DTranspose)):
            last_conv = l
        elif isinstance(l, tf.keras.Model):
            nested_conv = find_last_conv_in_model(l)
            if nested_conv is not None:
                return nested_conv
    return last_conv

def list_all_convs(model):
    """Recursively collect all Conv2D / Conv2DTranspose layers with shape + param count."""
    convs = []
    for l in model.layers:
        if isinstance(l, (layers.Conv2D, layers.Conv2DTranspose)):
            try:
                shape = str(l.output.shape)  # symbolic shape
            except Exception:
                shape = "N/A"
            params = l.count_params()
            convs.append((l.name, shape, params))
        elif isinstance(l, tf.keras.Model):  # recurse into nested models
            convs.extend(list_all_convs(l))
    return convs

def log_all_convs(model, model_name="model"):
    """Log conv layers as a W&B table with name, shape, and param count."""
    convs = list_all_convs(model)
    table = wandb.Table(columns=["Layer Name", "Output Shape", "Params"])
    for name, shape, params in convs:
        table.add_data(name, shape, params)

    if wandb.run:
        wandb.log({f"{model_name}/conv_layers": table})

# ---------------------------
# Save / Load wrappers (include L2Normalization automatically)
# ---------------------------
def save_model(model, path):
    # ensure .keras extension for Keras native format
    if not (str(path).endswith(".keras") or str(path).endswith(".h5")):
        path = str(path) + ".keras"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def warmup_model(model, input_shape):
    try:
        # Drop batch dimension if provided (None,128,128,3)
        if input_shape[0] is None:
            input_shape = input_shape[1:]

        # Validate
        if any(v is None for v in input_shape):
            raise ValueError(f"Invalid warmup shape: {input_shape}")

        input_shape = tuple(int(v) for v in input_shape)

        dummy = tf.zeros((1,) + input_shape)
        _ = model(dummy, training=False)

    except Exception as e:
        print(f"⚠️ Warmup skipped: {e}")

    return model


def load_model(path, compile=False, safe_mode=True, warmup=True):
    """
    Load a Keras model with L2Normalization custom object.
    Optionally run warm-up to ensure conv/output shapes are initialized.
    """
    model = tf.keras.models.load_model(
        path,
        compile=compile,
        safe_mode=safe_mode,
        custom_objects={"L2Normalization": L2Normalization}
    )

    if warmup:
        input_shape = getattr(model, "input_shape", None)
        if input_shape is None:
            # fallback if not saved in model
            input_shape = (None, 128, 128, 3)
        model = warmup_model(model, input_shape)

    return model


# ---------------------------
# Simple logger helper
# ---------------------------
def log_progress(epoch, step, total_steps, loss, every=10):
    if step % every == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{now}] Epoch {epoch} Step {step}/{total_steps} Loss={loss:.4f}"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
    if (step + 1) == total_steps:
        print()


def init_wandb(project, config):
    if project and wandb.run is None:
        wandb.init(project=project, config=config, reinit=True, resume="allow")
    return wandb
