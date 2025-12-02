
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import os
import argparse
from utils import get_image_datasets, simclr_augment
from utils import NTXentLoss
from utils import log_progress, init_wandb, log_all_convs, find_last_conv_in_model
from utils import L2Normalization

"""
feature_backbone.py

Purpose:
- Utilities to build and (optionally) pretrain a SimCLR-style encoder that maps images
  to compact embeddings. Includes dataset pairing for contrastive learning and a small
  training loop that uses the NT-Xent loss from `utils.py`.

Main functions:
- make_simclr_dataset: produce paired augmented views for contrastive loss
- build_simclr_encoder: ResNet50 backbone with optional spatial return
- train_simclr: simple training loop using NT-Xent
- save_encoder: save encoder to disk
"""

# ---------------------------
# Dataset pipeline
# ---------------------------
# make_simclr_dataset: creates TF dataset of paired augmentations for SimCLR
def make_simclr_dataset(train_dir, val_dir, img_size, batch_size):
    """
    Convert an image folder dataset into a SimCLR-style dataset that yields
    (augmented_view1, augmented_view2) pairs for each original image.
    """
    train_ds, val_ds, class_names = get_image_datasets(train_dir, val_dir, (img_size, img_size), batch_size)
    print(f"Found {len(class_names)} classes: {class_names}")

    def make_pairs(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        aug1 = simclr_augment(x, (img_size, img_size))
        aug2 = simclr_augment(x, (img_size, img_size))
        return aug1, aug2

    @tf.autograph.experimental.do_not_convert
    def pair_wrapper(x, y):
        return make_pairs(x, y)

    simclr_ds = train_ds.map(lambda x, y: pair_wrapper(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    simclr_ds = simclr_ds.prefetch(tf.data.AUTOTUNE)
    return simclr_ds, train_ds


# ---------------------------
# Build model
# ---------------------------
# build_simclr_encoder: build either a spatial output encoder (for recon) or
# a vector encoder with a projection head (for contrastive learning)
def build_simclr_encoder(input_shape=(128, 128, 3), projection_dim=128, return_spatial=False):
    inputs = layers.Input(shape=input_shape)

    base = ResNet50(
        include_top=False,
        weights="imagenet",
        pooling=None,        # keep conv feature maps
        input_tensor=inputs
    )

    if return_spatial:
        # Return the last conv feature map for reconstruction
        return models.Model(inputs, base.output, name="feature_backbone_encoder_spatial")
    else:
        # Projection head for contrastive learning
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(projection_dim)(x)
        outputs = L2Normalization()(x)
        return models.Model(inputs, outputs, name="feature_backbone_encoder")

# ---------------------------
# Training loop
# ---------------------------
def train_simclr(encoder, simclr_ds, args, wandb=None):
    """
    Simplified training loop for SimCLR-style contrastive pretraining.

    This is intentionally small so it's easy to read and adapt; for large-scale
    training you'd replace this with a compiled model + standard Keras training.
    """
    opt = tf.keras.optimizers.Adam(args.lr)
    loss_fn = NTXentLoss(args.temperature)

    @tf.function
    def step(x1, x2):
        with tf.GradientTape() as tape:
            z1 = encoder(x1, training=True)
            z2 = encoder(x2, training=True)
            loss = loss_fn(z1, z2)
        grads = tape.gradient(loss, encoder.trainable_variables)
        opt.apply_gradients(zip(grads, encoder.trainable_variables))
        return loss

    for epoch in range(args.epochs):
        for stepn, (a1, a2) in enumerate(simclr_ds.take(args.steps_per_epoch)):
            loss = step(a1, a2)
            log_progress(epoch, stepn, args.steps_per_epoch, loss.numpy(), every=10)

            if stepn == 0 and wandb:
                imgs = tf.concat([a1[:3], a2[:3]], axis=0)
                wandb.log({"augment_examples": [wandb.Image(img) for img in imgs.numpy()]})
            elif wandb:
                wandb.log({"pretrain_loss": float(loss)})

    return encoder


# ---------------------------
# Save + Log
# ---------------------------
def save_encoder(encoder, args, name="simclr_encoder.keras", wandb=None):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, name)
    encoder.save(path)
    return path


# ---------------------------
# Logging details - during pretrain
# ---------------------------
def log_details(encoder, train_ds, wandb):
    # Log conv layers
    log_all_convs(encoder, "simclr_encoder")
    last_conv = find_last_conv_in_model(encoder)
    if wandb:
        wandb.config.update({f"simclr_encoder_last_conv": last_conv.name}, allow_val_change=True)

    # Embedding dimension check
    x, _ = next(iter(train_ds.take(1)))
    z = encoder(x[:1])  # first call builds graph
    if wandb:
        wandb.config.update({"simclr_encoder_embedding_dim": z.shape[-1]}, allow_val_change=True)
    return last_conv, z


# ---------------------------
# Main wrapper
# ---------------------------
def pretrain_main(args):
    wandb = init_wandb(args.wandb_project, vars(args))

    simclr_ds, train_ds = make_simclr_dataset(args.train_dir, args.val_dir, args.img_size, args.batch_size)

    # For SimCLR contrastive pretraining

    # encoder = build_simclr_encoder(args.img_size, args.projection_dim)
    encoder_vec = build_simclr_encoder(return_spatial=False)

    encoder = train_simclr(encoder_vec, simclr_ds, args, wandb=wandb)
    save_encoder(encoder, args, wandb=wandb)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--projection_dim', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps_per_epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--output_dir', default='./saved_models')
    p.add_argument('--wandb_project', default="simclr-pretrain")
    args = p.parse_args()
    pretrain_main(args)


def main():
    return None