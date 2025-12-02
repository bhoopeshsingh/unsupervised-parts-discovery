"""
image_reconstructor.py

Purpose:
- Train a decoder to reconstruct images from encoder embeddings
- Evaluate reconstruction quality
"""

import os
import argparse
import tensorflow as tf
from utils import (
    get_image_datasets,
    load_model,
    build_image_reconstructor,
    init_wandb,
    L2Normalization
)


def train_image_reconstructor(args):
    print("Training image reconstructor...")

    # Load full encoder
    full_encoder = load_model(args.encoder_path, compile=False)

    print("\n" + "=" * 90)
    print("RECONSTRUCTOR DIAGNOSTICS: MODEL LOADING")
    print("=" * 90)

    print(f"✔ Loaded encoder from: {args.encoder_path},   name={full_encoder.name}, trainable={full_encoder.trainable}, shape={full_encoder.input_shape} -> {full_encoder.output_shape}")

    # Extract spatial feature stage
    last_conv = full_encoder.get_layer("conv5_block3_3_conv")

    print(f"✔ Found last conv layer: {last_conv.name}")

    encoder = tf.keras.Model(
        inputs=full_encoder.input,
        outputs=last_conv.output,
        name="spatial_encoder"
    )
    print(f"✔ Built spatial encoder: name={encoder.name}, trainable={encoder.trainable}, shape={encoder.input_shape} -> {encoder.output_shape}")

    # Datasets
    train_ds, val_ds, class_names = get_image_datasets(
        args.train_dir, args.val_dir,
        (args.img_size, args.img_size),
        args.batch_size
    )

    def prep(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, x

    train_ds = train_ds.map(prep).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep).prefetch(tf.data.AUTOTUNE)

    # Spatial encoder output
    encoder_output_shape = encoder.output_shape[1:]

    # Build decoder
    decoder = build_image_reconstructor(
        encoder_output_shape,
        target_shape=(args.img_size, args.img_size, 3)
    )

    # Autoencoder
    inputs = encoder.input
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    autoencoder = tf.keras.Model(inputs, decoded)
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='mse'
    )

    # Train
    history = autoencoder.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # Save
    decoder_path = os.path.join(args.output_dir, "image_reconstructor.keras")
    decoder.save(decoder_path)

    return decoder, autoencoder


def evaluate_image_reconstructor(encoder_path, decoder_path, val_ds, n=5, log_to_wandb=False):
    """
    Evaluate reconstruction quality using spatial encoder + decoder.
    """
    import numpy as np
    import wandb

    # Load full encoder
    full_encoder = load_model(encoder_path, compile=False)

    # IMPORTANT: extract spatial layer (same as training)
    try:
        last_conv = full_encoder.get_layer("conv5_block3_3_conv")
    except:
        raise ValueError("ERROR: Could not find spatial layer 'conv5_block3_3_conv' in encoder.")

    spatial_encoder = tf.keras.Model(
        inputs=full_encoder.input,
        outputs=last_conv.output,
        name="spatial_encoder_eval"
    )

    # Load decoder
    decoder = load_model(decoder_path, compile=False)

    # Collect n samples
    samples = []
    for x, y in val_ds.unbatch().take(n):
        img = tf.cast(x, tf.float32) / 255.0
        samples.append(img.numpy())

    # Reconstruction loop
    for i, img in enumerate(samples):
        inp = np.expand_dims(img, 0)  # (1,H,W,C)

        encoded = spatial_encoder.predict(inp, verbose=0)  # (1,4,4,2048)
        reconstructed = decoder.predict(encoded, verbose=0)[0]

        if log_to_wandb and wandb.run:
            wandb.log({
                f"reconstruction_{i}": [
                    wandb.Image((img * 255).astype('uint8'), caption="original"),
                    wandb.Image((reconstructed * 255).astype('uint8'), caption="reconstructed")
                ]
            })


def main(args):
    """Main training pipeline"""
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, vars(args))
    
    decoder, autoencoder = train_image_reconstructor(args)
    
    return decoder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--output_dir', default='./saved_models')
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wandb_project', default=None)
    args = p.parse_args()
    
    main(args)
