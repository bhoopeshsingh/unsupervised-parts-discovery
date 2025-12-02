"""
label_predictor.py

High-level purpose:
- Fine-tune a pretrained encoder for a small classification task using a two-stage protocol.
  1) Train only the new classification head while the encoder is frozen (fast, safe).
  2) Unfreeze the encoder and fine-tune end-to-end with a lower learning rate.

Notes for maintainers:
- This script expects an encoder saved in Keras format (see `utils.load_model`).
- It uses Weights & Biases (wandb) for optional logging; if you don't use W&B, pass None as the project.
- The code prefers tf.data Datasets created by `get_image_datasets` (so class_names are inferred).

"""

import argparse, os, numpy as np, tensorflow as tf
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from utils import (
    get_image_datasets, build_classifier_from_encoder, load_model,
    init_wandb, save_model, log_all_convs, find_last_conv_in_model, L2Normalization,
    WandbEpochLogger
)
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# -------------------------
# Data Prep Function
# -------------------------
# prepare_datasets: loads and prepares classification datasets (normalizes inputs)
def prepare_datasets(train_dir, val_dir, img_size, batch_size):
    train_ds, val_ds, class_names = get_image_datasets(train_dir, val_dir, (img_size, img_size), batch_size)

    #class_names = train_ds.class_names  # capture early

    def prep(x, y): return tf.cast(x, tf.float32)/255.0, y
    train_ds = train_ds.map(prep).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(prep).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


# -------------------------
# Stage 1: Train head only
# -------------------------
# train_classifier_head: attach head to encoder and train only the new layers
def train_classifier_head(encoder, train_ds, val_ds, num_classes, lr_head, epochs_head, output_dir):

    clf = build_classifier_from_encoder(
                encoder,
                num_classes,
                freeze=True  # Freeze encoder layers
        )

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Stage 1: Training classifier head (encoder frozen)...")

    clf.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        verbose=1,
        callbacks=[
            WandbEpochLogger(),
            WandbModelCheckpoint(filepath=os.path.join(output_dir, "clf_head_checkpoint.keras"))
        ]
    )
    return clf


# -------------------------
# Stage 2: Fine-tune encoder
# -------------------------
# fine_tune_classifier: unfreeze encoder and train entire model with low lr
def fine_tune_classifier(encoder, clf, train_ds, val_ds, lr_full, epochs_full, output_dir):

    encoder.trainable = True # Unfreeze encoder layers for fine-tuning
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(lr_full),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Stage 2: Fine-tuning entire model (encoder unfrozen)...")
    clf.fit(
        train_ds, validation_data=val_ds, epochs=epochs_full,
        callbacks=[WandbMetricsLogger(),
                   WandbModelCheckpoint(filepath=os.path.join(output_dir, "clf_full_checkpoint.keras"))]
    )
    return clf


# -------------------------
# Log Classifier Info
# -------------------------
# log_classifier_info: gather architecture metadata and log to W&B
def log_classifier_info(clf, train_ds, wandb_run=None):
    """
    Logs all the important details about our classifier to Weights & Biases.
    This includes layer info, embedding dimensions, etc. - helpful for tracking experiments!
    """
    # Conv layers from full classifier
    log_all_convs(clf, "classifier")

    # Last conv
    last_conv = find_last_conv_in_model(clf)
    if wandb_run:
        wandb.config.update({"classifier/last_conv": last_conv.name}, allow_val_change=True)

    # Find encoder or infer embedding layer
    encoder_submodel = None
    for l in clf.layers:
        if isinstance(l, tf.keras.Model):
            encoder_submodel = l
            break

    if encoder_submodel is None:
        # infer embedding as penultimate before head
        out_dim = clf.output_shape[-1]
        final_dense_index = None
        for i, l in enumerate(clf.layers):
            if isinstance(l, tf.keras.layers.Dense) and l.units == out_dim:
                final_dense_index = i
                break
        if final_dense_index is not None and final_dense_index > 0:
            encoder_submodel = tf.keras.Model(
                clf.inputs, clf.layers[final_dense_index - 1].output, name="inferred_encoder_submodel"
            )

    if encoder_submodel is None:
        raise ValueError("❌ Encoder submodel not found inside classifier and could not infer embedding layer")

    # Embedding dimension
    x, _ = next(iter(train_ds.take(1)))
    z = encoder_submodel(x[:1])
    if wandb_run:
        wandb.config.update({"classifier/embedding_dim": z.shape[-1]}, allow_val_change=True)

    # Output dimension
    if wandb_run:
        wandb.config.update({"classifier/output_dim": clf.output_shape[-1]}, allow_val_change=True)

    print("✅ Logged classifier info to W&B")
    return last_conv.name, z.shape[-1]


# -------------------------
# Confusion Matrix + Report
# -------------------------
# evaluate_classifier: run predictions on val set and generate report + confusion matrix
def evaluate_classifier(clf, val_ds, class_names, wandb_run=None):
    y_true, y_pred = [], []
    for x, y in val_ds.unbatch():
        preds = clf.predict(tf.expand_dims(x, 0), verbose=0)
        y_true.append(int(y.numpy()))
        y_pred.append(int(np.argmax(preds)))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if wandb_run:
        wandb.log({"confusion_matrix": wandb.Image(plt)})

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    if wandb_run:
        wandb.log({"classification_report": wandb.Html("<pre>" + report + "</pre>")})


# -------------------------
# Orchestration
# -------------------------
# finetune_main: top-level function to perform the full finetuning + eval flow
def finetune_main(args):
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, vars(args))

    # Prepare datasets
    train_ds, val_ds, _ = prepare_datasets(args.train_dir, args.val_dir, args.img_size, args.batch_size)

    # Load encoder
    encoder = load_model(args.encoder_path, compile=False, custom_objects={"L2Normalization": L2Normalization})

    # Stage 1
    clf = train_classifier_head(encoder, train_ds, val_ds, args.num_classes, args.lr_head, args.epochs_head, args.output_dir)

    # Stage 2
    clf = fine_tune_classifier(encoder, clf, train_ds, val_ds, args.lr_full, args.epochs_full, args.output_dir)


    # Evaluation
    evaluate_classifier(clf, val_ds, class_names=train_ds.class_names, wandb_run=wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--output_dir', default="./saved_models")
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs_head', type=int, default=5)
    p.add_argument('--epochs_full', type=int, default=10)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_full', type=float, default=1e-5)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--wandb_project', default="simclr-finetune")
    args = p.parse_args()
    finetune_main(args)


def main():
    return None