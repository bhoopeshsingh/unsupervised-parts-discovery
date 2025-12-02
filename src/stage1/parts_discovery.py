# parts_discovery.py
# This module finds "semantic parts" of an image without labels.
# It pulls out convolutional feature maps from a model and runs KMeans on the
# spatial feature vectors so that nearby pixels with similar features end up
# in the same cluster (an unsupervised part).

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans


def parts_discovery(img, classifier, last_conv_layer=None, n_parts=4):
    """
    Discovers semantic parts in an image by clustering conv features.

    This is where the magic happens - we take the spatial feature maps from the conv layer
    and run KMeans on them. Each spatial location gets assigned to a part cluster.

    Args:
        img: input image (numpy array or tensor)
        classifier: the full model (needs to contain the conv layer we want)
        last_conv_layer: which conv layer to use (we'll find it automatically if not provided)
        n_parts: how many parts to split the image into (default 4)

    Returns:
        A 2D array the same size as the input image, where each pixel value represents
        which part it belongs to. It's normalized to [0,1] for easy visualization.
    """
    # ensure numpy HWC float in [0,1]
    if isinstance(img, tf.Tensor):
        img_arr = img.numpy()
    else:
        img_arr = np.asarray(img, dtype=np.float32)

    # find conv layer inside classifier
    conv_layer_in_clf = None
    if isinstance(last_conv_layer, tf.keras.layers.Layer):
        # If a layer object is passed we try to map it into the classifier
        lname = last_conv_layer.name
        try:
            conv_layer_in_clf = classifier.get_layer(lname)
        except (ValueError, KeyError):
            # fallback: look for matching name suffix
            matches = [l for l in classifier.layers if l.name.endswith(lname)]
            if matches:
                conv_layer_in_clf = matches[-1]
    elif isinstance(last_conv_layer, str):
        # If a name was passed, try to get it directly
        try:
            conv_layer_in_clf = classifier.get_layer(last_conv_layer)
        except (ValueError, KeyError):
            matches = [l for l in classifier.layers if l.name.endswith(last_conv_layer)]
            if matches:
                conv_layer_in_clf = matches[-1]

    # if not provided or mapping failed, find last conv directly from classifier
    if conv_layer_in_clf is None:
        from utils import find_last_conv_in_model
        conv_layer_in_clf = find_last_conv_in_model(classifier)
        if conv_layer_in_clf is None:
            raise ValueError("No conv layer found in classifier for parts discovery.")

    # Build a submodel that maps classifier input -> conv features
    conv_model = tf.keras.models.Model(inputs=classifier.inputs, outputs=conv_layer_in_clf.output)

    img_tensor = tf.expand_dims(tf.convert_to_tensor(img_arr, dtype=tf.float32), 0)
    conv_features = conv_model(img_tensor, training=False)[0].numpy()  # Hc x Wc x C

    Hc, Wc, C = conv_features.shape
    flat = conv_features.reshape(-1, C)
    if flat.shape[0] < n_parts:
        # not enough spatial locations to cluster
        return np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.float32)

    # KMeans clustering in feature space
    kmeans = KMeans(n_clusters=n_parts, n_init=5, random_state=0).fit(flat)
    labels = kmeans.labels_.reshape(Hc, Wc).astype(np.float32)

    # Resize labels to original image resolution and normalize
    labels_resized = tf.image.resize(labels[..., np.newaxis], (img_arr.shape[0], img_arr.shape[1]), method="nearest").numpy()[..., 0]
    return (labels_resized / (n_parts - 1)) if n_parts > 1 else labels_resized
