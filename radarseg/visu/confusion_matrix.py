# Standard Libraries
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names, norm: bool = True, decimals: int = 2):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Arguments:
        cm: A confusion matrix of integer classes [num_classes, num_classes], <np.array or tf.Tensor>.
        class_names: String names of the integer classes [num_classes], <np.array or tf.Tensor>.
        norm: Whether to normalize the confusion matrix data (perceptual representation), <bool>.
        decimals: Number of valid decimal places, <int>.
    """
    # Convert confusion matrix.
    cm = tf.cast(cm, tf.float32)

    # Normalize the confusion matrix.
    if norm:
        cm = np.around(cm / tf.math.reduce_sum(cm, axis=1, keepdims=False)[:, np.newaxis], decimals=decimals)

    # Plot heat map
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark, otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
