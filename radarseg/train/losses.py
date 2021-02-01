# 3rd Party Libraries
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='Custom', name='CategoricalFocalCrossentropy')
class CategoricalFocalCrossentropy(LossFunctionWrapper):
    """
    Computes the categorical focal crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes. The
    lables are expacted to be provided in a `one_hot` representation.

    Class wrapper for the categorical focal crossentropy loss function.

    Arguments:
        from_logits: Whether prediction (y_pred) is expected to be a logits tensor, <bool>.
        alpha: Balancing parameter between positives and negatives, <list of floats or float>.
        gamma: Focusing parameter of the modulating factor, <float>.
        class_weight: List of class weights to weigh classes differently, <list>.
        reduction: Reduction type to apply to the loss, <tf.keras.losses.Reduction>.
    """
    def __init__(self,
                 from_logits=False,
                 alpha=None,
                 gamma=2.0,
                 class_weight=None,
                 reduction=tf.losses.Reduction.AUTO,
                 name='focal_loss'):
        # Call loss function wrapper
        super(CategoricalFocalCrossentropy, self).__init__(
            fn=categorical_focal_crossentropy,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
            class_weight=class_weight)


@tf.keras.utils.register_keras_serializable(package='Custom', name='categorical_focal_crossentropy')
@tf.function
def categorical_focal_crossentropy(y_true, y_pred, from_logits=False, alpha=None, gamma=2.0, class_weight=None):
    """
    Computes the categorical focal crossentropy loss.

    Modified focal loss implementation for multiple classes and class individual balancing factors.

    Reference:
        Lin, T., Focal Loss for Dense Object Detection. Available: https://arxiv.org/abs/1708.02002.

    Arguments:
        y_true: Tensor of one-hot true targets, <tf.Tensor>.
        y_pred: Tensor of predicted targets, <tf.Tensor>.
        from_logits: Whether prediction (y_pred) is expected to be a logits tensor, <bool>.
        alpha: Balancing parameter between positives and negatives, <list of floats or float>.
        gamma: Focusing parameter of the modulating factor, <float>.

    Returns:
        Categorical focal crossentropy loss value.
    """
    # Type casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(gamma)

    # Correction term for padded batching
    cor = K.sum(y_true, axis=-1, keepdims=True)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        y_pred = K.sigmoid(y_pred)

    # Clip prediction values for numerical stability
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    # Get categorical cross_entropy for each entry
    cce = y_true * K.log(y_pred) + ((1 - y_true) * K.log(1 - y_pred))

    # Get estimated class probability
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    # Get balancing factor
    if alpha is not None:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        balancing_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    else:
        balancing_factor = K.ones_like(y_true)

    # Get modulating factor
    if gamma != 0:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = K.pow((1 - p_t), gamma)
    else:
        modulating_factor = K.ones_like(y_true)

    # Get class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight, dtype=K.floatx())
        class_weight = y_true * class_weight
        class_weight = K.sum(class_weight, axis=-1, keepdims=True)
    else:
        class_weight = K.ones_like(cor)

    return -K.sum(balancing_factor * modulating_factor * cce, axis=-1, keepdims=True) * cor * class_weight
