# 3rd Party Libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# tensorflow backend
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.keras.utils import metrics_utils


class _TopKConfusionMatrixConditionCount(tf.keras.metrics.Metric):
    """
    Calculates the number of the given confusion matrix condition.

    Arguments:
        confusion_matrix_cond: One of metrics_utils.ConfusionMatrix conditions.
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 confusion_matrix_cond,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Initialize keras base metric instance
        super(_TopKConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)

        # Initialize the top K confusion matrix condition count
        self._confusion_matrix_cond = confusion_matrix_cond
        self.init_thresholds = thresholds if top_k is None else 0.0
        self.thresholds = metrics_utils.parse_init_thresholds(self.init_thresholds, default_threshold=0.5)
        self.top_k = top_k
        self.class_id = class_id
        self.sample_weight = sample_weight
        self.accumulator = self.add_weight('accumulator', shape=(len(self.thresholds),), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates the given confusion matrix condition statistics.

        Arguments:
            y_true: The ground truth values, <tf.Tensor>.
            y_pred: The predicted values, <tf.Tensor>.
            sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.

        Returns:
            Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
            {self._confusion_matrix_cond: self.accumulator},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
            multi_label=False,
            label_weights=None)

    def result(self):
        if len(self.thresholds) == 1:
            result = self.accumulator[0]
        else:
            result = self.accumulator

        return tf.convert_to_tensor(result)

    def reset_states(self):
        if isinstance(self.thresholds, list):
            num_thresholds = len(self.thresholds)
        else:
            num_thresholds = len(list(self.thresholds))

        K.batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        # Get top K confusion matrix condition count configuration
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id,
            'sample_weight': self.sample_weight
        }

        # Get keras base metric configuration
        base_config = super(_TopKConfusionMatrixConditionCount, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Custom', name='TopKCategoricalTruePositives')
class TopKCategoricalTruePositives(_TopKConfusionMatrixConditionCount):
    """
    Arguments:
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Call confusion matrix condition count
        super(TopKCategoricalTruePositives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            sample_weight=sample_weight,
            name=name,
            dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='TopKCategoricalTrueNegatives')
class TopKCategoricalTrueNegatives(_TopKConfusionMatrixConditionCount):
    """
    Arguments:
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Call confusion matrix condition count
        super(TopKCategoricalTrueNegatives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            sample_weight=sample_weight,
            name=name,
            dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='TopKCategoricalFalsePositives')
class TopKCategoricalFalsePositives(_TopKConfusionMatrixConditionCount):
    """
    Arguments:
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Call confusion matrix condition count
        super(TopKCategoricalFalsePositives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            sample_weight=sample_weight,
            name=name,
            dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='TopKCategoricalFalseNegatives')
class TopKCategoricalFalseNegatives(_TopKConfusionMatrixConditionCount):
    """
    Arguments:
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Call confusion matrix condition count
        super(TopKCategoricalFalseNegatives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            sample_weight=sample_weight,
            name=name,
            dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='ConfusionMatrix')
class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    Calculates the confusion matrix for a multiclass, single label problem.

    Inputs:
        y_true: Target values (class labels), <tf.Tensor>.
        y_pred: Predicted values (model output), <tf.Tensor>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.

    Arguments:
        num_classes: Number of classes of the model output (y_pred), <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.

    Returns:
        confusion_matrix: The confusion matrix of size (num_classes, num_classes), <tf.Tensor>.
    """
    def __init__(self,
                 num_classes,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Initialize keras base metric instance
        super(ConfusionMatrix, self).__init__(name=name, dtype=dtype)

        # Checks
        if type(num_classes) not in (int,):
            raise TypeError("The value of num_classes should be a python int, but a '{}' was given".format(type(num_classes)))

        # Initialize the ConfusionMatrix instance
        self.num_classes = num_classes
        self.sample_weight = sample_weight

        # Add metric states
        self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(self.num_classes, self.num_classes), initializer='zeros', dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast inputs
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Transform inputs
        [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_pred, y_true], sample_weight)

        # Check input values and adjust shapes
        with ops.control_dependencies([
            check_ops.assert_greater_equal(
                y_pred,
                tf.cast(0.0, dtype=y_pred.dtype),
                message='predictions must be >= 0'),
            check_ops.assert_less_equal(
                y_pred,
                tf.cast(1.0, dtype=y_pred.dtype),
                message='predictions must be <= 1')]):

            if sample_weight is None:
                y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
            else:
                y_pred, y_true, sample_weight = (tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight=sample_weight))

        # Check shape compatibility
        y_pred.shape.assert_is_compatible_with(y_true.shape)

        # Get prediction shape
        pred_shape = tf.shape(y_pred)
        num_predictions = pred_shape[0]

        # Get lables (decode one-hot)
        y_pred_labels = K.flatten(tf.argmax(y_pred, axis=-1))
        y_true_labels = K.flatten(tf.argmax(y_true, axis=-1))

        # Set sample weights
        if sample_weight is not None:
            sample_weight = weights_broadcast_ops.broadcast_weights(tf.cast(sample_weight, dtype=y_pred.dtype), y_pred)
            weights_tiled = tf.gather(K.flatten(sample_weight),
                                      tf.range(start=0, limit=num_predictions * self.num_classes, delta=self.num_classes, dtype=tf.int64))
        else:
            weights_tiled = None

        def _weighted_assign_add(label, pred, weights, var):

            return var.assign_add(tf.math.confusion_matrix(labels=label,
                                                           predictions=pred,
                                                           num_classes=self.num_classes,
                                                           weights=weights,
                                                           dtype=self.dtype))

        # Set return value
        update_ops = []

        # Update confusion matrix
        update_ops.append(_weighted_assign_add(y_true_labels, y_pred_labels, weights_tiled, self.confusion_matrix))

        return tf.group(update_ops)

    def result(self):
        return tf.convert_to_tensor(self.confusion_matrix)

    def get_config(self):
        # Get ConfusionMatrix metric configuration
        config = {
            'num_classes': self.num_classes,
            'sample_weight': self.sample_weight
        }

        # Get keras base metric configuration
        base_config = super(ConfusionMatrix, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes))) for v in self.variables])


@tf.keras.utils.register_keras_serializable(package='Custom', name='FBetaScore')
class FBetaScore(tf.keras.metrics.Metric):
    """
    Returns the F-Beta Score.

    Output range is [0, 1]. Works for both multi-class and multi-label classification.

    F-1 = (1 + Beta²) * (precision * recall) / (Beta² * precision + recall)

    Hint: Behavior for the average parameter:
        None: Score for each class is returned
        micro: True positivies, false positives and false negatives are computed globally.
        macro: True positivies, false positives and false negatives are computed for each class
               and their unweighted mean is returned.
        weighted: Metrics are computed for each class and returns the mean weighted by the
                  number of true instances in each class.

    Inputs:
        y_true: Target values (class labels), <tf.Tensor>.
        y_pred: Predicted values (model output), <tf.Tensor>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.

    Arguments:
        num_classes: Number of classes of the model output (y_pred), <int>.
        average: Utilized averaging method of the F1 score. One of either None, micro, macro
                 or weighted, <None or str>.
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.

    Returns:
        f_score: List of class specific F-Beta scores (average is None) or 'global' F-Beta score, <list or float>.
    """
    def __init__(self,
                 num_classes=1,
                 average=None,
                 beta=1.0,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Initialize keras base metric instance
        super(FBetaScore, self).__init__(name=name, dtype=dtype)

        # Checks
        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError("Unknown average type. Given value '{}' is not in [None, micro, macro, weighted]".format(average))

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float, but a '{}' was given".format(type(beta)))

        if beta <= 0.0:
            raise ValueError("Beta value should be greater than zero, but a value of '{}' was given".format(beta))

        if type(top_k) not in (int,) and top_k is not None:
            raise TypeError("The value of top_k should be either a python float or None, but a '{}' was given".format(type(top_k)))

        # Initialize the F-Beta score instance
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.init_thresholds = thresholds if top_k is None else 0.0
        self.thresholds = metrics_utils.parse_init_thresholds(self.init_thresholds, default_threshold=0.5)
        self.top_k = top_k
        self.class_id = class_id
        self.sample_weight = sample_weight
        self.init_shape = () if self.average == 'micro' else (self.num_classes,)
        self.axis = None if self.average == 'micro' else 0

        # Add metric states
        self.true_positives = self.add_weight(name='true_positives', shape=self.init_shape, initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', shape=self.init_shape, initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', shape=self.init_shape, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast inputs
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Transform inputs
        [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_pred, y_true], sample_weight)

        # Get threshold properties
        if isinstance(self.thresholds, list):
            num_thresholds = len(self.thresholds)
        else:
            num_thresholds = len(list(self.thresholds))

        # Check input values and adjust shapes
        with ops.control_dependencies([
            check_ops.assert_greater_equal(
                y_pred,
                tf.cast(0.0, dtype=y_pred.dtype),
                message='predictions must be >= 0'),
            check_ops.assert_less_equal(
                y_pred,
                tf.cast(1.0, dtype=y_pred.dtype),
                message='predictions must be <= 1')]):

            if sample_weight is None:
                y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
            else:
                y_pred, y_true, sample_weight = (tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight=sample_weight))

        # Check shape compatibility
        y_pred.shape.assert_is_compatible_with(y_true.shape)

        # Check if num_classes corresponds to y_pred
        if self.average != 'micro':
            tf.debugging.assert_shapes(shapes=[(y_pred, (..., self.num_classes))], data=y_pred,
                                       summarize=10, message='num_classes must correspond to the prediction')

        # Filter top k
        if self.top_k is not None:
            y_pred = metrics_utils._filter_top_k(y_pred, self.top_k)

        # Select class id
        if self.class_id is not None:
            y_true = y_true[..., self.class_id]
            y_pred = y_pred[..., self.class_id]

        # Get prediction shape
        pred_shape = tf.shape(y_pred)
        num_predictions = pred_shape[0]

        # Set label shapes
        if y_pred.shape.ndims == 1:
            num_labels = 1
        else:
            num_labels = K.prod(pred_shape[1:], axis=0)

        # Flatten predicitons and labels
        predictions_extra_dim = tf.reshape(y_pred, [1, -1])
        labels_extra_dim = tf.reshape(tf.cast(y_true, dtype=tf.bool), [1, -1])

        # Tile the thresholds for every prediction
        thresh_pretile_shape = [num_thresholds, -1]
        thresh_tiles = [1, num_predictions * num_labels]
        data_tiles = [num_thresholds, 1]

        thresh_tiled = tf.tile(tf.reshape(tf.constant(self.thresholds, dtype=y_pred.dtype), thresh_pretile_shape), tf.stack(thresh_tiles))

        # Tile the predictions for every threshold
        preds_tiled = tf.tile(predictions_extra_dim, data_tiles)

        # Compare predictions and threshold
        pred_is_pos = tf.greater(preds_tiled, thresh_tiled)

        # Tile labels by number of thresholds
        label_is_pos = tf.tile(labels_extra_dim, data_tiles)

        # Set sample weights
        if sample_weight is not None:
            sample_weight = weights_broadcast_ops.broadcast_weights(tf.cast(sample_weight, dtype=y_pred.dtype), y_pred)
            weights_tiled = tf.tile(tf.reshape(sample_weight, thresh_tiles), data_tiles)
        else:
            weights_tiled = None

        def _weighted_assign_add(label, pred, weights, var):
            label_and_pred = tf.cast(tf.logical_and(label, pred), dtype=y_pred.dtype)

            if weights is not None:
                label_and_pred *= weights

            if self.average != 'micro':
                label_and_pred = tf.reshape(label_and_pred, shape=[-1, self.num_classes])

            return var.assign_add(tf.reduce_sum(label_and_pred, self.axis))

        # Set return value
        update_ops = []

        # Update true positives
        update_ops.append(_weighted_assign_add(label_is_pos, pred_is_pos, weights_tiled, self.true_positives))

        # Update false negatives
        pred_is_neg = tf.logical_not(pred_is_pos)
        update_ops.append(_weighted_assign_add(label_is_pos, pred_is_neg, weights_tiled, self.false_negatives))

        # Update false positives
        label_is_neg = tf.logical_not(label_is_pos)
        update_ops.append(_weighted_assign_add(label_is_neg, pred_is_pos, weights_tiled, self.false_positives))

        return tf.group(update_ops)

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

        numerator = precision * recall
        denominator = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(numerator, denominator)
        f_score = mean * (1 + tf.math.square(self.beta))

        if self.average is not None:
            f_score = tf.reduce_mean(f_score)

        return f_score

    def get_config(self):
        # Get top K confusion matrix condition count configuration
        config = {
            'num_classes': self.num_classes,
            'average': self.average,
            'beta': self.beta,
            'thresholds': self.thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id,
            'sample_weight': self.sample_weight
        }

        # Get keras base metric configuration
        base_config = super(FBetaScore, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(self.init_shape)) for v in self.variables])


@tf.keras.utils.register_keras_serializable(package='Custom', name='F1Score')
class F1Score(FBetaScore):
    """
    Returns the F-Beta Score.

    It is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class and multi-label classification.

    F1 = 2 * (precision * recall) / (precision + recall)

    Arguments:
        num_classes: Number of classes of the model output (y_pred), <int>.
        average: Utilized averaging method of the F1 score. One of either None, micro, macro
                 or weighted, <None or str>.
        thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
                    A threshold is compared with prediction values to determine the truth value
                    of predictions. One metric value is generated for each threshold value, <float or list>.
        top_k: Number of top values to be interpreted as truth values of the prediction, <int>.
               (If top_k is set/is not None, thresholds is set to zero).
        class_id: limits the prediction and labels to the class specified by this argument, <int>.
        sample_weight: Sample weight tensor whose rank is either 0, or the same rank as y_true, <tf.Tensor>.
        name: Name of the metric instance, <str>.
        dtype: Data type of the metric result, <str or tf.dtype>.
    """
    def __init__(self,
                 num_classes=1,
                 average=None,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 sample_weight=None,
                 name=None,
                 dtype=None):
        # Call F-Beta score
        super(F1Score, self).__init__(
            num_classes=num_classes,
            average=average,
            beta=1.0,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            sample_weight=sample_weight,
            name=name,
            dtype=dtype)

    def get_config(self):
        # Get F-Beta metric configuration
        base_config = super(F1Score, self).get_config()
        del base_config["beta"]
        return base_config
