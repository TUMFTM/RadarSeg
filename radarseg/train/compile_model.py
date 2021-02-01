# Standard Libraries
import six

# 3rd Party Libraries
import tensorflow as tf

# Local imports
from radarseg.train import losses as l  # noqa
from radarseg.train import metrics as m  # noqa


class ModelCompiler():
    """
    Compiles a kears model with respect to the given specifications.

    Note: For more details see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile.

    Inputs:
        model: Tensorflow/keras model object, <tf.keras.Model>.
        optimizer: Optimizer name or optimizer instance or name-value-pair, <str or tf.keras.optimizers or dict>.
        loss: Loss function(s) of the model output(s), <str or tf.keras.losses or list or dict>.
        metrics: Metrics to be evaluated by the model during training and testing, <str or tf.keras.metrics or list or dict>.
        loss_weights: Scalar coefficients to weight the loss contributions of different model outputs, <list or dict>.
        sample_weight_mode: Mode (structure) of the sample weights, <str or list or dict>.
        weighted_metrics: Metrics to be evaluated and weighted by sample_weight or class_weight during training and testing, <list>.
    """
    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        self.call(*args, **kwargs)

    def get_optimizer(self, optimizer, custom_objects=None):
        if isinstance(optimizer, dict):
            return tf.keras.optimizers.deserialize(optimizer, custom_objects)

        elif isinstance(optimizer, six.string_types):
            config = {'class_name': str(optimizer), 'config': {}}
            return tf.keras.optimizers.deserialize(config, custom_objects)

        elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
            return optimizer

        else:
            raise ValueError('Could not interpret optimizer identifier: {}'.format(optimizer))

    def get_loss(self, losses, custom_objects=None):
        if losses is None:
            return None

        elif isinstance(losses, list):
            for i, loss in enumerate(losses):
                losses[i] = self.get_loss(loss)

            return losses

        elif isinstance(losses, dict):
            return tf.keras.losses.deserialize(losses, custom_objects)

        elif isinstance(losses, six.string_types):
            return tf.keras.losses.deserialize(str(losses), custom_objects)

        elif callable(losses):
            return losses

        else:
            raise ValueError('Could not interpret loss function identifier: {}'.format(losses))

    def get_metrics(self, metrics, custom_objects=None):
        if metrics is None:
            return None

        elif isinstance(metrics, list):
            for i, metric in enumerate(metrics):
                metrics[i] = self.get_metrics(metric)

            return metrics

        elif isinstance(metrics, dict):
            return tf.keras.metrics.deserialize(metrics, custom_objects)

        elif isinstance(metrics, six.string_types):
            return tf.keras.metrics.deserialize(str(metrics), custom_objects)

        elif callable(metrics):
            return metrics

        else:
            raise ValueError('Could not interpret metric function identifier: {}'.format(metrics))

    def call(self, model, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, **kwargs):
        # Get optimizer
        optimizer = self.get_optimizer(optimizer)

        # Get loss function
        loss = self.get_loss(loss)

        # Get metrics
        metrics = self.get_metrics(metrics)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, **kwargs)


def from_config(model, config):
    """
    Compiles the model with respect to the given configuration specifications.

    Arguments:
        model: Model object, <tf.keras.Model>.
        config: Nested dictionary of configuration specifications, <ConfigObj or dict>.
                OPTIMIZER: dict with a name key, <dict>.
                LOSSES: nested dicts defining the loss functions, <dict>.
                METRICS: nested dicts defining the metrics, <dict>.
                loss_weights: List or dict of loss weights, <list or dict>.
                sample_weight_mode: Mode (structure) of the sample weights, <str or list or dict>.
                weighted_metrics: List of metrics to be weighted by sample_weight or class_weight, <list>.
    """
    # Get optimizer instance
    for name, attributes in config['OPTIMIZER'].items():
        identifier = {'class_name': str(name), 'config': attributes}
        optimizer = tf.keras.optimizers.deserialize(identifier)

    # Get loss instances
    losses = []
    for name, attributes in config['LOSSES'].items():
        identifier = {'class_name': str(name), 'config': attributes}
        losses.append(tf.keras.losses.deserialize(identifier))

    # Get metric instances
    metrics = []
    for name, attributes in config['METRICS'].items():
        identifier = {'class_name': str(name), 'config': attributes}
        metrics.append(tf.keras.metrics.deserialize(identifier))

    # Get additional compiler settings
    loss_weights = config['loss_weights']
    sample_weight_mode = config['sample_weight_mode']
    weighted_metrics = config['weighted_metrics']

    # Compile model
    ModelCompiler()(model=model, optimizer=optimizer, loss=losses, metrics=metrics,
                    loss_weights=loss_weights, sample_weight_mode=sample_weight_mode,
                    weighted_metrics=weighted_metrics)
