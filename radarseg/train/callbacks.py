# Standard Libraries
import os
import six

# 3rd Party Libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2

# TensorFlow callbacks
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import CSVLogger  # noqa: F401
from tensorflow.keras.callbacks import EarlyStopping  # noqa: F401
from tensorflow.keras.callbacks import LambdaCallback  # noqa: F401
from tensorflow.keras.callbacks import ModelCheckpoint  # noqa: F401
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: F401
from tensorflow.keras.callbacks import RemoteMonitor  # noqa: F401
from tensorflow.keras.callbacks import TensorBoard  # noqa: F401
from tensorflow.keras.callbacks import TerminateOnNaN  # noqa: F401

# Local imports
from radarseg.train import learning_rate_schedule  # noqa: F401
from radarseg.visu.confusion_matrix import plot_confusion_matrix
from radarseg.visu.utils import decode_figure


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler.

    Note: Extends the original tensorflow implementation by allowig a
    serialized schedule function to be passed.

    Arguments:
        schedule: a function that takes an epoch index as input
                  (integer, indexed from 0) and returns a new
                  learning rate as output (float) or a serialized
                  schedule function, <tf.keras.optimizers.schedules or dict>.
        verbose: Verbosity (0: quiet, 1: update messages), <int>.
        tensorboard: TensorBoard instance, <tf.keras.callbacks.TensorBoard>.
        log_dir: Folder path to store the log files, <str>.
    """

    def __init__(self, schedule, verbose=0, tensorboard=None, log_dir=None):
        # Initialize keras base callback
        super(LearningRateScheduler, self).__init__()

        # Initialize the LearningRateScheduler instance
        if callable(schedule):
            self.schedule = schedule
        elif isinstance(schedule, dict):
            self.schedule = tf.keras.optimizers.schedules.deserialize(schedule)
        else:
            raise TypeError('Schedule has to be either a callable schedule '
                            'function or a serialized schedule identifier, '
                            'but a schedule of type {} was given.'.format(type(schedule)))

        self.verbose = verbose
        self.tensorboard = tensorboard
        self.log_dir = log_dir
        self._writer_name = 'train'
        self._writers = {}

    def _close_writers(self):
        """
        Close all remaining open file writers owned by this callback.
        If there are no such file writers, this is a no-op.
        """
        for writer in six.itervalues(self._writers):
            writer.close()
        self._writers.clear()

    def _get_writer(self, writer_name):
        """
        Get a summary writer of the tensorboard logger.

        Arguments:
            writer_name: The name of the writer, <str>.

        Returns:
            A summary writer object, <tf.summary.SummaryWriter>.
        """
        if writer_name not in self._writers:
            path = os.path.join(self._log_write_dir, writer_name)
            writer = tf.summary.create_file_writer(path)
            self._writers[writer_name] = writer

        return self._writers[writer_name]

    def _set_default_writer(self):
        """
        Sets the tensorboard writer as default writer.
        """
        if self.tensorboard is not None:
            # Grap tensorboard writer
            self._writer_name = self.tensorboard._train_run_name
            self._log_write_dir = self.tensorboard._log_write_dir
            self._writers = self.tensorboard._writers

        elif self.log_dir is not None:
            # Get writer
            self._log_write_dir = self.log_dir
            self._get_writer(self._writer_name)

    def on_epoch_begin(self, epoch, logs=None):
        # Check if optimizer has learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Calculate new learning rate
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:
            lr = self.schedule(epoch)

        # Check new learning rate
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')

        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')

        # Set new learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))

        # Print new learning rate
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        # Set log writer
        self._set_default_writer()

        # Write learning rate to training logs
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        # Write learning rate to log
        self._log_metrics(name='learning_rate', value=self.model.optimizer.lr,
                          prefix='epoch_', step=epoch)

    def on_train_end(self, logs=None):
        self._close_writers()

    def _log_metrics(self, name, value, prefix, step):
        """
        Writes metrics out as custom scalar summaries.

        Arguments:
            name: Name of the metric, <str>.
            value: A real numeric scalar value, <convertible to a float32 Tensor>.
            prefix: The prefix to apply to the scalar summary names, <str>.
            step: The global step to use for TensorBoard, <int>.
        """
        name = prefix + name

        try:
            writer = self._get_writer(self._writer_name)
        except KeyError:
            # No logging if no writer is specified
            pass
        else:
            with writer.as_default():
                tf.summary.scalar(name, value, step=step)


class ConfusionMatrixLogger(Callback):
    """
    Logs the confusion matrix of a training as image and text file.

    Arguments:
        tensorboard: TensorBoard instance, <tf.keras.callbacks.TensorBoard>.
        log_dir: Folder path to store the log files, <str>.
        class_names: Names of the prediction classes, <list>.
    """
    def __init__(self, log_dir=None, tensorboard=None, class_names=None):
        # Initialize keras base callback
        super(ConfusionMatrixLogger, self).__init__()

        # Initialize the ConfusionMatrixLogger instance
        self.log_dir = log_dir
        self.tensorboard = tensorboard
        self.class_names = class_names
        self._writers = {}
        self._train_run_name = 'train'
        self._validation_run_name = 'validation'
        self._validation_prefix = 'val_'

        # Checks
        assert self.log_dir is not None or self.tensorboard is not None

    def _close_writers(self):
        """
        Close all remaining open file writers owned by this callback.
        If there are no such file writers, this is a no-op.
        """
        for writer in six.itervalues(self._writers):
            writer.close()
        self._writers.clear()

    def _get_writer(self, writer_name):
        """
        Get a summary writer of the tensorboard logger.

        Arguments:
            writer_name: The name of the writer, <str>.

        Returns:
            A summary writer object, <tf.summary.SummaryWriter>.
        """
        if writer_name not in self._writers:
            path = os.path.join(self._log_write_dir, writer_name)
            writer = tf.summary.create_file_writer(path)
            self._writers[writer_name] = writer

        return self._writers[writer_name]

    def _set_default_writer(self, writer_name):
        """
        Sets the default writer.

        Arguments:
            writer_name: The name of the writer, <str>.
        """
        if self.tensorboard is not None:
            # Grap tensorboard writer
            self._log_write_dir = self.tensorboard._log_write_dir
            self._writers[writer_name] = self.tensorboard._writers[writer_name]
        else:
            # Get writer
            self._log_write_dir = self.log_dir
            self._get_writer(writer_name)

    def on_train_begin(self, logs=None):
        # Check or set class names
        if self.class_names is None:
            self.class_names = list(map(str, range(self.model.output_shape[-1])))

        elif len(self.class_names) != self.model.output_shape[-1]:
            raise ValueError('The number of class names has to be equal to the number '
                             'of model output classes, but {} != {}.'.format(len(self.class_names), self.model.output[-1]))

    def on_epoch_end(self, epoch, logs=None):
        # Check whether the confusion matrix metric exists
        if 'confusion_matrix' not in logs:
            raise KeyError('The confusion matrix metric must be included in the model '
                           'to log the confusion matrix!')

        # Create train summary writer
        if self.tensorboard is not None:
            self._train_run_name = self.tensorboard._train_run_name

        self._set_default_writer(self._train_run_name)

        # Create validation summary writer
        if self.tensorboard is not None:
            self._validation_run_name = self.tensorboard._validation_run_name

        self._set_default_writer(self._validation_run_name)

        # Log confusion matrix
        self._log_confusion_matrix(logs, prefix='epoch_', step=epoch)
        self._log_raw_confusion_matrix(logs, prefix='epoch_', step=epoch)

    def on_train_end(self, logs=None):
        self._close_writers()

    def _log_raw_confusion_matrix(self, logs, prefix, step):
        """
        Logs the confusion matrix values as raw text.

        Arguments:
            logs: Training log dictionary with metric nams as keys, <dict>.
            prefix: The prefix to apply to the summary names, <str>.
            step: The global step to use for TensorBoard, <int>.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
            self._train_run_name: [],
            self._validation_run_name: [],
        }

        # Get confusion matrix values
        for (name, value) in logs.items():
            if name.endswith('confusion_matrix'):
                # Assign writer
                if name.startswith(self._validation_prefix):
                    name = name[len(self._validation_prefix):]
                    writer_name = self._validation_run_name
                else:
                    writer_name = self._train_run_name

                # Add prefix and suffix
                name = prefix + name + '_values'

                # Convert to text
                value = tf.identity(value)
                value = tf.expand_dims(tf.keras.backend.flatten(value), axis=0)
                value = tf.strings.as_string(value, precision=tf.keras.backend.epsilon(), scientific=False)

                # Add to writer list
                logs_by_writer[writer_name].append((name, value))

        # Iterate over writers (train, val)
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Skip if empts (no validation metric)
                        continue

                    # Write logs
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            tf.summary.text(name, value, step=step)

                    writer.flush()

    def _log_confusion_matrix(self, logs, prefix, step):
        """
        Logs the confusion matrix as image.

        Arguments:
            logs: Training log dictionary with metric nams as keys, <dict>.
            prefix: The prefix to apply to the summary names, <str>.
            step: The global step to use for TensorBoard, <int>.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
            self._train_run_name: [],
            self._validation_run_name: [],
        }

        # Get confusion matrix values
        for (name, value) in logs.items():
            if name.endswith('confusion_matrix'):
                # Assign writer
                if name.startswith(self._validation_prefix):
                    name = name[len(self._validation_prefix):]
                    writer_name = self._validation_run_name
                else:
                    writer_name = self._train_run_name

                # Add prefix
                name = prefix + name

                # Plot confusion matrix and decode figure
                value = tf.identity(value)
                value = plot_confusion_matrix(value, class_names=self.class_names, norm=True)
                value = decode_figure(value)

                # Add to writer list
                logs_by_writer[writer_name].append((name, value))

        # Iterate over writers (train, val)
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Skip if empts (no validation metric)
                        continue

                    # Write logs
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            summary_ops_v2.image(name, value, step=step)


def from_config(config: dict):
    """
    Returns a list of keras callbacks form a given configuration.

    Arguments:
        config: Configuration specifications as key-value-pairs, where the key
                corresponds to the callback name and the value represents the
                callback attributes (initialization values), <ConfigObj or dict>.
    """
    # Define callback list
    callbacks = []

    # Initialize tensorboard first (if specified)
    try:
        attributes = config.pop('TensorBoard')
    except KeyError:
        tensorboard = None
    else:
        tensorboard = TensorBoard(**attributes)
        callbacks.append(tensorboard)

    # Get callbacks
    for name, attributes in config.items():
        # Get callback
        callback = globals()[name]

        # Initialize callback
        try:
            attributes = {**{'tensorboard': tensorboard}, **attributes}
            callbacks.append(callback(**attributes))
        except TypeError:
            del attributes['tensorboard']
            callbacks.append(callback(**attributes))

    return callbacks
