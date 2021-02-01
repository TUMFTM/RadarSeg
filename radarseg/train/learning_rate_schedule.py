# 3rd Party Libraries
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='ExponentialDecayClipping')
class ExponentialDecayClipping(tf.optimizers.schedules.LearningRateSchedule):
    """
    Applies exponential decay to the learning rate which is clipped at a minimum value.

    This schedule applies an exponential decay function to an optimizer step,
    given a provided initial learning rate. This decay, however, is clipped
    at a defined minimum value.

    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

        initial_learning_rate * decay_rate ^ (step / decay_steps)

    and then clipped at the minimum value.

    Reference: tf.keras.optimizers.schedules.ExponentialDecay

    Note: If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

    Hint: You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.

    Arguments:
        initial_learning_rate: The initial learning rate value, <float or tf.Tensor>.
        min_learning_rate: The minimun learning rate value (cut off at this value), <float or tf.Tensor>.
        decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
        decay_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The decay rate.
        staircase: Boolean.  If `True` decay the learning rate at discrete
            intervals
        name: String.  Optional name of the operation.  Defaults to
            'ExponentialDecay'.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
    """
    def __init__(self,
                 initial_learning_rate,
                 min_learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 name=None):
        # Initialize base learning rate scheduler
        super(ExponentialDecayClipping, self).__init__()

        # Initialize ExponentialDecayClipping learning rate scheduler
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecayClipping") as name:
            # Convert scheduler attributes
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
                                                         name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            min_learning_rate = tf.cast(self.min_learning_rate, dtype)
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            # Get exponent
            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.math.floor(p)

            # Get new learning rate
            learning_rate = tf.math.multiply(initial_learning_rate, tf.math.pow(decay_rate, p))
            return tf.maximum(learning_rate, min_learning_rate, name=name)

    def get_config(self):
        # Get ExponentialDecayClipping configuration
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name
        }

        return config
