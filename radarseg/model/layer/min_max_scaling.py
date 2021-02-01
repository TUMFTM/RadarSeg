# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers


class MinMaxScaling(layers.Layer):
    """
    Min-Max-Scaling layer.

    Scales the layer input channel values according to the provided minimum and maximum values. If no minimum or
    maximum values are provided, the minimum or maximium value of the current input is used for scaling. The scaling
    is given by: output = (input - min) / (max - min).

    Inputs:
        inputs: Tensor of arbitrary size and dimensions (batch_size, channels, ...) for channels_first, <tf.Tensor>.
                Tensor of arbitrary size and dimensions (batch_size, ..., channels) for channels_last, <tf.Tensor>.

    Arguments:
        minimum: Scalar (global) or per channel minimum values to be used for scaling, <tf.Tensor or convertible value>.
        maximum: Scalar (global) or per channel maximum values to be used for scaling, <tf.Tensor or convertible value>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        output: Scaled tensor with identical shape (batch_size, scaled channels, ...) for channels_first , <tf.Tensor>.
                Scaled tensor with identical shape (batch_size, ..., scaled channels) for channels_last , <tf.Tensor>.
    """
    def __init__(self,
                 minimum=None,
                 maximum=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(MinMaxScaling, self).__init__(name=name, **kwargs)

        # Initialize MinMaxScaling layer
        self.minimum = minimum
        self.maximum = maximum
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.minimum is not None and self.maximum is not None:
            assert tf.keras.backend.all(tf.math.less(tf.cast(self.minimum, dtype=self.dtype), tf.cast(self.maximum, dtype=self.dtype)))

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1

    def build(self, input_shape):
        # Convert input shape
        input_shape = tf.TensorShape(input_shape)

        # Get reduction axis
        self._chan_axis = input_shape.rank - 1 if self._chan_axis == -1 else self._chan_axis
        self._reduction_axis = list(range(input_shape.rank - 1))

        # Get permutation (required only for channels first)
        self._perm = sorted(set(range(input_shape.rank)) - set((self._chan_axis,)))
        self._perm.append(self._chan_axis)

        # Get reshape pattern
        self._reshape_pattern = [1] * (input_shape.rank - 1)
        self._reshape_pattern.append(input_shape[self._chan_axis])

        # Set minimum value(s)
        if self.minimum is not None:
            self._minimum = tf.convert_to_tensor(self.minimum, dtype=self.dtype, name='minimum')

            if self._minimum.shape != input_shape[self._chan_axis] and tf.size(self._minimum) != 1:
                raise ValueError('The number of input channles and the number of provided minimum '
                                 'values has to be equal, but {} != {}.'.format(len(self.minimum), input_shape[self._chan_axis]))

        # Set maximum value(s)
        if self.maximum is not None:
            self._maximum = tf.convert_to_tensor(self.maximum, dtype=self.dtype, name='maximum')

            if self._maximum.shape != input_shape[self._chan_axis] and tf.size(self._maximum) != 1:
                raise ValueError('The number of input channles and the number of provided maximum '
                                 'values has to be equal, but {} != {}.'.format(len(self.maximum), input_shape[self._chan_axis]))

        # Call build function of the keras base layer
        super(MinMaxScaling, self).build(input_shape)

    def call(self, inputs):
        # Get input shape
        input_shape = inputs.shape

        # Cast inputs
        if inputs.dtype != self.dtype:
            inputs = tf.cast(inputs, dtype=self.dtype)

        # Reshape inputs (if channels first)
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, perm=self._perm)

        # Get current per channel min and max values (if not provided)
        if self.minimum is not None:
            minimum = self._minimum
        else:
            minimum = tf.math.reduce_min(inputs, axis=self._reduction_axis)

        if self.maximum is not None:
            maximum = self._maximum
        else:
            maximum = tf.math.reduce_max(inputs, axis=self._reduction_axis)

        # Reshape scaling factors
        minimum = tf.reshape(tf.broadcast_to(minimum, [input_shape[self._chan_axis]]), self._reshape_pattern)
        maximum = tf.reshape(tf.broadcast_to(maximum, [input_shape[self._chan_axis]]), self._reshape_pattern)

        # Scale inputs
        outputs = tf.math.divide_no_nan(tf.math.subtract(inputs, minimum), tf.math.subtract(maximum, minimum))

        # Reshape outputs (if channels first)
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=self._perm)

        return outputs

    def get_config(self):
        # Get Decoder OutputModule configuration
        config = {
            'minimum': self.minimum,
            'maximum': self.maximum,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(MinMaxScaling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        return layers.Lambda(lambda x: x).compute_output_shape(input_shape)
