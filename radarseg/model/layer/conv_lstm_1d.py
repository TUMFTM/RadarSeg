# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers


class ConvLSTM1D(layers.Layer):
    """
    Convolutional Long Short-Term Memory (LSTM) - Shi 2015.

    Applies a convolutional LSTM layer to one dimensional data.

    Note: This is no real one dimensional convolutional LSTM layer, but rather a wrapper to apply a standard
    convolutional LSTM layer to one dimensional input data.

    Inputs:
        inputs: 4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.
                4D tensor (timesteps, batch_size, channels, ndataset) for channels_first and time_major, <tf.Tensor>.
                4D tensor (timesteps, batch_size, ndataset, channels) for channels_last and time_major, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        kernel_size: Specifying the height and width of the 2D convolution kernel, <list or tuple>.
        strides: Specifying the strides of the convolution along the height and width, <list or tuple>.
        padding: One of "valid" or "same", <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.
        dilation_rate: Specifying the dilation rate to use for dilated convolution, <int or list>.
        activation: Activation function to use, <str>.
        recurrent_activation: Activation function to use for the recurrent step, <str>.
        use_bias: Whether to use a bias vector within the convolution, <bool>.
        kernel_initializer: Initializer for the kernel weights matrix.
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Whether to add 1 to the bias of the forget gate at initialization, <bool>.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix.
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer.
        kernel_constraint: Constraint function applied to the kernel weights matrix.
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        return_sequences: Whether to return the last output in the output sequence, or the full sequence, <bool>.
        go_backwards: Whether to process the input sequence backwards and return the reversed sequence, <bool>.
        stateful: Whether to use the last state for each sample at index i in a batch as initial state for the sample of index i in the following batch, <bool>.
        dropout: Fraction of the units to drop for the linear transformation of the inputs, <float>.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the recurrent state, <float>.

    Returns:
        output: 3D tensor (batch_size, units, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, units) for channels_last, <tf.Tensor>.
                4D tensor (batch_size, timesteps, units, ndataset) for channels_first and return_sequences, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, units) for channels_last and return_sequences, <tf.Tensor>.
                4D tensor (timesteps, batch_size, units, ndataset) for channels_first, return_sequences and time_major, <tf.Tensor>.
                4D tensor (timesteps, batch_size, ndataset, units) for channels_last, return_sequences and time_major, <tf.Tensor>.
    """
    def __init__(self,
                 filters=1,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(ConvLSTM1D, self).__init__(name=name, **kwargs)

        # Initialize ConvLSTM1D layer
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        # Checks
        assert data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self._exp_axis = 3 if data_format == 'channels_last' else 4
        self._squ_axis = self._exp_axis if return_sequences else self._exp_axis - 1

    @staticmethod
    def is_recurrent():
        return True

    def build(self, input_shape):
        # Build the convolutional LSTM layer
        self.conv_lstm_layer = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            recurrent_initializer=self.recurrent_initializer,
            bias_initializer=self.bias_initializer,
            unit_forget_bias=self.unit_forget_bias,
            kernel_regularizer=self.kernel_regularizer,
            recurrent_regularizer=self.recurrent_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            recurrent_constraint=self.recurrent_constraint,
            bias_constraint=self.bias_constraint,
            return_sequences=self.return_sequences,
            go_backwards=self.go_backwards,
            stateful=self.stateful,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)

        # Call keras base layer build function
        super(ConvLSTM1D, self).build(input_shape)

    def call(self, inputs, initial_state=None, constants=None, training=None):
        output = tf.keras.backend.expand_dims(inputs, axis=self._exp_axis)
        output = self.conv_lstm_layer(output, initial_state=initial_state, constants=constants, training=training)
        output = tf.keras.backend.squeeze(output, axis=self._squ_axis)

        return output

    def get_config(self):
        # Get ConvLSTM1D layer configuration
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'recurrent_initializer': self.recurrent_initializer,
            'bias_initializer': self.bias_initializer,
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': self.kernel_regularizer,
            'recurrent_regularizer': self.recurrent_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'recurrent_constraint': self.recurrent_constraint,
            'bias_constraint': self.bias_constraint,
            'return_sequences': self.return_sequences,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout
        }

        # Get keras base layer configuration
        base_config = super(ConvLSTM1D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A shape tensors, <tf.TensorShape>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Compute output shape
        output_shape = tf.keras.backend.expand_dims(input_shape, axis=self._exp_axis)
        output_shape = self.conv_lstm_layer.compute_output_shape(output_shape)
        output_shape = tf.keras.backend.squeeze(output_shape, axis=self._squ_axis)

        return output_shape
