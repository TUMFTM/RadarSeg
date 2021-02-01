# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers


class LSTM1D(layers.Layer):
    """
    Long Short-Term Memory (LSTM) - Hochreiter 1997.

    Applies an LSTM layer to one dimensional data.

    Note: This is no real one dimensional LSTM layer, but rather a wrapper to apply a standard LSTM layer to
    one dimensional input data.

    Inputs:
        inputs: 4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.
                4D tensor (timesteps, batch_size, channels, ndataset) for channels_first and time_major, <tf.Tensor>.
                4D tensor (timesteps, batch_size, ndataset, channels) for channels_last and time_major, <tf.Tensor>.

    Arguments:
        units: dimensionality of the output space, <int>.
        activation: Activation function to use, <str>.
        recurrent_activation: Activation function to use for the recurrent step, <str>.
        use_bias: Whether to use a bias vector for the lstm operations, <bool>.
        reduction_mode: Reduction mode to reduce the input dimensionality by one, <str>.
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
        dropout: Fraction of the units to drop for the linear transformation of the inputs, <float>.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the recurrent state, <float>.
        implementation: Implementation mode, either 1 or 2 (see TensorFlow LSTM layer), <int>.
        return_sequences: Whether to return the last output in the output sequence, or the full sequence, <bool>.
        return_state: Whether to return the last state in addition to the output, <bool>.
        go_backwards: Whether to process the input sequence backwards and return the reversed sequence, <bool>.
        stateful: Whether to use the last state for each sample at index i in a batch as initial state for the sample of index i in the following batch, <bool>.
        time_major: Whether the time or the batch size dimension is the first dimension, <bool>.
        unroll: Whether to unroll the network or use a symbolic loop, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        output: 3D tensor (batch_size, units, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, units) for channels_last, <tf.Tensor>.
                4D tensor (batch_size, timesteps, units, ndataset) for channels_first and return_sequences, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, units) for channels_last and return_sequences, <tf.Tensor>.
                4D tensor (timesteps, batch_size, units, ndataset) for channels_first, return_sequences and time_major, <tf.Tensor>.
                4D tensor (timesteps, batch_size, ndataset, units) for channels_last, return_sequences and time_major, <tf.Tensor>.
    """
    def __init__(self,
                 units=1,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 reduction_mode='reshape',
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
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 time_major=False,
                 unroll=False,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(LSTM1D, self).__init__(name=name, **kwargs)

        # Initialize LSTM1D layer
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.reduction_mode = reduction_mode
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
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.implementation = implementation
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.time_major = time_major
        self.unroll = unroll
        self.data_format = data_format

        # Checks
        assert self.reduction_mode == 'reshape'
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.time_major:
            raise ValueError('Time major format is not supported with this wrapper.')

        # Set layer attributes
        self._ndataset_axis = -2 if self.data_format == 'channels_last' else -1

    @staticmethod
    def is_recurrent():
        return True

    def build(self, input_shape):
        # Build input reshape layer
        self.reshape_layer1 = layers.Reshape((-1, input_shape[-2] * input_shape[-1]))

        # Build LSTM layer
        self.lstm_layer = layers.LSTM(
            units=self.units * input_shape[self._ndataset_axis],
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
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            implementation=self.implementation,
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            go_backwards=self.go_backwards,
            stateful=self.stateful,
            time_major=self.time_major,
            unroll=self.unroll)

        # Build output reshape layer
        if self.return_sequences and self.data_format == 'channels_last':
            self.reshape_layer2 = layers.Reshape((-1, input_shape[self._ndataset_axis], self.units))
        elif self.return_sequences:
            self.reshape_layer2 = layers.Reshape((-1, self.units, input_shape[self._ndataset_axis]))
        elif self.data_format == 'channels_last':
            self.reshape_layer2 = layers.Reshape((-1, self.units))
        else:
            self.reshape_layer2 = layers.Reshape((-1, input_shape[self._ndataset_axis]))

        # Call keras base layer build function
        super(LSTM1D, self).build(input_shape)

    def call(self, inputs, initial_state=None, training=None):
        output = self.reshape_layer1(inputs)
        output = self.lstm_layer(output, initial_state=initial_state, training=training)
        output = self.reshape_layer2(output)

        return output

    def get_config(self):
        # Get LSTM layer configuration
        config = {
            'units': self.units,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'use_bias': self.use_bias,
            'reduction_mode': self.reduction_mode,
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
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'implementation': self.implementation,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'time_major': self.time_major,
            'unroll': self.unroll,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(LSTM1D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A shape tensors, <tf.TensorShape>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        if self.return_sequences and self.data_format == 'channels_last':
            output_shape = tf.TensorShape((input_shape[0], input_shape[1], input_shape[2], self.units))
        elif self.return_sequences and self.data_format == 'channels_first':
            output_shape = tf.TensorShape((input_shape[0], input_shape[1], self.units, input_shape[3]))
        elif not self.return_sequences and self.time_major and self.data_format == 'channels_last':
            output_shape = tf.TensorShape((input_shape[1], input_shape[2], self.units))
        elif not self.return_sequences and self.time_major and self.data_format == 'channels_first':
            output_shape = tf.TensorShape((input_shape[1], self.units, input_shape[3]))
        elif not self.return_sequences and not self.time_major and self.data_format == 'channels_last':
            output_shape = tf.TensorShape((input_shape[0], input_shape[2], self.units))
        else:
            output_shape = tf.TensorShape((input_shape[0], self.units, input_shape[3]))

        return output_shape
