# 3rd Party Libraries
import tensorflow as tf
from tensorflow.python.keras.layers import recurrent

# Local imports
from radarseg.model.layer import kpconv
from radarseg.model import kpfcnn


class KPConvRNN(tf.keras.layers.RNN):
    """
    Base class for kernel point convolutional recurrent layers.

    Arguments:
    cell: A RNN cell instance. A RNN cell is a class that has:
          - a `call(input_at_t, states_at_t)` method, returning
            `(output_at_t, states_at_t_plus_1)`. The call method of the
            cell can also take the optional argument `constants`, see
            section "Note on passing external constants" below.
          - a `state_size` attribute. This can be a single integer
            (single state) in which case it is
            the number of channels of the recurrent state
            (which should be the same as the number of channels of the cell
            output). This can also be a list/tuple of integers
            (one size per state). In this case, the first entry
            (`state_size[0]`) should be the same as
            the size of the cell output.
    return_sequences: Whether to return the last output in the output sequence, or the full sequence, <bool>.
    return_state: Whether to return the last states in addition to the output, <bool>.
    go_backwards: Whether to process the input sequence backwards and return the reversed sequence, <bool>.
    stateful: Whether the last state for each sample at index i in a batch will be used as initial
              state for the sample of index i in the following batch, <bool>.
    unroll: Whether the network should be unrolled or a symbolic loop should be used. Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive. Unrolling is only suitable for short sequences, <bool>.
    """
    def __init__(self,
                 cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        # Initialize KPConvRNN base class
        super(KPConvRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)

    def get_initial_state(self, inputs):
        """
        Returns the initial state of the RNN layer.

        Arguments:
            inputs: The input of the call function, <tf.Tensor>.
        """
        # Get initial cell state(s)
        init_state = self.cell.get_initial_state(inputs=inputs, dtype=inputs.dtype)

        # Keras RNN expect the states in a list, even if it's a single state tensor.
        if not tf.nest.is_nested(init_state):
            init_state = [init_state]

        return list(init_state)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A shape tensor, <tf.TensorShape>.
        """
        # Convert input shape
        input_shape = tf.keras.layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Determine number of output channels
        if self.cell.use_xyz:
            nbr_out_chan = 3 + self.cell.filters
        else:
            nbr_out_chan = self.cell.filters

        # Compute output shape
        if self.return_sequences and self.cell.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], nbr_out_chan])
        elif self.return_sequences:
            output_shape = tf.TensorShape([input_shape[0], input_shape[1], nbr_out_chan, input_shape[3]])
        elif self.cell.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[2], nbr_out_chan])
        else:
            output_shape = tf.TensorShape([input_shape[0], nbr_out_chan, input_shape[3]])

        if self.return_state and self.return_sequences:
            output_shape = (output_shape, output_shape[0] + output_shape[2:], output_shape[0] + output_shape[2:])
        elif self.return_state:
            output_shape = (output_shape, output_shape, output_shape)

        return output_shape


class KPConvLSTMCell(recurrent.DropoutRNNCellMixin, tf.keras.layers.Layer):
    """
    Cell class for the KPConvLSTM layer.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        k_points: Number of kernel points (similar to the kernel size), <int>.
        nsample: Maximum number of points in each local region, <int>.
        radius: Search radius in local region, <float>.
        kp_extend: Extension factor to define the area of influence of the kernel points, <float>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        knn: Whether to use kNN instead of radius search, <bool>.
        kp_influence: Association function for the influence of the kernel points (one of either 'constant', 'linear' or 'gaussian'), <str>.
        aggregation_mode: Method to aggregate the activation of the point convolution kernel (one of either 'closest' or 'sum'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        activation: Activation function of the states (cell state and hidden state), <str>.
        recurrent_activation: Activation function of the gates, <str>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        dropout: Fraction of the units to drop for the linear transformation of the inputs, <float>.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the recurrent state, <float>.
        upsample_mode: Method used to associate the features of the last cell state with the points of the current input
            (one of either 'nearest' or 'threenn'), <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        h: Output of the cell at time t, <tf.Tensor>.
        [h, c]: States (cell state and hidden state) of the cell at time t, <List>.
    """
    def __init__(self,
                 filters=1,
                 k_points=1,
                 nsample=1,
                 radius=1.0,
                 kp_extend=1.0,
                 fixed='center',
                 knn=False,
                 kp_influence='linear',
                 aggregation_mode='sum',
                 use_xyz=True,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 upsample_mode='nearest',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(KPConvLSTMCell, self).__init__(name=name, **kwargs)

        # Initialize KPConvLSTMCell
        self.filters = filters
        self.k_points = k_points
        self.nsample = nsample
        self.radius = radius
        self.kp_extend = kp_extend
        self.fixed = fixed
        self.knn = knn
        self.kp_influence = kp_influence
        self.aggregation_mode = aggregation_mode
        self.use_xyz = use_xyz
        self.activation = activation if activation is not None else 'linear'
        self.recurrent_activation = recurrent_activation if recurrent_activation is not None else 'linear'
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.upsample_mode = upsample_mode
        self.data_format = data_format
        self.state_size = (3 + self.filters, 3 + self.filters) if self.use_xyz else (self.filters, self.filters)

        # Checks
        assert self.filters > 0
        assert self.k_points > 0
        assert self.radius > 0
        assert self.kp_extend > 0
        assert self.fixed in set(('none', 'center', 'verticals'))
        assert self.kp_influence in set(('constant', 'linear', 'gaussian'))
        assert self.aggregation_mode in set(('closest', 'sum'))
        assert self.dropout >= 0.0 and self.dropout < 1.0
        assert self.recurrent_dropout >= 0.0 and self.recurrent_dropout < 1.0
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.nsample is not None:
            assert self.nsample > 0

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else -2
        self._ndataset_axis = -2 if self.data_format == 'channels_last' else -1

    def _get_kpfp(self):
        return kpconv.KPFP(filters=self.filters, k_points=self.k_points, nsample=self.nsample, radius=self.radius,
                           activation='linear', kp_extend=self.kp_extend, fixed=self.fixed, kernel_initializer=self.kernel_initializer,
                           kernel_regularizer=self.kernel_regularizer, knn=self.knn, kp_influence=self.kp_influence,
                           aggregation_mode=self.aggregation_mode, use_xyz=self.use_xyz, data_format=self.data_format)

    def build(self, input_shape):
        # Check input shapes
        if input_shape[self._chan_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        elif input_shape[self._chan_axis] < 3:
            raise ValueError('The input tensor must have at least three channels (xyz), '
                             'but an input with {} channels is given.'.format(input_shape[self._chan_axis]))

        # Build internal layer (required to calculate the initial state)
        self.internal_layer = self._get_kpfp()

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda_layer = tf.keras.layers.Lambda(lambda x: x[:, :, 0:3])
        elif self.data_format == 'channels_first':
            self.lambda_layer = tf.keras.layers.Lambda(lambda x: x[:, 0:3, :])

        # Build internal gate layers
        self.i_layer = self._get_kpfp()
        self.f_layer = self._get_kpfp()
        self.o_layer = self._get_kpfp()

        # Build internal cell state layers
        self.c_layer = self._get_kpfp()
        self.c_hat_layer = kpfcnn.UpsampleBlock(upsample_mode=self.upsample_mode, use_xyz=True, data_format=self.data_format)

        # Build activation layers
        self.activation_layer = tf.keras.layers.Activation(self.activation)
        self.recurrent_activation_layer = tf.keras.layers.Activation(self.recurrent_activation)

    def get_initial_state(self, inputs, dtype=None):
        """
        Returns the initial states (cell and hidden state) of the cell.

        Arguments:
            inputs: Inputs of the call function, <tf.Tensor>.
            dtype: Optional dtype of the initial state, <tf.dtype>.
        """
        # Set internal state
        internal_state = tf.keras.backend.zeros_like(inputs, dtype=dtype)
        internal_state = tf.keras.backend.sum(internal_state, axis=1)

        # Compute initial states
        initial_state = []
        for _ in self.state_size:
            initial_state.append(self.internal_layer([internal_state, internal_state]))

        return initial_state

    def call(self, inputs, states, training=None):
        # Get previous hidden and cell state (t - 1)
        h_tm1 = states[0]
        c_tm1 = states[1]

        # Get input point coordinates
        xyz = self.lambda_layer(inputs)

        # Get dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)

        # Get dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        # Apply input dropout mask
        if 0 < self.dropout < 1.0:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        # Apply recurrent dropout mask
        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        # Get gate values
        i = self.i_layer([h_tm1_i, inputs_i])
        f = self.f_layer([h_tm1_f, inputs_f])
        o = self.o_layer([h_tm1_o, inputs_o])

        # Get intermediate cell states
        c = self.c_layer([h_tm1_c, inputs_c])
        c_hat_tm1, _ = self.c_hat_layer([c_tm1, xyz])

        # Apply recurrent (gate) activation function
        i = self.recurrent_activation_layer(i)
        f = self.recurrent_activation_layer(f)
        o = self.recurrent_activation_layer(o)

        # Get new hidden and cell state (t)
        c = f * c_hat_tm1 + i * self.activation_layer(c)
        h = o * self.activation_layer(c)

        return h, [h, c]

    def get_config(self):
        # Get KPConvLSTMCell configuration
        config = {
            'filters': self.filters,
            'k_points': self.k_points,
            'nsample': self.nsample,
            'radius': self.radius,
            'kp_extend': self.kp_extend,
            'fixed': self.fixed,
            'knn': self.knn,
            'kp_influence': self.kp_influence,
            'aggregation_mode': self.aggregation_mode,
            'use_xyz': self.use_xyz,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'upsample_mode': self.upsample_mode,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(KPConvLSTMCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class KPConvLSTM(KPConvRNN):
    """
    Kernel point convolution long short-term memory layer.

    It is similar to an LSTM layer, but the input transformations
    and recurrent transformations are both replaced by a kernel point
    convolution operation and an additional step is introduced to
    associate the last cell state to the current input.

    This LSTM layer is based on the work of Hehe Fan and Hugues Thomas.

    References:
        - Fan, Hehe, PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing.
        [Online] Available: https://arxiv.org/abs/1910.08287, 2019.
        - Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
        [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Inputs:
        inputs: 4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        k_points: Number of kernel points (similar to the kernel size), <int>.
        nsample: Maximum number of points in each local region, <int>.
        radius: Search radius in local region, <float>.
        kp_extend: Extension factor to define the area of influence of the kernel points, <float>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        knn: Whether to use kNN instead of radius search, <bool>.
        kp_influence: Association function for the influence of the kernel points (one of either 'constant', 'linear' or 'gaussian'), <str>.
        aggregation_mode: Method to aggregate the activation of the point convolution kernel (one of either 'closest' or 'sum'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        activation: Activation function of the states (cell state and hidden state), <str>.
        recurrent_activation: Activation function of the gates, <str>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        dropout: Fraction of the units to drop for the linear transformation of the inputs, <float>.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the recurrent state, <float>.
        upsample_mode: Method used to associate the features of the last cell state with the points of the current input
            (one of either 'nearest' or 'threenn'), <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.
        return_sequences: Whether to return the last output in the output sequence, or the full sequence, <bool>.
        return_state: Whether to return the last states in addition to the output, <bool>.
        go_backwards: Whether to process the input sequence backwards and return the reversed sequence, <bool>.
        stateful: Whether the last state for each sample at index i in a batch will be used as initial
                  state for the sample of index i in the following batch, <bool>.

    Returns:
        - If return_sequences is true
          4D tensor (batch_size, timesteps, 3 + channels, ndataset) for channels_first and use_xyz, <tf.Tensor>.
          4D tensor (batch_size, timesteps, ndataset, 3 + channels) for channels_last and use_xyz, <tf.Tensor>.
          4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
          4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.
        - If return_sequences is false
          3D tensor (batch_size, 3 + channels, ndataset) for channels_first and use_xyz, <tf.Tensor>.
          3D tensor (batch_size, ndataset, 3 + channels) for channels_last and use_xyz, <tf.Tensor>.
          3D tensor (batch_size, channels, ndataset) for channels_first, <tf.Tensor>.
          3D tensor (batch_size, ndataset, channels) for channels_last, <tf.Tensor>.
        - If return_state is true the output will be a tuple of three tensors containing the output as defined above
          as well as the cell state and hidden state of the last time step.
    """
    def __init__(self,
                 filters=1,
                 k_points=2,
                 nsample=1,
                 radius=1.0,
                 kp_extend=1.0,
                 fixed='center',
                 knn=False,
                 kp_influence='linear',
                 aggregation_mode='sum',
                 use_xyz=True,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 upsample_mode='nearest',
                 data_format='channels_last',
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 **kwargs):
        # Initialize KPConvLSTMCell
        cell = KPConvLSTMCell(filters=filters,
                              k_points=k_points,
                              nsample=nsample,
                              radius=radius,
                              kp_extend=kp_extend,
                              fixed=fixed,
                              knn=knn,
                              kp_influence=kp_influence,
                              aggregation_mode=aggregation_mode,
                              use_xyz=use_xyz,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              upsample_mode=upsample_mode,
                              data_format=data_format,
                              dtype=kwargs.get('dtype'))

        # Check if layer is stateful (currently no supported)
        if stateful:
            # TODO: Add support for stateful KPConvLSTM
            raise NotImplementedError('Stateful KPConvLSTM is currently not supported.')

        # Initialize KPConvLSTM base class
        super(KPConvLSTM, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         return_state=return_state,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(KPConvLSTM, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    def get_config(self):
        # Get keras base layer configuration
        base_config = super(KPConvLSTM, self).get_config()
        del base_config['cell']

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
