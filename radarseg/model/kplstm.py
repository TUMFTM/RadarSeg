# Standard Libraries
import warnings

# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers

# Local imports
from radarseg.model.layer import kpconv
from radarseg.model.layer import kpconv_lstm
from radarseg.model.layer import timedistributed


class SAModule(layers.Layer):
    """
    Set abstraction (SA) module of the KPLSTM architecture.

    Inputs:
        inputs: 4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        kp_conv: List of dicts to define the attributes of the input KPConv layers (length of list represents the number of Conv layers), <List[Dict, ... , Dict]>.
        kp_conv_lstm: Dict to define the attributes of the KPConvLSTM layer, <Dict>.
        kp_conv2: List of dicts to define the attributes of the output KPConv layers (length of list represents the number of Conv layers), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        - If return_sequences is true
          4D tensor (batch_size, timesteps, 3 + filters, npoint) for channels_first and use_xyz, <tf.Tensor>.
          4D tensor (batch_size, timesteps, npoint, 3 + filters) for channels_last and use_xyz, <tf.Tensor>.
          4D tensor (batch_size, timesteps, filters, npoint) for channels_first, <tf.Tensor>.
          4D tensor (batch_size, timesteps, npoint, filters) for channels_last, <tf.Tensor>.
        - If return_sequences is false
          3D tensor (batch_size, 3 + filters, npoint) for channels_first and use_xyz, <tf.Tensor>.
          3D tensor (batch_size, npoint, 3 + filters) for channels_last and use_xyz, <tf.Tensor>.
          3D tensor (batch_size, filters, npoint) for channels_first, <tf.Tensor>.
          3D tensor (batch_size, npoint, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 kp_conv=None,
                 kp_conv_lstm=None,
                 kp_conv2=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(SAModule, self).__init__(name=name, **kwargs)

        # Initialize Encoder layer
        self.kp_conv = kp_conv if kp_conv is not None else []
        self.kp_conv_lstm = kp_conv_lstm if kp_conv_lstm is not None else {}
        self.kp_conv2 = kp_conv2 if kp_conv2 is not None else []
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to all child layers
        _ = self.kp_conv_lstm.update({'data_format': self.data_format})
        _ = [kp_conv_setting.update({'data_format': self.data_format}) for kp_conv_setting in self.kp_conv]
        _ = [kp_conv_setting.update({'data_format': self.data_format}) for kp_conv_setting in self.kp_conv2]

        # Set layer attributes
        self.kp_conv_list = []
        self.kp_conv2_list = []
        self._ndataset_axis = 2 if self.data_format == 'channels_last' else 3
        self._chan_axis = 3 if self.data_format == 'channels_last' else 2

    @staticmethod
    def is_recurrent():
        return True

    def build(self, input_shape):
        # Print warning if use_xyz is set to false for any layer
        if not self.kp_conv_lstm.get('use_xyz', True):
            warnings.warn("If the 'use_xyz' attribute is set to False for the KPConvLSTM layer "
                          "within the SAModule, the xyz information will be lost for all subsequent "
                          "KPConv layers.")
        elif any(not kp_conv_layer.get('use_xyz', True) for kp_conv_layer in self.kp_conv_list):
            warnings.warn("If the 'use_xyz' attribute is set to False for any KPConv layer "
                          "within the SAModule, the xyz information will be lost for all subsequent "
                          "layers.")
        elif any(not kp_conv_layer.get('use_xyz', True) for kp_conv_layer in self.kp_conv2_list):
            warnings.warn("If the 'use_xyz' attribute is set to False for any KPConv layer "
                          "within the SAModule, the xyz information will be lost for all subsequent "
                          "KPConv layers.")

        # Build input KPConv layers
        for kp_conv_setting in self.kp_conv:
            self.kp_conv_list.append(kpconv.KPConv(**kp_conv_setting))

        # Build KPConvLSTM layer
        self.kp_conv_lstm_layer = kpconv_lstm.KPConvLSTM(**self.kp_conv_lstm)

        # Build output KPConv layer
        for kp_conv_setting in self.kp_conv2:
            self.kp_conv2_list.append(kpconv.KPConv(**kp_conv_setting))

        # Call keras base layer build function
        super(SAModule, self).build(input_shape)

    def call(self, inputs, states=None, training=None):
        # Set initial output
        output = inputs

        # Encode input data
        for kp_conv_layer in self.kp_conv_list:
            output = timedistributed.TimeDistributed(kp_conv_layer)(output, training=training)

        # Apply RNN layer
        output = self.kp_conv_lstm_layer(output, initial_state=states, training=training)

        # Encode output data
        for kp_conv_layer in self.kp_conv2_list:
            if self.kp_conv_lstm.get('return_sequences', False):
                output = timedistributed.TimeDistributed(kp_conv_layer)(output, training=training)
            else:
                output = kp_conv_layer(output, training=training)

        return output

    def get_config(self):
        # Get SAModule layer configuration
        config = {
            'kp_conv': self.kp_conv,
            'kp_conv_lstm': self.kp_konv_lstm,
            'kp_conv2': self.kp_conv2,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(SAModule, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: An shape tuple, <tuple>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Compute number of output points
        last_layer = next((layer for layer in reversed(self.kp_conv_list + self.kp_conv2_list) if layer.npoint is not None), None)

        if last_layer is not None:
            nbr_out_points = last_layer.compute_output_shape(input_shape[0] + input_shape[2:])[self._ndataset_axis - 1]
        else:
            nbr_out_points = input_shape[self._ndataset_axis]

        # Compute number of output channels
        if self.kp_conv2:
            nbr_out_chan = self.kp_conv2_list[-1].compute_output_shape(input_shape[0] + input_shape[2:])[self._chan_axis - 1]
        elif self.kp_conv_lstm_layer.return_sequences:
            nbr_out_chan = self.kp_conv_lstm_layer.compute_output_shape(input_shape)[self._chan_axis]
        else:
            nbr_out_chan = self.kp_conv_lstm_layer.compute_output_shape(input_shape)[self._chan_axis - 1]

        # Compute output shape
        if self.kp_conv_lstm_layer.return_sequences and self.data_format == 'channels_first':
            output_shape = tf.TensorShape([input_shape[0], input_shape[1], nbr_out_chan, nbr_out_points])
        elif self.kp_conv_lstm_layer.return_sequences:
            output_shape = tf.TensorShape([input_shape[0], input_shape[1], nbr_out_points, nbr_out_chan])
        elif self.data_format == 'channels_first':
            output_shape = tf.TensorShape([input_shape[0], nbr_out_chan, nbr_out_points])
        else:
            output_shape = tf.TensorShape([input_shape[0], nbr_out_points, nbr_out_chan])

        return output_shape


class FPModule(layers.Layer):
    """
    Feature propagation (FP) module of the KPLSTM architecture.

    Inputs:
        inputs: List of two 3D tensors, whereas the second one defines the output shape.
        [(batch_size, ndataset1, channels1), (batch_size, ndataset2, channels2)], if channels_last, <List[tf.Tensor, tf.Tensor]>.
        [(batch_size, channels1, ndataset1), (batch_size, channels2, ndataset2)], if channels_first, <List[tf.Tensor, tf.Tensor]>.

    Arguments:
        kpfp: Dict to define the attributes of the KPFP layer, <Dict>.
        kp_conv: List of dicts to define the attributes of the KPConv layers (length of list represents the number of Conv layers), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, 3 + filters, ndataset2) for channels_first and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, ndataset2, 3 + filters) for channels_last and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, filters, ndataset2) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, ndataset2, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 kpfp=None,
                 kp_conv=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(FPModule, self).__init__(name=name, **kwargs)

        # Initialize FPModule layer
        self.kpfp = kpfp if kpfp is not None else {}
        self.kp_conv = kp_conv if kp_conv is not None else []
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to all kpconv layers
        _ = self.kpfp.update({'data_format': self.data_format})
        _ = [kp_conv_setting.update({'data_format': self.data_format}) for kp_conv_setting in self.kp_conv]

        # Set npoint to None for all kpconv layers (no downsampling)
        _ = [kp_conv_setting.update({'npoint': None}) for kp_conv_setting in self.kp_conv]

        # Set layer attributes
        self.kp_conv_list = []

    def build(self, input_shape):
        # Check input_shape
        if not isinstance(input_shape, (list, tuple)):
            raise TypeError('The FPModule input has to be a list or tuple of inputs, '
                            'but an input of type {} was given.'.format(type(input_shape)))
        elif len(input_shape) != 2:
            raise ValueError('The FPModule requires an input list/tuple of exactly 2 inputs, '
                             'but {} where given.'.format(len(input_shape)))

        # Print warning if use_xyz is set to false for any layer
        if not self.kpfp.get('use_xyz', True):
            warnings.warn("If the 'use_xyz' attribute is set to False for any KPFP layer "
                          "within the FPModule, the xyz information will be lost for all subsequent "
                          "KPConv layers.")
        elif any(not kp_conv_layer.use_xyz for kp_conv_layer in self.kp_conv_list):
            warnings.warn("If the 'use_xyz' attribute is set to False for any KPConv layer "
                          "within the FPModule, the xyz information will be lost for all subsequent "
                          "KPConv layers.")

        # Build KPFP layer
        self.kpfp_layer = kpconv.KPFP(**self.kpfp)

        # Build KPConv layers
        for kp_conv_setting in self.kp_conv:
            self.kp_conv_list.append(kpconv.KPConv(**kp_conv_setting))

        # Call build function of the keras base layer
        super(FPModule, self).build(input_shape)

    def call(self, inputs):
        # Set input
        decoding = self.kpfp_layer(inputs)

        # Point feature propagation
        for kp_conv_layer in self.kp_conv_list:
            decoding = kp_conv_layer(decoding)

        return decoding

    def get_config(self):
        # Get FPModule layer configuration
        config = {
            'kpfp': self.kp_conv,
            'kp_conv': self.kp_conv,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(FPModule, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per input tensor of the layer), <tuple>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, input_shape)

        # Get decoding shape
        decoding_shape = self.kpfp_layer.compute_output_shape(input_shape)

        # Compute channel shape
        if self.kp_conv:
            decoding_shape = self.kp_conv_list[-1].compute_output_shape(decoding_shape)

        return decoding_shape


class Encoder(layers.Layer):
    """
    Encoder of the KPLSTM architecture.

    Returns an encoded representation of the given point cloud and its internal states.

    Inputs:
        inputs: 4D tensor (batch_size, timesteps, channels, ndataset) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, timesteps, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        sa: List of dicts to define the attributes of the SAModule layers (length of list represents the number of SA modules), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        encoding: Encoded point cloud (batch_size, npoint, sa[-1]), if channels_last, <tf.Tensor>.
                  Encoded point cloud (batch_size, sa[-1], npoint), if channels_first, <tf.Tensor>.
        state: Input of the last encoding layer/SA module (batch_size, ndataset or npoint, sa[-2]), if channels_last, <tf.Tensor>.
               Input of the last encoding layer/SA module (batch_size, sa[-2], ndataset or npoint), if channels_first, <tf.Tensor>.
        states: List of internal states (inputs and outputs) of the encoding layers, except for the last one, <List[List[tf.Tensor, tf.Tensor], ..., List[tf.Tensor, tf.Tensor]]>.
                The first entry (element) of the list of states referes to the second last SA module, the last element to the first SA module.
    """
    def __init__(self,
                 sa=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(Encoder, self).__init__(name=name, **kwargs)

        # Initialize Encoder layer
        self.sa = sa if sa is not None else list()
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass encoder data_format setting to all SAModule layers
        _ = [sa_setting.update({'data_format': self.data_format}) for sa_setting in self.sa]

        # Force the perceiving of the temporal information (set return_sequences to True)
        _ = [sa_setting['kp_conv_lstm'].update({'return_sequences': True}) for sa_setting in self.sa]

        # Set layer attributes
        self.chan_axis = 3 if self.data_format == 'channels_last' else 2
        self.sa_list = []
        self.num_sa_modules = len(self.sa)

    @staticmethod
    def is_recurrent():
        return True

    def build(self, input_shape):
        # Build SA layers
        for sa_setting in self.sa:
            self.sa_list.append(SAModule(**sa_setting))

        # Call keras base layer build function
        super(Encoder, self).build(input_shape)

    def call(self, inputs, states=None, training=None):
        # Create list of internal states
        states = []

        # Set initial state
        encoding = inputs
        state = inputs

        # Encode point cloud
        for i, sa_layer in enumerate(self.sa_list):
            encoding = sa_layer(state)

            # Set internal states
            if i < self.num_sa_modules - 1:
                # Fill states list from the right
                states.append([encoding, state])

                # Output as new SA module input
                state = encoding

        # Revert states (last state first)
        states.reverse()

        if not states:
            return encoding, state

        return encoding, state, states

    def get_config(self):
        # Get Encoder layer configuration
        config = {
            'sa': self.sa,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(Encoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Get output shape
        if self.sa:
            states_shape = []
            state_shape = input_shape

            for i, sa_layer in enumerate(self.sa_list):
                encoding_shape = sa_layer.compute_output_shape(state_shape)

                if i < self.num_sa_modules - 1:
                    states_shape.append([encoding_shape, state_shape])
                    state_shape = encoding_shape

            states_shape.reverse()

            if not states_shape:
                return (encoding_shape, state_shape)

            return (encoding_shape, state_shape, states_shape)

        else:
            return (input_shape, input_shape)


class Decoder(layers.Layer):
    """
    Decoder of the KPLSTM architecture.

    Returns a decoded representation of the given point cloud and its corresponding origin.

    Inputs: List of two to three inputs (states is optional for no or one FP layer).
        encoding: Encoded point cloud (batch_size, ndataset1, 3 + channels1), if channels_last, <tf.Tensor>.
                  Encoded point cloud (batch_size, 3 + channels1, ndataset1), if channels_first, <tf.Tensor>.
        state: Target point cloud / supporting point coordinates and features (batch_size, ndataset2, 3 + channels2), if channels_last, <tf.Tensor>.
               Target point cloud / supporting point coordinates and features (batch_size, 3 + channels2, ndataset2), if channels_first, <tf.Tensor>.
        states: Internal states (inputs/state and outputs/encoding) of the Encoder - must match the number of FP layers - 1, <List(List(tf.Tensor, tf.Tensor), ... ,List(tf.Tensor, tf.Tensor))>

    Arguments:
        fp: List of dicts to define the attributes of the FPModule layers (length of list represents the number of FP modules), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        decoding: Decoded point cloud (batch_size, ndataset2, fp[-1]), if channels_last, <tf.Tensor>.
                  Decoded point cloud (batch_size, fp[-1], ndataset2), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 fp=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(Decoder, self).__init__(name=name, **kwargs)

        # Initialize Decoder layer
        self.fp = fp if fp is not None else []
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass encoder data_format setting to all SAModule layers
        _ = [fp_setting.update({'data_format': self.data_format}) for fp_setting in self.fp]

        # Set layer attributes
        self.chan_axis = 3 if self.data_format == 'channels_last' else 2
        self.fp_list = []

    def build(self, input_shape):
        # Check input_shape
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) < 2 or len(input_shape) > 3:
                raise ValueError('The Decoder layer requires an input list/tuple of 2 or 3 inputs, '
                                 'but {} where given.'.format(len(input_shape)))
        else:
            raise TypeError('The Decoder layer input has to be a list or tuple of inputs, '
                            'but an input of type {} was given.'.format(type(input_shape)))

        # Check input and initialization compatibility
        if len(self.fp) > 1 and len(input_shape) != 3:
            raise ValueError('A Decoder configuration with more than one FP module requires the '
                             'provision of the states input. {} != 3'.format(len(input_shape)))

        # Build FP layers
        for fp_setting in self.fp:
            self.fp_list.append(FPModule(**fp_setting))

    def call(self, inputs):
        # Get inputs
        decoding = inputs[0]
        state = inputs[1]

        # Get additional encoder states
        if len(inputs) == 3:
            states = inputs[2]
            assert len(states) == max(len(self.fp) - 1, 0)
            states.append([None, None])
        else:
            states = [[None, None]]

        # Encode point cloud
        for i, fp_layer in enumerate(self.fp_list):
            decoding = fp_layer([decoding, state])
            state = states[i][1]

        return decoding

    def get_config(self):
        # Get Decoder layer configuration
        config = {
            'fp': self.fp,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(Decoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per input tensor of the layer), <tuple>.

        Returns:
            output_shape: A shape tensors, <tf.TensorShape>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, input_shape)

        # Split input shapes
        decoding_shape = input_shape[0]
        state_shape = input_shape[1]

        if len(input_shape) == 3:
            states_shape = input_shape[2]
            assert len(states_shape) == max(len(self.fp) - 1, 0)
            states_shape.append([None, None])
        else:
            states_shape = [[None, None]]

        states_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, states_shape)

        # Get output shape
        if self.fp:
            for i, fp_layer in enumerate(self.fp_list):
                decoding_shape = fp_layer.compute_output_shape([decoding_shape, state_shape])
                state_shape = states_shape[i][1]

            return decoding_shape

        else:
            return decoding_shape
