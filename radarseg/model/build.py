# 3rd Party Libraries
import tensorflow as tf

# Local imports
from radarseg import model
from radarseg.model.layer.timedistributed import TimeDistributed


class Builder():
    """
    Builds the model based on the given instructions.

    Note: The builder only supports the dataformat 'channels_last' due to the limitations
    of keras RNN layers and the keras TimeDistributed wrapper.

    Arguments:
        input_shape: Shape of the model input features (channels), <tuple or list>.
        batch_size: Input batch size, <int or None>.
        input_names: Names of the model input features (channel names), <list>.
        input_dtypes: Data types of the model input features (channel types), <list>.
        sparse_input: Whether the model input is a sparse tensor or dense, <bool>.
        ragged_input: Whether the model input is a regged tensor or dense, <bool>.
        input_module: Module name of the input layer, <str>.
        input_layer: Name of the input layer, <str>.
        input_config: Configuration of the input layer (name-value-pairs), <dict>.
        encoder_module: Module name of the encoder layer, <str>.
        encoder_layer: Name of the encoder layer, <str>.
        encoder_config: Configuration of the encoder layer (name-value-pairs), <dict>.
        bridge_module: Module name of the bridge layer, <str>.
        bridge_layer: Name of the bridge layer, <str>.
        bridge_config: Configuration of the bridge layer (name-value-pairs), <dict>.
        decoder_module: Module name of the decoder layer, <str>.
        decoder_layer: Name of the decoder layer, <str>.
        decoder_config: Configuration of the decoder layer (name-value-pairs), <dict>.
        output_module: Module name of the output layer, <str>.
        output_layer: Name of the output layer, <str>.
        output_config: Configuration of the output layer (name-value-pairs), <dict>.
        data_format: Data format of the model (input), <str>.
        dtype: Data type of the model (input), <str>.
    """
    def __init__(self,
                 input_shape,
                 batch_size: int = None,
                 input_names: list = None,
                 input_dtypes: list = None,
                 sparse_input: bool = False,
                 ragged_input: bool = False,
                 input_module: str = None,
                 input_layer: str = None,
                 input_config: dict = None,
                 encoder_module: str = None,
                 encoder_layer: str = None,
                 encoder_config: dict = None,
                 bridge_module: str = None,
                 bridge_layer: str = None,
                 bridge_config: dict = None,
                 decoder_module: str = None,
                 decoder_layer: str = None,
                 decoder_config: dict = None,
                 output_module: str = None,
                 output_layer: str = None,
                 output_config: dict = None,
                 data_format: str = 'channels_last',
                 dtype=None):

        # Initialize Builder
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.input_names = input_names if input_names is not None else list()
        self.input_dtypes = input_dtypes if input_dtypes is not None else list()
        self.sparse_input = sparse_input
        self.ragged_input = ragged_input
        self.input_module = input_module
        self.input_layer = input_layer
        self.input_config = input_config if input_config is not None else dict()
        self.encoder_module = encoder_module
        self.encoder_layer = encoder_layer
        self.encoder_config = encoder_config if encoder_config is not None else dict()
        self.bridge_module = bridge_module
        self.bridge_layer = bridge_layer
        self.bridge_config = bridge_config if bridge_config is not None else dict()
        self.decoder_module = decoder_module
        self.decoder_layer = decoder_layer
        self.decoder_config = decoder_config if decoder_config is not None else dict()
        self.output_module = output_module
        self.output_layer = output_layer
        self.output_config = output_config if output_config is not None else dict()
        self.data_format = data_format
        self.dtype = dtype

        # Checks
        assert isinstance(self.input_shape, (tuple, list, tf.TensorShape))
        assert not (self.sparse_input and self.ragged_input)
        assert len(self.input_names) == len(self.input_dtypes)
        assert self.data_format == 'channels_last'

        # Set builder attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.expand_axis = -1 if self.data_format == 'channels_last' else 0

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if value is not None:
            self._dtype = tf.as_dtype(value)
        else:
            self._dtype = tf.float32

    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        return self.build(*args, **kwargs)

    def _get_layer(self, layer_name: str, module_name: str = None, config: dict = None):
        """
        Returns a layer from a given module.

        This function returns a tensorflow/keras layer based on the given layer and
        module name. If a configuration dictionary is provided, the layer will be
        initialized and a layer instance will be returned. Otherwise a layer class
        object will be returned. If the layer is no child of the RNN layer (no RNN
        layer), a keras time distributed instance of the layer will be returned.

        Note: All modules have to be registered (imported) in the model __init__ file.

        Arguments:
            layer_name: Name of the requested layer, <str>.
            module_name: Name of the module the layer belongs to, <str>.
            config: Initialization configuration (attributes) of the layer, <dict>.
        """
        # Set passthrough attributes
        if module_name is None or layer_name is None:
            module_name = 'layers'
            layer_name = 'Lambda'
            config = {'function': lambda x: x}

        # Get layer
        module = getattr(model, module_name)
        layer = getattr(module, layer_name)

        if config is not None:
            if not issubclass(layer, tf.keras.layers.RNN) and not hasattr(layer, 'is_recurrent'):
                # Initialize layer
                layer = layer(**config)

                # Wrapp layer
                layer = TimeDistributed(layer)

            else:
                # Initialize layer
                layer = layer(**config)

        return layer

    def build(self):
        # Build model inputs
        inputs = {}
        inputs_reshaped = []
        for name, dtype in zip(self.input_names, self.input_dtypes):
            # Skip any inputs named label
            if name == 'label':
                continue

            # Build symbolic input tensor (placeholder)
            inputs[name] = tf.keras.Input(shape=self.input_shape, batch_size=self.batch_size, dtype=tf.as_dtype(dtype),
                                          sparse=self.sparse_input, ragged=self.ragged_input, name=name)

            # Adjust input shape and dtype
            if tf.as_dtype(dtype) == self._dtype:
                inputs_reshaped.append(tf.keras.backend.expand_dims(inputs[name], axis=self.expand_axis))
            else:
                inputs_reshaped.append(tf.cast(tf.keras.backend.expand_dims(inputs[name], axis=self.expand_axis),
                                               dtype=self._dtype, name='cast_' + name))

        # Concatenate inputs
        outputs = tf.keras.layers.Concatenate(axis=self.chan_axis, name='Concatenate_1')(inputs_reshaped)

        # Build input submodel
        input_model = self._get_layer(layer_name=self.input_layer, module_name=self.input_module, config=self.input_config)
        outputs = input_model(outputs)

        # Build encoder submodel
        encoder_model = self._get_layer(layer_name=self.encoder_layer, module_name=self.encoder_module, config=self.encoder_config)
        outputs = encoder_model(outputs)

        # Build bridge submodel
        bridge_model = self._get_layer(layer_name=self.bridge_layer, module_name=self.bridge_module, config=self.bridge_config)
        if isinstance(outputs, (list, tuple)):
            encoding = bridge_model(outputs[0])
            outputs = (encoding,) + outputs[1:]
        else:
            outputs = bridge_model(outputs)

        # Build decoder submodel
        decoder_model = self._get_layer(layer_name=self.decoder_layer, module_name=self.decoder_module, config=self.decoder_config)
        outputs = decoder_model(outputs)

        # Build output submodel
        output_model = self._get_layer(layer_name=self.output_layer, module_name=self.output_module, config=self.output_config)
        outputs = output_model(outputs)

        # Build the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model


def from_config(config: dict):
    """
    Builds a model from given configuration specifications.

    Arguments:
        config: Nested dictionary of configuration specifications, <ConfigObj or dict>.
    """
    # Check input type
    assert isinstance(config, dict)

    # Get model specifications (builder attributes)
    input_shape = config['input_shape']
    batch_size = config['batch_size']
    input_names = config['input_names']
    input_dtypes = config['input_dtypes']
    sparse_input = config['sparse_input']
    ragged_input = config['ragged_input']
    input_config = config['INPUT']
    input_module = input_config.pop('module')
    input_layer = input_config.pop('layer')
    encoder_config = config['ENCODER']
    encoder_module = encoder_config.pop('module')
    encoder_layer = encoder_config.pop('layer')
    bridge_config = config['BRIDGE']
    bridge_module = bridge_config.pop('module')
    bridge_layer = bridge_config.pop('layer')
    decoder_config = config['DECODER']
    decoder_module = decoder_config.pop('module')
    decoder_layer = decoder_config.pop('layer')
    output_config = config['OUTPUT']
    output_module = output_config.pop('module')
    output_layer = output_config.pop('layer')
    data_format = config['data_format']
    dtype = config['dtype']

    # Build model
    model = Builder(input_shape=input_shape, batch_size=batch_size, input_names=input_names,
                    input_dtypes=input_dtypes, sparse_input=sparse_input, ragged_input=ragged_input,
                    input_module=input_module, input_layer=input_layer, input_config=input_config,
                    encoder_module=encoder_module, encoder_layer=encoder_layer, encoder_config=encoder_config,
                    bridge_module=bridge_module, bridge_layer=bridge_layer, bridge_config=bridge_config,
                    decoder_module=decoder_module, decoder_layer=decoder_layer, decoder_config=decoder_config,
                    output_module=output_module, output_layer=output_layer, output_config=output_config,
                    data_format=data_format, dtype=dtype)()

    return model
