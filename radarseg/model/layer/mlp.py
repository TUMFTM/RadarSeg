"""
Multi-layer perceptron (MLP).

Author: Charles R. Qi
Modified by: Felix Fent
Date: May 2020

References:
    - Qi, Charles R., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.
      [Online] Available: https://arxiv.org/abs/1612.00593, 2016.
    - Implementation: https://github.com/charlesq34/pointnet/blob/539db60eb63335ae00fe0da0c8e38c791c764d2b/utils/tf_util.py#L112
"""
# 3rd Party Libraries
from tensorflow.keras import layers


class MLP(layers.Layer):
    """
    Multi-layer perceptron (MLP).

    Inputs:
        inputs: 4D tensor (batch_size, channels, rows, cols) for channels_first, <tf.Tensor>.
                4D tensor (batch_size, rows, cols, channels) for channels_last, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        kernel_size: Specifying the height and width of the 2D convolution kernel, <list or tuple>.
        strides: Specifying the strides of the convolution along the height and width, <list or tuple>.
        padding: One of "valid" or "same", <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.
        activation: Activation function of the convolution.
        use_bias: Whether to use a bias vector within the convolution, <bool>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        bias_initializer: Initializer for the convolution bias vector.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        bias_regularizer:Regularizer function applied to the convolution bias vector.
        bn: Whether to use batch normalization, <bool>.
        momentum: Momentum for the batch normalization moving average.
        epsilon: Small float added to the batch normalization variance to avoid dividing by zero.
        center: Whether to add an offset of beta to batch normalized tensor, <bool>.
        scale: Whether to multiply the batch normalized tensor by gamma, <bool>.
        beta_initializer: Initializer for the batch normalization beta weight.
        gamma_initializer: Initializer for the batch normalization gamma weight.
        moving_mean_initializer: Initializer for the batch normalization moving mean.
        moving_variance_initializer: Initializer for the batch normalization moving variance.
        beta_regularizer: Regularizer for the batch normalization beta weight.
        gamma_regularizer: Regularizer for the batch normalization gamma weight.

    Returns:
        output: 4D tensor (batch_size, filters, new_rows, new_cols) for channels_first , <tf.Tensor>.
                4D tensor (batch_size, new_rows, new_cols, filters) for channels_last , <tf.Tensor>.
    """
    def __init__(self,
                 filters,
                 kernel_size=[1, 1],
                 strides=[1, 1],
                 padding='valid',
                 data_format='channels_last',
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 bn=False,
                 momentum=0.9,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(MLP, self).__init__(name=name, **kwargs)

        # Initialize MLP layer
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.bn = bn
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer

        if self.data_format == 'channels_last':
            self.axis = -1
        elif self.data_format == 'channels_first':
            self.axis = 1
        else:
            raise ValueError('The data_format has to be either channels_last or channels_first!')

    def build(self, input_shape):
        # Build mlp layer
        self.mlp = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                 data_format=self.data_format, activation=self.activation, use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer)

        # Build batch normalization layer
        if self.bn:
            self.batch_normalization = layers.BatchNormalization(axis=self.axis, momentum=self.momentum, epsilon=self.epsilon, center=self.center,
                                                                 scale=self.scale, beta_initializer=self.beta_initializer,
                                                                 gamma_initializer=self.gamma_initializer, moving_mean_initializer=self.moving_mean_initializer,
                                                                 moving_variance_initializer=self.moving_variance_initializer,
                                                                 beta_regularizer=self.beta_regularizer, gamma_regularizer=self.gamma_regularizer)

        # Call build function of the keras base layer
        super(MLP, self).build(input_shape)

    def call(self, inputs):
        # Apply mlp layer
        output = self.mlp(inputs)

        # Apply batch normalization layer
        if self.bn:
            output = self.batch_normalization(output)

        return output

    def get_config(self):
        # Get MLP layer configuration
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'bn': self.bn,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'moving_mean_initializer': self.moving_mean_initializer,
            'moving_variance_initializer': self.moving_variance_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer
        }

        # Get keras base layer configuration
        base_config = super(MLP, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per output tensor of the layer), <tuple>.

        Returns:
            output_shape: An shape tuple, <tuple>.
        """
        return layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                             data_format=self.data_format, activation=self.activation, use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                             kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer).compute_output_shape(input_shape)
