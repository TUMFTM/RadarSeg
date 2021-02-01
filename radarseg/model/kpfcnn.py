"""
KPConv KP-FCNN Model Architecture.

Author: Hugues, Thomas
Modified by: Felix Fent
Date: July 2020

References:
    - Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
      [Online] Available: https://arxiv.org/abs/1904.08889, 2019.
"""
# Standard Libraries
import warnings

# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers

# Local imports
from radarseg.model.tf_ops.interpolation.tf_interpolate import three_interpolate
from radarseg.model.tf_ops.interpolation.tf_interpolate import three_nn
from radarseg.model.tf_ops.sampling.tf_sampling import farthest_point_sample
from radarseg.model.tf_ops.grouping.tf_grouping import knn_point
from radarseg.model.layer.kpconv import KPConv
from radarseg.model.layer.mlp import MLP


class KPConvBlock(layers.Layer):
    """
    Kernel point convolution (KPConv) block.

    Combines a kernel point convolution (KPConv) layer with a batch normalization and dropout layer.

    Inputs:
        inputs: 3D tensor (batch_size, channels, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        k_points: Number of kernel points (similar to the kernel size), <int>.
        npoint: Number of points sampled in farthest point sampling (number of output points), <int>.
        activation: Activation function of the kernel point convolution, <str or tf.keras.activations>.
        alpha: Slope coefficient of the leaky rectified linear unit (if specified), <float>.
        nsample: Maximum number of points in each local region, <int>.
        radius: Search radius in local region, <float>.
        kp_extend: Extension factor to define the area of influence of the kernel points, <float>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        knn: Whether to use kNN instead of radius search, <bool>.
        kp_influence: Association function for the influence of the kernel points (one of either 'constant', 'linear' or 'gaussian'), <str>.
        aggregation_mode: Method to aggregate the activation of the point convolution kernel (one of either 'closest' or 'sum'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        seed: Random seed for the kernel and dropout initialization, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.
        bn: Whether to use batch normalization, <bool>.
        momentum: Momentum for the batch normalization moving average, <float>.
        epsilon: Small float added to the batch normalization variance to avoid dividing by zero, <float>.
        center: Whether to add an offset of beta to batch normalized tensor, <bool>.
        scale: Whether to multiply the batch normalized tensor by gamma, <bool>.
        beta_initializer: Initializer for the batch normalization beta weight.
        gamma_initializer: Initializer for the batch normalization gamma weight.
        moving_mean_initializer: Initializer for the batch normalization moving mean.
        moving_variance_initializer: Initializer for the batch normalization moving variance.
        beta_regularizer: Regularizer for the batch normalization beta weight.
        gamma_regularizer: Regularizer for the batch normalization gamma weight.
        dropout_rate: Fraction of the input units to drop, <float>.

    Returns:
        outputs: 3D tensor (batch_size, filters, npoint) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, npoint, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 filters=1,
                 k_points=2,
                 npoint=None,
                 activation='lrelu',
                 alpha=0.3,
                 nsample=1,
                 radius=1.0,
                 kp_extend=1.0,
                 fixed='center',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 knn=False,
                 kp_influence='linear',
                 aggregation_mode='sum',
                 use_xyz=True,
                 seed=42,
                 data_format='channels_last',
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
                 dropout_rate=0.0,
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(KPConvBlock, self).__init__(name=name, **kwargs)

        # Initialize KPConvBlock layer
        self.filters = filters
        self.k_points = k_points
        self.npoint = npoint
        self.activation = activation
        self.alpha = alpha
        self.nsample = nsample
        self.radius = radius
        self.kp_extend = kp_extend
        self.fixed = fixed
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.knn = knn
        self.kp_influence = kp_influence
        self.aggregation_mode = aggregation_mode
        self.use_xyz = use_xyz
        self.seed = seed
        self.data_format = data_format

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

        self.dropout_rate = dropout_rate

        # Checks
        assert self.filters > 0
        assert self.k_points > 0
        assert self.alpha > 0
        assert self.nsample > 0
        assert self.radius > 0
        assert self.kp_extend > 0
        assert self.fixed in set(('none', 'center', 'verticals'))
        assert self.kp_influence in set(('constant', 'linear', 'gaussian'))
        assert self.aggregation_mode in set(('closest', 'sum'))
        assert self.data_format in set(('channels_last', 'channels_first'))
        assert self.momentum > 0
        assert self.epsilon > 0
        assert self.dropout_rate >= 0 and self.dropout_rate <= 1

        if self.npoint is not None:
            assert self.npoint > 0

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1

    def build(self, input_shape):
        # Build KPConv layer
        self.kp_conv = KPConv(filters=self.filters, k_points=self.k_points, npoint=self.npoint, nsample=self.nsample, radius=self.radius,
                              activation=self.activation, alpha=self.alpha, kp_extend=self.kp_extend, fixed=self.fixed,
                              kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, knn=self.knn,
                              kp_influence=self.kp_influence, aggregation_mode=self.aggregation_mode, use_xyz=self.use_xyz,
                              seed=self.seed, data_format=self.data_format)

        # Build batch normalization layer
        if self.bn:
            self.batch_normalization = layers.BatchNormalization(axis=self._chan_axis, momentum=self.momentum, epsilon=self.epsilon,
                                                                 center=self.center, scale=self.scale, beta_initializer=self.beta_initializer,
                                                                 gamma_initializer=self.gamma_initializer,
                                                                 moving_mean_initializer=self.moving_mean_initializer,
                                                                 moving_variance_initializer=self.moving_variance_initializer,
                                                                 beta_regularizer=self.beta_regularizer, gamma_regularizer=self.gamma_regularizer)

        # Build dropout layer
        if self.dropout_rate:
            self.dropout_layer = layers.Dropout(rate=self.dropout_rate, seed=self.seed)

        # Call keras base layer build function
        super(KPConvBlock, self).build(input_shape)

    def call(self, inputs):
        # Execute kernel point convolution
        output = self.kp_conv(inputs)

        # Normalize batch (if specified)
        if self.bn:
            output = self.batch_normalization(output)

        # Apply dropout
        if self.dropout_rate:
            output = self.dropout_layer(output)

        return output

    def get_config(self):
        # Get KPConvBlock layer configuration
        config = {
            'filters': self.filters,
            'k_points': self.k_points,
            'npoint': self.npoint,
            'activation': self.activation,
            'alpha': self.alpha,
            'nsample': self.nsample,
            'radius': self.radius,
            'kp_extend': self.kp_extend,
            'fixed': self.fixed,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'knn': self.knn,
            'kp_influence': self.kp_influence,
            'aggregation_mode': self.aggregation_mode,
            'use_xyz': self.use_xyz,
            'seed': self.seed,
            'data_format': self.data_format,
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
            'gamma_regularizer': self.gamma_regularizer,
            'dropout_rate': self.dropout_rate
        }

        # Get keras base layer configuration
        base_config = super(KPConvBlock, self).get_config()

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

        return self.kp_conv.compute_output_shape(input_shape)


class UnaryBlock(layers.Layer):
    """
    Simple unary convolution block.

    Combines a simple unary convolution (1 x 1 convolution) layer with a batch normalization and dropout layer.

    Inputs:
        inputs: 3D tensor (batch_size, channels, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        activation: Activation function, <str or tf.keras.activations>.
        alpha: Slope coefficient of the leaky rectified linear unit (if specified), <float>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.
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
        dropout_rate: Fraction of the input units to drop, <float>.
        seed: Random seed for the dropout initialization, <int>.

    Returns:
        outputs: 3D tensor (batch_size, 3 + filters, ndataset) for channels_first and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, ndataset, 3 + filters) for channels_last and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, filters, ndataset) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, ndataset, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 filters=1,
                 activation='lrelu',
                 alpha=0.3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 use_xyz=True,
                 data_format='channels_last',
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
                 dropout_rate=0.0,
                 seed=42,
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(UnaryBlock, self).__init__(name=name, **kwargs)

        # Initialize UnaryBlock layer
        self.filters = filters
        self.activation = activation
        self.alpha = alpha
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_xyz = use_xyz
        self.data_format = data_format

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

        self.dropout_rate = dropout_rate
        self.seed = seed

        # Checks
        assert self.filters > 0
        assert self.alpha > 0
        assert self.momentum > 0
        assert self.epsilon > 0
        assert self.data_format in set(('channels_last', 'channels_first'))
        assert self.dropout_rate >= 0 and self.dropout_rate <= 1

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1
        self._exp_axis = 2 if self.data_format == 'channels_last' else 3

    def build(self, input_shape):
        # Check input shape
        if input_shape[self._chan_axis] < 3:
            raise ValueError('The input data must contain at least three channels (xyz), '
                             'but a number of {} channels are given.'.format(input_shape[self._chan_axis]))

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        else:
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build MLP layer (without activation and bias)
        self.mlp = MLP(filters=self.filters, data_format=self.data_format, activation=None, use_bias=False, kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer, bn=self.bn, momentum=self.momentum, epsilon=self.epsilon,
                       center=self.center, scale=self.scale, beta_initializer=self.beta_initializer, gamma_initializer=self.gamma_initializer,
                       moving_mean_initializer=self.moving_mean_initializer, moving_variance_initializer=self.moving_variance_initializer,
                       beta_regularizer=self.beta_regularizer, gamma_regularizer=self.gamma_regularizer)

        # Build activation layer
        if self.activation == 'lrelu':
            self.activation_layer = layers.LeakyReLU(alpha=self.alpha)
        else:
            self.activation_layer = layers.Activation(activation=self.activation)

        # Build concatenation layer
        self.concat_layer = layers.Concatenate(axis=self._chan_axis)

        # Build dropout layer
        if self.dropout_rate:
            self.dropout_layer = layers.Dropout(rate=self.dropout_rate, seed=self.seed)

        # Call keras base layer build function
        super(UnaryBlock, self).build(input_shape)

    def call(self, inputs):
        # Split input to get point coordinates [ndataset, 3] and point features [ndataset, dim]
        xyz = self.lambda1(inputs)
        features = self.lambda2(inputs)

        # Execute per point convolution [ndataset, filters]
        outputs = tf.keras.backend.expand_dims(features, axis=self._exp_axis)
        outputs = self.mlp(outputs)
        outputs = tf.keras.backend.squeeze(outputs, axis=self._exp_axis)
        outputs = self.activation_layer(outputs)

        # Add point coordinates to output features if use_xyz [ndataset, 3 + filters]
        if self.use_xyz:
            outputs = self.concat_layer([xyz, outputs])

        # Apply dropout
        if self.dropout_rate:
            outputs = self.dropout_layer(outputs)

        return outputs

    def get_config(self):
        # Get UnaryBlock layer configuration
        config = {
            'filters': self.filters,
            'activation': self.activation,
            'alpha': self.alpha,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'use_xyz': self.use_xyz,
            'data_format': self.data_format,
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
            'gamma_regularizer': self.gamma_regularizer,
            'dropout_rate': self.dropout_rate,
            'seed': self.seed
        }

        # Get keras base layer configuration
        base_config = super(UnaryBlock, self).get_config()

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

        # Split input shape in point coordinate shape and feature shape
        coordinates_shape = self.lambda1.compute_output_shape(input_shape)
        features_shape = self.lambda2.compute_output_shape(input_shape)

        # Compute output shape
        output_shape = self.mlp.compute_output_shape(features_shape)
        if self.use_xyz:
            output_shape = self.concat_layer.compute_output_shape([coordinates_shape, output_shape])

        return output_shape


class UpsampleBlock(layers.Layer):
    """
    Upsample layer.

    Applies the specified upsampling method to associate the features of the first input tensor to the
    points of the second input tensor and concatenate both features afterwards.

    Inputs:
        inputs: List of two 3D tensors (output and input of the corresponding downsampling step), with
                [(batch_size, ndataset1, channels1), (batch_size, ndataset2, channels2)], if channels_last, <List[tf.Tensor, tf.Tensor]> or
                [(batch_size, channels1, ndataset1), (batch_size, channels2, ndataset2)], if channels_first, <List[tf.Tensor, tf.Tensor]>.


    Arguments:
        upsample_mode: Method used to associate the features of the first input tensor with the points of the second input tensor
                       (one of either 'nearest', 'threenn' or 'kpconv'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, channels1 + channels2 - 3, ndataset) for channels_first and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, ndataset, channels1 + channels2 - 3) for channels_last and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, channels1 + channels2 - 6, ndataset) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, ndataset, channels1 + channels2 - 6) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 upsample_mode='nearest',
                 use_xyz=True,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(UpsampleBlock, self).__init__(name=name, **kwargs)

        # Initialize UpsampleBlock layer
        self.upsample_mode = upsample_mode
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert self.upsample_mode in set(('nearest', 'threenn', 'kpconv'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        # TODO: Add KPDeconv ('deconvolution') layer
        if self.upsample_mode == 'kpconv':
            raise NotImplementedError

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1
        self._ndataset_axis = 1 if self.data_format == 'channels_last' else -1

    def build(self, input_shape):
        # Check input
        if len(input_shape) != 2:
            raise ValueError('A UpsampleBlock layer should be called on a list of two '
                             'inputs, but {} inputs are given.'.format(len(input_shape)))
        elif input_shape[0][0] != input_shape[1][0]:
            raise ValueError('The batch size of both inputs has to be equal, but '
                             '{} != {}.'.format(input_shape[0][0], input_shape[1][0]))
        elif input_shape[0][self._chan_axis] < 3:
            raise ValueError('Both inputs must have at least three channels (xyz), but '
                             'the first input has only {} channels.'.format(input_shape[0][self._chan_axis]))
        elif input_shape[1][self._chan_axis] < 3:
            raise ValueError('Both inputs must have at least three channels (xyz), but '
                             'the second input has only {} channels.'.format(input_shape[0][self._chan_axis]))

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build concatenation layer
        self.concat_layer = layers.Concatenate(axis=2)

        # Call keras base layer build function
        super(UpsampleBlock, self).build(input_shape)

    def call(self, inputs):
        # Split input in point coordinates and point features
        xyz1 = self.lambda1(inputs[0])
        features1 = self.lambda2(inputs[0])
        xyz2 = self.lambda1(inputs[1])
        features2 = self.lambda2(inputs[1])

        # Adjust input format
        if self.data_format == 'channels_first':
            xyz1 = tf.transpose(xyz1, [0, 2, 1])
            features1 = tf.transpose(features1, [0, 2, 1])
            xyz2 = tf.transpose(xyz2, [0, 2, 1])
            features2 = tf.transpose(features2, [0, 2, 1])

        if self.upsample_mode == 'nearest':
            # Get neighbor points
            _, idx = knn_point(1, xyz1, xyz2)

            nearest_features = tf.gather(features1, idx, axis=1, batch_dims=1)

            nearest_features = tf.keras.backend.squeeze(nearest_features, axis=2)

            # Concatenate the point features
            new_features = self.concat_layer([nearest_features, features2])

        elif self.upsample_mode == 'threenn':
            # Get the distance to the three nearest neighbors
            dist, idx = three_nn(xyz2, xyz1)
            dist = tf.maximum(dist, tf.keras.backend.epsilon())

            # Calculate the interpolation weights
            norm = tf.reduce_sum((1.0 / dist), axis=-1, keepdims=True)
            norm = tf.tile(norm, [1, 1, 3])
            weight = (1.0 / dist) / norm

            # Interpolate the point features
            interpolated_features = three_interpolate(features1, idx, weight)

            # Concatenate the point features
            new_features = self.concat_layer([interpolated_features, features2])

        # Add point coordinates to output features if use_xyz
        if self.use_xyz:
            new_features = self.concat_layer([xyz2, new_features])

        # Adjust output format
        if self.data_format == 'channels_first':
            xyz2 = tf.transpose(xyz2, [0, 2, 1])
            new_features = tf.transpose(new_features, [0, 2, 1])

        return new_features, xyz2

    def get_config(self):
        # Get UpsampleBlock layer configuration
        config = {
            'upsample_mode': self.upsample_mode,
            'use_xyz': self.use_xyz,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(UpsampleBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per input tensor of the layer), <tuple>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Convert input shape
        input_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, input_shape)

        # Determine number of output channles
        if self.use_xyz:
            nbr_out_chan = input_shape[0][self._chan_axis] + input_shape[1][self._chan_axis] - 3
        else:
            nbr_out_chan = input_shape[0][self._chan_axis] + input_shape[1][self._chan_axis] - 6

        # Set output shape
        if self.data_format == 'channels_last':
            new_features_shape = tf.TensorShape([input_shape[1][0], input_shape[1][self._ndataset_axis], nbr_out_chan])
            xyz2_shape = tf.TensorShape([input_shape[1][0], input_shape[1][self._ndataset_axis], 3])
        else:
            new_features_shape = tf.TensorShape([input_shape[1][0], nbr_out_chan, input_shape[1][self._ndataset_axis]])
            xyz2_shape = tf.TensorShape([input_shape[1][0], 3, input_shape[1][self._ndataset_axis]])

        return new_features_shape, xyz2_shape


class ResnetBlock(layers.Layer):
    """
    Block performing a resnet convolution (1x1conv > KPConv > 1x1conv + shortcut).

    Note: The grid based batch subsampling of the skiplink data, which is used by the original
    KP-FCNN architecture (KPConv) is replaced by the iterative farthest point sampling method of the
    PointNet++ architecture.

    References:
        - Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
          [Online] Available: https://arxiv.org/abs/1904.08889, 2019.
        - Qi, Charles R, PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
          [Online] Available: https://arxiv.org/abs/1706.02413

    Inputs:
        inputs: 3D tensor (batch_size, channels, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        npoint: Number of points sampled in farthest point sampling (number of output points), <int>.
        activation: Activation function of the kernel point convolution, <str or tf.keras.activations>.
        alpha: Slope coefficient of the leaky rectified linear unit (if specified), <float>.
        unary: Dictionary to define the attributes of the first unary block layer, <dict>.
        kpconv: Dictionary to define the attributes of the first kpconv block layer, <dict>.
        unary2: Dictionary to define the attributes of the second unary block layer, <dict>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, 3 + unary2[filters], npoint) for channels_first and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, npoint, 3 + unary2[filters]) for channels_last and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, unary2[filters], npoint) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, npoint, unary2[filters]) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 npoint=None,
                 activation='lrelu',
                 alpha=0.3,
                 unary=None,
                 kpconv=None,
                 unary2=None,
                 use_xyz=True,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(ResnetBlock, self).__init__(name=name, **kwargs)

        # Initialize ResnetBlock layer
        self.npoint = npoint
        self.activation = activation
        self.alpha = alpha
        self.unary = unary if unary is not None else {}
        self.kpconv = kpconv if kpconv is not None else {}
        self.unary2 = unary2 if unary2 is not None else {}
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert alpha > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.npoint is not None:
            assert self.npoint > 0

        # Pass module data_format setting to child layers
        self.unary.update({'data_format': self.data_format})
        self.kpconv.update({'data_format': self.data_format})
        self.unary2.update({'data_format': self.data_format})

        # Set preceding child layers to use_xyz to pass through point coordinate information
        self.unary.update({'use_xyz': True})
        self.kpconv.update({'use_xyz': True})
        self.unary2.update({'use_xyz': False})

        # Set activation of the last unary block to None
        self.unary2.update({'activation': None})

        # Pass module npoint setting (stride) to the kpconv block
        self.kpconv.update({'npoint': self.npoint})

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1
        self._ndataset_axis = 1 if self.data_format == 'channels_last' else -1
        self._exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.adjust_features = False

    def build(self, input_shape):
        # Check input shape compatibility
        if input_shape[self._chan_axis] < 3:
            raise ValueError('The input tensor must have at least three channels (xyz), '
                             'but an input with {} channels is given.'.format(input_shape[self._chan_axis]))

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build unary convolution layer
        self.unary_layer_1 = UnaryBlock(**self.unary)

        # Build KPConv layer
        self.kpconv_layer = KPConvBlock(**self.kpconv)

        # Build second unary convolution layer
        self.unary_layer_2 = UnaryBlock(**self.unary2)

        # Build feature adjustment layer
        if self.unary_layer_2.filters != input_shape[self._chan_axis] - 3:
            self.adjust_features = True
            self.conv = layers.Conv2D(filters=self.unary_layer_2.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                      data_format=self.data_format, activation=None, use_bias=False)

        # Build addition layer
        self.addition_layer = layers.Add()

        # Build activation layer
        if self.activation == 'lrelu':
            self.activation_layer = layers.LeakyReLU(alpha=self.alpha)
        else:
            self.activation_layer = layers.Activation(activation=self.activation)

        # Build concatenation layer
        self.concat_layer = layers.Concatenate(axis=self._chan_axis)

        # Call keras base layer build function
        super(ResnetBlock, self).build(input_shape)

    def call(self, inputs):
        # Get point coordinates
        xyz = self.lambda1(inputs)
        features = self.lambda2(inputs)

        # Process input data
        outputs = self.unary_layer_1(inputs)
        outputs = self.kpconv_layer(outputs)
        outputs = self.unary_layer_2(outputs)

        # Create skiplink data
        if self.npoint is not None:
            # Adjust point coordinates shape
            if self.data_format == 'channels_first':
                xyz = tf.transpose(xyz, [0, 2, 1])

            # Subsample points
            idx = farthest_point_sample(self.npoint, xyz)
            xyz = tf.gather(xyz, idx, axis=1, batch_dims=1)
            skiplink = tf.gather(features, idx, axis=self._ndataset_axis, batch_dims=1)

            # Reset point coordinates shape
            if self.data_format == 'channels_first':
                xyz = tf.transpose(xyz, [0, 2, 1])

        else:
            skiplink = features

        # Adjust skiplink features
        if self.adjust_features:
            skiplink = tf.keras.backend.expand_dims(skiplink, axis=self._exp_axis)
            skiplink = self.conv(skiplink)
            skiplink = tf.keras.backend.squeeze(skiplink, axis=self._exp_axis)

        # Add skiplink to the processed data
        outputs = self.addition_layer([outputs, skiplink])

        # Apply activation function
        outputs = self.activation_layer(outputs)

        # Add point coordinates to output features if use_xyz
        if self.use_xyz:
            outputs = self.concat_layer([xyz, outputs])

        return outputs

    def get_config(self):
        # Get ResnetBlock layer configuration
        config = {
            'npoint': self.npoint,
            'activation': self.activation,
            'alpha': self.alpha,
            'unary': self.unary,
            'kpconv': self.kpconv,
            'unary2': self.unary2,
            'use_xyz': self.use_xyz,
            'data_format': self.data_format,
        }

        # Get keras base layer configuration
        base_config = super(ResnetBlock, self).get_config()

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

        # Get point coordinate shape
        if self.npoint is not None and self.data_format == 'channels_last':
            coordinates_shape = tf.TensorShape([input_shape[0], self.npoint, 3])

        elif self.npoint is not None and self.data_format == 'channels_first':
            coordinates_shape = tf.TensorShape([input_shape[0], 3, self.npoint])

        else:
            coordinates_shape = self.lambda1.compute_output_shape(input_shape)

        # Compute output shape
        output_shape = self.unary_layer_1.compute_output_shape(input_shape)
        output_shape = self.kpconv_layer.compute_output_shape(output_shape)
        output_shape = self.unary_layer_2.compute_output_shape(output_shape)

        if self.use_xyz:
            output_shape = self.concat_layer.compute_output_shape([coordinates_shape, output_shape])

        return output_shape


class FPBlock(layers.Layer):
    """
    Feature propogation (FP) block.

    The feature propagation block combines an upsampling layer with an unary layer and
    builds the basis of the KPConv decoder.

    Inputs:
        inputs: List of two 3D tensors (output and input of the corresponding encoding step), with
                [(batch_size, ndataset1, channels1), (batch_size, ndataset2, channels2)], if channels_last, <List[tf.Tensor, tf.Tensor]> or
                [(batch_size, channels1, ndataset1), (batch_size, channels2, ndataset2)], if channels_first, <List[tf.Tensor, tf.Tensor]>.

    Arguments:
        upsample: Dictionary to define the attributes of the upsample block layer, <dict>.
        unary: Dictionary to define the attributes of the unary block layer, <dict>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, 3 + unary[filters], ndataset2) for channels_first and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, ndataset2, 3 + unary[filters]) for channels_last and use_xyz, <tf.Tensor>.
                 3D tensor (batch_size, unary[filters], ndataset2) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, ndataset2, unary[filters]) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 upsample=None,
                 unary=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(FPBlock, self).__init__(name=name, **kwargs)

        # Initialize FPBlock layer
        self.upsample = upsample if upsample is not None else {}
        self.unary = unary if unary is not None else {}
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to child layers
        self.upsample.update({'data_format': self.data_format})
        self.unary.update({'data_format': self.data_format})

        # Set preceding child layers to use_xyz to pass through point coordinate information
        self.upsample.update({'use_xyz': True})

    def build(self, input_shape):
        # Check number of inputs
        if len(input_shape) != 2:
            raise ValueError('A FPBlock layer should be called on a list of two '
                             'inputs, but {} inputs are given.'.format(len(input_shape)))

        # Build upsample layer
        self.upsample_layer = UpsampleBlock(**self.upsample)

        # Build unary layer
        self.unary_layer = UnaryBlock(**self.unary)

        # Call keras base layer build function
        super(FPBlock, self).build(input_shape)

    def call(self, inputs):
        # Upsample points
        new_features, xyz2 = self.upsample_layer(inputs)

        # Merge features
        new_features = self.unary_layer(new_features)

        return new_features, xyz2

    def get_config(self):
        # Get FPBlock layer configuration
        config = {
            'upsample': self.upsample,
            'unary': self.unary,
            'data_format': self.data_format,
        }

        # Get keras base layer configuration
        base_config = super(FPBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per input tensor of the layer), <tuple>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Convert input shapes
        input_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, input_shape)

        # Compute output shapes
        new_features_shape, xyz2_shape = self.upsample_layer.compute_output_shape(input_shape)
        new_features_shape = self.unary_layer.compute_output_shape(new_features_shape)

        return new_features_shape, xyz2_shape


class Encoder(layers.Layer):
    """
    Encoder layer of the KP-FCNN model.

    Returns an encoded representation of the given input tensor and its internal states.

    Inputs:
        inputs: 3D tensor (batch_size, ndataset, channels), if channels_last, <tf.Tensor>.
                3D tensor (batch_size, channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        resnet: List of dicts to define the attributes of the ResnetBlock layers
                (length of list represents the number of resnet blocks), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        encoding: Encoded 3D tensor (batch_size, resnet[-1]['npoint'], resnet[-1]['unary']['filters']), if channels_last, <tf.Tensor>.
                  Encoded 3D tensor  (batch_size, resnet[-1]['unary']['filters'], resnet[-1]['npoint']), if channels_first, <tf.Tensor>.
        state: Input of the last encoding layer/resnet block (batch_size, resnet[-2]['npoint'], resnet[-2]['unary']['filters']), if channels_last, <tf.Tensor>.
               Input of the last encoding layer/resnet block (batch_size, resnet[-2]['unary']['filters'], resnet[-2]['npoint']), if channels_first, <tf.Tensor>.
        states: List of internal states (inputs and outputs) of the encoding layers, except for the last one, <List[List[tf.Tensor, tf.Tensor], ..., List[tf.Tensor, tf.Tensor]]>.
                The first entry (element) of the list of states referes to the second last resnet bloxk, the last element to the first resnet block.
    """
    def __init__(self,
                 resnet=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(Encoder, self).__init__(name=name, **kwargs)

        # Initialize Encoder layer
        self.resnet = resnet if resnet is not None else []
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to child layers
        _ = [resnet_setting.update({'data_format': self.data_format}) for resnet_setting in self.resnet]

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.resnet_list = []
        self.num_resnet_layers = len(self.resnet)

    def build(self, input_shape):
        # Build ResnetBolck layers
        for resnet_setting in self.resnet:
            self.resnet_list.append(ResnetBlock(**resnet_setting))

        # Print warning if use_xyz is set to false for any layer
        if any(not resnet_layer.use_xyz for resnet_layer in self.resnet_list):
            warnings.warn("If the 'use_xyz' attribute is set to False for any ResnetBlock layer "
                          "within the Encoder, the xyz information will be lost for all subsequent "
                          "ResnetBlocks.")

        # Call keras base layer build function
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        # Create list of internal states
        states = []

        # Set initial state and encoding
        state = inputs
        encoding = inputs

        # Encode point cloud
        for i, resnet_layer in enumerate(self.resnet_list):
            encoding = resnet_layer(state)

            # Set internal states
            if i < self.num_resnet_layers - 1:
                # Fill states list from the right
                states.append([encoding, state])

                # Set output as new input for the next encoder layer
                state = encoding

        # Revert states (last state first)
        states.reverse()

        if not states:
            return encoding, state

        return encoding, state, states

    def get_config(self):
        # Get Encoder layer configuration
        config = {
            'resnet': self.resnet,
            'data_format': self.data_format,
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

        # Compute output shape
        if self.resnet:
            states_shape = []
            state_shape = input_shape

            for i, resnet_layer in enumerate(self.resnet_list):
                encoding_shape = resnet_layer.compute_output_shape(state_shape)

                if i < self.num_resnet_layers - 1:
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
    Decoder layer of the KP-FCNN model.

    Returns an decoded representation of the given input and state tensors.

    Inputs: List of two to three inputs (the states input is not required for no more than one FPBlock layers).
        encoding: Encoded 3D tensor (batch_size, ndataset1, channels1), if channels_last, <tf.Tensor>.
                  Encoded 3D tensor (batch_size, channels1, ndataset1), if channels_first, <tf.Tensor>.
        state: Target 3D tensor / supporting points (batch_size, ndataset2, channels2), if channels_last, <tf.Tensor>.
               Target 3D tensor / supporting points (batch_size, channels2, ndataset2), if channels_first, <tf.Tensor>.
        states: Internal states (inputs/state and outputs/encoding) of the Encoder - must match the number of FPBlock layers - 1, <List(List(tf.Tensor, tf.Tensor), ... ,List(tf.Tensor, tf.Tensor))>

    Arguments:
        fp: List of dicts to define the attributes of the FPBlock layers (length of list represents the number of FPBlocks), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        decoding: Decoded 3D tensor  (batch_size, fp[-1] or ndataset2, fp[-1]['unary'] or channels2), if channels_last, <tf.Tensor>.
                  Decoded 3D tensor  (batch_size, fp[-1]['unary'] or channels2, fp[-1] or ndataset2), if channels_first, <tf.Tensor>.
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

        # Pass module data_format setting to child layers
        _ = [fp_setting.update({'data_format': self.data_format}) for fp_setting in self.fp]

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.fp_list = []

    def build(self, input_shape):
        # Check number of inputs
        if not isinstance(input_shape, (list, tuple)):
            raise TypeError('The Decoder layer should be called on a list of two or three'
                            'inputs, but an input of type {} was given.'.format(type(input_shape)))
        elif len(input_shape) < 2:
            raise ValueError('The Decoder layer should be called on a list of two or three'
                             'inputs, but {} inputs are given.'.format(len(input_shape)))
        elif len(input_shape) == 2 and len(self.fp) > 1:
            raise ValueError('If the Decoder layer contains more than one FPBlock the "states" '
                             'input has to be provided, but only 2 inputs were given.')
        elif len(input_shape) > 3:
            raise ValueError('The Decoder layer should be called on a list of two or three'
                             'inputs, but {} inputs are given.'.format(len(input_shape)))
        elif len(input_shape) == 2:
            pass
        elif len(input_shape[2]) != max(len(self.fp) - 1, 0):
            raise ValueError('The "states" input (the third input) of the Decoder layer has to contain'
                             'as many elements as FPBlocks - 1 are specified, '
                             'but {} elements != {} FPBlocks - 1.'.format(len(input_shape[2]), len(self.fp)))

        # Build Decoder layers
        for fp_setting in self.fp:
            self.fp_list.append(FPBlock(**fp_setting))

        # Call keras base layer build function
        super(Decoder, self).build(input_shape)

    def call(self, inputs):
        # Get inputs
        decoding = inputs[0]
        state = inputs[1]

        # Get additional encoder states
        if len(inputs) == 3:
            states = inputs[2]
            states.append([None, None])
        else:
            states = [[None, None]]

        # Decode point cloud
        for i, fp_layer in enumerate(self.fp_list):
            decoding, _ = fp_layer([decoding, state])
            state = states[i][1]

        return decoding

    def get_config(self):
        # Get Decoder layer configuration
        config = {
            'fp': self.fp,
            'data_format': self.data_format,
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
            states_shape.append([None, None])
        else:
            states_shape = [[None, None]]

        states_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, states_shape)

        # Compute output shape
        if self.fp:
            for i, fp_layer in enumerate(self.fp_list):
                decoding_shape, _ = fp_layer.compute_output_shape([decoding_shape, state_shape])
                state_shape = states_shape[i][1]

            return decoding_shape

        else:
            return decoding_shape


class OutputBlock(layers.Layer):
    """
    Output block.

    Combines point convolution layers with dropout layers, which are followed by an unary layer that
    maps the values to the desired number of output classes.

    Inputs:
        inputs: 3D tensor (batch_size, ndataset, channels), if channels_last, <tf.Tensor>.
                3D tensor (batch_size, channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        num_classes: Number of different output classes/labels, <int>.
        kpconv: List of dicts to define the attributes of the KPConvBlock layers (length of the list represents the number of KPConvBlock layers), <List[Dict, ... , Dict]>.
        dropout: List of dicts to define the attributes of the dropout layers (length of list represents the number of dropout layers), <List[Dict, ... , Dict]>.
        out_activation: Activation function of the output layer.
        out_use_bias: Whether to use a bias vector within the output layer, <bool>.
        out_kernel_initializer: Initializer for the output layer kernel weights matrix.
        out_bias_initializer: Initializer for the output layer bias vector.
        out_kernel_regularizer: Regularizer function applied to the output layer kernel weights matrix.
        out_bias_regularizer: Regularizer function applied to the output layer bias vector.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        prediction: 3D tensor representing the class labels (batch_size, ndataset, num_classes), if channels_last, <tf.Tensor>.
                    3D tensor representing the class labels (batch_size, num_classes, ndataset), if channels_first, <tf.Tensor>.

    """
    def __init__(self,
                 num_classes=1,
                 kpconv=None,
                 dropout=None,
                 out_activation=None,
                 out_use_bias=False,
                 out_kernel_initializer='glorot_uniform',
                 out_bias_initializer='zeros',
                 out_kernel_regularizer=None,
                 out_bias_regularizer=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(OutputBlock, self).__init__(name=name, **kwargs)

        # Initialize OutputBlock layer
        self.num_classes = num_classes
        self.kpconv = kpconv if kpconv is not None else []
        self.dropout = dropout if dropout is not None else []
        self.out_activation = out_activation
        self.out_use_bias = out_use_bias
        self.out_kernel_initializer = out_kernel_initializer
        self.out_bias_initializer = out_bias_initializer
        self.out_kernel_regularizer = out_kernel_regularizer
        self.out_bias_regularizer = out_bias_regularizer
        self.data_format = data_format

        # Checks
        assert self.num_classes > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self._ndataset_axis = 1 if self.data_format == 'channels_last' else -1
        self._exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.kpconv_list = []
        self.dropout_list = []

        # Pass module data_format setting to all child layers
        _ = [kpconv_setting.update({'data_format': self.data_format}) for kpconv_setting in self.kpconv]

        # Set npoint to None to preserve the number of points
        _ = [kpconv_setting.update({'npoint': None}) for kpconv_setting in self.kpconv]

        # Adjust the length of the dropout layers to match with the kpconv layers
        self.dropout += [None] * (len(self.kpconv) - len(self.dropout))

    def build(self, input_shape):
        # Build KPConvBlock layers
        for kpconv_setting in self.kpconv:
            self.kpconv_list.append(KPConvBlock(**kpconv_setting))

        # Print warning if use_xyz is set to False for any child layer
        if any(not kpconv_layer.use_xyz for kpconv_layer in self.kpconv_list):
            warnings.warn("If the 'use_xyz' attribute is set to False for any KPConvBlock layer "
                          "within the OutputBlock, the xyz information will be lost for all subsequent "
                          "KPConvBlocks.")

        # Build dropout layers
        for dropout_setting in self.dropout:
            if dropout_setting is not None:
                self.dropout_list.append(layers.Dropout(**dropout_setting))
            else:
                self.dropout_list.append(None)

        # Build output layer
        self.output_layer = layers.Conv2D(filters=self.num_classes, kernel_size=[1, 1], strides=[1, 1], padding='valid',
                                          data_format=self.data_format, activation=self.out_activation, use_bias=self.out_use_bias,
                                          kernel_initializer=self.out_kernel_initializer, bias_initializer=self.out_bias_initializer,
                                          kernel_regularizer=self.out_kernel_regularizer, bias_regularizer=self.out_bias_regularizer)

        # Call keras base layer build function
        super(OutputBlock, self).build(input_shape)

    def call(self, inputs):
        # Declare
        prediction = inputs

        # Process layer inputs
        for kpconv_layer, dropout_layer in zip(self.kpconv_list, self.dropout_list):
            prediction = kpconv_layer(prediction)
            if dropout_layer is not None:
                prediction = dropout_layer(prediction)

        # Compute output
        prediction = tf.keras.backend.expand_dims(prediction, axis=self._exp_axis)
        prediction = self.output_layer(prediction)
        prediction = tf.keras.backend.squeeze(prediction, axis=self._exp_axis)

        return prediction

    def get_config(self):
        # Get Decoder OutputBlock configuration
        config = {
            'num_classes': self.num_classes,
            'kpconv': self.kpconv,
            'dropout': self.dropout,
            'out_activation': self.out_activation,
            'out_use_bias': self.out_use_bias,
            'out_kernel_initializer': self.out_kernel_initializer,
            'out_bias_initializer': self.out_bias_initializer,
            'out_kernel_regularizer': self.out_kernel_regularizer,
            'out_bias_regularizer': self.out_bias_regularizer,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(OutputBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Get output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[self._ndataset_axis], self.num_classes])
        else:
            output_shape = tf.TensorShape([input_shape[0], self.num_classes, input_shape[self._ndataset_axis]])

        return output_shape


class KPFCNN(layers.Layer):
    """
    Example implementation of the KP-FCNN model.

    Note: This implementation of the KP-FCNN model differes form the original one in the
    following points: the radius, the number of points in each local neighborhood (nsample)
    and the standard deviation value of the kernel initializer. Moreover, the KPConv base
    layer differes form the original one, as discussed in the KPConv layer docstring.

    Reference:
        - Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
          [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Inputs:
        inputs: 3D tensor (batch_size, ndataset, channels), if channels_last, <tf.Tensor>.
                3D tensor (batch_size, channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        num_classes: Number of different output classes/labels, <int>.
        seed: Random seed of the kernel initializer, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        prediction: 3D tensor representing the class labels (batch_size, ndataset, num_classes), if channels_last, <tf.Tensor>.
                    3D tensor representing the class labels (batch_size, num_classes, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 num_classes=1,
                 seed=42,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(KPFCNN, self).__init__(name=name, **kwargs)

        # Initialize KPFCNN layer
        self.num_classes = num_classes
        self.seed = seed
        self.data_format = data_format

        # Checks
        assert self.num_classes > 0
        assert self.seed > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1

        # Set radius (first_subsample_dl * kp_extend * 2.5)
        self.radius = 0.04 * 1.0 * 2.5

        # Set maximum number of points in each local neighborhood. The original paper uses the
        # 80th percentile of the neighbor points within the local regions, further dicussed
        # here: https://github.com/HuguesTHOMAS/KPConv/issues/12.
        self.nsample = 64

    def build(self, input_shape):
        # Set kernel initializer. The standard deviation value is set as default value (not
        # specified), since the unkonwn point dimension (ndataset) can not be used to calculate
        # the standard deviation.
        kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, seed=self.seed)

        # Build input layer
        self.kpconv_layer = KPConvBlock(filters=64, k_points=15, npoint=None, activation='lrelu', nsample=self.nsample, radius=self.radius,
                                        kernel_initializer=kernel_initializer, data_format=self.data_format, bn=True, momentum=0.98)

        self.resnet_layer = ResnetBlock(
            npoint=None,
            activation='lrelu',
            unary={'filters': 32, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98},
            kpconv={'filters': 32, 'k_points': 15, 'nsample': self.nsample, 'radius': self.radius, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98},
            unary2={'filters': 64, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98},
            data_format=self.data_format
        )

        # Build Encoder
        self.encoder_layer = Encoder(
            resnet=[
                {'npoint': 512, 'activation': 'lrelu', 'unary': {'filters': 64, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'kpconv': {'filters': 64, 'k_points': 15, 'nsample': self.nsample, 'radius': self.radius, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'unary2': {'filters': 128, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'npoint': 256, 'activation': 'lrelu', 'unary': {'filters': 128, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'kpconv': {'filters': 128, 'k_points': 15, 'nsample': self.nsample, 'radius': self.radius, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'unary2': {'filters': 256, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'npoint': 128, 'activation': 'lrelu', 'unary': {'filters': 256, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'kpconv': {'filters': 256, 'k_points': 15, 'nsample': self.nsample, 'radius': self.radius, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'unary2': {'filters': 512, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'npoint': 64, 'activation': 'lrelu', 'unary': {'filters': 512, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'kpconv': {'filters': 256, 'k_points': 15, 'nsample': self.nsample, 'radius': self.radius, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}, 'unary2': {'filters': 1024, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}}
            ],
            data_format=self.data_format
        )

        # Build Decoder
        self.decoder_layer = Decoder(
            fp=[
                {'upsample': {'upsample_mode': 'nearest'}, 'unary': {'filters': 512, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'upsample': {'upsample_mode': 'nearest'}, 'unary': {'filters': 256, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'upsample': {'upsample_mode': 'nearest'}, 'unary': {'filters': 128, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}},
                {'upsample': {'upsample_mode': 'nearest'}, 'unary': {'filters': 64, 'kernel_initializer': kernel_initializer, 'bn': True, 'momentum': 0.98}}
            ],
            data_format=self.data_format
        )

        # Build output layer
        self.output_layer = UnaryBlock(filters=self.num_classes, activation='softmax', use_xyz=False, kernel_initializer=kernel_initializer, data_format=self.data_format)

    def call(self, inputs):
        # Process input data
        prediction = self.kpconv_layer(inputs)
        prediction = self.resnet_layer(prediction)

        # Autoencode input data
        prediction = self.encoder_layer(prediction)
        prediction = self.decoder_layer(prediction)

        # Make prediction
        prediction = self.output_layer(prediction)

        return prediction

    def get_config(self):
        # Get Decoder KPFCNN configuration
        config = {
            'num_classes': self.num_classes,
            'seed': self.seed,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(KPFCNN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Get output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[self.ndataset_axis], self.num_classes])
        else:
            output_shape = tf.TensorShape([input_shape[0], self.num_classes, input_shape[self.ndataset_axis]])

        return output_shape
