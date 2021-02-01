"""
Kernel point convolution (KPConv) layer.

Author: Hugues, Thomas
Modified by: Felix Fent
Date: July 2020

References:
    - Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
      [Online] Available: https://arxiv.org/abs/1904.08889, 2019.
"""
# 3rd Party Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Local imports
from radarseg.model.tf_ops.sampling.tf_sampling import farthest_point_sample
from radarseg.model.tf_ops.grouping.tf_grouping import query_ball_point
from radarseg.model.tf_ops.grouping.tf_grouping import knn_point


def kernel_point_optimization(radius: float,
                              num_points: int,
                              num_kernels: int = 1,
                              dimension: int = 3,
                              fixed: str = 'center',
                              ratio: float = 1.0,
                              verbose: bool = False,
                              seed: int = 42):
    """
    Creation of kernel points via optimization of potentials.

    Optimization process to find a set of points with stable dispositions within
    a sphere of a defined radius.

    Note: This function is one-to-one transferred from the repository of Hugues Thomas.
    For more information please see his repository.

    Reference:
        Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
        [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Arguments:
        radius: Radius of the kernel (maximum distance of the kernel points to the center), <float>.
        num_points: Number of kernel points (kernel size), <int>.
        num_kernels: Number of returned kernels, <int>.
        dimension: Dimension of the kernel (space), <int>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        ratio: Ratio of the kernel radius (scaling factor), <float>.
        verbose: Verbosity option, <bool>
        seed: Random seed for the kernel point initialization, <int>.

    Returns:
        Kernel points (num_kernels, num_points, dimension), <np.array>.
    """
    # Checks (Currently only 3d point clouds and kernels with more than one point are supported!)
    assert radius > 0
    assert num_points > 1
    assert num_kernels > 0
    assert dimension == 3
    assert fixed in set(('none', 'center', 'verticals'))

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Initialize random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    # Optimize kernel points
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, axis=-1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, axis=-1) * gradients / np.expand_dims(gradients_norms + 1e-6, axis=-1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def create_kernel_points(radius: float, num_kpoints: int, num_kernels: int, dimension: int, fixed: str, seed: int = 42):
    """
    Returns a set of kernel points placed in a sphere with stable dispositions.

    Note: This function is one-to-one transferred from the repository of Hugues Thomas.
    For more information please see his repository.

    Reference:
        Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
        [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Arguments:
        radius: Radius of the kernel (maximum distance of the kernel points to the center), <float>.
        num_kpoints: Number of kernel points, <int>.
        dimension: Dimension of the kernel, <int>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        seed: Random seed for the kernel point initialization and random rotation, <int>.

    Reurns:
        kernels: Kernel points (num_kernels, num_points, dimension), <np.array>.
    """
    # Number of tries in the optimization process, to ensure we get the most stable disposition
    NUM_TRIES = 100

    # Checks (Currently only 3d point clouds and kernels with more than one point are supported!)
    assert dimension == 3
    assert num_kpoints > 1
    assert fixed in set(('none', 'center', 'verticals'))

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Create kernels
    kernel_points, grad_norms = kernel_point_optimization(radius=1.0,
                                                          num_points=num_kpoints,
                                                          num_kernels=NUM_TRIES,
                                                          dimension=dimension,
                                                          fixed=fixed,
                                                          verbose=0,
                                                          seed=seed)

    # Find best candidate
    best_k = np.argmin(grad_norms[-1, :])
    original_kernel = kernel_points[best_k, :, :]

    # Random rotations depending of the fixed points
    if fixed == 'verticals':
        # Create random rotations
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, 0, 0] = c
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        R[:, 0, 1] = s
        R[:, 1, 0] = -s

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, axis=0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

    else:
        # Create random rotations
        u = np.ones((num_kernels, 3))
        v = np.ones((num_kernels, 3))
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        while np.any(wrongs):
            new_u = np.random.rand(num_kernels, 3) * 2 - 1
            new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, axis=-1)
            u[wrongs, :] = new_u[wrongs, :]
            new_v = np.random.rand(num_kernels, 3) * 2 - 1
            new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, axis=-1)
            v[wrongs, :] = new_v[wrongs, :]
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

        # Make v perpendicular to u
        v -= np.expand_dims(np.sum(u * v, axis=1), axis=-1) * u
        v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, axis=-1)

        # Last rotation vector
        w = np.cross(u, v)
        R = np.stack((u, v, w), axis=-1)

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, axis=0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

        # Add a small noise
        kernels = kernels
        kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)

    return kernels


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Returns a radius gaussian (gaussian of distance)

    Arguments:
        sq_r: Input radiuses (dn, ..., d1, d0), <tf.Tensor>.
        sig: Extents of gaussians (d1, d0) or scalar, <tf.Tensor>.
    """
    return tf.exp(-sq_r / (2 * tf.square(sig) + eps))


class KPConv(layers.Layer):
    """
    Kernel point convolution (KPConv) layer.

    Note: This is a modified version of the original KPConv layer. Instead of pre-calculating
    the point indices and kernel points, everything is handeld within the layer itself.
    Moreover, a ball query search or a knn approch is used to determine the local neighbors
    instead of the original batch neighbors operation.

    Reference:
        Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
        [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Inputs:
        inputs: 3D tensor (batch_size, channels, ndataset) for channels_first, <tf.Tensor>.
                3D tensor (batch_size, ndataset, channels) for channels_last, <tf.Tensor>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        k_points: Number of kernel points (similar to the kernel size), <int>.
        npoint: Number of points sampled in farthest point sampling (number of output points), <int>.
        nsample: Maximum number of points in each local region, <int>.
        radius: Search radius in local region, <float>.
        activation: Activation function of the kernel point convolution, <str or tf.keras.activations>.
        alpha: Slope coefficient of the leaky rectified linear unit (if specified), <float>.
        kp_extend: Extension factor to define the area of influence of the kernel points, <float>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        knn: Whether to use kNN instead of radius search, <bool>.
        kp_influence: Association function for the influence of the kernel points (one of either 'constant', 'linear' or 'gaussian'), <str>.
        aggregation_mode: Method to aggregate the activation of the point convolution kernel (one of either 'closest' or 'sum'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        seed: Random seed for the kernel point initialization, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, filters, npoint) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, npoint, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 filters=1,
                 k_points=2,
                 npoint=None,
                 nsample=1,
                 radius=1.0,
                 activation='lrelu',
                 alpha=0.3,
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
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(KPConv, self).__init__(name=name, **kwargs)

        # Initialize KPConv layer
        self.filters = filters
        self.k_points = k_points
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.activation = activation
        self.alpha = alpha
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

        # Checks
        assert self.filters > 0
        assert self.k_points > 0
        assert self.nsample > 0
        assert self.radius > 0
        assert self.alpha > 0
        assert self.kp_extend > 0
        assert self.fixed in set(('none', 'center', 'verticals'))
        assert self.kp_influence in set(('constant', 'linear', 'gaussian'))
        assert self.aggregation_mode in set(('closest', 'sum'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.npoint is not None:
            assert self.npoint > 0

        # Set layer attributes
        self._chan_axis = -1 if self.data_format == 'channels_last' else 1
        self._ndataset_axis = 1 if self.data_format == 'channels_last' else -1

    def build(self, input_shape):
        # Convert input shape
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        elif isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()
        else:
            input_shape = tf.keras.backend.int_shape(input_shape)

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

        # Build kernel
        self.kernel_values = self.add_weight(name='kernel_values',
                                             shape=(self.k_points, input_shape[self._chan_axis] - 3, self.filters),
                                             dtype=self.dtype,
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer)

        self.kernel_points = create_kernel_points(radius=self.kp_extend,
                                                  num_kpoints=self.k_points,
                                                  num_kernels=1,
                                                  dimension=3,
                                                  fixed=self.fixed,
                                                  seed=self.seed)

        # Build activation layer
        if self.activation == 'lrelu':
            self.activation_layer = layers.LeakyReLU(alpha=self.alpha)
        else:
            self.activation_layer = layers.Activation(activation=self.activation)

        # Call keras base layer build function
        super(KPConv, self).build(input_shape)

    def call(self, inputs, training=None):
        # Split input in point coordinates [ndataset, dim] and point features [ndataset, in_fdim]
        xyz = self.lambda1(inputs)
        features = self.lambda2(inputs)

        if self.data_format == 'channels_first':
            xyz = tf.transpose(xyz, [0, 2, 1])
            features = tf.transpose(features, [0, 2, 1])

        # Subsample points (get center points/query points) [npoint, dim]
        if self.npoint is not None:
            idx = farthest_point_sample(self.npoint, xyz)
            center_points = tf.gather(xyz, idx, axis=1, batch_dims=1)
        else:
            center_points = xyz

        # Get neighbor points [npoint, nsample, dim]
        if self.knn:
            _, idx = knn_point(self.nsample, xyz, center_points)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, xyz, center_points)

        neighbors = tf.gather(xyz, idx, axis=1, batch_dims=1)

        # Center every neighborhood [npoint, nsample, dim]
        neighbors -= tf.tile(tf.expand_dims(center_points, axis=2), [1, 1, self.nsample, 1])

        # Get all difference matrices [npoint, nsample, k_points, dim]
        # TODO: Add deformable KPConv layer option (deformable kernel points)
        neighbors = tf.expand_dims(neighbors, axis=3)
        neighbors = tf.tile(neighbors, [1, 1, 1, self.k_points, 1])
        differences = neighbors - self.kernel_points

        # Get the square distances [npoint, nsample, k_points]
        sq_distances = tf.reduce_sum(tf.square(differences), axis=4)

        # Get Kernel point influences [npoint, k_points, nsample]
        if self.kp_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = tf.ones_like(sq_distances)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        elif self.kp_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = kp_extend.
            all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.kp_extend, 0.0)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        else:
            # Influence in gaussian of the distance.
            sigma = self.kp_extend * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = tf.argmin(sq_distances, axis=3, output_type=tf.int32)
            all_weights *= tf.one_hot(neighbors_1nn, self.k_points, axis=2, dtype=self.dtype)

        # Get the features of each neighborhood [npoint, nsample, in_fdim]
        neighborhood_features = tf.gather(features, idx, axis=1, batch_dims=1)

        # Apply distance weights [npoint, k_points, in_fdim]
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        # Apply network weights [k_points, npoint, out_fdim]
        weighted_features = tf.transpose(weighted_features, [0, 2, 1, 3])
        kernel_outputs = tf.matmul(weighted_features, self.kernel_values)

        # Kernel point aggregation [npoint, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=1)

        # Apply activation function
        output_features = self.activation_layer(output_features)

        # Add point coordinates to output features if use_xyz [npoint, 3 + out_fdim]
        if self.use_xyz:
            output_features = tf.concat([center_points, output_features], axis=2)

        # Adjust output format
        if self.data_format == 'channels_first':
            output_features = tf.transpose(output_features, [0, 2, 1])

        return output_features

    def get_config(self):
        # Get KPConv layer configuration
        config = {
            'filters': self.filters,
            'k_points': self.k_points,
            'npoint': self.npoint,
            'nsample': self.nsample,
            'radius': self.radius,
            'activation': self.activation,
            'alpha': self.alpha,
            'kp_extend': self.kp_extend,
            'fixed': self.fixed,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'knn': self.knn,
            'kp_influence': self.kp_influence,
            'aggregation_mode': self.aggregation_mode,
            'use_xyz': self.use_xyz,
            'seed': self.seed,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(KPConv, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: An shape tuple, <tuple>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Compute number of output points
        if self.npoint is not None:
            npoint = self.npoint
        else:
            npoint = input_shape[self._ndataset_axis]

        # Compute number of output channels
        if self.use_xyz:
            nbr_out_chan = 3 + self.filters
        else:
            nbr_out_chan = self.filters

        # Compute output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], npoint, nbr_out_chan])
        else:
            output_shape = tf.TensorShape([input_shape[0], nbr_out_chan, npoint])

        return output_shape


class KPFP(layers.Layer):
    """
    Kernel point based feature propagation (KPFP) layer.

    Reference:
        Thomas, Hugues, KPConv: Flexible and Deformable Convolution for Point Clouds.
        [Online] Available: https://arxiv.org/abs/1904.08889, 2019.

    Inputs:
        inputs: List of two 3D tensors, whereas the second one defines the output shape.
        [(batch_size, ndataset1, channels1), (batch_size, ndataset2, channels2)], if channels_last, <List[tf.Tensor, tf.Tensor]>.
        [(batch_size, channels1, ndataset1), (batch_size, channels2, ndataset2)], if channels_first, <List[tf.Tensor, tf.Tensor]>.

    Arguments:
        filters: The dimensionality of the output space (i.e. the number of output filters in the convolution), <int>.
        k_points: Number of kernel points (similar to the kernel size), <int>.
        nsample: Maximum number of points in each local region, <int>.
        radius: Search radius in local region, <float>.
        activation: Activation function of the kernel point convolution, <str or tf.keras.activations>.
        alpha: Slope coefficient of the leaky rectified linear unit (if specified), <float>.
        kp_extend: Extension factor to define the area of influance of the kernel points, <float>.
        fixed: Whether to fix the position of certain kernel points (one of either 'none', 'center' or 'verticals'), <str>.
        kernel_initializer: Initializer for the convolution kernel weights matrix.
        kernel_regularizer: Regularizer function applied to the convolution kernel weights matrix.
        knn: Whether to use kNN instead of radius search, <bool>.
        kp_influence: Association function for the influance of the kernel points (one of either 'constant', 'linear' or 'gaussian'), <str>.
        aggregation_mode: Method to aggregate the activation of the point convolution kernel (one of either 'closest' or 'sum'), <str>.
        use_xyz: Whether to concat xyz with the output point features, otherwise just use point features, <bool>.
        seed: Random seed for the kernel point initialization, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        outputs: 3D tensor (batch_size, filters, ndataset2) for channels_first, <tf.Tensor>.
                 3D tensor (batch_size, ndataset2, filters) for channels_last, <tf.Tensor>.
    """
    def __init__(self,
                 filters=1,
                 k_points=2,
                 nsample=None,
                 radius=1.0,
                 activation='lrelu',
                 alpha=0.3,
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
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(KPFP, self).__init__(name=name, **kwargs)

        # Initialize KPFP layer
        self.filters = filters
        self.k_points = k_points
        self.nsample = nsample
        self.radius = radius
        self.activation = activation
        self.alpha = alpha
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

        # Checks
        assert self.filters > 0
        assert self.k_points > 0
        assert self.radius > 0
        assert self.alpha > 0
        assert self.kp_extend > 0
        assert self.fixed in set(('none', 'center', 'verticals'))
        assert self.kp_influence in set(('constant', 'linear', 'gaussian'))
        assert self.aggregation_mode in set(('closest', 'sum'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        if self.nsample is not None:
            assert self.nsample > 0

        # Set layer attributes
        self._dim_xyz = 3
        self._chan_axis = 2 if self.data_format == 'channels_last' else 1
        self._ndataset_axis = 1 if self.data_format == 'channels_last' else 2

    def build(self, input_shape):
        # Check if two input tensors are provided
        if not isinstance(input_shape, (list, tuple)):
            raise TypeError('The input of the KPFP layer must be a list or tuple, '
                            'of two tensors, but an input of type {} is given.'.format(type(input_shape)))
        elif len(input_shape) != 2:
            raise ValueError('The input of the KPFP layer must contain exactly two '
                             'tensors, but an input of lenght {} is given.'.format(len(input_shape)))

        # Convert input shape
        def to_list(shape):
            if isinstance(shape, tf.TensorShape):
                return shape.as_list()

            return tf.keras.backend.int_shape(shape)

        input_shape = tf.nest.map_structure(to_list, input_shape)

        # Check input shape compatibility
        if any(shape[self._chan_axis] < 3 for shape in input_shape):
            raise ValueError('Every input tensor must have at least three channels (xyz), '
                             'but an input of shape {} is given.'.format(input_shape))

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build kernel
        self.kernel_values = self.add_weight(name='kernel_values',
                                             shape=(self.k_points, input_shape[0][self._chan_axis] + input_shape[1][self._chan_axis] - 2 * self._dim_xyz, self.filters),
                                             dtype=self.dtype,
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer)

        self.kernel_points = create_kernel_points(radius=self.kp_extend,
                                                  num_kpoints=self.k_points,
                                                  num_kernels=1,
                                                  dimension=3,
                                                  fixed=self.fixed,
                                                  seed=self.seed)

        # Build activation layer
        if self.activation == 'lrelu':
            self.activation_layer = layers.LeakyReLU(alpha=self.alpha)
        else:
            self.activation_layer = layers.Activation(activation=self.activation)

        # Call keras base layer build function
        super(KPFP, self).build(input_shape)

    def call(self, inputs, training=None):
        # Split inputs into point coordinates [ndataset, dim] and point features [ndataset, in_fdim]
        xyz1 = self.lambda1(inputs[0])
        features1 = self.lambda2(inputs[0])
        xyz2 = self.lambda1(inputs[1])
        features2 = self.lambda2(inputs[1])

        if self.data_format == 'channels_first':
            xyz1 = tf.transpose(xyz1, [0, 2, 1])
            features1 = tf.transpose(features1, [0, 2, 1])
            xyz2 = tf.transpose(xyz2, [0, 2, 1])
            features2 = tf.transpose(features2, [0, 2, 1])

        # Set nsample (number of neighboring points)
        if self.nsample is not None:
            nsample = self.nsample
        else:
            nsample = xyz1.shape[1]

        # Get neighbor points [npoint, nsample, dim]
        if self.knn:
            _, idx = knn_point(nsample, xyz1, xyz2)
        else:
            idx, _ = query_ball_point(self.radius, nsample, xyz1, xyz2)

        neighbors = tf.gather(xyz1, idx, axis=1, batch_dims=1)

        # Center every neighborhood [npoint, nsample, dim]
        neighbors -= tf.tile(tf.expand_dims(xyz2, axis=2), [1, 1, nsample, 1])

        # Get all difference matrices [npoint, nsample, k_points, dim]
        # TODO: Add deformable KPConv layer option (deformable kernel points)
        neighbors = tf.expand_dims(neighbors, axis=3)
        neighbors = tf.tile(neighbors, [1, 1, 1, self.k_points, 1])
        differences = neighbors - self.kernel_points

        # Get the square distances [npoint, nsample, k_points]
        sq_distances = tf.reduce_sum(tf.square(differences), axis=4)

        # Get Kernel point influences [npoint, k_points, nsample]
        if self.kp_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = tf.ones_like(sq_distances)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        elif self.kp_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = kp_extend.
            all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.kp_extend, 0.0)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        else:
            # Influence in gaussian of the distance.
            sigma = self.kp_extend * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = tf.transpose(all_weights, [0, 1, 3, 2])

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = tf.argmin(sq_distances, axis=3, output_type=tf.int32)
            all_weights *= tf.one_hot(neighbors_1nn, self.k_points, axis=2, dtype=self.dtype)

        # Get the features of each neighborhood [npoint, nsample, in_fdim]
        neighborhood_features = tf.gather(features1, idx, axis=1, batch_dims=1)

        # Concatenate features [npoint, nsample, in_fdim]
        features2 = tf.tile(tf.expand_dims(features2, axis=2), [1, 1, nsample, 1])
        neighborhood_features = tf.concat([neighborhood_features, features2], axis=3)

        # Apply distance weights [npoint, k_points, in_fdim]
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        # Apply network weights [k_points, npoint, out_fdim]
        weighted_features = tf.transpose(weighted_features, [0, 2, 1, 3])
        kernel_outputs = tf.matmul(weighted_features, self.kernel_values)

        # Kernel point aggregation [npoint, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=1)

        # Apply activation function
        output_features = self.activation_layer(output_features)

        # Add point coordinates to output features if use_xyz [npoint, 3 + out_fdim]
        if self.use_xyz:
            output_features = tf.concat([xyz2, output_features], axis=2)

        # Adjust output format
        if self.data_format == 'channels_first':
            output_features = tf.transpose(output_features, [0, 2, 1])

        return output_features

    def get_config(self):
        # Get KPFP layer configuration
        config = {
            'filters': self.filters,
            'k_points': self.k_points,
            'nsample': self.nsample,
            'radius': self.radius,
            'activation': self.activation,
            'alpha': self.alpha,
            'kp_extend': self.kp_extend,
            'fixed': self.fixed,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'knn': self.knn,
            'kp_influence': self.kp_influence,
            'aggregation_mode': self.aggregation_mode,
            'use_xyz': self.use_xyz,
            'seed': self.seed,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(KPFP, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or list of shape tuples (one per input tensor of the layer), <tuple>.

        Returns:
            output_shape: A shape tensor, <tf.TensorShape>.
        """
        # Convert input shape
        input_shape = tf.nest.map_structure(layers.Lambda(lambda x: x).compute_output_shape, input_shape)

        # Select state input
        input_shape2 = input_shape[1]

        # Compute number of output channels
        if self.use_xyz:
            nbr_out_chan = 3 + self.filters
        else:
            nbr_out_chan = self.filters

        # Compute output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape2[0], input_shape2[self._ndataset_axis], nbr_out_chan])
        else:
            output_shape = tf.TensorShape([input_shape2[0], nbr_out_chan, input_shape2[self._ndataset_axis]])

        return output_shape
