"""
PointNet++ Model Architecture

Author: Charles R. Qi
Modified by: Felix Fent
Date: May 2020

References:
    - Qi, Charles R, PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
      [Online] Available: https://arxiv.org/abs/1706.02413
"""
# 3rd Party Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Local imports
from radarseg.model.tf_ops.sampling.tf_sampling import farthest_point_sample
from radarseg.model.tf_ops.sampling.tf_sampling import gather_point
from radarseg.model.tf_ops.grouping.tf_grouping import query_ball_point
from radarseg.model.tf_ops.grouping.tf_grouping import group_point
from radarseg.model.tf_ops.grouping.tf_grouping import knn_point
from radarseg.model.tf_ops.interpolation.tf_interpolate import three_interpolate
from radarseg.model.tf_ops.interpolation.tf_interpolate import three_nn
from radarseg.model.layer.mlp import MLP


class SampleAndGroup(layers.Layer):
    """
    Sampling and grouping layer.

    Samples a number of 'npoint' centroid points and forms groups of 'nsample' points, which are within a defined
    radius 'radius' around the centroid points. The determination of the neighboring points is executed by either a
    a ball query search or a k-Nearest-Neighbor (kNN) search.

    Performance advices:
    The performance of this layer is best for the data_format 'channels_last' and knn disabled.

    Note: The number of points in each local region 'nsample' has to be less or equal th the number of points 'ndataset'
          in the sample, if knn is enabled (this is not required for knn=False).

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        npoint: Number of points sampled in farthest point sampling, <int32>.
        radius: Search radius in local region, <float32>.
        nsample: Maximum number of points in each local region, <int32>.
        knn: Whether to use kNN instead of radius search, <bool>.
        use_xyz: Whether to concat xyz with local point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        new_xyz: Centroid point coordinates (batch_size, npoint, 3), if channels_last, <tf.Tensor>.
                 Centroid point coordinates (batch_size, 3, npoint), if channels_first, <tf.Tensor>.
        new_points: Local regions, coordinates and features (batch_size, npoint, nsample, 3 + channels), if channels_last and use_xyz, <tf.Tensor>.
                    Local regions, coordinates and features (batch_size, 3 + channels, npoint, nsample), if channels_first and use_xyz, <tf.Tensor>.
                    Local regions, coordinates and features (batch_size, npoint, nsample, channels), if channels_last, <tf.Tensor>.
                    Local regions, coordinates and features (batch_size, channels, npoint, nsample), if channels_first, <tf.Tensor>.
        idx: Indices of the points in the local regions as in ndataset points (batch_size, npoint, nsample), <tf.Tensor>.
        grouped_xyz: Normalized point coordinates (subtracted by seed point xyz) in local regions (batch_size, npoint, nsample, 3), if channels_last, <tf.Tensor>.
                     Normalized point coordinates (subtracted by seed point xyz) in local regions (batch_size, 3, npoint, nsample), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 npoint=1,
                 radius=1.0,
                 nsample=1,
                 knn=False,
                 use_xyz=True,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(SampleAndGroup, self).__init__(name=name, **kwargs)

        # Initialize SampleAndGroup layer
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert self.npoint > 0
        assert self.radius > 0
        assert self.nsample > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1

    def build(self, input_shape):
        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Call build function of the keras base layer
        super(SampleAndGroup, self).build(input_shape)

    def call(self, inputs):
        # Split input in point coordinates and point features
        xyz = self.lambda1(inputs)
        points = self.lambda2(inputs)

        if self.data_format == 'channels_first':
            xyz = tf.transpose(xyz, [0, 2, 1]) if self.data_format == 'channels_first' else xyz
            points = tf.transpose(points, [0, 2, 1]) if self.data_format == 'channels_first' else points

        # Determine centroid points (coordinates)
        idx = farthest_point_sample(self.npoint, xyz)

        # Subsample points
        new_xyz = gather_point(xyz, idx)

        # Determine neighboring points (coordinates)
        if self.knn:
            _, idx = knn_point(self.nsample, xyz, new_xyz)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        # Group points (coordinates)
        grouped_xyz = group_point(xyz, idx)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, self.nsample, 1])

        # Group point features (if provided)
        grouped_points = group_point(points, idx)

        if self.use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points

        # Adjust output format
        if self.data_format == 'channels_first':
            new_xyz = tf.transpose(new_xyz, [0, 2, 1])
            new_points = tf.transpose(new_points, [0, 3, 1, 2])
            grouped_xyz = tf.transpose(grouped_xyz, [0, 3, 1, 2])

        return new_xyz, new_points, idx, grouped_xyz

    def get_config(self):
        # Get SampleAndGroup layer configuration
        config = {
            'npoint': self.npoint,
            'radius': self.radius,
            'nsample': self.nsample,
            'knn': self.knn,
            'use_xyz': self.use_xyz,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(SampleAndGroup, self).get_config()

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

        # Get number of output channles
        if self.use_xyz:
            nbr_out_chan = input_shape[self.chan_axis]
        else:
            nbr_out_chan = input_shape[self.chan_axis] - 3

        # Get output shapes
        if self.data_format == 'channels_last':
            new_xyz_shape = tf.TensorShape([input_shape[0], self.npoint, 3])
            new_points_shape = tf.TensorShape([input_shape[0], self.npoint, self.nsample, nbr_out_chan])
            idx_shape = tf.TensorShape([input_shape[0], self.npoint, self.nsample])
            grouped_xyz_shape = tf.TensorShape([input_shape[0], self.npoint, self.nsample, 3])

        else:
            new_xyz_shape = tf.TensorShape([input_shape[0], 3, self.npoint])
            new_points_shape = tf.TensorShape([input_shape[0], nbr_out_chan, self.npoint, self.nsample])
            idx_shape = tf.TensorShape([input_shape[0], self.npoint, self.nsample])
            grouped_xyz_shape = tf.TensorShape([input_shape[0], 3, self.npoint, self.nsample])

        return (new_xyz_shape, new_points_shape, idx_shape, grouped_xyz_shape)


class SampleAndGroupAll(layers.Layer):
    """
    Sampling and grouping layer.

    Groups all points within the origin (0 , 0 , 0).

    Performance advices:
    The performance of this layer is best for the data_format 'channels_last'.

    Note: Equivalent to SampleAndGroup with npoint=1, nsample=ndataset, radius=inf, uses (0, 0, 0) as the centroid.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        use_xyz: Whether to concat xyz with local point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        new_xyz: Centroid point coordinates (batch_size, 1, 3) as (0, 0, 0), if channels_last, <tf.Tensor>.
                 Centroid point coordinates (batch_size, 3, 1) as (0, 0, 0), if channels_first, <tf.Tensor>.
        new_points: Point coordinates and features (batch_size, 1, ndataset, 3 + channels), if channels_last and use_xyz, <tf.Tensor>.
                    Point coordinates and features (batch_size, 3 + channels, 1, ndataset), if channels_first and use_xyz, <tf.Tensor>.
                    Point coordinates and features (batch_size, 1, ndataset, channels), if channels_last, <tf.Tensor>.
                    Point coordinates and features (batch_size, channels, 1, ndataset), if channels_first, <tf.Tensor>.
        idx: Indices of the points as in ndataset points (batch_size, 1, ndataset), <tf.Tensor>.
        grouped_xyz: Normalized point coordinates (subtracted by seed point xyz) (batch_size, 1, ndataset, 3), if channels_last, <tf.Tensor>.
                     Normalized point coordinates (subtracted by seed point xyz) (batch_size, 3, 1, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 use_xyz=True,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(SampleAndGroupAll, self).__init__(name=name, **kwargs)

        # Initialize SampleAndGroupAll layer
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1

    def build(self, input_shape):
        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Call build function of the keras base layer
        super(SampleAndGroupAll, self).build(input_shape)

    def call(self, inputs):
        # Split input in point coordinates and point features
        xyz = self.lambda1(inputs)
        points = self.lambda2(inputs)

        if self.data_format == 'channels_first':
            xyz = tf.transpose(xyz, [0, 2, 1]) if self.data_format == 'channels_first' else xyz
            points = tf.transpose(points, [0, 2, 1]) if self.data_format == 'channels_first' else points

        # Subsample points (to just one point (0 ,0 ,0))
        self.batch_size = tf.shape(xyz)[0]
        self.ndataset = tf.shape(xyz)[1]
        new_xyz = tf.tile(tf.reshape(tf.zeros((3,), dtype=tf.dtypes.float32), (1, 1, 3)), (self.batch_size, 1, 1))

        # Set point indices
        idx = tf.tile(tf.reshape(tf.range(start=0, limit=self.ndataset, dtype=tf.dtypes.int64), (1, 1, self.ndataset)), (self.batch_size, 1, 1))

        # Group all points (coordinates)
        grouped_xyz = tf.reshape(xyz, (self.batch_size, 1, self.ndataset, 3))

        # Group all point features (if provided)
        if self.use_xyz:
            new_points = tf.concat([xyz, points], axis=2)
        else:
            new_points = points

        new_points = tf.expand_dims(new_points, 1)

        # Adjust output format
        if self.data_format == 'channels_first':
            new_xyz = tf.transpose(new_xyz, [0, 2, 1])
            new_points = tf.transpose(new_points, [0, 3, 1, 2])
            grouped_xyz = tf.transpose(grouped_xyz, [0, 3, 1, 2])

        return new_xyz, new_points, idx, grouped_xyz

    def get_config(self):
        # Get SampleAndGroupAll layer configuration
        config = {
            'use_xyz': self.use_xyz,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(SampleAndGroupAll, self).get_config()

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

        # Get number of output channles
        if self.use_xyz:
            nbr_out_chan = input_shape[self.chan_axis]
        else:
            nbr_out_chan = input_shape[self.chan_axis] - 3

        # Get output shapes
        if self.data_format == 'channels_last':
            new_xyz_shape = tf.TensorShape([input_shape[0], 1, 3])
            new_points_shape = tf.TensorShape([input_shape[0], 1, input_shape[self.ndataset_axis], nbr_out_chan])
            idx_shape = tf.TensorShape([input_shape[0], 1, input_shape[self.ndataset_axis]])
            grouped_xyz_shape = tf.TensorShape([input_shape[0], 1, input_shape[self.ndataset_axis], 3])

        else:
            new_xyz_shape = tf.TensorShape([input_shape[0], 3, 1])
            new_points_shape = tf.TensorShape([input_shape[0], nbr_out_chan, 1, input_shape[self.ndataset_axis]])
            idx_shape = tf.TensorShape([input_shape[0], 1, input_shape[self.ndataset_axis]])
            grouped_xyz_shape = tf.TensorShape([input_shape[0], 3, 1, input_shape[self.ndataset_axis]])

        return (new_xyz_shape, new_points_shape, idx_shape, grouped_xyz_shape)


class SAModule(layers.Layer):
    """
    PointNet++ Set Abstraction (SA) Module

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        npoint: Number of points sampled in farthest point sampling, <int32>.
        radius: Search radius in local region, <float32>.
        nsample: Maximum number of points in each local region, <int32>.
        mlp: List of dicts to define the attributes of the first MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>
        mlp2: List of dicts to define the attributes of the second MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>
        pooling: Pooling operation to encode the local regions, one of either 'max' or 'avg', <str>.
        knn: Whether to use kNN instead of radius search, <bool>.
        use_xyz: Whether to concat xyz with local point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        new_xyz: Centroid point coordinates (batch_size, 1, 3), if channels_last and group_all, <tf.Tensor>.
                 Centroid point coordinates (batch_size, 3, 1), if channels_first and group_all, <tf.Tensor>.
                 Centroid point coordinates (batch_size, npoint, 3), if channels_last, <tf.Tensor>.
                 Centroid point coordinates (batch_size, 3, npoint), if channels_first, <tf.Tensor>.
        new_points: Centroid point coordinates and features (batch_size, npoint, mlp[-1] or mlp2[-1] or channels + 3), if channels_last and use_xyz, <tf.Tensor>.
                    Centroid point coordinates and features (batch_size, mlp[-1] or mlp2[-1] or channels + 3, npoint), if channels_first and use_xyz, <tf.Tensor>.
                    Centroid point coordinates and features (batch_size, npoint, mlp[-1] or mlp2[-1] or channels), if channels_last, <tf.Tensor>.
                    Centroid point coordinates and features (batch_size, mlp[-1] or mlp2[-1] or channels, npoint), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 npoint=1,
                 radius=1.0,
                 nsample=1,
                 mlp=None,
                 mlp2=None,
                 group_all=False,
                 pooling='max',
                 knn=False,
                 use_xyz=True,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(SAModule, self).__init__(name=name, **kwargs)

        # Initialize SAModule layer
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = [{'filters': 1}] if mlp is None else mlp
        self.mlp2 = [] if mlp2 is None else mlp2
        self.group_all = group_all
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert self.npoint > 0
        assert self.radius > 0
        assert self.nsample > 0
        assert self.pooling in set(('max', 'avg'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to all mlp layers
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp2]

        # Set layer attributes
        self.ndataset_axis = 2 if data_format == 'channels_last' else 3
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.mlp_list = []
        self.mlp2_list = []

    def build(self, input_shape):
        # Build sample and group layer
        if self.group_all:
            self.sample_and_group_all = SampleAndGroupAll(use_xyz=self.use_xyz, data_format=self.data_format, name='sample_and_group_all')
        else:
            self.sample_and_group = SampleAndGroup(npoint=self.npoint, radius=self.radius,
                                                   nsample=self.nsample, knn=self.knn,
                                                   use_xyz=self.use_xyz, data_format=self.data_format,
                                                   name='sample_and_group')

        # Build first MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # [Optional] Build second MLP layer
        for mlp_setting in self.mlp2:
            self.mlp2_list.append(MLP(**mlp_setting))

        # Call build function of the keras base layer
        super(SAModule, self).build(input_shape)

    def call(self, inputs):
        # Sample and Grouping
        if self.group_all:
            new_xyz, new_points, _, _ = self.sample_and_group_all(inputs)
        else:
            new_xyz, new_points, _, _ = self.sample_and_group(inputs)

        # Point Feature Embedding
        for mlp_layer in self.mlp_list:
            new_points = mlp_layer(new_points)

        # Pool local regions
        if self.pooling == 'max':
            new_points = keras.backend.max(new_points, axis=self.ndataset_axis, keepdims=True)
        elif self.pooling == 'avg':
            new_points = keras.backend.mean(new_points, axis=self.ndataset_axis, keepdims=True)

        # [Optional] Further processing
        for mlp_layer in self.mlp2_list:
            new_points = mlp_layer(new_points)

        new_points = keras.backend.squeeze(new_points, axis=self.ndataset_axis)

        return new_xyz, new_points

    def get_config(self):
        # Get SAModule layer configuration
        config = {
            'npoint': self.npoint,
            'radius': self.radius,
            'nsample': self.nsample,
            'mlp': self.mlp,
            'mlp2': self.mlp2,
            'group_all': self.group_all,
            'pooling': self.pooling,
            'knn': self.knn,
            'use_xyz': self.use_xyz,
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
            output_shape: A tuple of shape tensors, <tuple>.
        """
        # Call layer build function to get input_shape dependant attributes
        if tf.executing_eagerly() and not self.built:
            self.build(input_shape)

        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        # Get nuber of output channels
        if self.mlp2:
            nbr_out_chan = self.mlp2_list[-1].compute_output_shape(input_shape)[self.chan_axis]
        elif self.mlp:
            nbr_out_chan = self.mlp_list[-1].compute_output_shape(input_shape)[self.chan_axis]
        elif self.use_xyz:
            nbr_out_chan = input_shape[self.chan_axis]
        else:
            nbr_out_chan = input_shape[self.chan_axis] - 3

        # Get number of new points
        if self.group_all:
            nbr_new_points = 1
        else:
            nbr_new_points = self.npoint

        # Get output shapes
        if self.data_format == 'channels_last':
            new_xyz_shape = tf.TensorShape([input_shape[0], nbr_new_points, 3])
            new_points_shape = tf.TensorShape([input_shape[0], nbr_new_points, nbr_out_chan])

        else:
            new_xyz_shape = tf.TensorShape([input_shape[0], 3, nbr_new_points])
            new_points_shape = tf.TensorShape([input_shape[0], nbr_out_chan, nbr_new_points])

        return (new_xyz_shape, new_points_shape)


class FPModule(layers.Layer):
    """
    PointNet Feature Propogation (FP) Module

    Performance advices:
    The performance of this layer is best for the data_format 'channels_last'.

    Input:
        inputs: List of two point clouds with each 3 coordinates and additional channels (output and input of the corresponding SA module)
                [(batch_size, ndataset1, 3 + channels1), (batch_size, ndataset2, 3 + channels2)], if channels_last, <List[tf.Tensor, tf.Tensor]>.
                List of two point clouds with each 3 coordinates and additional channels (output and input of the corresponding SA module)
                [(batch_size, 3 + channels1, ndataset1), (batch_size, 3 + channels2, ndataset2)], if channels_first, <List[tf.Tensor, tf.Tensor]>.
    Arguments:
        mlp: List of dicts to define the attributes of the MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        use_xyz: Whether to concat xyz with local point features, otherwise just use point features, <bool>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        new_points: Interpolated point features (batch_size, ndataset2, mlp[-1] + 3 or channels1 + channels2 + 3), if channels_last and use_xyz, <tf.Tensor>.
                    Interpolated point features (batch_size, mlp[-1] + 3 or channels1 + channels2 + 3, ndataset2), if channels_first and use_xyz, <tf.Tensor>.
                    Interpolated point features (batch_size, ndataset2, mlp[-1] or channels1 + channels2), if channels_last, <tf.Tensor>.
                    Interpolated point features (batch_size, mlp[-1] or channels1 + channels2, ndataset2), if channels_first, <tf.Tensor>.
        xyz2: Centroid point coordinates (batch_size, ndataset2, 3), if channels_last, <tf.Tensor>.
              Centroid point coordinates (batch_size, 3, ndataset2), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 mlp=None,
                 use_xyz=False,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(FPModule, self).__init__(name=name, **kwargs)

        # Initialize FPModule layer
        self.mlp = [{'filters: 1'}] if mlp is None else mlp
        self.use_xyz = use_xyz
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass module data_format setting to all mlp layers
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1
        self.mlp_list = []

    def build(self, input_shape):
        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        else:
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build concatenate layer
        self.concat1 = layers.Concatenate(axis=-1)
        self.concat2 = layers.Concatenate(axis=self.chan_axis)

        # Build MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # Call build function of the keras base layer
        super(FPModule, self).build(input_shape)

    def call(self, inputs):
        # Split input in point coordinates and point features
        xyz1 = self.lambda1(inputs[0])
        points1 = self.lambda2(inputs[0])
        xyz2 = self.lambda1(inputs[1])
        points2 = self.lambda2(inputs[1])

        # Adjust input format
        if self.data_format == 'channels_first':
            xyz1 = tf.transpose(xyz1, [0, 2, 1])
            points1 = tf.transpose(points1, [0, 2, 1])
            xyz2 = tf.transpose(xyz2, [0, 2, 1])
            points2 = tf.transpose(points2, [0, 2, 1])

        # Get the distance to the three nearest neighbors
        dist, idx = three_nn(xyz2, xyz1)
        dist = tf.maximum(dist, tf.keras.backend.epsilon())

        # Calculate the interpolation weights
        norm = tf.reduce_sum((1.0 / dist), axis=-1, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm

        # Interpolate the points
        interpolated_points = three_interpolate(points1, idx, weight)

        # Concatenate the point features
        new_points = self.concat1([interpolated_points, points2])

        # Adjust output format
        if self.data_format == 'channels_first':
            xyz2 = tf.transpose(xyz2, [0, 2, 1])
            new_points = tf.transpose(new_points, [0, 2, 1])

        new_points = tf.keras.backend.expand_dims(new_points, axis=self.exp_axis)

        # Point feature processing
        for mlp_layer in self.mlp_list:
            new_points = mlp_layer(new_points)

        new_points = tf.keras.backend.squeeze(new_points, axis=self.exp_axis)

        # Add the point coordinates
        if self.use_xyz:
            new_points = self.concat2([xyz2, new_points])

        return new_points, xyz2

    def get_config(self):
        # Get FPModule layer configuration
        config = {
            'mlp': self.mlp,
            'use_xyz': self.use_xyz,
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

        # Select state input
        encoding_shape = input_shape[0]
        state_shape = input_shape[1]

        # Get intermediate new_points shape
        if self.data_format == 'channels_last':
            new_points_shape = tf.TensorShape([state_shape[0], state_shape[self.ndataset_axis], encoding_shape[self.chan_axis] + state_shape[self.chan_axis] - 3])
        else:
            new_points_shape = tf.TensorShape([state_shape[0], encoding_shape[self.chan_axis] + state_shape[self.chan_axis] - 3, state_shape[self.ndataset_axis]])

        # Get number of output channels
        if self.mlp and self.use_xyz:
            nbr_out_chan = self.mlp_list[-1].compute_output_shape(new_points_shape)[self.chan_axis] + 3
        elif self.mlp:
            nbr_out_chan = self.mlp_list[-1].compute_output_shape(new_points_shape)[self.chan_axis]
        elif self.use_xyz:
            nbr_out_chan = new_points_shape[self.chan_axis]
        else:
            nbr_out_chan = new_points_shape[self.chan_axis] - 3

        # Get output shapes
        if self.data_format == 'channels_last':
            new_points_shape = tf.TensorShape([state_shape[0], state_shape[self.ndataset_axis], nbr_out_chan])
            xyz1_shape = tf.TensorShape([state_shape[0], state_shape[self.ndataset_axis], 3])

        else:
            new_points_shape = tf.TensorShape([state_shape[0], nbr_out_chan, state_shape[self.ndataset_axis]])
            xyz1_shape = tf.TensorShape([state_shape[0], 3, state_shape[self.ndataset_axis]])

        return (new_points_shape, xyz1_shape)


class OutputModule(layers.Layer):
    """
    Output module of the PointNet++ architecture.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        num_classes: Number of different output classes/labels, <int>.
        mlp: List of dicts to define the attributes of the MLP layers (length of the list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        dropout: List of dicts to define the attributes of the dropout layers (length of list represents the number of dropout layers), <List[Dict, ... , Dict]>.
        out_activation: Activation function of the output layer.
        out_use_bias: Whether to use a bias vector within the output layer, <bool>.
        out_kernel_initializer: Initializer for the output layer kernel weights matrix.
        out_bias_initializer: Initializer for the output layer bias vector.
        out_kernel_regularizer: Regularizer function applied to the output layer kernel weights matrix.
        out_bias_regularizer: Regularizer function applied to the output layer bias vector.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        prediction: Point class labels (batch_size, ndataset, num_classes), if channels_last, <tf.Tensor>.
                    Point class labels (batch_size, num_classes, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 num_classes=1,
                 mlp=None,
                 dropout=None,
                 out_activation=None,
                 out_use_bias=True,
                 out_kernel_initializer='glorot_uniform',
                 out_bias_initializer='zeros',
                 out_kernel_regularizer=None,
                 out_bias_regularizer=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(OutputModule, self).__init__(name=name, **kwargs)

        # Initialize OutputModule layer
        self.num_classes = num_classes
        self.mlp = mlp if mlp is not None else []
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
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1
        self.exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.mlp_list = []
        self.dropout_list = []

        # Pass module data_format setting to all child layers
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]

        # Adjust the length of the dropout layers to match with the mlp layers
        self.dropout += [None] * (len(self.mlp) - len(self.dropout))

    def build(self, input_shape):
        # Build MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # Build dropout layer
        for dropout_setting in self.dropout:
            if dropout_setting is not None:
                self.dropout_list.append(layers.Dropout(**dropout_setting))
            else:
                self.dropout_list.append(None)

        # Build output layer
        self.out_layer = layers.Conv2D(filters=self.num_classes, kernel_size=[1, 1], strides=[1, 1], padding='valid',
                                       data_format=self.data_format, activation=self.out_activation, use_bias=self.out_use_bias,
                                       kernel_initializer=self.out_kernel_initializer, bias_initializer=self.out_bias_initializer,
                                       kernel_regularizer=self.out_kernel_regularizer, bias_regularizer=self.out_bias_regularizer)

    def call(self, inputs):
        # Process layer inputs
        prediction = tf.keras.backend.expand_dims(inputs, axis=self.exp_axis)

        for mlp_layer, dropout_layer in zip(self.mlp_list, self.dropout_list):
            prediction = mlp_layer(prediction)
            if dropout_layer is not None:
                prediction = dropout_layer(prediction)

        # Compute output
        prediction = self.out_layer(prediction)

        prediction = tf.keras.backend.squeeze(prediction, axis=self.exp_axis)

        return prediction

    def get_config(self):
        # Get Decoder OutputModule configuration
        config = {
            'num_classes': self.num_classes,
            'mlp': self.mlp,
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
        base_config = super(OutputModule, self).get_config()

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


class Encoder(layers.Layer):
    """
    Encoder of the PointNet++ architecture.

    Returns an encoded representation of the given point cloud and its internal states.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        sa: List of dicts to define the attributes of the SAModule layers (length of list represents the number of SA modules), <List[Dict, ... , Dict]>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        encoding: Encoded point cloud (batch_size, ndataset or 1, sa[-1] new_points + 3 or 3 + channels), if channels_last, <tf.Tensor>.
                  Encoded point cloud (batch_size, sa[-1] new_points + 3 or 3 + channels, ndataset or 1), if channels_first, <tf.Tensor>.
        state: Input of the last encoding layer/SA module (batch_size, ndataset or 1, sa[-2] new_points + 3 or 3 + channels), if channels_last, <tf.Tensor>.
               Input of the last encoding layer/SA module (batch_size, sa[-2] new_points + 3 or 3 + channels, ndataset or 1), if channels_first, <tf.Tensor>.
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

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.sa_list = []
        self.num_sa_modules = len(self.sa)

    def build(self, input_shape):
        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        else:
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build concatenate layer
        self.concat = layers.Concatenate(axis=self.chan_axis)

        # Build SA layers
        for sa_setting in self.sa:
            self.sa_list.append(SAModule(**sa_setting))

    def call(self, inputs):
        # Create list of internal states
        states = []

        # Split input in point coordinates and point features
        new_xyz = self.lambda1(inputs)
        new_points = self.lambda2(inputs)

        # Concatenate input state
        state = self.concat([new_xyz, new_points])

        # Encode point cloud
        for i, sa_layer in enumerate(self.sa_list):
            new_xyz, new_points = sa_layer(state)

            # Set internal states
            if i < self.num_sa_modules - 1:
                # Fill states list from the right
                states.append([self.concat([new_xyz, new_points]), state])

                # Output as new SA module input
                state = self.concat([new_xyz, new_points])

        # Concatenate encoded point cloud
        encoding = self.concat([new_xyz, new_points])

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
                new_xyz_shape, new_point_shape = sa_layer.compute_output_shape(state_shape)
                encoding_shape = self.concat.compute_output_shape([new_xyz_shape, new_point_shape])
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
    Decoder of the PointNet++ architecture.

    Returns an decoded representation of the given point cloud and its corresponding origin.

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
        new_points  Interpolated point features (batch_size, ndataset2, fp[-1] or channels2), if channels_last, <tf.Tensor>.
                    Interpolated point features (batch_size, fp[-1] or channels2, ndataset2), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 fp=None,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(Decoder, self).__init__(name=name, **kwargs)

        # Initialize Decoder layer
        self.fp = fp if fp is not None else list()
        self.data_format = data_format

        # Checks
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Pass encoder data_format setting to all SAModule layers
        _ = [fp_setting.update({'data_format': self.data_format}) for fp_setting in self.fp]

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.fp_list = []

    def build(self, input_shape):
        # Check input_shape
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) < 2 or len(input_shape) > 3:
                raise ValueError('The Decoder layer requires an input list/tuple of 2 or 3 inputs, but {} where given.'.format(len(input_shape)))
        else:
            raise ValueError('The Decoder layer input has to be a list or tuple of inputs.')

        # Check input and initialization compatibility
        if len(self.fp) > 1 and len(input_shape) != 3:
            raise ValueError('A Decoder configuration with more than one FP module requires the provision of the states input. {} != 3'.format(len(input_shape)))

        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        else:
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build concatenate layer
        self.concat = layers.Concatenate(axis=self.chan_axis)

        # Build FP layers
        for fp_setting in self.fp:
            self.fp_list.append(FPModule(**fp_setting))

    def call(self, inputs):
        # Get inputs
        encoding = inputs[0]
        state = inputs[1]

        # Get additional encoder states
        if len(inputs) == 3:
            states = inputs[2]
            assert len(states) == max(len(self.fp) - 1, 0)
            states.append([None, None])
        else:
            states = [[None, None]]

        # Split inputs
        new_xyz = self.lambda1(encoding)
        new_points = self.lambda2(encoding)

        # Encode point cloud
        for i, fp_layer in enumerate(self.fp_list):
            new_points, new_xyz = fp_layer([self.concat([new_xyz, new_points]), state])
            state = states[i][1]

        return new_points

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
        encoding_shape = input_shape[0]
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
            new_xyz_shape = self.lambda1.compute_output_shape(encoding_shape)
            new_points_shape = self.lambda2.compute_output_shape(encoding_shape)

            for i, fp_layer in enumerate(self.fp_list):
                encoding_shape = self.concat.compute_output_shape([new_xyz_shape, new_points_shape])
                new_points_shape, new_xyz_shape = fp_layer.compute_output_shape([encoding_shape, state_shape])
                state_shape = states_shape[i][1]

            return new_points_shape

        else:
            return self.lambda2.compute_output_shape(encoding_shape)


class PointNet2(layers.Layer):
    """
    PointNet++

    Example implementation of the PointNet++ semantic segmentaion model as defined by Charles R. Qi.
    https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        num_classes: Number of different output classes/labels, <int>.
        seed: Random seed of the dropout layer, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        prediction: Point class labels (batch_size, ndataset, num_classes), if channels_last, <tf.Tensor>.
                    Point class labels (batch_size, num_classes, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 num_classes=1,
                 seed=42,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(PointNet2, self).__init__(name=name, **kwargs)

        # Initialize PointNet2 layer
        self.num_classes = num_classes
        self.seed = seed
        self.data_format = data_format

        # Checks
        assert self.num_classes > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.chan_axis = 2 if self.data_format == 'channels_last' else 1
        self.exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1

    def build(self, input_shape):
        # Build concatenate layer
        self.concat = layers.Concatenate(axis=self.chan_axis)

        # Build encoder (SA modules)
        self.encoder = Encoder(
            sa=[
                {'npoint': 1024, 'radius': 0.1, 'nsample': 32, 'mlp': [{'filters': 32, 'bn': True}, {'filters': 32, 'bn': True}, {'filters': 64, 'bn': True}], 'data_format': self.data_format, 'name': 'sa_module_0'},
                {'npoint': 256, 'radius': 0.2, 'nsample': 32, 'mlp': [{'filters': 64, 'bn': True}, {'filters': 64, 'bn': True}, {'filters': 128, 'bn': True}], 'data_format': self.data_format, 'name': 'sa_module_1'},
                {'npoint': 64, 'radius': 0.4, 'nsample': 32, 'mlp': [{'filters': 128, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 256, 'bn': True}], 'data_format': self.data_format, 'name': 'sa_module_2'},
                {'npoint': 16, 'radius': 0.8, 'nsample': 32, 'mlp': [{'filters': 256, 'bn': True}, {'filters': 256, 'bn': True}, {'filters': 512, 'bn': True}], 'data_format': self.data_format, 'name': 'sa_module_3'}
            ],
            data_format=self.data_format
        )

        # Build decoder (FP modules)
        self.decoder = Decoder(
            fp=[
                {'mlp': [{'filters': 256, 'bn': True}, {'filters': 256, 'bn': True}], 'data_format': self.data_format, 'name': 'fp_module_0'},
                {'mlp': [{'filters': 256, 'bn': True}, {'filters': 256, 'bn': True}], 'data_format': self.data_format, 'name': 'fp_module_1'},
                {'mlp': [{'filters': 256, 'bn': True}, {'filters': 128, 'bn': True}], 'data_format': self.data_format, 'name': 'fp_module_2'},
                {'mlp': [{'filters': 128, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 128, 'bn': True}], 'data_format': self.data_format, 'name': 'fp_module_3'}
            ],
            data_format=self.data_format
        )

        # Build output layers
        self.mlp_0 = MLP(filters=128, bn=True, data_format=self.data_format, name='mlp_0')
        self.dropout = layers.Dropout(rate=0.5, seed=self.seed, name='dropout')
        self.mlp_1 = MLP(filters=self.num_classes, activation=None, data_format=self.data_format, name='mlp_1')

        # Call build function of the keras base layer
        super(PointNet2, self).build(input_shape)

    def call(self, inputs):
        # Encoding
        encoding, state, states = self.encoder(inputs)

        # Decoding
        prediction = self.decoder([encoding, state, states])

        # Output
        prediction = tf.keras.backend.expand_dims(prediction, axis=self.exp_axis)

        prediction = self.mlp_0(prediction)
        prediction = self.dropout(prediction)
        prediction = self.mlp_1(prediction)

        prediction = tf.keras.backend.squeeze(prediction, axis=self.exp_axis)

        return prediction

    def get_config(self):
        # Get PointNet2 layer configuration
        config = {
            'num_classes': self.num_classes,
            'seed': self.seed,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(PointNet2, self).get_config()

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

        # Get output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[self.ndataset_axis], self.num_classes])
        else:
            output_shape = tf.TensorShape([input_shape[0], self.num_classes, input_shape[self.ndataset_axis]])

        return output_shape
