"""
PointNet Model Architecture

Author: Charles R. Qi
Modified by: Felix Fent
Date: May 2020

References:
    - Qi, Charles R., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.
      [Online] Available: https://arxiv.org/abs/1612.00593, 2016.
"""
# 3rd Party Libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Local imports
from radarseg.model.layer.mlp import MLP


class InputTransformNet(layers.Layer):
    """
    Input transformation network.

    Returns a 3 by 3 transformation matrix based on the encoded point coordinates (xyz).

    Note: A minimum of one mlp layer has to be defined.

    Inputs:
        inputs: Point cloud of 3 point coordinates (batch_size, ndataset, 3), if channels_last, <tf.Tensor>.
                Point cloud of 3 point coordinates (batch_size, 3, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        mlp: List of dicts to define the attributes of the MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        dense: List of dicts to define the attributes of the Dense layers (length of list represents the number of Dense layers), <List[Dict, ... , Dict]>.
        batch_norm: List of dicts to define the attributes of the BatchNormalization layers (length of list represents the number of BatchNormalization layers), <List[Dict, ... , Dict]>.
        pooling: Pooling operation to encode the point coordinates, one of either 'max' or 'avg', <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        trans_mat: Transformation matrix for the point coordinates (batch_size, 3, 3), <tf.Tensor>.
    """
    def __init__(self,
                 mlp=[{'filters': 1}],
                 dense=[],
                 batch_norm=[],
                 pooling='max',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(InputTransformNet, self).__init__(name=name, **kwargs)

        # Initialize InputTransformNet layer
        self.mlp = mlp
        self.dense = dense
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.data_format = data_format

        # Checks
        assert self.pooling in set(('max', 'avg'))
        assert self.data_format in set(('channels_last', 'channels_first'))
        assert self.mlp

        # Set layer attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.mlp_list = []
        self.dense_list = []
        self.batch_norm_list = []

        # Pass module data_format setting to all mlp layers
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]
        _ = [batch_norm_setting.update({'axis': self.chan_axis}) for batch_norm_setting in self.batch_norm if batch_norm_setting is not None]

        # Set kernel of the first mlp layer
        if self.mlp and data_format == 'channels_last':
            self.mlp[0].update({'kernel_size': [1, 3]})
        elif self.mlp and data_format == 'channels_first':
            self.mlp[0].update({'kernel_size': [3, 1]})

        # Adjust the length of the batch_norm layers to match with the dense layers
        self.batch_norm += [None] * (len(self.dense) - len(self.batch_norm))

    def build(self, input_shape):
        # Build MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # Build pooling layer
        if self.pooling == 'max':
            self.pooling_layer = layers.GlobalMaxPool2D(data_format=self.data_format)
        else:
            self.pooling_layer = layers.GlobalAveragePooling2D(data_format=self.data_format)

        # Build dense layer
        for dense_setting, batch_norm_setting in zip(self.dense, self.batch_norm):
            self.dense_list.append(layers.Dense(**dense_setting))
            if batch_norm_setting is not None:
                self.batch_norm_list.append(layers.BatchNormalization(**batch_norm_setting))
            else:
                self.batch_norm_list.append(None)

        # Add weights
        if self.dense:
            shape = (self.dense[-1]['units'], 9)
        elif self.mlp:
            shape = (self.mlp[-1]['filters'], 9)
        else:
            shape = (input_shape[self.chan_axis], 9)

        self.w = self.add_weight(name='w', shape=shape, initializer='zeros', trainable=True)
        self.b = self.add_weight(name='b', shape=(3, 3), initializer='identity', trainable=True)

        # Build reshape layer
        self.reshape = layers.Reshape(target_shape=(3, 3))

        # Call build function of the keras base layer
        super(InputTransformNet, self).build(input_shape)

    def call(self, inputs):
        # Expand input dimensions (to match the input format of the mlp layer)
        trans_mat = K.expand_dims(inputs, axis=self.chan_axis)

        # Point coordinate encoding
        for mlp_layer in self.mlp_list:
            trans_mat = mlp_layer(trans_mat)

        # Pool encoded point channels (coordinates)
        trans_mat = self.pooling_layer(trans_mat)

        # Process encoded information
        for dense_layer, batch_norm_layer in zip(self.dense_list, self.batch_norm_list):
            trans_mat = dense_layer(trans_mat)
            if batch_norm_layer is not None:
                trans_mat = batch_norm_layer(trans_mat)

        # Apply transformation matrix
        trans_mat = K.dot(trans_mat, self.w)
        trans_mat = K.bias_add(trans_mat, K.flatten(self.b))

        # Adjust output format
        trans_mat = self.reshape(trans_mat)

        return trans_mat

    def get_config(self):
        # Get InputTransformNet layer configuration
        config = {
            'mlp': self.mlp,
            'dense': self.dense,
            'batch_norm': self.batch_norm,
            'pooling': self.pooling,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(InputTransformNet, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A shape tensor, <tf.TensorShape>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        return tf.TensorShape([input_shape[0], 3, 3])


class FeatureTransformNet(layers.Layer):
    """
    Feature transformation network.

    Returns a transformation matrix of the size of the channel axis based on the encoded point features.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        mlp: List of dicts to define the attributes of the MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        dense: List of dicts to define the attributes of the Dense layers (length of list represents the number of Dense layers), <List[Dict, ... , Dict]>.
        batch_norm: List of dicts to define the attributes of the BatchNormalization layers (length of list represents the number of BatchNormalization layers), <List[Dict, ... , Dict]>.
        pooling: Pooling operation to encode the point features, one of either 'max' or 'avg', <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        trans_mat: Transformation matrix for the point coordinates (batch_size, 3 + channels, 3 + channels), <tf.Tensor>.
    """
    def __init__(self,
                 mlp=[],
                 dense=[],
                 batch_norm=[],
                 pooling='max',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(FeatureTransformNet, self).__init__(name=name, **kwargs)

        # Initialize FeatureTransformNet layer
        self.mlp = mlp
        self.dense = dense
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.data_format = data_format

        # Checks
        assert self.pooling in set(('max', 'avg'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.mlp_list = []
        self.dense_list = []
        self.batch_norm_list = []

        # Pass module data_format setting to all mlp layers
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]
        _ = [batch_norm_setting.update({'axis': self.chan_axis}) for batch_norm_setting in self.batch_norm if batch_norm_setting is not None]

        # Adjust the length of the batch_norm layers to match with the dense layers
        self.batch_norm += [None] * (len(self.dense) - len(self.batch_norm))

    def build(self, input_shape):
        # Build MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # Build pooling layer
        if self.pooling == 'max':
            self.pooling_layer = layers.GlobalMaxPool2D(data_format=self.data_format)
        else:
            self.pooling_layer = layers.GlobalAveragePooling2D(data_format=self.data_format)

        # Build dense layer
        for dense_setting, batch_norm_setting in zip(self.dense, self.batch_norm):
            self.dense_list.append(layers.Dense(**dense_setting))
            if batch_norm_setting is not None:
                self.batch_norm_list.append(layers.BatchNormalization(**batch_norm_setting))
            else:
                self.batch_norm_list.append(None)

        # Add weights
        if self.dense:
            shape = (self.dense[-1]['units'], input_shape[self.chan_axis] * input_shape[self.chan_axis])
        elif self.mlp:
            shape = (self.mlp[-1]['filters'], input_shape[self.chan_axis] * input_shape[self.chan_axis])
        else:
            shape = (input_shape[self.chan_axis], input_shape[self.chan_axis] * input_shape[self.chan_axis])

        self.w = self.add_weight(name='w', shape=shape, initializer='zeros', trainable=True)
        self.b = self.add_weight(name='b', shape=(input_shape[self.chan_axis], input_shape[self.chan_axis]), initializer='identity', trainable=True)

        # Build reshape layer
        self.reshape = layers.Reshape(target_shape=(input_shape[self.chan_axis], input_shape[self.chan_axis]))

        # Call build function of the keras base layer
        super(FeatureTransformNet, self).build(input_shape)

    def call(self, inputs):
        # Expand input dimensions (to match the input format of the mlp layer)
        trans_mat = K.expand_dims(inputs, axis=self.exp_axis)

        # Point coordinate encoding
        for mlp_layer in self.mlp_list:
            trans_mat = mlp_layer(trans_mat)

        # Pool encoded point channels (coordinates)
        trans_mat = self.pooling_layer(trans_mat)

        # Process encoded information
        for dense_layer, batch_norm_layer in zip(self.dense_list, self.batch_norm_list):
            trans_mat = dense_layer(trans_mat)
            if batch_norm_layer is not None:
                trans_mat = batch_norm_layer(trans_mat)

        # Apply transformation matrix
        trans_mat = K.dot(trans_mat, self.w)

        trans_mat = K.bias_add(trans_mat, K.flatten(self.b))

        # Adjust output format
        trans_mat = self.reshape(trans_mat)

        return trans_mat

    def get_config(self):
        # Get FeatureTransformNet layer configuration
        config = {
            'mlp': self.mlp,
            'dense': self.dense,
            'batch_norm': self.batch_norm,
            'pooling': self.pooling,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(FeatureTransformNet, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple or shape list or shape tensor, <tuple or list or tf.TensorShape>.

        Returns:
            output_shape: A shape tensor, <tf.TensorShape>.
        """
        # Convert input shape
        input_shape = layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        return tf.TensorShape([input_shape[0], input_shape[self.chan_axis], input_shape[self.chan_axis]])


class Encoder(layers.Layer):
    """
    Encoder of the PointNet architecture.

    Returns a global feature vector as encoded representation of the given point cloud.

    Performance advices:
    The performance of this layer is best for the data_format 'channels_last'.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        input_trans: Dict to define the attributes of the input transformation network, <Dict>.
        mlp: List of dicts to define the attributes of the first MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        feature_trans: Dict to define the attributes of the feature transformation network, <Dict>.
        mlp2: List of dicts to define the attributes of the second MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        pooling: Pooling operation to encode the point cloud, one of either 'max' or 'avg', <str>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        glob_feat: Global feature vector of the point cloud (batch_size, 1, mlp2[-1] or mlp[-1] or 3 + channels), if channels_last, <tf.Tensor>.
                   Global feature vector of the point cloud (batch_size, mlp2[-1] or mlp[-1] or 3 + channels, 1), if channels_first, <tf.Tensor>.
        inputs: Pass through of the input values - see input, <tf.Tensor>.
    """
    def __init__(self,
                 input_trans={},
                 mlp=[{'filters': 1}],
                 feature_trans={},
                 mlp2=[],
                 pooling='max',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(Encoder, self).__init__(name=name, **kwargs)

        # Initialize Encoder layer
        self.input_trans = input_trans
        self.mlp = mlp
        self.feature_trans = feature_trans
        self.mlp2 = mlp2
        self.pooling = pooling
        self.data_format = data_format

        # Checks
        assert self.mlp
        assert self.pooling in set(('max', 'avg'))
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.chan_axis = -1 if self.data_format == 'channels_last' else 1
        self.exp_axis = 2 if self.data_format == 'channels_last' else 3
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1
        self.col_axis = 2
        self.mlp_list = []
        self.mlp2_list = []

        # Pass module data_format setting to all child layers
        self.input_trans.update({'data_format': self.data_format})
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp]
        self.feature_trans.update({'data_format': self.data_format})
        _ = [mlp_setting.update({'data_format': self.data_format}) for mlp_setting in self.mlp2]

    def build(self, input_shape):
        # Build split layer (split input in point coordinates and point features)
        if self.data_format == 'channels_last':
            self.lambda1 = layers.Lambda(lambda x: x[:, :, 0:3])
            self.lambda2 = layers.Lambda(lambda x: x[:, :, 3:])
        elif self.data_format == 'channels_first':
            self.lambda1 = layers.Lambda(lambda x: x[:, 0:3, :])
            self.lambda2 = layers.Lambda(lambda x: x[:, 3:, :])

        # Build permutation (transponation) layer
        if self.data_format == 'channels_first':
            self.permut = layers.Permute(dims=(2, 1))

        # Build input transformation layer
        self.input_trans_net = InputTransformNet(**self.input_trans)

        # Build concatenation layer
        self.concat = layers.Concatenate(axis=self.chan_axis)

        # Set kernel size of the first mlp layer
        if self.data_format == 'channels_last':
            self.mlp[0].update({'kernel_size': [1, input_shape[self.chan_axis]]})
        else:
            self.mlp[0].update({'kernel_size': [input_shape[self.chan_axis], 1]})

        # Build first MLP layer
        for mlp_setting in self.mlp:
            self.mlp_list.append(MLP(**mlp_setting))

        # Build feature transformation layer
        self.feat_trans_net = FeatureTransformNet(**self.feature_trans)

        # Build second MLP layer
        for mlp_setting in self.mlp2:
            self.mlp2_list.append(MLP(**mlp_setting))

        # Build pooling layer
        if self.pooling == 'max':
            self.pooling_layer = layers.GlobalMaxPool2D(data_format=self.data_format)
        else:
            self.pooling_layer = layers.GlobalAveragePooling2D(data_format=self.data_format)

        # Call build function of the keras base layer
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        # Split input in point coordinates and point features
        xyz = self.lambda1(inputs)
        features = self.lambda2(inputs)

        # Transform input coordinates
        input_trans_mat = self.input_trans_net(xyz)
        if self.data_format == 'channels_first':
            xyz = self.permut(xyz)

        xyz = tf.linalg.matmul(xyz, input_trans_mat)

        if self.data_format == 'channels_first':
            xyz = self.permut(xyz)

        # Concatenate transformed point coordinates and point features
        glob_feat = self.concat([xyz, features])

        # Encode point cloud
        glob_feat = K.expand_dims(glob_feat, axis=self.chan_axis)

        for mlp_layer in self.mlp_list:
            glob_feat = mlp_layer(glob_feat)

        glob_feat = K.squeeze(glob_feat, axis=self.col_axis)

        # Transform encoded point features
        feat_trans_mat = self.feat_trans_net(glob_feat)
        if self.data_format == 'channels_first':
            glob_feat = self.permut(glob_feat)

        glob_feat = tf.linalg.matmul(glob_feat, feat_trans_mat)

        if self.data_format == 'channels_first':
            glob_feat = self.permut(glob_feat)

        # Encode point cloud
        glob_feat = K.expand_dims(glob_feat, axis=self.exp_axis)

        for mlp_layer in self.mlp2_list:
            glob_feat = mlp_layer(glob_feat)

        # Pool point cloud to global feature vector
        glob_feat = self.pooling_layer(glob_feat)
        glob_feat = K.expand_dims(glob_feat, axis=self.ndataset_axis)

        return glob_feat, inputs

    def get_config(self):
        # Get Encoder layer configuration
        config = {
            'input_trans': self.input_trans,
            'mlp': self.mlp,
            'feature_trans': self.feature_trans,
            'mlp2': self.mlp2,
            'pooling': self.pooling,
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
        if self.mlp2:
            nbr_out_chan = self.mlp2_list[-1].compute_output_shape(input_shape)[self.chan_axis]
        elif self.mlp:
            nbr_out_chan = self.mlp_list[-1].compute_output_shape(input_shape)[self.chan_axis]
        else:
            nbr_out_chan = input_shape[self.chan_axis]

        if self.data_format == 'channels_last':
            glob_feat_shape = tf.TensorShape([input_shape[0], 1, nbr_out_chan])
        else:
            glob_feat_shape = tf.TensorShape([input_shape[0], nbr_out_chan, 1])

        return (glob_feat_shape, input_shape)


class Decoder(layers.Layer):
    """
    Decoder of the PointNet architecture.

    Returns a class label for every point in the given point cloud in consideration of the provided global feature vector.

    Inputs:
        inputs: Either a list/tuple of exactly two input tensors or a single input tensor, <List[tf.Tensor, tf.Tensor] or tf.Tensor>.
                Tensor input:
                    Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last.
                    Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first.
                List/Tuple input:
                    Gloabel feature vector and point cloud [(batch_size, 1, feat), (batch_size, ndataset, 3 + channels)], if channels_last.
                    Gloabel feature vector and point cloud [(batch_size, feat, 1), (batch_size, 3 + channels, ndataset)], if channels_first.

    Arguments:
        num_classes: Number of different output classes/labels, <int>.
        mlp: List of dicts to define the attributes of the first MLP layers (length of list represents the number of MLP layers), <List[Dict, ... , Dict]>.
        dropout: List of dicts to define the attributes of the dropout layers (length of list represents the number of dropout layers), <List[Dict, ... , Dict]>.
        out_activation: Activation function of the output layer.
        out_use_bias: Whether to use a bias vector within the output layer, <bool>.
        out_kernel_initializer: Initializer for the output layer kernel weights matrix.
        out_bias_initializer: Initializer for the output layer bias vector.
        out_kernel_regularizer: Regularizer function applied to the output layer kernel weights matrix.
        out_bias_regularizer: Regularizer function applied to the output layer bias vector.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Returns:
        prediction: Global feature vector of the point cloud (batch_size, ndataset, num_class), if channels_last, <tf.Tensor>.
                    Global feature vector of the point cloud (batch_size, num_class, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 num_classes=1,
                 mlp=[],
                 dropout=[],
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
        super(Decoder, self).__init__(name=name, **kwargs)

        # Initialize Decoder layer
        self.num_classes = num_classes
        self.mlp = mlp
        self.dropout = dropout
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
        # Check input shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) != 2:
            raise ValueError('A Decoder layer should be called on a list of exactly 2 inputs or a single tensor.')

        # Build concatenation layer
        self.concat = layers.Concatenate(axis=self.chan_axis)

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
        # Determine layer inputs (only point cloud (tensor) or point cloud and global feature vector (list))
        if not isinstance(inputs, (list, tuple)):
            prediction = inputs

        else:
            # Determine number of points (ndataset)
            ndataset = tf.shape(inputs[1])[self.ndataset_axis]

            # Expand the dimensions of the global feature vector ([None, feat] -> [None, ndataset, feat])
            if self.data_format == 'channels_last':
                new_shape = K.concatenate([[1], [ndataset], [1]], axis=0)
            else:
                new_shape = K.concatenate([[1], [1], [ndataset]], axis=0)

            glob_feat = K.tile(inputs[0], new_shape)

            # Concatenate the point cloud and the global feature vector
            prediction = self.concat([inputs[1], glob_feat])

        # Process layer inputs
        prediction = K.expand_dims(prediction, axis=self.exp_axis)

        for mlp_layer, dropout_layer in zip(self.mlp_list, self.dropout_list):
            prediction = mlp_layer(prediction)
            if dropout_layer is not None:
                prediction = dropout_layer(prediction)

        # Compute decoder output
        prediction = self.out_layer(prediction)

        prediction = K.squeeze(prediction, axis=self.exp_axis)

        return prediction

    def get_config(self):
        # Get Decoder layer configuration
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
        base_config = super(Decoder, self).get_config()

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

        # Select state input
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]

        # Get output shape
        if self.data_format == 'channels_last':
            output_shape = tf.TensorShape([input_shape[0], input_shape[self.ndataset_axis], self.num_classes])
        else:
            output_shape = tf.TensorShape([input_shape[0], self.num_classes, input_shape[self.ndataset_axis]])

        return output_shape


class PointNet(layers.Layer):
    """
    PointNet

    Example implementation of the PointNet semantic segmentaion model as defined by Charles R. Qi.
    https://github.com/charlesq34/pointnet/blob/master/models/pointnet_seg.py

    Performance advices:
    The performance of PointNet layer is best for the data_format 'channels_last'.

    Inputs:
        inputs: Point cloud with 3 coordinates and additional channels (batch_size, ndataset, 3 + channels), if channels_last, <tf.Tensor>.
                Point cloud with 3 coordinates and additional channels (batch_size, 3 + channels, ndataset), if channels_first, <tf.Tensor>.

    Arguments:
        num_class: Number of different output classes/labels, <int>.
        data_format: Whether the channles are the first or the last dimension of the input tensor (except for the batch size), <str>.

    Return:
        prediction: Point class labels (batch_size, ndataset, num_class), if channels_last, <tf.Tensor>.
                    Point class labels (batch_size, num_class, ndataset), if channels_first, <tf.Tensor>.
    """
    def __init__(self,
                 num_classes=1,
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        # Initialize keras base layer
        super(PointNet, self).__init__(name=name, **kwargs)

        # Initialize PointNet layer
        self.num_classes = num_classes
        self.data_format = data_format

        # Checks
        assert self.num_classes > 0
        assert self.data_format in set(('channels_last', 'channels_first'))

        # Set layer attributes
        self.ndataset_axis = 1 if self.data_format == 'channels_last' else -1

    def build(self, input_shape):
        # Build encoder layer
        self.encoder = Encoder(
            input_trans={'mlp': [{'filters': 64, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 1024, 'bn': True}], 'dense': [{'units': 512, 'activation': 'relu'}, {'units': 256, 'activation': 'relu'}], 'batch_norm': [{'momentum': 0.9}, {'momentum': 0.9}]},
            mlp=[{'filters': 64, 'kernel_size': [1, 3], 'bn': True}, {'filters': 64, 'bn': True}],
            feature_trans={'mlp': [{'filters': 64, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 1024, 'bn': True}], 'dense': [{'units': 512, 'activation': 'relu'}, {'units': 256, 'activation': 'relu'}], 'batch_norm': [{'momentum': 0.9}, {'momentum': 0.9}]},
            mlp2=[{'filters': 64, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 1024, 'bn': True}],
            pooling='max',
            data_format=self.data_format,
            name='decoder'
        )

        # Build decoder layer
        self.decoder = Decoder(
            num_classes=self.num_classes,
            mlp=[{'filters': 512, 'bn': True}, {'filters': 256, 'bn': True}, {'filters': 128, 'bn': True}, {'filters': 128, 'bn': True}],
            out_activation=None,
            out_use_bias=False,
            data_format=self.data_format,
            name='encoder'
        )

    def call(self, inputs):
        # Encode point cloud
        glob_feat, state = self.encoder(inputs)

        # Decode point features and determine class labels
        prediction = self.decoder([glob_feat, state])

        return prediction

    def get_config(self):
        # Get PointNet layer configuration
        config = {
            'num_classes': self.num_classes,
            'data_format': self.data_format
        }

        # Get keras base layer configuration
        base_config = super(PointNet, self).get_config()

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
