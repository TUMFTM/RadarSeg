"""
Configuration specifications for all models handled by the ConfigFileHandler.

This validation file containes one class for each model that is handled
(checkt) by the config file handler. Therefore, each class contains a
static method for each layer used within the particular model. This
static method returns the config specification of this layer in the
context of the particular model (class). The config specification is
defined as a dict, whereas the dict keys correspond with the layer
attributes. The value (of the key value pair) defines the datatype,
boundary values and default value of the layer attribute (key).

The model validation class can also contain additional static methods to
specify the elements of complex attribute values like lists or tuples.
These static methods provid a check of the validator module or a custom
validation functions - defined in the config file handler.

Note: Each configspec dict has to contain the associated model name as
its first key-value-pair, like so: 'model': name.

Hint: All permissible values of the configspec dicts can be found in the
documentation of the validate module, except for the dict validation
function, which is defined in the ConfigFileHandler class.

References:
ConfigObj documentation: https://configobj.readthedocs.io/en/latest/configobj.html#
Validate documentation: http://www.voidspace.org.uk/python/validate.html

Author: Felix Fent
Date: June 2020
"""


class validate_pointnet():
    """
    Configuration specifications of all PointNet layers.
    """
    @staticmethod
    def validate_pointnet():
        configspecs = {
            'module': "option('pointnet', default='pointnet')",
            'layer': "option('PointNet', default='PointNet')",
            'num_classes': "integer(min=1, default=1)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_encoder():
        configspecs = {
            'module': "option('pointnet', default='pointnet')",
            'layer': "option('Encoder', default='Encoder')",
            'input_trans': "dict(default=dict())",
            'mlp': "dict_list(min=1, default=list('dict()'))",
            'feature_trans': "dict(default=dict())",
            'mlp2': "dict_list(default=list())",
            'pooling': "option('max', 'avg', default='max')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_decoder():
        configspecs = {
            'module': "option('pointnet', default='pointnet')",
            'layer': "option('Decoder', default='Decoder')",
            'num_classes': "integer(min=1, default=1)",
            'mlp': "dict_list(default=list())",
            'dropout': "dict_list(default=list())",
            'out_activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'out_use_bias': "boolean(default=True)",
            'out_kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'out_bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'out_kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'out_bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_input_trans():
        configspecs = {
            'mlp': "dict_list(min=1, default=list('dict()'))",
            'dense': "dict_list(default=list())",
            'batch_norm': "dict_list(default=list())",
            'pooling': "option('max', 'avg', default='max')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_mlp():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'kernel_size': "int_list(min=2, max=2, default=list('1', '1'))",
            'strides': "int_list(min=2, max=2, default=list('1', '1'))",
            'padding': "option('valid', 'same', default='valid')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bn': "boolean(default=True)",
            'momentum': "float(default=0.9)",
            'epsilon': "float(default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_feature_trans():
        configspecs = {
            'mlp': "dict_list(default=list())",
            'dense': "dict_list(default=list())",
            'batch_norm': "dict_list(default=list())",
            'pooling': "option('max', 'avg', default='max')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_mlp2():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'kernel_size': "int_list(min=2, max=2, default=list('1', '1'))",
            'strides': "int_list(min=2, max=2, default=list('1', '1'))",
            'padding': "option('valid', 'same', default='valid')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bn': "boolean(default=True)",
            'momentum': "float(default=0.9)",
            'epsilon': "float(default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_dense():
        configspecs = {
            'units': "integer(min=1, default=1)",
            'activation': "string(default=None)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'activity_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'kernel_constraint': "string(default=None)",
            'bias_constraint': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_batch_norm():
        configspecs = {
            'axis': "integer(default=-1)",
            'momentum': "float(default=0.99)",
            'epsilon': "float(default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kernel_size():
        return "integer(min=1)"

    @staticmethod
    def validate_strides():
        return "integer(min=1)"


class validate_pointnet2():
    """
    Configuration specifications of all PointNet++ layers.
    """
    @staticmethod
    def validate_pointnet2():
        configspecs = {
            'module': "option('pointnet2', default='pointnet2')",
            'layer': "option('PointNet2', default='PointNet2')",
            'num_classes': "integer(min=1, default=1)",
            'seed': "integer(min=0, default=42)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_encoder():
        configspecs = {
            'module': "option('pointnet2', default='pointnet2')",
            'layer': "option('Encoder', default='Encoder')",
            'sa': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_decoder():
        configspecs = {
            'module': "option('pointnet2', default='pointnet2')",
            'layer': "option('Decoder', default='Decoder')",
            'fp': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_sa():
        configspecs = {
            'npoint': "integer(min=0, default=1)",
            'radius': "float(min=0.0, default=1.0)",
            'nsample': "integer(min=0, default=1)",
            'mlp': "dict_list(min=1, default=list('dict()'))",
            'mlp2': "dict_list(default=list())",
            'group_all': "boolean(default=False)",
            'pooling': "option('max', 'avg', default='max')",
            'knn': "boolean(default=False)",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_fp():
        configspecs = {
            'mlp': "dict_list(min=1, default=list('dict()'))",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_outputmodule():
        configspecs = {
            'module': "option('pointnet2', default='pointnet2')",
            'layer': "option('OutputModule', default='OutputModule')",
            'num_classes': "integer(min=1, default=1)",
            'mlp': "dict_list(default=list())",
            'dropout': "dict_list(default=list())",
            'out_activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'out_use_bias': "boolean(default=True)",
            'out_kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'out_bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'out_kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'out_bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_mlp():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'kernel_size': "int_list(min=2, max=2, default=list('1', '1'))",
            'strides': "int_list(min=2, max=2, default=list('1', '1'))",
            'padding': "option('valid', 'same', default='valid')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bn': "boolean(default=True)",
            'momentum': "float(default=0.9)",
            'epsilon': "float(default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_mlp2():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'kernel_size': "int_list(min=2, max=2, default=list('1', '1'))",
            'strides': "int_list(min=2, max=2, default=list('1', '1'))",
            'padding': "option('valid', 'same', default='valid')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'activation': "option('None', 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh', default=None)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bn': "boolean(default=True)",
            'momentum': "float(default=0.9)",
            'epsilon': "float(default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_dropout():
        configspecs = {
            'rate': "float(min=0, max=1, default=0.5)",
            'noise_shape': "int_list_or_none(default=None)",
            'seed': "integer_or_none(min=0, default=42)"
        }
        return configspecs

    @staticmethod
    def validate_kernel_size():
        return "integer(min=1)"

    @staticmethod
    def validate_strides():
        return "integer(min=1)"


class validate_kpfcnn():
    """
    Configuration specifications of all KPFCNN layers.
    """
    @staticmethod
    def validate_kpfcnn():
        configspecs = {
            'module': "option('kpfcnn', default='kpfcnn')",
            'layer': "option('KPFCNN', default='KPFCNN')",
            'num_classes': "integer(min=1, default=1)",
            'seed': "integer(min=0, default=1)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_outputblock():
        configspecs = {
            'module': "option('kpfcnn', default='kpfcnn')",
            'layer': "option('OutputBlock', default='OutputBlock')",
            'num_classes': "integer(min=0, default=1)",
            'kpconv': "dict_list(default=list())",
            'dropout': "dict_list(default=list())",
            'out_activation': "string(default=softmax)",
            'out_use_bias': "boolean(default=False)",
            'out_kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'out_bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'out_kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'out_bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kpconv():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=1)",
            'npoint': "integer_or_none(min=1, default=None)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'nsample': "integer(min=1, default=1)",
            'radius': "float(min=0, default=1.0)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'seed': "integer_or_none(min=0, default=42)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'bn': "boolean(default=False)",
            'momentum': "float(min=0, default=0.9)",
            'epsilon': "float(min=0, default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout_rate': "float(min=0, max=1, default=0.0)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_dropout():
        configspecs = {
            'rate': "float(min=0, max=1, default=0.5)",
            'noise_shape': "int_list_or_none(default=None)",
            'seed': "integer_or_none(min=0, default=42)"
        }
        return configspecs

    @staticmethod
    def validate_decoder():
        configspecs = {
            'module': "option('kpfcnn', default='kpfcnn')",
            'layer': "option('Decoder', default='Decoder')",
            'fp': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_fp():
        configspecs = {
            'upsample': "dict(default=dict())",
            'unary': "dict(default=dict())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_upsample():
        configspecs = {
            'upsample_mode': "option('nearest', 'threenn', 'kpconv', default='nearest')",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_unary():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'bn': "boolean(default=False)",
            'momentum': "float(min=0, default=0.9)",
            'epsilon': "float(min=0, default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout_rate': "float(min=0, max=1, default=0.0)",
            'seed': "integer_or_none(min=0, default=42)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_encoder():
        configspecs = {
            'module': "option('kpfcnn', default='kpfcnn')",
            'layer': "option('Encoder', default='Encoder')",
            'resnet': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_resnet():
        configspecs = {
            'npoint': "integer_or_none(min=1, default=None)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'unary': "dict(default=dict())",
            'kpconv': "dict(default=dict())",
            'unary2': "dict(default=dict())",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_unary2():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'bn': "boolean(default=False)",
            'momentum': "float(min=0, default=0.9)",
            'epsilon': "float(min=0, default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout_rate': "float(min=0, max=1, default=0.0)",
            'seed': "integer_or_none(min=0, default=42)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_fpblock():
        configspecs = {
            'upsample': "dict(default=dict())",
            'unary': "dict(default=dict())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_resnetblock():
        configspecs = {
            'npoint': "integer_or_none(min=1, default=None)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'unary': "dict(default=dict())",
            'kpconv': "dict(default=dict())",
            'unary2': "dict(default=dict())",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_upsampleblock():
        configspecs = {
            'upsample_mode': "option('nearest', 'threenn', 'kpconv', default='nearest')",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_unaryblock():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'bn': "boolean(default=False)",
            'momentum': "float(min=0, default=0.9)",
            'epsilon': "float(min=0, default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kpconvblock():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=1)",
            'npoint': "integer_or_none(min=1, default=None)",
            'activation': "string(default=lrelu)",
            'alpha': "float(min=0, default=0.3)",
            'nsample': "integer(min=1, default=1)",
            'radius': "float(min=0, default=1.0)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'seed': "integer_or_none(min=0, default=42)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'bn': "boolean(default=False)",
            'momentum': "float(min=0, default=0.9)",
            'epsilon': "float(min=0, default=0.001)",
            'center': "boolean(default=True)",
            'scale': "boolean(default=True)",
            'beta_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'gamma_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'moving_mean_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'moving_variance_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='ones')",
            'beta_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'gamma_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'name': "string(default=None)"
        }
        return configspecs


class validate_kplstm():
    """
    Configuration specifications of all KPLSTM layers.
    """
    @staticmethod
    def validate_encoder():
        configspecs = {
            'module': "option('kplstm', default='kplstm')",
            'layer': "option('Encoder', default='Encoder')",
            'sa': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_sa():
        configspecs = {
            'kp_conv': "dict_list(default=list())",
            'kp_conv_lstm': "dict(default=dict())",
            'kp_conv2': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_decoder():
        configspecs = {
            'module': "option('kplstm', default='kplstm')",
            'layer': "option('Decoder', default='Decoder')",
            'fp': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_fp():
        configspecs = {
            'kpfp': "dict(default=dict())",
            'kp_conv': "dict_list(default=list())",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kpfp():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=2)",
            'nsample': "integer_or_none(min=1, default=None)",
            'radius': "float(min=0, default=1.0)",
            'activation': "string(default=relu)",
            'alpha': "float(min=0, default=0.3)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kp_conv():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=2)",
            'npoint': "integer_or_none(min=1, default=None)",
            'nsample': "integer(min=1, default=1)",
            'radius': "float(min=0, default=1.0)",
            'activation': "string(default=relu)",
            'alpha': "float(min=0, default=0.3)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'seed': "integer_or_none(min=0, default=42)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kp_conv_lstm():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=2)",
            'nsample': "integer(min=1, default=1)",
            'radius': "float(min=0, default=1.0)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'activation': "string(default=tanh)",
            'recurrent_activation': "string(default=hard_sigmoid)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout': "float(min=0, max=1, default=0.0)",
            'recurrent_dropout': "float(min=0, max=1, default=0.0)",
            'upsample_mode': "option('nearest', 'threenn', default='nearest')",
            'return_sequences': "boolean(default=True)",
            'return_state': "boolean(default=False)",
            'go_backwards': "boolean(default=False)",
            'stateful': "boolean(default=False)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_kp_conv2():
        configspecs = {
            'filters': "integer(min=1, default=1)",
            'k_points': "integer(min=1, default=2)",
            'npoint': "integer_or_none(min=1, default=None)",
            'nsample': "integer(min=1, default=1)",
            'radius': "float(min=0, default=1.0)",
            'activation': "string(default=relu)",
            'alpha': "float(min=0, default=0.3)",
            'kp_extend': "float(min=0, default=1.0)",
            'fixed': "option('none', 'center', 'verticals', default='center')",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'knn': "boolean(default=False)",
            'kp_influence': "option('constant', 'linear', 'gaussian', default='linear')",
            'aggregation_mode': "option('closest', 'sum', default='sum')",
            'use_xyz': "boolean(default=True)",
            'seed': "integer_or_none(min=0, default=42)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs


class validate_lstm_1d():
    """
    Configuration specifications of the LSTM1D layer.
    """
    @staticmethod
    def validate_lstm1d():
        configspecs = {
            'module': "option('lstm_1d', default='lstm_1d')",
            'layer': "option('LSTM1D', default='LSTM1D')",
            'units': "integer(min=1, default=1)",
            'activation': "string(default=tanh)",
            'recurrent_activation': "string(default=sigmoid)",
            'use_bias': "boolean(default=True)",
            'reduction_mode': "string(default=reshape)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'recurrent_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='orthogonal')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'unit_forget_bias': "boolean(default=True)",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'recurrent_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'activity_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout': "float(default=0.0)",
            'recurrent_dropout': "float(default=0.0)",
            'implementation': "integer(min=1, max=2, default=2)",
            'return_sequences': "boolean(default=True)",
            'return_state': "boolean(default=False)",
            'go_backwards': "boolean(default=False)",
            'stateful': "boolean(default=False)",
            'time_major': "boolean(default=False)",
            'unroll': "boolean(default=False)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs


class validate_conv_lstm_1d():
    """
    Configuration specifications of the ConvLSTM1D layer.
    """
    @staticmethod
    def validate_convlstm1d():
        configspecs = {
            'module': "option('conv_lstm_1d', default='conv_lstm_1d')",
            'layer': "option('ConvLSTM1D', default='ConvLSTM1D')",
            'filters': "integer(min=1, default=1)",
            'kernel_size': "int_list(min=2, max=2, default=list('1', '1'))",
            'strides': "int_list(min=2, max=2, default=list('1', '1'))",
            'padding': "option('valid', 'same', default='valid')",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'dilation_rate': "int_list(min=2, max=2, default=list('1', '1'))",
            'activation': "string(default=tanh)",
            'recurrent_activation': "string(default=hard_sigmoid)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'recurrent_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='orthogonal')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'unit_forget_bias': "boolean(default=True)",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'recurrent_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'activity_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'return_sequences': "boolean(default=True)",
            'go_backwards': "boolean(default=False)",
            'stateful': "boolean(default=False)",
            'dropout': "float(default=0.0)",
            'recurrent_dropout': "float(default=0.0)",
            'name': "string(default=None)"
        }
        return configspecs


class validate_min_max_scaling():
    """
    Configuration specifications of the MinMaxScaling layer.
    """
    @staticmethod
    def validate_minmaxscaling():
        configspecs = {
            'module': "option('min_max_scaling', default='min_max_scaling')",
            'layer': "option('MinMaxScaling', default='MinMaxScaling')",
            'minimum': "float_list_or_none(default=None)",
            'maximum': "float_list_or_none(default=None)",
            'data_format': "option('channels_last', 'channels_first', default='channels_last')",
            'name': "string(default=None)"
        }
        return configspecs


class validate_layers():
    """
    Configuration specifications of keras layers.
    """
    @staticmethod
    def validate_lstm():
        configspecs = {
            'module': "option('layers', default='layers')",
            'layer': "option('LSTM', default='LSTM')",
            'units': "integer(min=1, default=1)",
            'activation': "string(default=tanh)",
            'recurrent_activation': "string(default=sigmoid)",
            'use_bias': "boolean(default=True)",
            'kernel_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='glorot_uniform')",
            'recurrent_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='orthogonal')",
            'bias_initializer': "option('None', 'zeros', 'ones', 'constant', 'random_uniform', 'random_normal', 'truncated_normal', 'identity', 'orthogonal', 'glorot_normal', 'glorot_uniform', default='zeros')",
            'unit_forget_bias': "boolean(default=True)",
            'kernel_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'recurrent_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'bias_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'activity_regularizer': "option('None', 'l1', 'l2', 'l1_l2', default=None)",
            'dropout': "float(default=0.0)",
            'recurrent_dropout': "float(default=0.0)",
            'implementation': "integer(min=1, max=2, default=2)",
            'return_sequences': "boolean(default=True)",
            'return_state': "boolean(default=False)",
            'go_backwards': "boolean(default=False)",
            'stateful': "boolean(default=False)",
            'time_major': "boolean(default=False)",
            'unroll': "boolean(default=False)"
        }
        return configspecs
