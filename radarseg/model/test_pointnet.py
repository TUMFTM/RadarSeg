# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model import pointnet


# Helper function
def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


class TestInputTransformNet(tf.test.TestCase):
    """
    Unit test of the InputTransformNet layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestInputTransformNet, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the InputTransformNet class
        self.keywords = [
            'mlp',
            'dense',
            'batch_norm',
            'pooling',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestInputTransformNet, self).tearDown()

    def test_call(self):
        """
        Test of the InputTransformNet call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        mlp = [[]] + [[{'filters': 8}, {'filters': 16}]]
        dense = [[]] + [[{'units': 8}, {'units': 16}]]
        batch_norm = [[]] + [[{}], [{}, {}]]
        pooling = ['max', 'avg', 'no-valid-pooling']
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the InputTransformNet settings (the order has to fit to self.keywords)
        settings = list(itertools.product(mlp, dense, batch_norm, pooling, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points = [0] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (Exactly three input channels have to be provided (xyz))
        nbr_of_channels = [3]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute InputTransformNet layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        trans_mat = pointnet.InputTransformNet(**init)(test_data)

                elif init['pooling'] not in set(('max', 'avg')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        trans_mat = pointnet.InputTransformNet(**init)(test_data)

                elif not init['mlp']:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        trans_mat = pointnet.InputTransformNet(**init)(test_data)

                else:
                    # Valid initialization and test data
                    input_transform_net = pointnet.InputTransformNet(**init)
                    trans_mat = input_transform_net(test_data)

                    # Check output shapes
                    self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, 3]), trans_mat, msg=error_msg)
                    self.assertEqual(input_transform_net.compute_output_shape(test_data.shape), trans_mat.shape, msg=error_msg)

                    # Check output values
                    if all(input_shape):
                        # Check against a 3d identity matrix
                        ref_trans_mat = np.broadcast_to(np.identity(3)[None, ...], (input_shape[0], 3, 3))
                        self.assertAllClose(ref_trans_mat, trans_mat, msg=error_msg)
                    else:
                        # Check against a 3d matrix of nan values
                        ref_trans_mat = np.empty(shape=[input_shape[0], 3, 3])
                        ref_trans_mat[:] = np.nan
                        self.assertAllEqual(ref_trans_mat, trans_mat, msg=error_msg)


class TestFeatureTransformNet(tf.test.TestCase):
    """
    Unit test of the FeatureTransformNet layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestFeatureTransformNet, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the FeatureTransformNet class
        self.keywords = [
            'mlp',
            'dense',
            'batch_norm',
            'pooling',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestFeatureTransformNet, self).tearDown()

    def test_call(self):
        """
        Test of the FeatureTransformNet call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        mlp = [[]] + [[{'filters': 8}, {'filters': 16}]]
        dense = [[]] + [[{'units': 8}, {'units': 16}]]
        batch_norm = [[]] + [[{}], [{}, {}]]
        pooling = ['max', 'avg', 'no-valid-pooling']
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the FeatureTransformNet settings (the order has to fit to self.keywords)
        settings = list(itertools.product(mlp, dense, batch_norm, pooling, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an
        # Fatal Python error: Floating point exception, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_channels = 0) leads to an
        # Fatal Python error: Floating point exception, why it can not be tested.
        nbr_of_channels = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute FeatureTransformNet layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        trans_mat = pointnet.FeatureTransformNet(**init)(test_data)

                elif init['pooling'] not in set(('max', 'avg')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        trans_mat = pointnet.FeatureTransformNet(**init)(test_data)

                else:
                    # Valid initialization and test data
                    feature_transform_net = pointnet.FeatureTransformNet(**init)
                    trans_mat = feature_transform_net(test_data)

                    # Check output shapes
                    self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], input_shape[2]]), trans_mat, msg=error_msg)
                    self.assertEqual(feature_transform_net.compute_output_shape(test_data.shape), trans_mat.shape, msg=error_msg)

                    # Check output values
                    if all(input_shape):
                        # Check against a 3d identity matrix
                        ref_trans_mat = np.broadcast_to(np.identity(input_shape[2])[None, ...], (input_shape[0], input_shape[2], input_shape[2]))
                        self.assertAllClose(ref_trans_mat, trans_mat, msg=error_msg)
                    else:
                        # Check against a 3d matrix of nan values
                        ref_trans_mat = np.empty(shape=[input_shape[0], input_shape[2], input_shape[2]])
                        ref_trans_mat[:] = np.nan
                        self.assertAllEqual(ref_trans_mat, trans_mat, msg=error_msg)


class TestEncoder(tf.test.TestCase):
    """
    Unit test of the Encoder layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestEncoder, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the Encoder class
        self.keywords = [
            'input_trans',
            'mlp',
            'feature_trans',
            'mlp2',
            'pooling',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestEncoder, self).tearDown()

    def test_call(self):
        """
        Test of the Encoder call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        input_trans = [{}] + [{'mlp': [{'filters': 8}, {'filters': 16}], 'dense': [{'units': 8}, {'units': 16}], 'batch_norm': [{}]}]
        mlp = [[]] + [[{'filters': 8}, {'filters': 16}]]
        feature_trans = [{}] + [{'mlp': [{'filters': 8}, {'filters': 16}], 'dense': [{'units': 8}, {'units': 16}], 'batch_norm': [{}]}]
        mlp2 = [[]] + [[{'filters': 8}, {'filters': 16}]]
        pooling = ['max', 'avg', 'no-valid-pooling']
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the Encoder settings (the order has to fit to self.keywords)
        settings = list(itertools.product(input_trans, mlp, feature_trans, mlp2, pooling, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # A minimum of one input point has to be provided
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of three input channels have to be provided (xyz)
        nbr_of_channels = [3] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute Encoder layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat, state = pointnet.Encoder(**init)(test_data)

                elif init['pooling'] not in set(('max', 'avg')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat, state = pointnet.Encoder(**init)(test_data)

                elif not init['mlp']:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat, state = pointnet.Encoder(**init)(test_data)

                else:
                    # Valid initialization and test data
                    encoder = pointnet.Encoder(**init)
                    glob_feat, state = encoder(test_data)

                    # Determine number of output channels
                    if init['mlp2']:
                        nbr_out_chan = init['mlp2'][-1]['filters']
                    elif init['mlp']:
                        nbr_out_chan = init['mlp'][-1]['filters']
                    else:
                        nbr_out_chan = input_shape[2]

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, nbr_out_chan]), glob_feat, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_chan, 1]), glob_feat, msg=error_msg)
                    self.assertShapeEqual(test_data.numpy(), state, msg=error_msg)
                    self.assertTupleEqual(encoder.compute_output_shape(test_data.shape), (glob_feat.shape, state.shape), msg=error_msg)

                    # Check output state values
                    self.assertAllClose(test_data, state)


class TestDecoder(tf.test.TestCase):
    """
    Unit test of the Decoder layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestDecoder, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the Decoder class
        self.keywords = [
            'num_classes',
            'mlp',
            'dropout',
            'out_activation',
            'out_use_bias',
            'out_kernel_initializer',
            'out_bias_initializer',
            'out_kernel_regularizer',
            'out_bias_regularizer',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestDecoder, self).tearDown()

    def test_call(self):
        """
        Test of the Decoder call function.
            - Test 1: Point cloud only
            - Test 2: Point cloud and global feature vector
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        mlp = [[]] + [[{'filters': 8}, {'filters': 16}]]
        dropout = [[]] + [[{'rate': 0.5}]]
        out_activation = [None]
        out_use_bias = [True, False]
        out_kernel_initializer = ['ones'] + ['glorot_uniform']
        out_bias_initializer = ['zeros', 'ones']
        out_kernel_regularizer = [None] + [tf.keras.regularizers.l2]
        out_bias_regularizer = [None] + [tf.keras.regularizers.l2]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the Decoder settings (the order has to fit to self.keywords)
        def get_settings():
            settings = itertools.product(num_classes, mlp, dropout, out_activation, out_use_bias, out_kernel_initializer,
                                         out_bias_initializer, out_kernel_regularizer, out_bias_regularizer, data_format, name)
            return settings

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # A minimum of one input point has to be provided
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of one input channels have to be provided
        nbr_of_channels = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

        # Test 1: Point cloud only
        for input_shape in input_shapes:
            settings = get_settings()
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute Decoder layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat = pointnet.Decoder(**init)(test_data)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat = pointnet.Decoder(**init)(test_data)

                else:
                    # Valid initialization and test data
                    decoder = pointnet.Decoder(**init)
                    glob_feat = decoder(test_data)

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), glob_feat, msg=error_msg)
                        self.assertEqual(decoder.compute_output_shape(test_data.shape), glob_feat.shape, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), glob_feat, msg=error_msg)
                        self.assertEqual(decoder.compute_output_shape(test_data.shape), glob_feat.shape, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

        # Test 2: Point cloud and global feature vector
        # Define number of features of the global feature vector (the value of high is chosen completely arbitrary)
        nbr_of_feat = [0] + [int(np.random.randint(low=2, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels, nbr_of_feat))

        for input_shape in input_shapes:
            settings = get_settings()
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    data_shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=data_shape), dtype=tf.float32, shape=data_shape, name='test_data')
                else:
                    data_shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=data_shape), dtype=tf.float32, shape=data_shape, name='test_data')

                # Generate synthetic global feature vector
                if init['data_format'] == 'channels_first':
                    glob_feat_shape = [input_shape[0], input_shape[3], 1]
                    test_glob_feat = tf.constant(np.random.random(size=glob_feat_shape), dtype=tf.float32, shape=glob_feat_shape, name='test_glob_feat')
                else:
                    glob_feat_shape = [input_shape[0], 1, input_shape[3]]
                    test_glob_feat = tf.constant(np.random.random(size=glob_feat_shape), dtype=tf.float32, shape=glob_feat_shape, name='test_glob_feat')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape 1: {}. \n Input shape 2: {}.'.format(str(init), str(data_shape), str(glob_feat_shape))

                # Execute Decoder layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat = pointnet.Decoder(**init)([test_glob_feat, test_data])

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        glob_feat = pointnet.Decoder(**init)([test_glob_feat, test_data])

                else:
                    # Valid initialization and test data
                    decoder = pointnet.Decoder(**init)
                    glob_feat = decoder([test_glob_feat, test_data])

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), glob_feat, msg=error_msg)
                        self.assertEqual(decoder.compute_output_shape(test_data.shape), glob_feat.shape, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), glob_feat, msg=error_msg)
                        self.assertEqual(decoder.compute_output_shape(test_data.shape), glob_feat.shape, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')


class TestPointNet(tf.test.TestCase):
    """
    Unit test of the PointNet layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestPointNet, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the PointNet class
        self.keywords = [
            'num_classes',
            'data_format',
            'name'
        ]

        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the PointNet settings (the order has to fit to self.keywords)
        self.settings = list(itertools.product(num_classes, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # A minimum of one input point has to be provided
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of three input channels have to be provided (xyz)
        nbr_of_channels = [3] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        self.input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

    def tearDown(self):
        # Call teardown of the base test class
        super(TestPointNet, self).tearDown()

    def test_call(self):
        """
        Test of the PointNet call function.
        """
        for input_shape in self.input_shapes:
            for setting in self.settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[2], input_shape[1]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(shape))

                # Execute PointNet layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet.PointNet(**init)(test_data)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet.PointNet(**init)(test_data)

                else:
                    # Valid initialization and test data
                    point_net = pointnet.PointNet(**init)
                    prediction = point_net(test_data)

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), prediction, msg=error_msg)
                        self.assertEqual(point_net.compute_output_shape(test_data.shape), prediction.shape, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), prediction, msg=error_msg)
                        self.assertEqual(point_net.compute_output_shape(test_data.shape), prediction.shape, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

    def test_model(self):
        """
        Test of a PointNet model execution.
        """
        for input_shape in self.input_shapes:
            for setting in self.settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Define model input
                if init['data_format'] == 'channels_first':
                    test_input = tf.keras.Input(shape=(input_shape[2], None), name='test_input', dtype=tf.float32)
                else:
                    test_input = tf.keras.Input(shape=(None, input_shape[2]), name='test_input', dtype=tf.float32)

                # Define model layers
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet.PointNet(**init)(test_input)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet.PointNet(**init)(test_input)

                else:
                    # Valid layer initialization
                    prediction = pointnet.PointNet(**init)(test_input)

                    # Compile model
                    model = tf.keras.Model(inputs=[test_input], outputs=[prediction])

                    # Generate synthetic test data (point cloud)
                    if init['data_format'] == 'channels_first':
                        shape = [input_shape[0], input_shape[2], input_shape[1]]
                        test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                    else:
                        shape = [input_shape[0], input_shape[1], input_shape[2]]
                        test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                    # Make prediction (execute model)
                    prediction = model.predict(test_data)

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), tf.constant(prediction), msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), tf.constant(prediction), msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')


if __name__ == "__main__":
    # Set GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # Run tests
    tf.test.main()
