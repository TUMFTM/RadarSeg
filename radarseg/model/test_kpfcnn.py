# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model import kpfcnn


# Helper function
def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def is_not_none_and_less(value, minimum):
    if value is None:
        return False
    elif value < minimum:
        return True
    else:
        return False


def to_list(value):
    if isinstance(value, list):
        return value
    else:
        return [value]


class TestKPConvBlock(tf.test.TestCase):
    """
    Unit test of the KPConvBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestKPConvBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the KPConvBlock class
        self.keywords = [
            'filters',
            'npoint',
            'use_xyz',
            'data_format',
            'bn',
            'momentum',
            'epsilon',
            'center',
            'scale',
            'beta_initializer',
            'gamma_initializer',
            'moving_mean_initializer',
            'moving_variance_initializer',
            'beta_regularizer',
            'gamma_regularizer',
            'dropout_rate'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestKPConvBlock, self).tearDown()

    def test_call(self):
        """
        Test of the KPConvBlock call function.

        Note: This test only covers settings, which are not subject to the KPConv layer
        itself. All settings of the KPConv layer are tested by the dedicated unit test of the
        KPConv layer. The only exceptions are settings, which have an influance on the output
        shape of the KPConvBlock layer.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        filters = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        npoint = [None, 0] + [int(np.random.randint(low=1, high=16, size=1))]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        bn = [True, False]
        momentum = [0.9]
        epsilon = [0.001]
        center = [True, False]
        scale = [True, False]
        beta_initializer = [None] + ['zeros']
        gamma_initializer = [None] + ['ones']
        moving_mean_initializer = [None] + ['zeros']
        moving_variance_initializer = [None] + ['ones']
        beta_regularizer = [None] + ['l2']
        gamma_regularizer = [None] + ['l2']
        dropout_rate = [0.0] + [0.5]

        # Get all possible combinations of the KPConvBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(filters, npoint, use_xyz, data_format, bn, momentum, epsilon, center, scale, beta_initializer,
                                          gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                                          gamma_regularizer, dropout_rate))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

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

                # Execute KPConvBlock layer
                if init['filters'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpconv_block = kpfcnn.KPConvBlock(**init)

                elif is_not_none_and_less(init['npoint'], 1):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpconv_block = kpfcnn.KPConvBlock(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpconv_block = kpfcnn.KPConvBlock(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpfcnn.KPConvBlock(**init)(test_data)

                else:
                    # Valid initialization and test data
                    kpconv_block = kpfcnn.KPConvBlock(**init)
                    try:
                        output = kpconv_block(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output points
                    if init['npoint'] is not None:
                        num_out_points = init['npoint']
                    else:
                        num_out_points = input_shape[1]

                    # Determine number of output channels
                    if init['use_xyz']:
                        num_out_chan = 3 + init['filters']
                    else:
                        num_out_chan = init['filters']

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, num_out_points]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_points, num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(kpconv_block.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestUnaryBlock(tf.test.TestCase):
    """
    Unit test of the UnaryBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestUnaryBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the UnaryBlock class
        self.keywords = [
            'filters',
            'activation',
            'alpha',
            'use_xyz',
            'data_format',
            'dropout_rate',
            'seed'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestUnaryBlock, self).tearDown()

    def test_call(self):
        """
        Test of the UnaryBlock call function.

        Note: This test only covers settings, which are not subject to the MLP layer
        itself. All settings of the MLP layer are tested by the dedicated unit test of the
        MLP layer. The only exceptions are settings, which have an influance on the output
        shape of the UnaryBlock layer.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        filters = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        activation = [None] + ['lrelu']
        alpha = [0] + [0.3]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        dropout_rate = [0.0] + [0.5]
        seed = [42]

        # Get all possible combinations of the UnaryBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(filters, activation, alpha, use_xyz, data_format, dropout_rate, seed))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

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

                # Execute UnaryBlock layer
                if init['filters'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        unary_block = kpfcnn.UnaryBlock(**init)

                elif init['alpha'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        unary_block = kpfcnn.UnaryBlock(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        unary_block = kpfcnn.UnaryBlock(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpfcnn.UnaryBlock(**init)(test_data)

                else:
                    # Valid initialization and test data
                    unary_block = kpfcnn.UnaryBlock(**init)
                    try:
                        output = unary_block(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output channels
                    if init['use_xyz']:
                        num_out_chan = 3 + init['filters']
                    else:
                        num_out_chan = init['filters']

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[1]]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(unary_block.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestUpsampleBlock(tf.test.TestCase):
    """
    Unit test of the UpsampleBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestUpsampleBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the UpsampleBlock class
        self.keywords = [
            'upsample_mode',
            'use_xyz',
            'data_format'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestUpsampleBlock, self).tearDown()

    def test_call(self):
        """
        Test of the UpsampleBlock call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        upsample_mode = ['nearest', 'threenn', 'kpconv', 'no-valid-upsample-mode']
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the UpsampleBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(upsample_mode, use_xyz, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size1 = [1] + [int(np.random.randint(low=2, high=16, size=1))]
        batch_size2 = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points1 = [1] + [int(np.random.randint(low=2, high=128, size=1))]
        nbr_of_points2 = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels1 = [1] + [int(np.random.randint(low=4, high=16, size=1))]
        nbr_of_channels2 = [1] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size1, nbr_of_points1, nbr_of_channels1, batch_size2, nbr_of_points2, nbr_of_channels2))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape1 = [input_shape[0], input_shape[2], input_shape[1]]
                    shape2 = [input_shape[3], input_shape[5], input_shape[4]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')
                else:
                    shape1 = [input_shape[0], input_shape[1], input_shape[2]]
                    shape2 = [input_shape[3], input_shape[4], input_shape[5]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute UpsampleBlock layer
                if init['upsample_mode'] not in set(('nearest', 'threenn', 'kpconv')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        upsample_block = kpfcnn.UpsampleBlock(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        upsample_block = kpfcnn.UpsampleBlock(**init)

                elif init['upsample_mode'] == 'kpconv':
                    # TODO: Remove if implemented
                    with self.assertRaises(NotImplementedError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.UpsampleBlock(**init)([test_data1, test_data2])

                elif input_shape[0] != input_shape[3]:
                    with self.assertRaises(ValueError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.UpsampleBlock(**init)([test_data1, test_data2])

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.UpsampleBlock(**init)([test_data1, test_data2])

                elif input_shape[5] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.UpsampleBlock(**init)([test_data1, test_data2])

                else:
                    # Valid initialization and test data
                    upsample_block = kpfcnn.UpsampleBlock(**init)
                    try:
                        new_features, xyz2 = upsample_block([test_data1, test_data2])
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output channels
                    if init['use_xyz']:
                        num_out_chan = input_shape[2] + input_shape[5] - 3
                    else:
                        num_out_chan = input_shape[2] + input_shape[5] - 6

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[4]]), new_features, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, input_shape[4]]), xyz2, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[4], num_out_chan]), new_features, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[4], 3]), xyz2, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertTupleEqual(upsample_block.compute_output_shape([test_data1.shape, test_data2.shape]), (new_features.shape, xyz2.shape), msg=error_msg)


class TestResnetBlock(tf.test.TestCase):
    """
    Unit test of the ResnetBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestResnetBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the ResnetBlock class
        self.keywords = [
            'npoint',
            'activation',
            'alpha',
            'unary',
            'kpconv',
            'unary2',
            'use_xyz',
            'data_format'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestResnetBlock, self).tearDown()

    def test_call(self):
        """
        Test of the ResnetBlock call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        npoint = [None, 0] + [int(np.random.randint(low=1, high=16, size=1))]
        activation = [None] + ['lrelu']
        alpha = [0] + [0.3]
        unary = [None] + [{'filters': 16}]
        kpconv = [None] + [{'filters': 16}]
        unary2 = [None] + [{'filters': 32}]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the ResnetBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(npoint, activation, alpha, unary, kpconv, unary2, use_xyz, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

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

                # Execute ResnetBlock layer
                if init['alpha'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        resnet_block = kpfcnn.ResnetBlock(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        resnet_block = kpfcnn.ResnetBlock(**init)

                elif is_not_none_and_less(init['npoint'], 1):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        resnet_block = kpfcnn.ResnetBlock(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpfcnn.ResnetBlock(**init)(test_data)

                else:
                    # Valid initialization and test data
                    resnet_block = kpfcnn.ResnetBlock(**init)
                    try:
                        output = resnet_block(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output points
                    if init['npoint'] is not None:
                        num_out_points = init['npoint']
                    else:
                        num_out_points = input_shape[1]

                    # Determine number of output channels
                    if init['use_xyz'] and init['unary2']:
                        num_out_chan = 3 + init['unary2']['filters']
                    elif init['unary2']:
                        num_out_chan = init['unary2']['filters']
                    elif init['use_xyz']:
                        num_out_chan = 4
                    else:
                        num_out_chan = 1

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, num_out_points]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_points, num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(resnet_block.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestFPBlock(tf.test.TestCase):
    """
    Unit test of the FPBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestFPBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the FPBlock class
        self.keywords = [
            'upsample',
            'unary',
            'data_format'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestFPBlock, self).tearDown()

    def test_call(self):
        """
        Test of the FPBlock call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        upsample = [None] + [{'upsample_mode': 'nearest'}]
        unary = [None] + [{'filters': 16, 'use_xyz': True}, {'filters': 16, 'use_xyz': False}]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the FPBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(upsample, unary, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size1 = [1] + [int(np.random.randint(low=2, high=16, size=1))]
        batch_size2 = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points1 = [1] + [int(np.random.randint(low=2, high=128, size=1))]
        nbr_of_points2 = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels1 = [1] + [int(np.random.randint(low=4, high=16, size=1))]
        nbr_of_channels2 = [1] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size1, nbr_of_points1, nbr_of_channels1, batch_size2, nbr_of_points2, nbr_of_channels2))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape1 = [input_shape[0], input_shape[2], input_shape[1]]
                    shape2 = [input_shape[3], input_shape[5], input_shape[4]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')
                else:
                    shape1 = [input_shape[0], input_shape[1], input_shape[2]]
                    shape2 = [input_shape[3], input_shape[4], input_shape[5]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute FPBlock layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        fp_block = kpfcnn.FPBlock(**init)

                elif input_shape[2] < 3 or input_shape[5] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.FPBlock(**init)([test_data1, test_data2])

                elif input_shape[0] != input_shape[3]:
                    with self.assertRaises(ValueError, msg=error_msg):
                        new_features, xyz2 = kpfcnn.FPBlock(**init)([test_data1, test_data2])

                else:
                    # Valid initialization and test data
                    fp_block = kpfcnn.FPBlock(**init)
                    try:
                        new_features, xyz2 = fp_block([test_data1, test_data2])
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output channels
                    if init['unary']:
                        if init['unary']['use_xyz']:
                            num_out_chan = 3 + init['unary']['filters']
                        else:
                            num_out_chan = init['unary']['filters']
                    else:
                        num_out_chan = 4

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[4]]), new_features, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, input_shape[4]]), xyz2, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[4], num_out_chan]), new_features, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[4], 3]), xyz2, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertTupleEqual(fp_block.compute_output_shape([test_data1.shape, test_data2.shape]), (new_features.shape, xyz2.shape), msg=error_msg)


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
            'resnet',
            'data_format'
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
        resnet = [None] + [[{'npoint': 8, 'unary2': {'filters': 16}, 'use_xyz': True}], [{'npoint': 8, 'unary2': {'filters': 16}, 'use_xyz': False}, {'npoint': 8, 'unary2': {'filters': 16}, 'use_xyz': True}]]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the Encoder layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(resnet, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

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
                        encoder = kpfcnn.Encoder(**init)

                elif input_shape[2] < 3 and init['resnet'] and len(to_list(init['resnet'])) < 2:
                    with self.assertRaises(ValueError, msg=error_msg):
                        encoding, state = kpfcnn.Encoder(**init)(test_data)

                elif input_shape[2] < 3 and init['resnet']:
                    with self.assertRaises(ValueError, msg=error_msg):
                        encoding, state, states = kpfcnn.Encoder(**init)(test_data)

                else:
                    # Valid initialization and test data
                    encoder = kpfcnn.Encoder(**init)
                    try:
                        if len(to_list(init['resnet'])) < 2:
                            encoding, state = encoder(test_data)
                        else:
                            encoding, state, states = encoder(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    def get_resnet_output_shape(resnet_setting):
                        if init['data_format'] == 'channels_first' and resnet_setting['use_xyz']:
                            return (input_shape[0], 3 + resnet_setting['unary2']['filters'], resnet_setting['npoint'])
                        elif init['data_format'] == 'channels_first':
                            return (input_shape[0], resnet_setting['unary2']['filters'], resnet_setting['npoint'])
                        elif resnet_setting['use_xyz']:
                            return (input_shape[0], resnet_setting['npoint'], 3 + resnet_setting['unary2']['filters'])
                        else:
                            return (input_shape[0], resnet_setting['npoint'], resnet_setting['unary2']['filters'])

                    # Determine encoding shape
                    if init['resnet']:
                        encoding_shape = get_resnet_output_shape(init['resnet'][-1])
                    else:
                        encoding_shape = test_data.shape

                    # Determine state shape
                    if len(to_list(init['resnet'])) > 1:
                        state_shape = get_resnet_output_shape(init['resnet'][-2])
                    else:
                        state_shape = test_data.shape

                    # Determine states shape
                    if len(to_list(init['resnet'])) > 1:
                        states_shape = []
                        resnet_input_shape = test_data.shape
                        for resnet_setting in init['resnet']:
                            resnet_output_shape = get_resnet_output_shape(resnet_setting)
                            states_shape.append([resnet_output_shape, resnet_input_shape])
                            resnet_input_shape = resnet_output_shape

                        del states_shape[-1]
                        states_shape.reverse()

                    else:
                        states_shape = []

                    # Check output shapes
                    self.assertShapeEqual(np.empty(shape=encoding_shape), encoding, msg=error_msg)
                    self.assertShapeEqual(np.empty(shape=state_shape), state, msg=error_msg)

                    if len(to_list(init['resnet'])) > 1:
                        for internal_state_shape, internal_state in zip(states_shape, states):
                            self.assertShapeEqual(np.empty(shape=internal_state_shape[0]), internal_state[0], msg=error_msg)
                            self.assertShapeEqual(np.empty(shape=internal_state_shape[1]), internal_state[1], msg=error_msg)

                    # Test compute_output_shape function
                    if len(to_list(init['resnet'])) < 2:
                        encoding_shape, state_shape = encoder.compute_output_shape(test_data.shape)
                    else:
                        encoding_shape, state_shape, states_shape = encoder.compute_output_shape(test_data.shape)

                    self.assertEqual(encoding_shape, encoding.shape, msg=error_msg)
                    self.assertEqual(state_shape, state.shape, msg=error_msg)
                    if len(to_list(init['resnet'])) > 1:
                        self.assertListEqual(states_shape, tf.nest.map_structure(tf.shape, states), msg=error_msg)


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
            'fp',
            'data_format'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestDecoder, self).tearDown()

    def test_call(self):
        """
        Test of the Decoder call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        fp = [None] + [[{'unary': {'filters': 1, 'use_xyz': True}}], [{'unary': {'filters': 8, 'use_xyz': False}}, {'unary': {'filters': 16, 'use_xyz': True}}]]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the Decoder settings (the order has to fit to self.keywords)
        settings = list(itertools.product(fp, data_format))

        # Define random batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = int(np.random.randint(low=2, high=16, size=1))

        def _get_rand_nbr_points():
            # Define random number of points of the test data (the value of high is chosen completely arbitrary)
            # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
            return int(np.random.randint(low=1, high=128, size=1))

        def _get_rand_nbr_chan():
            # Define random number of channels of the test data (the value of high is chosen completely arbitrary)
            # A minimum of three input channels have to be provided for the first input (xyz)
            return int(np.random.randint(low=4, high=16, size=1))

        def _get_rand_states(nbr_of_states: int = 0, batch_size: int = 1):
            states = []
            for _ in range(nbr_of_states):
                states.append([tf.TensorShape([batch_size, _get_rand_nbr_points(), _get_rand_nbr_chan()]),
                               tf.TensorShape([batch_size, _get_rand_nbr_points(), _get_rand_nbr_chan()])])

            return states

        # Define test input data shapes
        encoding_shape = [[batch_size, 1, 4]] + [[batch_size, _get_rand_nbr_points(), _get_rand_nbr_chan()]]
        state_shape = [[batch_size, 1, 4]] + [[batch_size, _get_rand_nbr_points(), _get_rand_nbr_chan()]]
        states_shape = [[]] + [_get_rand_states(nbr_of_states=1, batch_size=batch_size), _get_rand_states(nbr_of_states=2, batch_size=batch_size)]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(encoding_shape, state_shape, states_shape))

        # Define shape indices
        encoding_shape_ind = 0
        state_shape_ind = 1
        states_shape_ind = 2
        batch_dim_ind = 0
        point_dim_ind = 1
        chan_dim_ind = 2

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_last':
                    # Generate encoding test data
                    encoding_test_data = tf.constant(np.random.random(size=input_shape[encoding_shape_ind]), dtype=tf.float32, shape=input_shape[encoding_shape_ind], name='encoding_test_data')

                    # Generate state test data
                    state_test_data = tf.constant(np.random.random(size=input_shape[state_shape_ind]), dtype=tf.float32, shape=input_shape[state_shape_ind], name='state_test_data')

                    # Generate states test data
                    states_test_data = []
                    for int_stat_shape in input_shape[states_shape_ind]:
                        states_test_data.append([tf.constant(np.random.random(size=int_stat_shape[0]), dtype=tf.float32, shape=int_stat_shape[0], name='states_test_data1'),
                                                 tf.constant(np.random.random(size=int_stat_shape[1]), dtype=tf.float32, shape=int_stat_shape[1], name='states_test_data2')])
                else:
                    # Generate encoding test data
                    temp_shape = [input_shape[encoding_shape_ind][batch_dim_ind], input_shape[encoding_shape_ind][chan_dim_ind], input_shape[encoding_shape_ind][point_dim_ind]]
                    encoding_test_data = tf.constant(np.random.random(size=temp_shape), dtype=tf.float32, shape=temp_shape, name='encoding_test_data')

                    # Generate state test data
                    temp_shape = [input_shape[state_shape_ind][batch_dim_ind], input_shape[state_shape_ind][chan_dim_ind], input_shape[state_shape_ind][point_dim_ind]]
                    state_test_data = tf.constant(np.random.random(size=temp_shape), dtype=tf.float32, shape=temp_shape, name='state_test_data')

                    # Generate states test data
                    states_test_data = []
                    for int_stat_shape in input_shape[states_shape_ind]:
                        temp_shape1 = [int_stat_shape[0][batch_dim_ind], int_stat_shape[0][chan_dim_ind], int_stat_shape[0][point_dim_ind]]
                        temp_shape2 = [int_stat_shape[1][batch_dim_ind], int_stat_shape[1][chan_dim_ind], int_stat_shape[1][point_dim_ind]]
                        states_test_data.append([tf.constant(np.random.random(size=temp_shape1), dtype=tf.float32, shape=temp_shape1, name='states_test_data1'),
                                                 tf.constant(np.random.random(size=temp_shape2), dtype=tf.float32, shape=temp_shape2, name='states_test_data2')])

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute Decoder layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        decoder = kpfcnn.Decoder(**init)

                elif len(to_list(states_test_data)) != max(len(to_list(init['fp'])) - 1, 0):
                    with self.assertRaises(ValueError, msg=error_msg):
                        decoding = kpfcnn.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                else:
                    # Valid initialization and test data
                    decoder = kpfcnn.Decoder(**init)
                    try:
                        decoding = decoder([encoding_test_data, state_test_data, states_test_data])
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output points
                    if not init['fp']:
                        nbr_out_points = input_shape[encoding_shape_ind][point_dim_ind]
                    elif states_test_data:
                        nbr_out_points = input_shape[states_shape_ind][-1][state_shape_ind][point_dim_ind]
                    else:
                        nbr_out_points = input_shape[state_shape_ind][point_dim_ind]

                    # Determine number of output channels
                    if init['fp']:
                        # Get settings of the unary layer of the last fp_layer
                        last_unary_settings = init['fp'][-1]['unary']

                        if last_unary_settings['use_xyz']:
                            nbr_out_chan = 3 + last_unary_settings['filters']
                        else:
                            nbr_out_chan = last_unary_settings['filters']

                    else:
                        nbr_out_chan = input_shape[encoding_shape_ind][chan_dim_ind]

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0][batch_dim_ind], nbr_out_points, nbr_out_chan]), decoding, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0][batch_dim_ind], nbr_out_chan, nbr_out_points]), decoding, msg=error_msg)

                    # Test compute_output_shape function
                    states_test_data_shape = tf.nest.map_structure(lambda x: x.shape, states_test_data)
                    self.assertEqual(decoder.compute_output_shape((encoding_test_data.shape, state_test_data.shape, states_test_data_shape)),
                                     decoding.shape,
                                     msg=error_msg)


class TestOutputBlock(tf.test.TestCase):
    """
    Unit test of the OutputBlock layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestOutputBlock, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the OutputBlock class
        self.keywords = [
            'num_classes',
            'kpconv',
            'dropout',
            'out_activation',
            'out_use_bias',
            'out_kernel_initializer',
            'out_bias_initializer',
            'out_kernel_regularizer',
            'out_bias_regularizer',
            'data_format'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestOutputBlock, self).tearDown()

    def test_call(self):
        """
        Test of the OutputBlock call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [0, 1] + [int(np.random.randint(low=1, high=16, size=1))]
        kpconv = [None] + [[{'filters': 16, 'use_xyz': True}], [{'filters': 32, 'use_xyz': False}, {'filters': 16, 'use_xyz': True}]]
        dropout = [None] + [[{'rate': 0.5}], [{'rate': 0.5}, {'rate': 0.5}], [{'rate': 0.5}, {'rate': 0.5}, {'rate': 0.5}]]
        out_activation = [None] + ['softmax']
        out_use_bias = [True, False]
        out_kernel_initializer = [None] + ['glorot_uniform']
        out_bias_initializer = [None] + ['zeros']
        out_kernel_regularizer = [None] + ['l2']
        out_bias_regularizer = [None] + ['l2']
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the OutputBlock layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(num_classes, kpconv, dropout, out_activation, out_use_bias, out_kernel_initializer, out_bias_initializer,
                                          out_kernel_regularizer, out_bias_regularizer, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

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

                # Execute OutputBlock layer
                if init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output_block = kpfcnn.OutputBlock(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output_block = kpfcnn.OutputBlock(**init)

                elif input_shape[2] < 3 and init['kpconv']:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpfcnn.OutputBlock(**init)(test_data)

                else:
                    # Valid initialization and test data
                    output_block = kpfcnn.OutputBlock(**init)
                    try:
                        output = output_block(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(output_block.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestKPFCNN(tf.test.TestCase):
    """
    Unit test of the KPFCNN layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestKPFCNN, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the KPFCNN class
        self.keywords = [
            'num_classes',
            'seed',
            'data_format'
        ]

        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [0, 1] + [int(np.random.randint(low=1, high=16, size=1))]
        seed = [42]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']

        # Get all possible combinations of the KPFCNN layer settings (the order has to fit to self.keywords)
        self.settings = list(itertools.product(num_classes, seed, data_format))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=4, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=32, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=8, size=1))]

        # Get all possible input data shape combinations
        self.input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

    def tearDown(self):
        # Call teardown of the base test class
        super(TestKPFCNN, self).tearDown()

    def test_call(self):
        """
        Test of the KPFCNN call function.
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
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute KPFCNN layer
                if init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif init['seed'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpfcnn.KPFCNN(**init)(test_data)

                else:
                    # Valid initialization and test data
                    kp_fcnn = kpfcnn.KPFCNN(**init)
                    try:
                        output = kp_fcnn(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(kp_fcnn.compute_output_shape(test_data.shape), output.shape, msg=error_msg)

    def test_model(self):
        """
        Test of a KPFCNN model execution.
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
                if init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif init['seed'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_fcnn = kpfcnn.KPFCNN(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        prediction = kpfcnn.KPFCNN(**init)(test_input)

                else:
                    # Valid layer initialization
                    kp_fcnn = kpfcnn.KPFCNN(**init)
                    prediction = kp_fcnn(test_input)

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

                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), tf.constant(prediction), msg=error_msg)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    tf.test.main()
