# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model.layer import kpconv


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


class TestKPConv(tf.test.TestCase):
    """
    Unit test of the KPConv layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestKPConv, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the KPConv class
        self.keywords = [
            'filters',
            'k_points',
            'npoint',
            'nsample',
            'radius',
            'activation',
            'alpha',
            'kp_extend',
            'fixed',
            'kernel_initializer',
            'kernel_regularizer',
            'knn',
            'kp_influence',
            'aggregation_mode',
            'use_xyz',
            'seed',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestKPConv, self).tearDown()

    def test_call(self):
        """
        Test of the KPConv call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        filters = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        k_points = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        npoint = [None, 0] + [int(np.random.randint(low=1, high=16, size=1))]
        nsample = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        radius = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        activation = [None] + ['lrelu']
        alpha = [0] + [0.3]
        kp_extend = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        fixed = ['none', 'center', 'verticals']
        kernel_initializer = [None] + ['glorot_uniform']
        kernel_regularizer = [None] + ['l2']
        knn = [True, False]
        kp_influence = ['constant', 'linear', 'gaussian']
        aggregation_mode = ['closest', 'sum']
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first']
        seed = [None] + [int(np.random.randint(low=1, high=16, size=1))]
        name = [None]

        # Get all possible combinations of the KPConv layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(filters, k_points, npoint, nsample, radius, activation, alpha, kp_extend, fixed, kernel_initializer,
                                          kernel_regularizer, knn, kp_influence, aggregation_mode, use_xyz, seed, data_format, name))

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

                # Execute KPConv layer
                if init['filters'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['k_points'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['k_points'] < 2:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)(test_data)

                elif is_not_none_and_less(init['npoint'], 1):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['nsample'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['radius'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['alpha'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['kp_extend'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['fixed'] not in set(('none', 'center', 'verticals')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['kp_influence'] not in set(('constant', 'linear', 'gaussian')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['aggregation_mode'] not in set(('closest', 'sum')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = kpconv.KPConv(**init)

                elif input_shape[2] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpconv.KPConv(**init)(test_data)

                elif input_shape[1] < init['nsample'] and init['knn']:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kpconv.KPConv(**init)(test_data)

                else:
                    # Valid initialization and test data
                    kp_conv = kpconv.KPConv(**init)
                    try:
                        output = kp_conv(test_data)
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
                    self.assertEqual(kp_conv.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestKPFP(tf.test.TestCase):
    """
    Unit test of the KPFP layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestKPFP, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the KPFP class
        self.keywords = [
            'filters',
            'k_points',
            'nsample',
            'radius',
            'activation',
            'alpha',
            'kp_extend',
            'fixed',
            'kernel_initializer',
            'kernel_regularizer',
            'knn',
            'kp_influence',
            'aggregation_mode',
            'use_xyz',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestKPFP, self).tearDown()

    def test_call(self):
        """
        Test of the KPFP call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        filters = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        k_points = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        nsample = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        radius = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        activation = [None] + ['lrelu']
        alpha = [0] + [0.3]
        kp_extend = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        fixed = ['none', 'center', 'verticals']
        kernel_initializer = [None] + ['glorot_uniform']
        kernel_regularizer = [None] + ['l2']
        knn = [True, False]
        kp_influence = ['constant', 'linear', 'gaussian']
        aggregation_mode = ['closest', 'sum']
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first']
        name = [None]

        # Get all possible combinations of the KPFP layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(filters, k_points, nsample, radius, activation, alpha, kp_extend, fixed, kernel_initializer,
                                          kernel_regularizer, knn, kp_influence, aggregation_mode, use_xyz, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points1 = [1] + [int(np.random.randint(low=2, high=128, size=1))]
        nbr_of_points2 = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels1 = [1] + [int(np.random.randint(low=4, high=16, size=1))]
        nbr_of_channels2 = [1] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points1, nbr_of_channels1, nbr_of_points2, nbr_of_channels2))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape1 = [input_shape[0], input_shape[2], input_shape[1]]
                    shape2 = [input_shape[0], input_shape[4], input_shape[3]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')
                else:
                    shape1 = [input_shape[0], input_shape[1], input_shape[2]]
                    shape2 = [input_shape[0], input_shape[3], input_shape[4]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute KPFP layer
                if init['filters'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['k_points'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['k_points'] < 2:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif is_not_none_and_less(init['nsample'], 1):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['radius'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['alpha'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['kp_extend'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['fixed'] not in set(('none', 'center', 'verticals')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['kp_influence'] not in set(('constant', 'linear', 'gaussian')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['aggregation_mode'] not in set(('closest', 'sum')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kpfp = kpconv.KPFP(**init)

                elif input_shape[2] < 3 or input_shape[4] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpconv.KPFP(**init)([test_data1, test_data2])

                elif input_shape[1] < init['nsample'] and init['knn']:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kpconv.KPFP(**init)([test_data1, test_data2])

                else:
                    # Valid initialization and test data
                    kpfp = kpconv.KPFP(**init)
                    try:
                        output = kpfp([test_data1, test_data2])
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
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[3]]), output, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[3], num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(kpfp.compute_output_shape([test_data1.shape, test_data2.shape]), output.shape, msg=error_msg)


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
