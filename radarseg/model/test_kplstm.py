# Standard Libraries
import copy
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model import kplstm


# Helper function
def is_not_none_and_less(value, minimum):
    if value is None:
        return False
    elif value < minimum:
        return True
    else:
        return False


def is_not_none_and_greater(value, maximum):
    if value is None:
        return False
    elif value > maximum:
        return True
    else:
        return False


class TestSAModule(tf.test.TestCase):
    """
    Unit test of the kplstm SAModule layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestSAModule, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the SAModule class
        self.keywords = [
            'kp_conv',
            'kp_conv_lstm',
            'kp_conv2',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestSAModule, self).tearDown()

    def _is_valide_k_for_knn(self, settings, input_points):
        # Evaluates if k (number of sampled points) in knn is not grater than the actual number of points
        for i, setting in enumerate(settings):
            # Skip if knn is not set
            if not setting.get('knn', False):
                continue

            # Check if k is grater than the current number of points
            current_nbr_of_points = next((previous.get('npoint', None) for previous in reversed(settings[:i]) if previous.get('npoint', None) is not None), input_points)
            if is_not_none_and_greater(setting.get('nsample', None), current_nbr_of_points):
                return False

        return True

    def test_call(self):
        """
        Test of the SAModule call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        kp_conv = [[]] + [[{'filters': 1, 'k_points': 1, 'npoint': None, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                          [{'filters': 1, 'k_points': 2, 'npoint': None, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': True}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': False, 'knn': False}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': False, 'knn': False},
                           {'filters': 8, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}],
                          [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False},
                           {'filters': 8, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}]]
        kp_conv_lstm = [{}] + [{'filters': 1, 'k_points': 1, 'nsample': 1, 'use_xyz': True, 'knn': True, 'return_sequences': True, 'go_backwards': True, 'stateful': True},
                               {'filters': 1, 'k_points': 2, 'nsample': 1, 'use_xyz': True, 'knn': True, 'return_sequences': True, 'go_backwards': True, 'stateful': True},
                               {'filters': 1, 'k_points': 2, 'nsample': 2, 'use_xyz': True, 'knn': True, 'return_sequences': True, 'go_backwards': True, 'stateful': True},
                               {'filters': 1, 'k_points': 2, 'nsample': 2, 'use_xyz': True, 'knn': False, 'return_sequences': True, 'go_backwards': True, 'stateful': True},
                               {'filters': 1, 'k_points': 2, 'nsample': 2, 'use_xyz': False, 'knn': False, 'return_sequences': True, 'go_backwards': True, 'stateful': True},
                               {'filters': 8, 'k_points': 2, 'nsample': 2, 'use_xyz': True, 'knn': False, 'return_sequences': False, 'go_backwards': False, 'stateful': False},
                               {'filters': 8, 'k_points': 2, 'nsample': 2, 'use_xyz': True, 'knn': False, 'return_sequences': True, 'go_backwards': False, 'stateful': False}]
        kp_conv2 = [[]] + [[{'filters': 1, 'k_points': 1, 'npoint': None, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                           [{'filters': 1, 'k_points': 2, 'npoint': None, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 1, 'use_xyz': True, 'knn': True}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': True}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': False, 'knn': False}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': False, 'knn': False},
                            {'filters': 8, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}],
                           [{'filters': 1, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False},
                            {'filters': 8, 'k_points': 2, 'npoint': 2, 'nsample': 2, 'use_xyz': True, 'knn': False}]]
        data_format = ['channels_last', 'channels_first']
        name = [None]

        # Get all possible combinations of the SAModule layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(kp_conv, kp_conv_lstm, kp_conv2, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=8, size=1))]

        # Define number of timesteps of the test data (the value of high is chosen completely arbitrary)
        timesteps = [1] + [int(np.random.randint(low=2, high=8, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=32, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, timesteps, nbr_of_points, nbr_of_channels))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[1], input_shape[3], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute SAModule layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        sa_module = kplstm.SAModule(**init)

                elif init['kp_conv_lstm'].get('stateful', False):
                    with self.assertRaises(NotImplementedError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                elif input_shape[3] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                elif init['kp_conv_lstm'].get('k_points', 1) < 2 and init['kp_conv_lstm']:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        sa_module = kplstm.SAModule(**init)(test_data)

                elif any(setting.get('k_points', 2) < 2 for setting in init['kp_conv']):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        sa_module = kplstm.SAModule(**init)(test_data)

                elif not self._is_valide_k_for_knn(init['kp_conv'], input_shape[2]):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                elif init['kp_conv_lstm'].get('k_points', 2) < 2:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        sa_module = kplstm.SAModule(**init)(test_data)

                elif not self._is_valide_k_for_knn(init['kp_conv'] + [init['kp_conv_lstm']], input_shape[2]):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                elif any(setting.get('filters', 1) < 3 and not setting.get('use_xyz', True) for setting in init['kp_conv'] + [init['kp_conv_lstm']] + init['kp_conv2'][:-1]):
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                elif any(setting.get('k_points', 2) < 2 for setting in init['kp_conv2']):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        sa_module = kplstm.SAModule(**init)(test_data)

                elif not self._is_valide_k_for_knn(init['kp_conv'] + [init['kp_conv_lstm']] + init['kp_conv2'], input_shape[2]):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kplstm.SAModule(**init)(test_data)

                else:
                    # Valid initialization and test data
                    sa_module = kplstm.SAModule(**init)
                    try:
                        output = sa_module(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    def _get_last_setting_with_attribute(settings: list, attribute: str):
                        return next((setting for setting in reversed(settings) if setting.get(attribute, None) is not None), None)

                    # Determine number of output points
                    last_setting = _get_last_setting_with_attribute(init['kp_conv'] + init['kp_conv2'], 'npoint')
                    if last_setting is not None:
                        nbr_out_points = last_setting['npoint']
                    else:
                        nbr_out_points = input_shape[2]

                    # Determine number of output channels
                    if not init['kp_conv2'] and init['kp_conv_lstm'].get('use_xyz', True):
                        num_out_chan = 3 + init['kp_conv_lstm'].get('filters', 1)
                    elif not init['kp_conv2']:
                        num_out_chan = init['kp_conv_lstm'].get('filters', 1)
                    elif init['kp_conv2'][-1].get('use_xyz', True):
                        num_out_chan = 3 + init['kp_conv2'][-1].get('filters', 1)
                    else:
                        num_out_chan = init['kp_conv2'][-1].get('filters', 1)

                    # Check output shapes
                    if init['kp_conv_lstm'].get('return_sequences', False) and init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], num_out_chan, nbr_out_points]), output, msg=error_msg)

                    elif init['kp_conv_lstm'].get('return_sequences', False):
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], nbr_out_points, num_out_chan]), output, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, nbr_out_points]), output, msg=error_msg)

                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_points, num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(sa_module.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


class TestFPModule(tf.test.TestCase):
    """
    Unit test of the kplstm FPModule layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestFPModule, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the FPModule class
        self.keywords = [
            'kpfp',
            'kp_conv',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestFPModule, self).tearDown()

    def test_call(self):
        """
        Test of the FPModule call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        kpfp = [{}] + [{'filters': 1, 'k_points': 2, 'nsample': None, 'knn': False, 'use_xyz': True, 'data_format': 'channels_last'},
                       {'filters': 8, 'k_points': 5, 'nsample': 4, 'knn': True, 'use_xyz': False, 'data_format': 'channels_last'}]
        kp_conv = [[]] + [[{'filters': 1, 'k_points': 2, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True, 'data_format': 'channels_last'}],
                          [{'filters': 8, 'k_points': 5, 'npoint': 4, 'nsample': 4, 'knn': True, 'use_xyz': True, 'data_format': 'channels_first'},
                           {'filters': 16, 'k_points': 7, 'npoint': 4, 'nsample': 4, 'knn': False, 'use_xyz': False, 'data_format': 'channels_last'}]]
        data_format = ['channels_last', 'channels_first']
        name = [None]

        # Get all possible combinations of the FPModule layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(kpfp, kp_conv, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=8, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points1 = [1] + [int(np.random.randint(low=2, high=32, size=1))]
        nbr_of_points2 = [1] + [int(np.random.randint(low=2, high=32, size=1))]

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

                # Execute FPModule layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        fp_module = kplstm.FPModule(**init)

                elif input_shape[2] < 3 or input_shape[4] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kplstm.FPModule(**init)([test_data1, test_data2])

                elif is_not_none_and_greater(init['kpfp'].get('nsample', None), input_shape[1]) and init['kpfp'].get('knn', False):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kplstm.FPModule(**init)([test_data1, test_data2])

                elif any(is_not_none_and_greater(kp_conv_setting.get('nsample', None), input_shape[3]) and kp_conv_setting.get('knn', False) for kp_conv_setting in init['kp_conv']):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kplstm.FPModule(**init)([test_data1, test_data2])

                else:
                    # Valid initialization and test data
                    fp_module = kplstm.FPModule(**init)
                    try:
                        output = fp_module([test_data1, test_data2])
                    except Exception as e:
                        output = fp_module([test_data1, test_data2])
                        print(error_msg)
                        raise e

                    # Get number of output points
                    if init['kp_conv']:
                        if init['kp_conv'][-1]['npoint'] is not None:
                            nbr_out_points = init['kp_conv'][-1]['npoint']
                        else:
                            nbr_out_points = input_shape[3]
                    else:
                        nbr_out_points = input_shape[3]

                    # Get number of output channels
                    if init['kp_conv']:
                        if init['kp_conv'][-1]['use_xyz']:
                            nbr_out_chan = 3 + init['kp_conv'][-1]['filters']
                        else:
                            nbr_out_chan = init['kp_conv'][-1]['filters']
                    else:
                        if init['kpfp'].get('use_xyz', True):
                            nbr_out_chan = 3 + init['kpfp'].get('filters', 1)
                        else:
                            nbr_out_chan = init['kpfp'].get('filters', 1)

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_chan, nbr_out_points]), output, msg=error_msg)

                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_points, nbr_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    self.assertEqual(fp_module.compute_output_shape([test_data1.shape, test_data2.shape]), output.shape, msg=error_msg)


class TestEncoder(tf.test.TestCase):
    """
    Unit test of the kplstm Encoder layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestEncoder, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the Encoder class
        self.keywords = [
            'sa',
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
        sa = [[]] + [[{'kp_conv_lstm': {'filters': 1, 'k_points': 2, 'nsample': 1, 'knn': False, 'use_xyz': True, 'return_sequences': False, 'stateful': False, 'data_format': 'channels_last'}}],
                     [{'kp_conv_lstm': {'filters': 1, 'k_points': 2, 'nsample': 1, 'knn': False, 'use_xyz': True, 'return_sequences': False, 'stateful': True, 'data_format': 'channels_last'}}],
                     [{'kp_conv_lstm': {'filters': 1, 'k_points': 2, 'nsample': 4, 'knn': True, 'use_xyz': True, 'return_sequences': False, 'stateful': False, 'data_format': 'channels_last'}}],
                     [{'kp_conv': [{'filters': 8, 'k_points': 5, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}],
                       'kp_conv_lstm': {'filters': 8, 'k_points': 5, 'nsample': 1, 'knn': False, 'use_xyz': True, 'return_sequences': True, 'stateful': False, 'data_format': 'channels_last'},
                       'kp_conv2': [{'filters': 8, 'k_points': 5, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}]},
                      {'kp_conv': [{'filters': 8, 'k_points': 5, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}, {'filters': 8, 'k_points': 5, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}],
                       'kp_conv_lstm': {'filters': 16, 'k_points': 7, 'nsample': 4, 'knn': False, 'use_xyz': False, 'return_sequences': False, 'stateful': False, 'data_format': 'channels_first'}}]]
        data_format = ['channels_last', 'channels_first']
        name = [None]

        # Get all possible combinations of the Encoder layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(sa, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=8, size=1))]

        # Define number of timesteps of the test data (the value of high is chosen completely arbitrary)
        timesteps = [1] + [int(np.random.randint(low=2, high=8, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points = [1] + [int(np.random.randint(low=2, high=32, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, timesteps, nbr_of_points, nbr_of_channels))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[1], input_shape[3], input_shape[2]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute Encoder layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        encoder = kplstm.Encoder(**init)

                elif any(sa_setting['kp_conv_lstm'].get('stateful', False) for sa_setting in init['sa']) and len(init['sa']) > 1:
                    with self.assertRaises(NotImplementedError, msg=error_msg):
                        encoding, state, states = kplstm.Encoder(**init)(test_data)

                elif any(sa_setting['kp_conv_lstm'].get('stateful', False) for sa_setting in init['sa']):
                    with self.assertRaises(NotImplementedError, msg=error_msg):
                        encoding, state = kplstm.Encoder(**init)(test_data)

                elif input_shape[3] < 3 and init['sa'] and len(init['sa']) > 1:
                    with self.assertRaises(ValueError, msg=error_msg):
                        encoding, state, states = kplstm.Encoder(**init)(test_data)

                elif input_shape[3] < 3 and init['sa']:
                    with self.assertRaises(ValueError, msg=error_msg):
                        encoding, state = kplstm.Encoder(**init)(test_data)

                elif any(is_not_none_and_greater(sa_setting['kp_conv_lstm'].get('nsample', None), input_shape[2]) and sa_setting['kp_conv_lstm'].get('knn', False) for sa_setting in init['sa']) and len(init['sa']) > 1:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        encoding, state, states = kplstm.Encoder(**init)(test_data)

                elif any(is_not_none_and_greater(sa_setting['kp_conv_lstm'].get('nsample', None), input_shape[2]) and sa_setting['kp_conv_lstm'].get('knn', False) for sa_setting in init['sa']):
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        encoding, state, states = kplstm.Encoder(**init)(test_data)

                else:
                    # Valid initialization and test data
                    encoder = kplstm.Encoder(**init)
                    try:
                        if len(init['sa']) > 1:
                            encoding, state, states = encoder(test_data)
                        else:
                            encoding, state = encoder(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Helper function
                    def _get_output_shape(sa_settings, state=False):
                        if state:
                            sa_settings = sa_settings[:-1]

                        if sa_settings:
                            # Determine number of output points
                            for sa_setting in reversed(sa_settings):
                                nbr_out_points = next((kp_conv_setting['npoint'] for kp_conv_setting in reversed(sa_setting.get('kp_conv', []) + sa_setting.get('kp_conv2', [])) if kp_conv_setting['npoint'] is not None), None)
                                if nbr_out_points is not None:
                                    break

                            if nbr_out_points is None:
                                nbr_out_points = input_shape[2]

                            # Determine number of output channels
                            if sa_settings[-1].get('kp_conv2', False) and sa_settings[-1]['kp_conv2'][-1].get('use_xyz', True):
                                nbr_out_chan = 3 + sa_settings[-1]['kp_conv2'][-1].get('filters', 1)
                            elif sa_settings[-1].get('kp_conv2', False):
                                nbr_out_chan = sa_settings[-1]['kp_conv2'][-1].get('filters', 1)
                            elif sa_settings[-1]['kp_conv_lstm'].get('use_xyz', True):
                                nbr_out_chan = 3 + sa_settings[-1]['kp_conv_lstm'].get('filters', 1)
                            else:
                                nbr_out_chan = sa_settings[-1]['kp_conv_lstm'].get('filters', 1)
                        else:
                            nbr_out_points = input_shape[2]
                            nbr_out_chan = input_shape[3]

                        return nbr_out_points, nbr_out_chan

                    # Determine encoding shape
                    nbr_out_points, nbr_out_chan = _get_output_shape(init['sa'])

                    if init['data_format'] == 'channels_first':
                        encoding_shape = [input_shape[0], input_shape[1], nbr_out_chan, nbr_out_points]
                    else:
                        encoding_shape = [input_shape[0], input_shape[1], nbr_out_points, nbr_out_chan]

                    # Determine state shape
                    nbr_out_points, nbr_out_chan = _get_output_shape(init['sa'], state=True)

                    if init['data_format'] == 'channels_first':
                        state_shape = [input_shape[0], input_shape[1], nbr_out_chan, nbr_out_points]
                    else:
                        state_shape = [input_shape[0], input_shape[1], nbr_out_points, nbr_out_chan]

                    # Determine states shape
                    if len(init['sa']) > 1:
                        states_shape = []
                        for i, sa_setting in enumerate(init['sa']):
                            if i < len(init['sa']) - 1:
                                nbr_encoding_out_points, nbr_encoding_out_chan = _get_output_shape(init['sa'][:i + 1])
                                nbr_state_out_points, nbr_state_out_chan = _get_output_shape(init['sa'][:i + 1], state=True)

                                if init['data_format'] == 'channels_first':
                                    states_shape.append([[input_shape[0], input_shape[1], nbr_encoding_out_chan, nbr_encoding_out_points],
                                                        [input_shape[0], input_shape[1], nbr_state_out_chan, nbr_state_out_points]])
                                else:
                                    states_shape.append([[input_shape[0], input_shape[1], nbr_encoding_out_points, nbr_encoding_out_chan],
                                                        [input_shape[0], input_shape[1], nbr_state_out_points, nbr_state_out_chan]])
                    else:
                        states_shape = []

                    # Check output shapes
                    self.assertShapeEqual(np.empty(shape=encoding_shape), encoding, msg=error_msg)
                    self.assertShapeEqual(np.empty(shape=state_shape), state, msg=error_msg)
                    if len(init['sa']) > 1:
                        states = tf.nest.map_structure(lambda x: x.shape, states)
                        self.assertListEqual(states_shape, states, msg=error_msg)

                    # Test compute_output_shape function
                    if len(init['sa']) > 1:
                        self.assertTupleEqual(encoder.compute_output_shape(test_data.shape), (encoding.shape, state.shape, states), msg=error_msg)
                    else:
                        self.assertTupleEqual(encoder.compute_output_shape(test_data.shape), (encoding.shape, state.shape), msg=error_msg)


class TestDecoder(tf.test.TestCase):
    """
    Unit test of the kplstm Decoder layer.
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
            'data_format',
            'name'
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
        fp = [[]] + [[{'kpfp': {'filters': 1, 'k_points': 2, 'nsample': None, 'knn': False, 'use_xyz': True},
                       'kp_conv': [{'filters': 1, 'k_points': 2, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}]}],
                     [{'kpfp': {'filters': 1, 'k_points': 2, 'nsample': None, 'knn': False, 'use_xyz': True},
                       'kp_conv': [{'filters': 1, 'k_points': 2, 'npoint': 4, 'nsample': 1, 'knn': True, 'use_xyz': True}]}],
                     [{'kpfp': {'filters': 1, 'k_points': 2, 'nsample': None, 'knn': False, 'use_xyz': False},
                       'kp_conv': [{'filters': 1, 'k_points': 2, 'npoint': 4, 'nsample': 4, 'knn': False, 'use_xyz': False}]}],
                     [{'kpfp': {'filters': 8, 'k_points': 2, 'nsample': None, 'knn': False, 'use_xyz': True},
                       'kp_conv': [{'filters': 8, 'k_points': 5, 'npoint': None, 'nsample': 1, 'knn': False, 'use_xyz': True}]},
                      {'kpfp': {'filters': 16, 'k_points': 5, 'nsample': None, 'knn': False, 'use_xyz': True},
                       'kp_conv': [{'filters': 16}, {'filters': 16}]}]]
        data_format = ['channels_last', 'channels_first']
        name = [None]

        # Get all possible combinations of the Decoder settings (the order has to fit to self.keywords)
        settings = list(itertools.product(fp, data_format, name))

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
                        decoder = kplstm.Decoder(**init)

                elif len(states_test_data) != max(len(init['fp']) - 1, 0):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        decoding = kplstm.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                elif input_shape[encoding_shape_ind][chan_dim_ind] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        decoding = kplstm.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                elif input_shape[state_shape_ind][chan_dim_ind] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        decoding = kplstm.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                elif any(fp_setting.get('kpfp', {}).get('filters', 1) < 3 and not fp_setting.get('kpfp', {}).get('use_xyz', True) for fp_setting in init['fp']):
                    with self.assertRaises(ValueError, msg=error_msg):
                        decoding = kplstm.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                else:
                    # Valid initialization and test data
                    decoder = kplstm.Decoder(**init)
                    try:
                        decoding = decoder([encoding_test_data, state_test_data, states_test_data])
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output points
                    if len(init['fp']) > 1:
                        nbr_out_points = input_shape[states_shape_ind][-1][state_shape_ind][point_dim_ind]
                    elif init['fp']:
                        nbr_out_points = input_shape[state_shape_ind][point_dim_ind]
                    else:
                        nbr_out_points = input_shape[encoding_shape_ind][point_dim_ind]

                    # Determine number of output channels
                    if init['fp']:
                        if init['fp'][-1].get('kp_conv', []):
                            if init['fp'][-1]['kp_conv'][-1].get('use_xyz', True):
                                nbr_out_chan = 3 + init['fp'][-1]['kp_conv'][-1].get('filters', 1)
                            else:
                                nbr_out_chan = init['fp'][-1]['kp_conv'][-1].get('filters', 1)
                        elif init['fp'][-1].get('kpfp', {}):
                            if init['fp'][-1]['kpfp'].get('use_xyz', True):
                                nbr_out_chan = 3 + init['fp'][-1]['kpfp'].get('filters', 1)
                            else:
                                nbr_out_chan = init['fp'][-1]['kpfp'].get('filters', 1)
                        else:
                            # xyz + filters
                            nbr_out_chan = 3 + 1
                    else:
                        nbr_out_chan = input_shape[encoding_shape_ind][chan_dim_ind]

                    # Check output shapes
                    if init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[encoding_shape_ind][batch_dim_ind], nbr_out_chan, nbr_out_points]), decoding, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[encoding_shape_ind][batch_dim_ind], nbr_out_points, nbr_out_chan]), decoding, msg=error_msg)

                    # Test compute_output_shape function
                    states_test_data_shape = tf.nest.map_structure(lambda x: x.shape, states_test_data)
                    self.assertEqual(decoder.compute_output_shape([encoding_test_data.shape, state_test_data.shape, states_test_data_shape]), decoding.shape, msg=error_msg)


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
