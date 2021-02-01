# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model import pointnet2


# Helper function
def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


class TestSampleAndGroup(tf.test.TestCase):
    """
    Unit test of the SampleAndGroup layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestSampleAndGroup, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the SampleAndGroup class
        self.keywords = [
            'npoint',
            'radius',
            'nsample',
            'knn',
            'use_xyz',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestSampleAndGroup, self).tearDown()

    def test_call(self):
        """
        Test of the SampleAndGroup call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        npoint = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        radius = [0] + [float(np.random.uniform(low=0.1, high=128.0, size=1))]
        nsample = [0] + [int(np.random.randint(low=1, high=32, size=1))]
        knn = [True, False]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the SampleAndGroup layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(npoint, radius, nsample, knn, use_xyz, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [int(np.random.randint(low=1, high=128, size=1))]

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

                # Execute SampleAndGroup layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroup(**init)(test_data)

                elif init['npoint'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroup(**init)(test_data)

                elif init['radius'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroup(**init)(test_data)

                elif init['nsample'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroup(**init)(test_data)

                elif input_shape[1] < init['nsample']:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroup(**init)(test_data)

                else:
                    # Valid initialization and test data
                    sample_and_group = pointnet2.SampleAndGroup(**init)
                    new_xyz, new_points, idx, grouped_xyz = sample_and_group(test_data)

                    # Check output shapes
                    if init['use_xyz'] and init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], 3]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample'], input_shape[2]]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample']]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample'], 3]), grouped_xyz, msg=error_msg)

                    elif init['use_xyz'] and init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, init['npoint']]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], init['npoint'], init['nsample']]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample']]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, init['npoint'], init['nsample']]), grouped_xyz, msg=error_msg)

                    elif init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], 3]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample'], input_shape[2] - 3]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample']]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample'], 3]), grouped_xyz, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, init['npoint']]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2] - 3, init['npoint'], init['nsample']]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['npoint'], init['nsample']]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, init['npoint'], init['nsample']]), grouped_xyz, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertTupleEqual(sample_and_group.compute_output_shape(test_data.shape),
                                          (new_xyz.shape, new_points.shape, idx.shape, grouped_xyz.shape),
                                          msg=error_msg)


class TestSampleAndGroupAll(tf.test.TestCase):
    """
    Unit test of the SampleAndGroupAll layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestSampleAndGroupAll, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the SampleAndGroupAll class
        self.keywords = [
            'use_xyz',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestSampleAndGroupAll, self).tearDown()

    def test_call(self):
        """
        Test of the SampleAndGroupAll call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the SampleAndGroupAll layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(use_xyz, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [int(np.random.randint(low=1, high=128, size=1))]

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

                # Execute SampleAndGroupAll layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points, idx, grouped_xyz = pointnet2.SampleAndGroupAll(**init)(test_data)
                else:
                    # Valid initialization and test data
                    sample_and_group_all = pointnet2.SampleAndGroupAll(**init)
                    new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(test_data)

                    # Check output shapes
                    if init['use_xyz'] and init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, 3]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1], input_shape[2]]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1]]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1], 3]), grouped_xyz, msg=error_msg)

                    elif init['use_xyz'] and init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, 1]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], 1, input_shape[1]]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1]]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, 1, input_shape[1]]), grouped_xyz, msg=error_msg)

                    elif init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, 3]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1], input_shape[2] - 3]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1]]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1], 3]), grouped_xyz, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, 1]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2] - 3, 1, input_shape[1]]), new_points, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 1, input_shape[1]]), idx, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, 1, input_shape[1]]), grouped_xyz, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertTupleEqual(sample_and_group_all.compute_output_shape(test_data.shape),
                                          (new_xyz.shape, new_points.shape, idx.shape, grouped_xyz.shape),
                                          msg=error_msg)


class TestSAModule(tf.test.TestCase):
    """
    Unit test of the SAModule.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestSAModule, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the SAModule class
        self.keywords = [
            'npoint',
            'radius',
            'nsample',
            'mlp',
            'mlp2',
            'group_all',
            'pooling',
            'knn',
            'use_xyz',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestSAModule, self).tearDown()

    def test_call(self):
        """
        Test of the SAModule call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        npoint = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        radius = [0] + [float(np.random.uniform(low=0.1, high=128.0, size=1))]
        nsample = [0] + [int(np.random.randint(low=1, high=32, size=1))]
        mlp = [[]] + [[{'filters': 16}, {'filters': 8}, {'filters': 4}]]
        mlp2 = [[]] + [[{'filters': 4}, {'filters': 4}]]
        group_all = [True, False]
        pooling = ['max', 'avg', 'no-valid-pooling']
        knn = [True, False]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, '']

        # Get all possible combinations of the SAModule settings (the order has to fit to self.keywords)
        settings = list(itertools.product(npoint, radius, nsample, mlp, mlp2, group_all, pooling, knn, use_xyz, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

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

                # Execute SAModule layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif init['pooling'] not in set(('max', 'avg')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif init['npoint'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif init['radius'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif init['nsample'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif input_shape[1] < init['nsample'] and not init['group_all'] and init['knn']:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        new_xyz, new_points = pointnet2.SAModule(**init)(test_data)

                elif input_shape[2] <= 3 and not init['use_xyz'] and (init['mlp'] or init['mlp2']):
                    # Call of the keras Conv2D layer with an empty tensor (one dimension equals zero)
                    # leads to an fatal python error (FloatingPointError), why this test needs to
                    # be skipped. -> Aborted (core dumped).
                    continue

                else:
                    # Valid initialization and test data
                    sa_module = pointnet2.SAModule(**init)
                    new_xyz, new_points = sa_module(test_data)

                    # Determine output channel dimensions
                    if init['mlp2']:
                        nbr_out_chan = init['mlp2'][-1]['filters']
                    elif init['mlp']:
                        nbr_out_chan = init['mlp'][-1]['filters']
                    elif init['use_xyz']:
                        nbr_out_chan = input_shape[2]
                    else:
                        nbr_out_chan = input_shape[2] - 3

                    # Determine output point dimension (number of output points)
                    if init['group_all']:
                        nbr_out_points = 1
                    else:
                        nbr_out_points = init['npoint']

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_points, 3]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_points, nbr_out_chan]), new_points, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], 3, nbr_out_points]), new_xyz, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_chan, nbr_out_points]), new_points, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertTupleEqual(sa_module.compute_output_shape(test_data.shape),
                                          (new_xyz.shape, new_points.shape),
                                          msg=error_msg)


class TestFPModule(tf.test.TestCase):
    """
    Unit test of the FPModule.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestFPModule, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the FPModule class
        self.keywords = [
            'mlp',
            'use_xyz',
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
        mlp = [[]] + [[{'filters': 4}, {'filters': 8}, {'filters': 16}]]
        use_xyz = [True, False]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the FPModule settings (the order has to fit to self.keywords)
        settings = list(itertools.product(mlp, use_xyz, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points1 = [1] + [int(np.random.randint(low=1, high=128, size=1))]
        nbr_of_points2 = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of three input channels have to be provided for the first input (xyz)
        nbr_of_channels1 = [3] + [int(np.random.randint(low=4, high=16, size=1))]
        nbr_of_channels2 = [3] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        input_shapes = list(itertools.product(batch_size, nbr_of_points1, nbr_of_points2, nbr_of_channels1, nbr_of_channels2))

        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape1 = [input_shape[0], input_shape[3], input_shape[1]]
                    shape2 = [input_shape[0], input_shape[4], input_shape[2]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')
                else:
                    shape1 = [input_shape[0], input_shape[1], input_shape[3]]
                    shape2 = [input_shape[0], input_shape[2], input_shape[4]]
                    test_data1 = tf.constant(np.random.random(size=shape1), dtype=tf.float32, shape=shape1, name='test_data1')
                    test_data2 = tf.constant(np.random.random(size=shape2), dtype=tf.float32, shape=shape2, name='test_data2')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape1: {}. \n Input shape2: {}.'.format(str(init), str(shape1), str(shape2))

                # Execute FPModule layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_points, new_xyz = pointnet2.FPModule(**init)([test_data1, test_data2])

                elif input_shape[-1] <= 3 and init['mlp']:
                    # Call of the keras Conv2D layer with an empty tensor (one dimension equals zero)
                    # leads to an fatal python error (FloatingPointError), why this test needs to
                    # be skipped. -> Aborted (core dumped).
                    continue

                else:
                    # Valid initialization and test data
                    fp_module = pointnet2.FPModule(**init)
                    new_points, new_xyz = fp_module([test_data1, test_data2])

                    # Determine output channel dimensions
                    if init['mlp'] and init['use_xyz']:
                        nbr_out_chan = 3 + init['mlp'][-1]['filters']
                    elif init['mlp']:
                        nbr_out_chan = init['mlp'][-1]['filters']
                    elif init['use_xyz']:
                        nbr_out_chan = input_shape[4] + input_shape[3] - 3
                    else:
                        nbr_out_chan = input_shape[4] + input_shape[3] - 6

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], nbr_out_chan]), new_points, msg=error_msg)
                        self.assertAllClose(test_data2[:, :, :3], new_xyz, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], nbr_out_chan, input_shape[2]]), new_points, msg=error_msg)
                        self.assertAllClose(test_data2[:, :3, :], new_xyz, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertTupleEqual(fp_module.compute_output_shape((test_data1.shape, test_data2.shape)),
                                          (new_points.shape, new_xyz.shape),
                                          msg=error_msg)


class TestOutputModule(tf.test.TestCase):
    """
    Unit test of the OutputModule.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestOutputModule, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the OutputModule class
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
        super(TestOutputModule, self).tearDown()

    def test_call(self):
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        mlp = [None] + [[{'filters': 8}, {'filters': 16}]]
        dropout = [None] + [[{'rate': 0.5}]]
        out_activation = [None]
        out_use_bias = [True, False]
        out_kernel_initializer = ['ones'] + ['glorot_uniform']
        out_bias_initializer = ['zeros', 'ones']
        out_kernel_regularizer = [None] + [tf.keras.regularizers.l2]
        out_bias_regularizer = [None] + [tf.keras.regularizers.l2]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the OutputModule settings (the order has to fit to self.keywords)
        settings = list(itertools.product(num_classes, mlp, dropout, out_activation, out_use_bias, out_kernel_initializer,
                        out_bias_initializer, out_kernel_regularizer, out_bias_regularizer, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
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

                # Execute OutputModule layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet2.OutputModule(**init)(test_data)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet2.OutputModule(**init)(test_data)

                else:
                    # Valid initialization and test data
                    output_module = pointnet2.OutputModule(**init)
                    prediction = output_module(test_data)

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), prediction, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), prediction, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertEqual(output_module.compute_output_shape(test_data.shape), prediction.shape, msg=error_msg)


class TestEncoder(tf.test.TestCase):
    """
    Unit test of the Encoder.
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
        sa = [[]] + [[{'mlp': [{'filters': 1}]}], [{'mlp': [{'filters': 1}]}, {'mlp': [{'filters': 2}]}]]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the Encoder settings (the order has to fit to self.keywords)
        settings = list(itertools.product(sa, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of three input channels have to be provided for the first input (xyz)
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
                        if len(init['sa']) < 2:
                            encoding, state = pointnet2.Encoder(**init)(test_data)
                        else:
                            encoding, state, states = pointnet2.Encoder(**init)(test_data)

                else:
                    # Valid initialization and test data
                    encoder = pointnet2.Encoder(**init)
                    if len(init['sa']) < 2:
                        encoding, state = encoder(test_data)
                    else:
                        encoding, state, states = encoder(test_data)

                    # Determine output shapes
                    def _get_sa_out_shape(sa_setting, input_shape):
                        # Determine channel dimension
                        if 'mlp2' in sa_setting:
                            nbr_out_chan = sa_setting['mlp2'][-1]['filters']
                        elif 'mlp' in sa_setting:
                            nbr_out_chan = sa_setting['mlp'][-1]['filters']
                        elif 'use_xyz' in sa_setting:
                            if sa_setting['use_xyz']:
                                nbr_out_chan = 3 + input_shape[2]
                            else:
                                nbr_out_chan = input_shape[2]
                        else:
                            nbr_out_chan = input_shape[2]

                        # Determine number of points
                        if 'group_all' in sa_setting and 'npoint' in sa_setting:
                            if sa_setting['group_all']:
                                nbr_out_points = 1
                            else:
                                nbr_out_points = sa_setting['npoint']
                        elif 'group_all' in sa_setting:
                            nbr_out_points = 1
                        elif 'npoint' in sa_setting:
                            nbr_out_points = sa_setting['npoint']
                        else:
                            nbr_out_points = 1

                        # Set output shape
                        if init['data_format'] == 'channels_last':
                            output_shape = ([input_shape[0], nbr_out_points, 3], [input_shape[0], nbr_out_points, nbr_out_chan])
                        else:
                            output_shape = ([input_shape[0], 3, nbr_out_points], [input_shape[0], nbr_out_chan, nbr_out_points])

                        return output_shape

                    # Set input shape
                    encoding_shape = test_data.shape.as_list()
                    state_shape = encoding_shape

                    # Initialize shape of the states output
                    states_shape = [[None, None]] * (len(init['sa']))

                    # Get output shapes
                    for i, sa_setting in enumerate(init['sa']):
                        state_shape = encoding_shape
                        new_xyz_shape, new_point_shape = _get_sa_out_shape(sa_setting, state_shape)
                        if init['data_format'] == 'channels_last':
                            encoding_shape = np.add(new_xyz_shape, np.array([0, 0, new_point_shape[2]]))
                        else:
                            encoding_shape = np.add(new_xyz_shape, np.array([0, new_point_shape[1], 0]))
                        states_shape[i] = [encoding_shape, state_shape]

                    states_shape = states_shape[:-1]
                    states_shape.reverse()

                    # Check output shapes
                    self.assertShapeEqual(np.empty(shape=encoding_shape), encoding, msg=error_msg)
                    self.assertShapeEqual(np.empty(shape=state_shape), state, msg=error_msg)

                    if len(init['sa']) > 1:
                        for internal_state_shape, internal_state in zip(states_shape, states):
                            self.assertShapeEqual(np.empty(shape=internal_state_shape[0]), internal_state[0], msg=error_msg)
                            self.assertShapeEqual(np.empty(shape=internal_state_shape[1]), internal_state[1], msg=error_msg)

                    # Test compute output shape method
                    if len(init['sa']) < 2:
                        encoding_shape, state_shape = encoder.compute_output_shape(test_data.shape)
                    else:
                        encoding_shape, state_shape, states_shape = encoder.compute_output_shape(test_data.shape)

                    self.assertEqual(encoding_shape, encoding.shape, msg=error_msg)
                    self.assertEqual(state_shape, state.shape, msg=error_msg)
                    if len(init['sa']) > 1:
                        self.assertListEqual(states_shape, tf.nest.map_structure(tf.shape, states), msg=error_msg)


class TestDecoder(tf.test.TestCase):
    """
    Unit test of the Decoder.
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
        fp = [[]] + [[{'mlp': [{'filters': 1}]}], [{'mlp': [{'filters': 1}]}, {'mlp': [{'filters': 2}]}]]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

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
                        new_points = pointnet2.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                elif len(states_test_data) != max(len(init['fp']) - 1, 0):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        new_points = pointnet2.Decoder(**init)([encoding_test_data, state_test_data, states_test_data])

                else:
                    # Valid initialization and test data
                    decoder = pointnet2.Decoder(**init)
                    new_points = decoder([encoding_test_data, state_test_data, states_test_data])

                    # Determine number of output points
                    if not init['fp']:
                        nbr_out_points = input_shape[encoding_shape_ind][point_dim_ind]
                    elif states_test_data:
                        nbr_out_points = input_shape[states_shape_ind][-1][-1][point_dim_ind]
                    else:
                        nbr_out_points = input_shape[state_shape_ind][point_dim_ind]

                    # Determine output channel dimensions
                    if init['fp']:
                        # Get settings of the last FP layer
                        nbr_fp_layers = len(init['fp']) - 1
                        fp_setting = init['fp'][nbr_fp_layers]

                        # Determine channel shape of the last fp_layer
                        if 'mlp' in fp_setting and 'use_xyz' in fp_setting:
                            if fp_setting['use_xyz']:
                                nbr_out_chan = 3 + fp_setting['mlp'][-1]['filters']
                            else:
                                nbr_out_chan = input_shape[states_shape_ind][nbr_fp_layers][0][chan_dim_ind]

                        elif 'mlp' in fp_setting:
                            nbr_out_chan = fp_setting['mlp'][-1]['filters']

                        elif 'use_xyz' in fp_setting:
                            if fp_setting['use_xyz']:
                                nbr_out_chan = input_shape[states_shape_ind][nbr_fp_layers][0][chan_dim_ind]
                            else:
                                nbr_out_chan = input_shape[states_shape_ind][nbr_fp_layers][0][chan_dim_ind] - 3

                        else:
                            nbr_out_chan = input_shape[states_shape_ind][nbr_fp_layers][0][chan_dim_ind] - 3

                    else:
                        nbr_out_chan = input_shape[encoding_shape_ind][chan_dim_ind] - 3

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0][batch_dim_ind], nbr_out_points, nbr_out_chan]), new_points, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0][batch_dim_ind], nbr_out_chan, nbr_out_points]), new_points, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    states_test_data_shape = tf.nest.map_structure(lambda x: x.shape, states_test_data)
                    self.assertEqual(decoder.compute_output_shape((encoding_test_data.shape, state_test_data.shape, states_test_data_shape)),
                                     new_points.shape,
                                     msg=error_msg)


class TestPointNet2(tf.test.TestCase):
    """
    Unit test of the PointNet2 semantic segmentation example.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestPointNet2, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the SequenceExampleGenerator class
        self.keywords = [
            'num_classes',
            'seed',
            'data_format',
            'name'
        ]

        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        num_classes = [1] + [int(np.random.randint(low=1, high=16, size=1))]
        seed = [None] + [int(np.random.randint(low=1, high=128, size=1))]
        data_format = ['channels_last', 'channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the Encoder settings (the order has to fit to self.keywords)
        self.settings = list(itertools.product(num_classes, seed, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        # The assignment of an empty input tensor (nbr_of_points = 0) leads to an CUDA_ERROR, why it can not be tested.
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        # A minimum of three input channels have to be provided for the first input (xyz)
        nbr_of_channels = [3] + [int(np.random.randint(low=4, high=16, size=1))]

        # Get all possible input data shape combinations
        self.input_shapes = list(itertools.product(batch_size, nbr_of_points, nbr_of_channels))

    def tearDown(self):
        # Call teardown of the base test class
        super(TestPointNet2, self).tearDown()

    def test_call(self):
        """
        Test of the Encoder call function.
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

                # Execute SampleAndGroup layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet2.PointNet2(**init)(test_data)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet2.PointNet2(**init)(test_data)

                else:
                    # Valid initialization and test data
                    point_net_2 = pointnet2.PointNet2(**init)
                    prediction = point_net_2(test_data)

                    # Check output shapes
                    if init['data_format'] == 'channels_last':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], init['num_classes']]), prediction, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], init['num_classes'], input_shape[1]]), prediction, msg=error_msg)

                    else:
                        raise Exception('Test setting not handled!')

                    self.assertEqual(point_net_2.compute_output_shape(test_data.shape),
                                     prediction.shape,
                                     msg=error_msg)

    def test_model(self):
        """
        Test of a PointNet++ model execution.
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
                        prediction = pointnet2.PointNet2(**init)(test_input)

                elif init['num_classes'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        prediction = pointnet2.PointNet2(**init)(test_input)

                else:
                    # Valid layer initialization
                    point_net_2 = pointnet2.PointNet2(**init)
                    prediction = point_net_2(test_input)

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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    tf.test.main()
