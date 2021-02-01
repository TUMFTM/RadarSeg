# Standard Libraries
import copy
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model.layer import kpconv_lstm


# Helper function
def is_not_none_and_less(value, minimum):
    if value is None:
        return False
    elif value < minimum:
        return True
    else:
        return False


class TestKPConvLSTM(tf.test.TestCase):
    """
    Unit test of the KPConvLSTM layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestKPConvLSTM, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the KPConvLSTM class
        self.keywords = [
            'filters',
            'k_points',
            'nsample',
            'radius',
            'kp_extend',
            'fixed',
            'knn',
            'kp_influence',
            'aggregation_mode',
            'use_xyz',
            'activation',
            'recurrent_activation',
            'kernel_initializer',
            'kernel_regularizer',
            'dropout',
            'recurrent_dropout',
            'upsample_mode',
            'data_format',
            'return_sequences',
            'return_state',
            'go_backwards',
            'stateful'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestKPConvLSTM, self).tearDown()

    def test_call(self):
        """
        Test of the KPConvLSTM call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        filters = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        k_points = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        nsample = [0] + [int(np.random.randint(low=1, high=16, size=1))]
        radius = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        kp_extend = [0] + [float(np.random.uniform(low=0.1, high=8.0, size=1))]
        fixed = ['none', 'center', 'verticals']
        knn = [True, False]
        kp_influence = ['constant', 'linear', 'gaussian']
        aggregation_mode = ['closest', 'sum']
        use_xyz = [True, False]
        activation = [None] + ['tanh']
        recurrent_activation = [None] + ['sigmoid']
        kernel_initializer = [None] + ['glorot_uniform']
        kernel_regularizer = [None] + ['l2']
        dropout = [0.0] + [float(np.random.uniform(low=0.1, high=1.0, size=1))]
        recurrent_dropout = [0.0] + [float(np.random.uniform(low=0.1, high=1.0, size=1))]
        upsample_mode = ['nearest', 'threenn']
        data_format = ['channels_last', 'channels_first']
        return_sequences = [True, False]
        return_state = [True, False]
        go_backwards = [True, False]
        stateful = [True, False]

        # Get all possible combinations of the KPConvLSTM layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(filters, k_points, nsample, radius, kp_extend, fixed, knn, kp_influence, aggregation_mode, use_xyz,
                                          activation, recurrent_activation, kernel_initializer, kernel_regularizer, dropout, recurrent_dropout,
                                          upsample_mode, data_format, return_sequences, return_state, go_backwards, stateful))

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

                # Execute KPConvLSTM layer
                if init['filters'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['k_points'] < 1:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['k_points'] < 2:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif is_not_none_and_less(init['nsample'], 1):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['radius'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['kp_extend'] <= 0:
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['fixed'] not in set(('none', 'center', 'verticals')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['kp_influence'] not in set(('constant', 'linear', 'gaussian')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['aggregation_mode'] not in set(('closest', 'sum')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)

                elif init['stateful']:
                    with self.assertRaises(NotImplementedError, msg=error_msg):
                        output = kpconv_lstm.KPConvLSTM(**init)(test_data)

                elif input_shape[3] < 3:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = kpconv_lstm.KPConvLSTM(**init)(test_data)

                elif input_shape[2] < init['nsample'] and init['knn']:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        output = kpconv_lstm.KPConvLSTM(**init)(test_data)

                else:
                    # Valid initialization and test data
                    kp_conv_lstm = kpconv_lstm.KPConvLSTM(**init)
                    try:
                        output = kp_conv_lstm(test_data)
                    except Exception as e:
                        print(error_msg)
                        raise e

                    # Determine number of output channels
                    if init['use_xyz']:
                        num_out_chan = 3 + init['filters']
                    else:
                        num_out_chan = init['filters']

                    # Check output shapes
                    if init['return_state'] and init['return_sequences'] and init['data_format'] == 'channels_first':
                        output, hidden_state, cell_state = output
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], num_out_chan, input_shape[2]]), output, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), hidden_state, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), cell_state, msg=error_msg)

                    elif init['return_state'] and init['return_sequences']:
                        output, hidden_state, cell_state = output
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], input_shape[2], num_out_chan]), output, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), hidden_state, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), cell_state, msg=error_msg)

                    elif init['return_state'] and init['data_format'] == 'channels_first':
                        output, hidden_state, cell_state = output
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), output, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), hidden_state, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), cell_state, msg=error_msg)

                    elif init['return_state']:
                        output, hidden_state, cell_state = output
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), output, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), hidden_state, msg=error_msg)
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), cell_state, msg=error_msg)

                    elif init['return_sequences'] and init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], num_out_chan, input_shape[2]]), output, msg=error_msg)

                    elif init['return_sequences']:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[1], input_shape[2], num_out_chan]), output, msg=error_msg)

                    elif init['data_format'] == 'channels_first':
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], num_out_chan, input_shape[2]]), output, msg=error_msg)

                    else:
                        self.assertShapeEqual(np.empty(shape=[input_shape[0], input_shape[2], num_out_chan]), output, msg=error_msg)

                    # Test compute_output_shape function
                    if init['return_state']:
                        self.assertTupleEqual(kp_conv_lstm.compute_output_shape(test_data.shape),
                                              (output.shape, hidden_state.shape, cell_state.shape),
                                              msg=error_msg)
                    else:
                        self.assertEqual(kp_conv_lstm.compute_output_shape(test_data.shape), output.shape, msg=error_msg)


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
