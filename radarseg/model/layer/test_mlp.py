# Standard Libraries
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model.layer.mlp import MLP


# Helper function
def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


class TestMLP(tf.test.TestCase):
    """
    Unit test of the MLP layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestMLP, self).setUp()

        # Define keywords of the SequenceExampleGenerator class
        self.keywords = [
            'filters',
            'kernel_size',
            'strides',
            'padding',
            'data_format',
            'activation',
            'use_bias',
            'kernel_initializer',
            'bias_initializer',
            'kernel_regularizer',
            'bias_regularizer',
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
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestMLP, self).tearDown()

    def test_call(self):
        """
        Test of the MLP call function.

        Compares the dimensions and values of the layer output with corresponding reference dimensions and values.

        The number of tested attribute settings is limited to representative scenarios for the intended use case
        of point cloud processing. This limitation was necessary due to limited CI server capacities and the hight
        number of initialization possibilities (> 10⁹). This limitation affects the following attributes: kernel_size,
        strides, padding, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, bn, momentum,
        epsilon, beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer,
        beta_regularizer and gamma_regularizer.

        Note: No tests are executed for bn = True (as mentioned above and indicated by the TODO below).

        Tests:
            Test 1: Zero assignment (Input tensor with all values equals zero).
            Test 2: Identity assignment (Input tensor with all values equals one).
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        # TODO: Extend test for bn = True
        filters = [1] + [int(np.random.randint(low=1, high=16, size=1))]
        kernel_size = [[1, 1]]
        strides = [[1, 1]]
        padding = ['valid']
        data_format = ['channels_last', 'channels_first']
        activation = [None] + ['relu']
        use_bias = [True, False]
        kernel_initializer = ['zeros', 'ones']
        bias_initializer = ['zeros', 'ones']
        kernel_regularizer = [None]
        bias_regularizer = [None]
        bn = [False]
        momentum = [0.99]
        epsilon = [0.001]
        center = [True, False]
        scale = [True, False]
        beta_initializer = ['ones']
        gamma_initializer = ['ones']
        moving_mean_initializer = ['zeros']
        moving_variance_initializer = ['ones']
        beta_regularizer = [None]
        gamma_regularizer = [None]
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of sequence example generator settings (the order has to fit to self.keywords)
        def get_settings():
            settings = itertools.product(filters, kernel_size, strides, padding, data_format, activation, use_bias, kernel_initializer,
                                         bias_initializer, kernel_regularizer, bias_regularizer, bn, momentum, epsilon, center, scale,
                                         beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer,
                                         beta_regularizer, gamma_regularizer, name)
            return settings

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of rows of the test data (the value of high is chosen completely arbitrary)
        nbr_of_rows = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of columns of the test data (the value of high is chosen completely arbitrary)
        nbr_of_cols = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of channels of the test data (the value of high is chosen completely arbitrary)
        nbr_of_channels = [1] + [int(np.random.randint(low=1, high=128, size=1))]

        # Get all possible input data shape combinations
        def get_input_shapes():
            input_shapes = itertools.product(batch_size, nbr_of_rows, nbr_of_cols, nbr_of_channels)
            return input_shapes

        # Get test settings and test data input shapes
        settings = get_settings()
        input_shapes = get_input_shapes()

        # Test 1: Zero assignment
        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = dict(zip(self.keywords, list(setting)))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.zeros(shape=shape, dtype=np.float32), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
                    test_data = tf.constant(np.zeros(shape=shape, dtype=np.float32), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute MLP layer
                output = MLP(**init)(test_data)

                # Check output shape
                if init['padding'] == 'valid':
                    nbr_of_new_rows = int(np.ceil((input_shape[1] - init['kernel_size'][0] + 1) / init['kernel_size'][0]))
                    nbr_of_new_cols = int(np.ceil((input_shape[2] - init['kernel_size'][1] + 1) / init['kernel_size'][1]))
                else:
                    nbr_of_new_rows = int(np.ceil(input_shape[1] / init['kernel_size'][0]))
                    nbr_of_new_cols = int(np.ceil(input_shape[2] / init['kernel_size'][1]))

                if init['data_format'] == 'channels_first':
                    new_shape = [input_shape[0], init['filters'], nbr_of_new_rows, nbr_of_new_cols]
                    self.assertShapeEqual(np.empty(shape=new_shape), output, msg=error_msg)
                    self.assertAllEqual(MLP(**init).compute_output_shape(test_data.shape), tf.shape(output))
                elif init['data_format'] == 'channels_last':
                    new_shape = [input_shape[0], nbr_of_new_rows, nbr_of_new_cols, init['filters']]
                    self.assertShapeEqual(np.empty(shape=new_shape), output, msg=error_msg)
                    self.assertAllEqual(MLP(**init).compute_output_shape(test_data.shape), tf.shape(output))

                # Check output values
                # TODO: Add checks for bn = True
                if init['use_bias'] and init['bias_initializer'] == 'ones':
                    self.assertAllClose(tf.ones_like(output, dtype=tf.float32), output, msg=error_msg)
                else:
                    self.assertAllClose(tf.zeros_like(output, dtype=tf.float32), output, msg=error_msg)

    # Get test settings and test data input shapes
        settings = get_settings()
        input_shapes = get_input_shapes()

    # Test 2: Identity assignment
        for input_shape in input_shapes:
            for setting in settings:
                # Initialization dictionary
                init = dict(zip(self.keywords, list(setting)))

                # Generate synthetic test data (point cloud)
                if init['data_format'] == 'channels_first':
                    shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
                    test_data = tf.constant(np.ones(shape=shape, dtype=np.float32), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
                    test_data = tf.constant(np.ones(shape=shape, dtype=np.float32), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute MLP layer
                output = MLP(**init)(test_data)

                # Check output shape
                if init['padding'] == 'valid':
                    nbr_of_new_rows = int(np.ceil((input_shape[1] - init['kernel_size'][0] + 1) / init['kernel_size'][0]))
                    nbr_of_new_cols = int(np.ceil((input_shape[2] - init['kernel_size'][1] + 1) / init['kernel_size'][1]))
                else:
                    nbr_of_new_rows = int(np.ceil(input_shape[1] / init['kernel_size'][0]))
                    nbr_of_new_cols = int(np.ceil(input_shape[2] / init['kernel_size'][1]))

                if init['data_format'] == 'channels_first':
                    new_shape = [input_shape[0], init['filters'], nbr_of_new_rows, nbr_of_new_cols]
                    self.assertShapeEqual(np.empty(shape=new_shape), output, msg=error_msg)
                    self.assertAllEqual(MLP(**init).compute_output_shape(test_data.shape), tf.shape(output))
                elif init['data_format'] == 'channels_last':
                    new_shape = [input_shape[0], nbr_of_new_rows, nbr_of_new_cols, init['filters']]
                    self.assertShapeEqual(np.empty(shape=new_shape), output, msg=error_msg)
                    self.assertAllEqual(MLP(**init).compute_output_shape(test_data.shape), tf.shape(output))

                # Check output values
                # TODO: Add checks for bn = True
                if init['kernel_initializer'] == 'ones' and init['use_bias'] and init['bias_initializer'] == 'ones':
                    self.assertAllClose(tf.add(tf.ones_like(output, dtype=tf.float32), tf.constant([1], dtype=tf.float32)), output, msg=error_msg)

                elif init['kernel_initializer'] == 'ones' and init['use_bias'] and init['bias_initializer'] == 'zeros':
                    self.assertAllClose(tf.ones_like(output, dtype=tf.float32), output, msg=error_msg)

                elif init['kernel_initializer'] == 'ones' and not init['use_bias']:
                    self.assertAllClose(tf.ones_like(output, dtype=tf.float32), output, msg=error_msg)

                elif init['kernel_initializer'] == 'zeros' and init['use_bias'] and init['bias_initializer'] == 'ones':
                    self.assertAllClose(tf.ones_like(output, dtype=tf.float32), output, msg=error_msg)

                elif init['kernel_initializer'] == 'zeros' and init['use_bias'] and init['bias_initializer'] == 'zeros':
                    self.assertAllClose(tf.zeros_like(output, dtype=tf.float32), output, msg=error_msg)

                elif init['kernel_initializer'] == 'zeros' and not init['use_bias']:
                    self.assertAllClose(tf.zeros_like(output, dtype=tf.float32), output, msg=error_msg)

                else:
                    print(error_msg)
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
