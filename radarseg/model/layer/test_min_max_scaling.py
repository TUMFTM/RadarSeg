# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.model.layer import min_max_scaling


# Helper function
def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def to_list(value):
    if isinstance(value, list):
        return value

    return [value]


def is_not_none_and_greater_equal(x1, x2):
    if x1 is not None and x2 is not None:
        return np.any(np.greater_equal(x1, x2))
    else:
        return False


class TestMinMaxScaling(tf.test.TestCase):
    """
    Unit test of the MinMaxScaling layer.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestMinMaxScaling, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the MinMaxScaling class
        self.keywords = [
            'minimum',
            'maximum',
            'data_format',
            'name'
        ]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestMinMaxScaling, self).tearDown()

    def test_call(self):
        """
        Test of the MinMaxScaling call function.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        minimum = [None, 1] + [float(np.random.uniform(low=0.1, high=128.0, size=1))]
        maximum = [None, 1] + [float(np.random.uniform(low=0.1, high=128.0, size=1))]
        data_format = ['channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the MinMaxScaling layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(minimum, maximum, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

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

                # Execute MinMaxScaling layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = min_max_scaling.MinMaxScaling(**init)(test_data)

                elif is_not_none_and_greater_equal(init['minimum'], init['maximum']):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        output = min_max_scaling.MinMaxScaling(**init)(test_data)

                elif init['minimum'] is not None and len(to_list(init['minimum'])) != input_shape[2] and len(to_list(init['minimum'])) != 1:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = min_max_scaling.MinMaxScaling(**init)(test_data)

                elif init['maximum'] is not None and len(to_list(init['maximum'])) != input_shape[2] and len(to_list(init['maximum'])) != 1:
                    with self.assertRaises(ValueError, msg=error_msg):
                        output = min_max_scaling.MinMaxScaling(**init)(test_data)

                else:
                    # Valid initialization and test data
                    scaling_layer = min_max_scaling.MinMaxScaling(**init)
                    output = scaling_layer(test_data)

                    # Check output shape
                    self.assertShapeEqual(np.empty(shape=shape), output, msg=error_msg)

                    # Check compute_output_shape function
                    self.assertEqual(scaling_layer.compute_output_shape(test_data.shape), test_data.shape)

                    # Check output values
                    if init['minimum'] is None and init['maximum'] is None:
                        self.assertAllGreaterEqual(output, 0.0)
                        self.assertAllLessEqual(output, 1.0)

    def test_model(self):
        """
        Test of the MinMaxScaling model execution.
        """
        # Define layer attribute settings
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        minimum = [None] + [0]
        maximum = [None] + [1]
        data_format = ['channels_first', 'no-valid-data-format']
        name = [None, ''] + [get_random_name(length=16)]

        # Get all possible combinations of the MinMaxScaling layer settings (the order has to fit to self.keywords)
        settings = list(itertools.product(minimum, maximum, data_format, name))

        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        nbr_of_points = [1] + [int(np.random.randint(low=1, high=128, size=1))]

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
                    inputs = tf.keras.Input(shape=(input_shape[2], None))
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')
                else:
                    shape = [input_shape[0], input_shape[1], input_shape[2]]
                    inputs = tf.keras.Input(shape=(None, input_shape[2]))
                    test_data = tf.constant(np.random.random(size=shape), dtype=tf.float32, shape=shape, name='test_data')

                # Set error message (if error occurs)
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Execute MinMaxScaling layer
                if init['data_format'] not in set(('channels_last', 'channels_first')):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        outputs = min_max_scaling.MinMaxScaling(**init)(inputs)

                elif is_not_none_and_greater_equal(init['minimum'], init['maximum']):
                    with self.assertRaises(AssertionError, msg=error_msg):
                        outputs = min_max_scaling.MinMaxScaling(**init)(test_data)

                elif init['minimum'] is not None and len(to_list(init['minimum'])) != input_shape[2] and len(to_list(init['minimum'])) != 1:
                    with self.assertRaises(ValueError, msg=error_msg):
                        outputs = min_max_scaling.MinMaxScaling(**init)(inputs)

                elif init['maximum'] is not None and len(to_list(init['maximum'])) != input_shape[2] and len(to_list(init['maximum'])) != 1:
                    with self.assertRaises(ValueError, msg=error_msg):
                        outputs = min_max_scaling.MinMaxScaling(**init)(inputs)

                else:
                    # Valid initialization and test data
                    scaling_layer = min_max_scaling.MinMaxScaling(**init)
                    outputs = scaling_layer(inputs)

                    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

                    prediction = model.predict(test_data)

                    # Check output shape
                    self.assertShapeEqual(np.empty(shape=shape), tf.convert_to_tensor(prediction, dtype=tf.float32), msg=error_msg)

                    # Check output values
                    self.assertAllGreaterEqual(tf.convert_to_tensor(prediction, dtype=tf.float32), 0.0)
                    self.assertAllLessEqual(tf.convert_to_tensor(prediction, dtype=tf.float32), 1.0)


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
