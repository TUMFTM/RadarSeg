# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.train import losses


# Helper function
def get_random_one_hot(size) -> np.array:
    """
    Returns an array of random one hot encoded vectors of a defined size.

    Arguments:
        size: List or tuple with three elements (batch size, number of points, number of classes), <list or tuple>.
    """
    return tf.one_hot(indices=np.random.randint(low=-1, high=size[2], size=(size[0], size[1])), depth=size[2])


def get_random_name(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


class TestCategoricalFocalCrossentropy(tf.test.TestCase):
    """
    Unit test of the CategoricalFocalCrossentropy loss.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestCategoricalFocalCrossentropy, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the CategoricalFocalCrossentropy class
        self.keywords = [
            'from_logits',
            'alpha',
            'gamma',
            'class_weight',
            'reduction',
            'name'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [int(np.random.randint(low=1, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [int(np.random.randint(low=1, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [int(np.random.randint(low=1, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.from_logits = [False, True]
        self.alpha = [None, 0.0, 1.0] + [np.random.random(size=(self.nbr_of_classes[0],)).tolist()]
        self.gamma = [0.0] + [float(np.random.uniform(low=0, high=8, size=1))]
        self.class_weight = [None] + [np.random.random(size=(self.nbr_of_classes[0],)).tolist()]
        self.reduction = [tf.losses.Reduction.AUTO, tf.losses.Reduction.NONE, tf.losses.Reduction.SUM, tf.losses.Reduction.SUM_OVER_BATCH_SIZE]
        self.name = [None, ''] + [get_random_name(length=16)]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestCategoricalFocalCrossentropy, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.from_logits, self.alpha, self.gamma, self.class_weight, self.reduction, self.name)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        categorical_focal_crossentropy = tf.keras.losses.get('Custom>CategoricalFocalCrossentropy')

        # Check instance
        self.assertIsInstance(categorical_focal_crossentropy, losses.CategoricalFocalCrossentropy)

    def test_output_shapes(self):
        """
        Tests the loss function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the loss function settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get loss values
                loss = losses.CategoricalFocalCrossentropy(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output shape
                if init['reduction'] == tf.losses.Reduction.NONE:
                    self.assertShapeEqual(np.empty(shape=(input_shape[0], input_shape[1], 1)), loss, msg=error_msg)

                elif init['reduction'] == tf.losses.Reduction.AUTO:
                    self.assertShapeEqual(np.empty(shape=()), loss, msg=error_msg)

                elif init['reduction'] == tf.losses.Reduction.SUM:
                    self.assertShapeEqual(np.empty(shape=()), loss, msg=error_msg)

                elif init['reduction'] == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
                    self.assertShapeEqual(np.empty(shape=()), loss, msg=error_msg)

                else:
                    Exception('Test setting not handled!')

    def test_perfect_prediction(self):
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the loss function settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Generate synthetic test data
                test_y_pred = tf.cast(get_random_one_hot(input_shape), dtype=tf.float32, name='test_y_pred')
                test_y_true = tf.cast(test_y_pred, dtype=tf.int64, name='test_y_true')

                # Get loss values
                loss = losses.CategoricalFocalCrossentropy(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output values
                if init['reduction'] == 'none' and init['from_logits'] and not tf.math.count_nonzero(test_y_true):
                    self.assertAllClose(np.zeros(shape=(input_shape[0], input_shape[1], 1)), loss, rtol=1e-04, atol=1e-04, msg=error_msg)

                if init['reduction'] == 'none' and init['from_logits']:
                    self.assertNotAllClose(np.zeros(shape=(input_shape[0], input_shape[1], 1)), loss, msg=error_msg)

                elif init['reduction'] == 'none':
                    self.assertAllClose(np.zeros(shape=(input_shape[0], input_shape[1], 1)), loss, rtol=1e-04, atol=1e-04, msg=error_msg)

                elif init['from_logits'] and not tf.math.count_nonzero(test_y_true):
                    self.assertAllClose(np.zeros(shape=()), loss, rtol=1e-04, atol=1e-04, msg=error_msg)

                elif init['from_logits']:
                    self.assertNotAllClose(np.zeros(shape=()), loss, msg=error_msg)

                elif init['reduction'] == 'sum':
                    atol = 2 * tf.keras.backend.epsilon() * tf.cast(tf.math.reduce_prod(input_shape), dtype=tf.keras.backend.floatx())
                    self.assertAllClose(np.zeros(shape=()), loss, rtol=1e-04, atol=atol, msg=error_msg)

                else:
                    self.assertAllClose(np.zeros(shape=()), loss, rtol=1e-04, atol=1e-04, msg=error_msg)


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
