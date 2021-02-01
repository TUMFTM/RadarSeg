# Standard Libraries
import copy
import string
import random
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.train import metrics


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


class TestTopKCategoricalTruePositives(tf.test.TestCase):
    """
    Unit test of the TopKCategoricalTruePositives metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestTopKCategoricalTruePositives, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the TopKCategoricalTruePositives class
        self.keywords = [
            'thresholds',
            'top_k',
            'class_id',
            'sample_weight',
            'name',
            'dtype'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.thresholds = [None, 0.0, 1.0] + [np.random.random(size=self.nbr_of_classes[-1]).tolist()]
        self.top_k = [None, 1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.class_id = [None, 0] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.sample_weight = [None] + [float(np.random.random(size=1))]
        self.name = [None, ''] + [get_random_name(length=16)]
        self.dtype = [None]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestTopKCategoricalTruePositives, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.thresholds, self.top_k, self.class_id, self.sample_weight, self.name, self.dtype)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        top_k_categorical_true_positives = tf.keras.losses.get('Custom>TopKCategoricalTruePositives')

        # Check instance
        self.assertIsInstance(top_k_categorical_true_positives, metrics.TopKCategoricalTruePositives)

    def test_output_shapes(self):
        """
        Tests the metric function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the metric settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Get current number of classes
                nbr_of_classes = input_shape[2]

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get metric values
                if init['class_id'] is None:
                    metric = metrics.TopKCategoricalTruePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                elif init['class_id'] > nbr_of_classes - 1:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        metric = metrics.TopKCategoricalTruePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                else:
                    metric = metrics.TopKCategoricalTruePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output shape
                if init['thresholds'] is not None and isinstance(init['thresholds'], (list, tuple)) and init['top_k'] is None:
                    if len(init['thresholds']) > 1:
                        self.assertShapeEqual(np.empty(shape=(len(init['thresholds']),)), metric, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)

                else:
                    self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)


class TestTopKCategoricalTrueNegatives(tf.test.TestCase):
    """
    Unit test of the TopKCategoricalTrueNegatives metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestTopKCategoricalTrueNegatives, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the TopKCategoricalTrueNegatives class
        self.keywords = [
            'thresholds',
            'top_k',
            'class_id',
            'sample_weight',
            'name',
            'dtype'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.thresholds = [None, 0.0, 1.0] + [np.random.random(size=self.nbr_of_classes[-1]).tolist()]
        self.top_k = [None, 1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.class_id = [None, 0] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.sample_weight = [None] + [float(np.random.random(size=1))]
        self.name = [None, ''] + [get_random_name(length=16)]
        self.dtype = [None]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestTopKCategoricalTrueNegatives, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.thresholds, self.top_k, self.class_id, self.sample_weight, self.name, self.dtype)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        top_k_categorical_true_negatives = tf.keras.losses.get('Custom>TopKCategoricalTrueNegatives')

        # Check instance
        self.assertIsInstance(top_k_categorical_true_negatives, metrics.TopKCategoricalTrueNegatives)

    def test_output_shapes(self):
        """
        Tests the metric function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the metric settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Get current number of classes
                nbr_of_classes = input_shape[2]

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get metric values
                if init['class_id'] is None:
                    metric = metrics.TopKCategoricalTrueNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                elif init['class_id'] > nbr_of_classes - 1:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        metric = metrics.TopKCategoricalTrueNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                else:
                    metric = metrics.TopKCategoricalTrueNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output shape
                if init['thresholds'] is not None and isinstance(init['thresholds'], (list, tuple)) and init['top_k'] is None:
                    if len(init['thresholds']) > 1:
                        self.assertShapeEqual(np.empty(shape=(len(init['thresholds']),)), metric, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)

                else:
                    self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)


class TestTopKCategoricalFalsePositives(tf.test.TestCase):
    """
    Unit test of the TopKCategoricalFalsePositives metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestTopKCategoricalFalsePositives, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the TopKCategoricalFalsePositives class
        self.keywords = [
            'thresholds',
            'top_k',
            'class_id',
            'sample_weight',
            'name',
            'dtype'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.thresholds = [None, 0.0, 1.0] + [np.random.random(size=self.nbr_of_classes[-1]).tolist()]
        self.top_k = [None, 1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.class_id = [None, 0] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.sample_weight = [None] + [float(np.random.random(size=1))]
        self.name = [None, ''] + [get_random_name(length=16)]
        self.dtype = [None]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestTopKCategoricalFalsePositives, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.thresholds, self.top_k, self.class_id, self.sample_weight, self.name, self.dtype)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        top_k_categorical_false_positives = tf.keras.losses.get('Custom>TopKCategoricalFalsePositives')

        # Check instance
        self.assertIsInstance(top_k_categorical_false_positives, metrics.TopKCategoricalFalsePositives)

    def test_output_shapes(self):
        """
        Tests the metric function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the metric settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Get current number of classes
                nbr_of_classes = input_shape[2]

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get metric values
                if init['class_id'] is None:
                    metric = metrics.TopKCategoricalFalsePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                elif init['class_id'] > nbr_of_classes - 1:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        metric = metrics.TopKCategoricalFalsePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                else:
                    metric = metrics.TopKCategoricalFalsePositives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output shape
                if init['thresholds'] is not None and isinstance(init['thresholds'], (list, tuple)) and init['top_k'] is None:
                    if len(init['thresholds']) > 1:
                        self.assertShapeEqual(np.empty(shape=(len(init['thresholds']),)), metric, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)

                else:
                    self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)


class TestTopKCategoricalFalseNegatives(tf.test.TestCase):
    """
    Unit test of the TopKCategoricalFalseNegatives metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestTopKCategoricalFalseNegatives, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the TopKCategoricalFalseNegatives class
        self.keywords = [
            'thresholds',
            'top_k',
            'class_id',
            'sample_weight',
            'name',
            'dtype'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.thresholds = [None, 0.0, 1.0] + [np.random.random(size=self.nbr_of_classes[-1]).tolist()]
        self.top_k = [None, 1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.class_id = [None, 0] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.sample_weight = [None] + [float(np.random.random(size=1))]
        self.name = [None, ''] + [get_random_name(length=16)]
        self.dtype = [None]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestTopKCategoricalFalseNegatives, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.thresholds, self.top_k, self.class_id, self.sample_weight, self.name, self.dtype)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        top_k_categorical_false_negatives = tf.keras.losses.get('Custom>TopKCategoricalFalseNegatives')

        # Check instance
        self.assertIsInstance(top_k_categorical_false_negatives, metrics.TopKCategoricalFalseNegatives)

    def test_output_shapes(self):
        """
        Tests the metric function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the metric settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Get current number of classes
                nbr_of_classes = input_shape[2]

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get metric values
                if init['class_id'] is None:
                    metric = metrics.TopKCategoricalFalseNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                elif init['class_id'] > nbr_of_classes - 1:
                    with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                        metric = metrics.TopKCategoricalFalseNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                else:
                    metric = metrics.TopKCategoricalFalseNegatives(**init)(y_true=test_y_true, y_pred=test_y_pred)

                # Check output shape
                if init['thresholds'] is not None and isinstance(init['thresholds'], (list, tuple)) and init['top_k'] is None:
                    if len(init['thresholds']) > 1:
                        self.assertShapeEqual(np.empty(shape=(len(init['thresholds']),)), metric, msg=error_msg)
                    else:
                        self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)

                else:
                    self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)


class TestFBetaScore(tf.test.TestCase):
    """
    Unit test of the FBetaScore metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestFBetaScore, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define keywords of the FBetaScore class
        self.keywords = [
            'num_classes',
            'average',
            'beta',
            'thresholds',
            'top_k',
            'class_id',
            'sample_weight',
            'name',
            'dtype'
        ]
        # Define batch size of the test data (the value of high is chosen completely arbitrary)
        self.batch_size = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define number of points of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_points = [1] + [int(np.random.randint(low=2, high=128, size=1))]

        # Define number of classes of the test data (the value of high is chosen completely arbitrary)
        self.nbr_of_classes = [1] + [int(np.random.randint(low=2, high=16, size=1))]

        # Define loss function arguments (initialization)
        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).
        self.num_classes = [1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.average = [None, 'micro', 'macro', 'weighted', 'no-valid-average']
        self.beta = [0.0, 1.0] + [float(np.random.uniform(low=2.0, high=16.0, size=1))]
        self.thresholds = [None, 0.0, 1.0] + [np.random.random(size=self.nbr_of_classes[-1]).tolist()]
        self.top_k = [None, 1] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.class_id = [None, 0] + [int(np.random.randint(low=1, high=self.nbr_of_classes[-1], size=1))]
        self.sample_weight = [None] + [float(np.random.random(size=1))]
        self.name = [None, ''] + [get_random_name(length=16)]
        self.dtype = [None]

    def tearDown(self):
        # Call teardown of the base test class
        super(TestFBetaScore, self).tearDown()

    def get_settings(self):
        # Returns all possible combinations of the loss function settings (the order has to fit to self.keywords)
        return itertools.product(self.num_classes, self.average, self.beta, self.thresholds, self.top_k, self.class_id, self.sample_weight, self.name, self.dtype)

    def get_input_shapes(self):
        # Returns all possible combinations of the input shapes
        return itertools.product(self.batch_size, self.nbr_of_points, self.nbr_of_classes)

    def test_keras_deserialization(self):
        """
        Tests the registration within the Keras serialization framework.
        """
        # Get loss function
        f_beta_score = tf.keras.losses.get('Custom>FBetaScore')

        # Check instance
        self.assertIsInstance(f_beta_score, metrics.FBetaScore)

    def test_output_shapes(self):
        """
        Tests the metric function call.
        """
        # Get all possible input data shape combinations
        input_shapes = self.get_input_shapes()

        for input_shape in input_shapes:
            # Get all possible combinations of the metric settings
            settings = self.get_settings()

            for setting in settings:
                # Initialization dictionary
                init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

                # Set error message
                error_msg = 'Setting: {}. \n Input shape: {}.'.format(str(init), str(input_shape))

                # Get current input shape
                nbr_of_classes = input_shape[2]

                # Generate synthetic test data
                test_y_pred = tf.constant(np.random.random(size=input_shape), dtype=tf.float32, shape=input_shape, name='test_y_pred')
                test_y_true = tf.cast(get_random_one_hot(input_shape), dtype=tf.int64, name='test_y_true')

                # Get metric values
                if init['beta'] <= 0.0:
                    with self.assertRaises(ValueError, msg=error_msg):
                        metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                    continue

                elif init['average'] not in set((None, "micro", "macro", "weighted")):
                    with self.assertRaises(ValueError, msg=error_msg):
                        metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                    continue

                elif init['num_classes'] != nbr_of_classes and init['average'] != 'micro':
                    with self.assertRaises(ValueError, msg=error_msg):
                        metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                    continue

                elif init['top_k'] is not None:
                    if init['top_k'] > nbr_of_classes:
                        with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                            metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                        continue
                    elif init['class_id'] is not None:
                        if init['class_id'] > nbr_of_classes - 1:
                            with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                                metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                            continue
                    else:
                        pass

                elif init['class_id'] is not None:
                    if init['class_id'] > nbr_of_classes - 1:
                        with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                            metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                        continue
                    elif init['top_k'] is not None:
                        if init['top_k'] > nbr_of_classes:
                            with self.assertRaises(tf.errors.InvalidArgumentError, msg=error_msg):
                                metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                            continue
                    else:
                        pass

                else:
                    pass

                try:
                    metric = metrics.FBetaScore(**init)(y_true=test_y_true, y_pred=test_y_pred)
                except Exception as e:
                    print(error_msg)
                    raise e

                # Check output shape
                if init['average'] is not None:
                    self.assertShapeEqual(np.empty(shape=()), metric, msg=error_msg)
                else:
                    self.assertShapeEqual(np.empty(shape=((init['num_classes'],))), metric, msg=error_msg)


class TestF1Score(tf.test.TestCase):
    """
    Unit test of the F1Score metric.
    """
    def setUp(self):
        # Call setup of the base test class
        super(TestF1Score, self).setUp()

        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

    def tearDown(self):
        # Call teardown of the base test class
        super(TestF1Score, self).tearDown()

    def test_example(self):
        """
        Tests the return value of an example test case.
        """
        # Generate synthetic test data
        test_y_pred = tf.constant([[[0.6, 0.5, 0.3, 0.1], [0.1, 0.5, 0.3, 0.1]],
                                  [[0.6, 0.5, 0.3, 0.1], [0.1, 0.5, 0.3, 0.1]]], dtype=tf.float32, shape=(2, 2, 4), name='test_y_pred')
        test_y_true = tf.constant([[[1, 0, 0, 0], [0, 1, 0, 0]],
                                  [[1, 0, 0, 0], [0, 0, 1, 0]]], dtype=tf.float32, shape=(2, 2, 4), name='test_y_true')

        # Test class individual F1 score (threshold)
        metric = metrics.F1Score(num_classes=4, average=None, thresholds=0.5, top_k=None)(y_true=test_y_true, y_pred=test_y_pred)
        self.assertAllClose(np.array([1.0, 0.0, 0.0, 0.0]), metric)

        # Test class individual F1 score (top 1)
        metric = metrics.F1Score(num_classes=4, average=None, thresholds=0.5, top_k=1)(y_true=test_y_true, y_pred=test_y_pred)
        self.assertAllClose(np.array([1.0, 0.6666666, 0.0, 0.0]), metric)

        # Test micro averaged F1 score
        metric = metrics.F1Score(num_classes=4, average='micro', top_k=1)(y_true=test_y_true, y_pred=test_y_pred)
        self.assertAllClose(tf.constant(0.75, dtype=tf.float32), metric)

        # Test macro averaged F1 score
        metric = metrics.F1Score(num_classes=4, average='macro', top_k=1)(y_true=test_y_true, y_pred=test_y_pred)
        self.assertAllClose(tf.constant(0.4166666, dtype=tf.float32), metric)


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
