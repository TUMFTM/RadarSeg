# Standard Libraries
import os
import copy
import shutil
import string
import random
import unittest
import itertools

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.data.generate import SequenceExampleGenerator


# Helper functions
def get_random_string(length: int = 1) -> str:
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def get_random_byte_list(size: int = 1, max_byte_len: int = 1) -> list:
    """
    Returns a list of random byte values with a defined number of elements.
    Arguments:
        size: Size of the returned list (number of list elements), <int>.
        max_byte_len: Maximum lenght of the byte values, <int>.
    """
    byte_list = []
    for _ in range(size):
        string_length = np.random.randint(low=1, high=max_byte_len)
        random_str_value = get_random_string(length=string_length)
        byte_list.append(bytes(random_str_value, 'utf-8'))

    return byte_list


def pad_or_truncate_list(some_list: list, target_len: int, fillup_value=None) -> list:
    return some_list[:target_len] + [fillup_value] * (target_len - len(some_list))


def iterable(obj):
    """
    Checks whether the passed object is iterable.
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def flatten(x):
    """
    Flattens a python iterable (list or tuple).
    """
    result = []

    for elem in x:
        if iterable(elem) and not isinstance(elem, (str, bytes)):
            result.extend(flatten(elem))
        else:
            result.append(elem)

    return result


class TestSequenceExampleGenerator(unittest.TestCase):
    """
    Unit test of the SequenceExampleGenerator class and it's functions.
    """
    @classmethod
    def setUpClass(cls):
        cls.TempLogDir = os.path.join(os.getcwd(), "logs/temp_test_log")
        os.makedirs(cls.TempLogDir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.TempLogDir != '/':
            shutil.rmtree(cls.TempLogDir, ignore_errors=True)

    def setUp(self):
        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define temporal test directory
        self.TempLogDir = os.path.join(os.getcwd(), "logs/temp_test_log")

        # Define keywords of the SequenceExampleGenerator class
        self.keywords = [
            'labels',
            'batch_size',
            'shuffle',
            'buffer_size',
            'seed',
            'cache',
            'one_hot',
            'nbr_of_classes',
            'feature_names',
            'feature_types',
            'context_names',
            'context_types',
            'label_major',
            'dense'
        ]

        # Create SequenceExampleGenerator instance
        self.seq_ex_generator = SequenceExampleGenerator(feature_names=[], feature_types=[])

    def tearDown(self):
        del self.seq_ex_generator

    def assertSparseEqual(self, test_data, ref_data, dtype=None, msg=None):
        """
        Checks the equality of a sparse tensors and an array.
        Arguments:
            test_data: Sparse data element to be tested, <tf.SparseTensor>.
            ref_data: Reference data element, <np.array or List>.
            dtype: Data type of the data elements (has to be the same for both), <str>.
            msg: Error message to be printed if the test fails, <str>.
        """
        # Transform sparse test data
        test_data = tf.sparse.to_dense(test_data).numpy()

        # Adjust test and reference data dimensions
        test_data = np.squeeze(test_data)
        ref_data = np.squeeze(ref_data)

        # Remove padding values
        test_data = test_data[np.nonzero(test_data)].tolist()
        ref_data = ref_data[np.nonzero(ref_data)].tolist()

        # Check equality
        if dtype == 'float':
            self.assertTrue(np.allclose(test_data, ref_data), msg=msg)
        else:
            self.assertListEqual(test_data, ref_data, msg=msg)

    def test_batch_size_property(self):
        """
        Tests the functionality of the batch_size property.
        """
        # Set random batch size value (the value of higth is an arbitrary number, could be anything greater than low)
        test_value = int(np.random.randint(low=1, high=128, size=1))

        # Initialize generator property
        self.seq_ex_generator.batch_size = test_value

        # Test set property / internal (private) state
        self.assertEqual(self.seq_ex_generator._batch_size, test_value)

        # Test get property / (public) state
        self.assertEqual(self.seq_ex_generator.batch_size, test_value)

        # Test invalid property value (no valid batch size)
        test_value = int(np.random.randint(low=-128, high=0, size=1))
        with self.assertRaises(ValueError):
            self.seq_ex_generator.batch_size = test_value

    def test_buffer_size_property(self):
        """
        Tests the functionality of the buffer_size property.
        """
        # Set random buffer size value (the value of higth is an arbitrary number, could be anything greater than low)
        test_values = [-1, int(np.random.randint(low=1, high=128, size=1))]

        for test_value in test_values:
            # Initialize generator property
            self.seq_ex_generator.buffer_size = test_value

            # Test set property / internal (private) state
            self.assertEqual(self.seq_ex_generator._buffer_size, test_value)

            # Test get property / (public) state
            self.assertEqual(self.seq_ex_generator.buffer_size, test_value)

        # Test invalid property value (no valid buffer size)
        test_values = [0, int(np.random.randint(low=-128, high=-2, size=1))]
        for test_value in test_values:
            with self.assertRaises(ValueError):
                self.seq_ex_generator.buffer_size = test_value

    def test_nbr_of_classes_property(self):
        """
        Tests the functionality of the nbr_of_classes property.
        """
        # Set random number of classes value (the value of higth is an arbitrary number, could be anything greater than low)
        test_value = int(np.random.randint(low=1, high=128, size=1))

        # Initialize generator property
        self.seq_ex_generator.nbr_of_classes = test_value

        # Test set property / internal (private) state
        self.assertEqual(self.seq_ex_generator._nbr_of_classes, test_value)

        # Test get property / (public) state
        self.assertEqual(self.seq_ex_generator.nbr_of_classes, test_value)

        # Test invalid property value (no valid number of classes)
        test_value = int(np.random.randint(low=-128, high=0, size=1))
        with self.assertRaises(ValueError):
            self.seq_ex_generator.nbr_of_classes = test_value

    def test_call(self):
        """
        Tests the correct behavior of the call function.

        The overall test consists of 3 main tests which compare the output of the call function with a defined reference
        output. Within each of the three main tests every possible configuration of the sequence example generator is
        tested (further described in the following comments). The three main tests are defined as:
            - Test 1: test empty test data (empty scene).
            - Test 2: test single test data (one scene, one sample/frame and one feature/channel).
            - Test 3: test multiple test data (two scenes, two samples/frames and three features/channels).
        """
        # Define data unrelated sequence example generator settings.

        # The setting values represent the edge cases of the attributes
        # and one random value within the specific boundaries of the
        # particular attribute (separated by the '+' sign).

        # Note: Shuffle is always set to False just for testing reasons.
        labels = [True, False]
        batch_size = [1] + [int(np.random.randint(low=1, high=128, size=1))]
        shuffle = [False]
        buffer_size = [-1, 1] + [int(np.random.randint(low=1, high=128, size=1))]
        seed = [0] + [int(np.random.randint(low=1, high=128, size=1))]
        cache = [False, True]
        one_hot = [True, False]
        nbr_of_classes = [1] + [int(np.random.randint(low=1, high=128, size=1))]
        label_major = [True, False]
        dense = [True, False]

        # Test 1: empty test data (empty scene)
        # Generate synthetic test data
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={}
            )
        )

        # Serialize synthetic test data
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Write synthetic test data to temporal test directory
        with tf.io.TFRecordWriter(self.TempLogDir + '/' + 'test_scene.tfrecord') as writer:
            writer.write(ref_seq_ex)

        # Define random feature name and type (Empty list, because the SequenceExample has no features)
        test_feat_names = []
        test_feature_types = []

        # Define random list of test context names (Empty list, because the SequenceExample has no context)
        test_context_names = []
        test_context_types = []

        # Get all possible combinations of sequence example generator settings (the order has to fit to self.keywords)
        settings = itertools.product(labels, batch_size, shuffle, buffer_size, seed, cache, one_hot, nbr_of_classes, [test_feat_names],
                                     [test_feature_types], [test_context_names], [test_context_types], label_major, dense)

        # Test all possible attribute setting combinations
        for setting in settings:
            # Generate initialization dictionary
            gen_init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

            if gen_init['labels']:
                # Initialize sequence example generator
                with self.assertRaises(AssertionError, msg=str(gen_init)):
                    self.seq_ex_generator = SequenceExampleGenerator(**gen_init)
            else:
                # Initialize sequence example generator
                self.seq_ex_generator = SequenceExampleGenerator(**gen_init)

                # Get test dataset
                test_dataset = self.seq_ex_generator(self.TempLogDir)

                # Check number of dataset elements
                self.assertEqual(len(list(test_dataset)), 1, msg=str(gen_init))

                # Check if dataset element is an empty tensor
                element = tf.data.experimental.get_single_element(test_dataset.take(1))
                self.assertEqual(element.shape, 0, msg=str(gen_init))

        # Clear temporal test log (delete synthetic test data)
        os.remove(self.TempLogDir + '/' + 'test_scene.tfrecord')

        # ----------------------------------------------------------------------

        # Test 2: singel test data (one scene)
        # Define number of test features (the limitation to one feature is not mandatory)
        nbr_test_feat = 1

        # Define number of test context elements (the limitation to one context element is not mandatory)
        nbr_test_context = 1

        # Define random list of test feature names and types (the value of high is chosen completely arbitrary)
        string_length = int(np.random.randint(low=1, high=32, size=1))
        test_feat_names = [''.join(random.choice(string.printable) for _ in range(string_length)) for _ in range(nbr_test_feat)]
        test_feature_types = ['byte']

        # Define random list of test context names
        test_context_names = [''.join(random.choice(string.printable) for _ in range(string_length)) for _ in range(nbr_test_context)]
        test_context_types = ['byte']

        # Define random test values (the value of high is chosen completely arbitrary)
        nbr_test_values = int(np.random.randint(low=2, high=32, size=1))
        max_byte_len = int(np.random.randint(low=2, high=32, size=1))
        test_feat_values = get_random_byte_list(size=nbr_test_values, max_byte_len=max_byte_len)
        test_context_values = get_random_byte_list(size=nbr_test_values, max_byte_len=max_byte_len)

        # Generate synthetic test data
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    test_feat_names[0]: tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(bytes_list=tf.train.BytesList(value=test_feat_values))
                        ]
                    )
                }
            ),
            context=tf.train.Features(
                feature={
                    test_context_names[0]: tf.train.Feature(bytes_list=tf.train.BytesList(value=test_context_values))
                }
            )
        )
        # Serialize synthetic test data
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Write synthetic test data to temporal test directory
        with tf.io.TFRecordWriter(self.TempLogDir + '/' + 'test_scene.tfrecord') as writer:
            writer.write(ref_seq_ex)

        # Get all possible combinations of sequence example generator settings (the order has to fit to self.keywords)
        settings = itertools.product(labels, batch_size, shuffle, buffer_size, seed, cache, one_hot, nbr_of_classes, [test_feat_names],
                                     [test_feature_types], [test_context_names], [test_context_types], label_major, dense)

        # Test all possible attribute setting combinations
        for setting in settings:
            # Generate initialization dictionary
            gen_init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

            if gen_init['labels']:
                # Initialize sequence example generator
                with self.assertRaises(AssertionError, msg=str(gen_init)):
                    self.seq_ex_generator = SequenceExampleGenerator(**gen_init)
            else:
                # Initialize sequence example generator
                self.seq_ex_generator = SequenceExampleGenerator(**gen_init)

                # Get test dataset
                test_dataset = self.seq_ex_generator(self.TempLogDir)

                # Check number of dataset elements
                self.assertEqual(len(list(test_dataset)), 1, msg=str(gen_init))

                # Check if dataset element is equal to the test (reference) data
                element = tf.data.experimental.get_single_element(test_dataset.take(1))
                if gen_init['dense']:
                    self.assertListEqual(np.squeeze(element[test_feat_names[0]].numpy()).tolist(), test_feat_values, msg=str(gen_init))
                else:
                    self.assertListEqual(np.squeeze(tf.sparse.to_dense(element[test_feat_names[0]]).numpy()).tolist(), test_feat_values, msg=str(gen_init))

                # Test compute output shape method
                if not (gen_init['batch_size'] > len(os.listdir(self.TempLogDir))):
                    # Skip compute_output_shape test if batch size exceeds number of data files
                    output_shape = self.seq_ex_generator.compute_output_shape(element[test_feat_names[0]].shape[1:])
                    self.assertDictEqual(output_shape[0], tf.nest.map_structure(tf.shape, element), msg=str(gen_init))

        # Clear temporal test log (delete synthetic test data)
        os.remove(self.TempLogDir + '/' + 'test_scene.tfrecord')

        # ----------------------------------------------------------------------

        # Test 3: test multiple test data (two scenes)
        # Define number of test scenes
        nbr_test_scenes = 2

        # Define number of test samples
        # Note: The sequence example message has to be changed if this number changes
        nbr_test_samples = 2

        # Define number of test features (one byte, float and int feature)
        # Note: The sequence example message has to be changed if this number changes
        nbr_test_feat = 3

        # Define number of test context elements (one byte, float and int feature)
        # Note: The sequence example message has to be changed if this number changes
        nbr_test_context = 3

        # Define random list of test feature names and types
        string_length = int(np.random.randint(low=1, high=32, size=1))
        test_feat_names = [get_random_string(length=string_length) for _ in range(nbr_test_feat)]
        test_feature_types = ['byte', 'float', 'int']

        # Define random list of test context names
        test_context_names = [get_random_string(length=string_length) for _ in range(nbr_test_context)]
        test_context_types = ['byte', 'float', 'int']

        # Generate synthetic test data (two scenes/files)
        test_feat_byte_values = [[] for _ in range(nbr_test_samples)]
        test_feat_float_values = [[] for _ in range(nbr_test_samples)]
        test_feat_int_values = [[] for _ in range(nbr_test_samples)]
        test_context_byte_values = []
        test_context_float_values = []
        test_context_int_values = []

        for scene in range(nbr_test_scenes):
            # Define random scene context
            nbr_test_values = int(np.random.randint(low=1, high=32, size=1))
            max_byte_len = int(np.random.randint(low=2, high=32, size=1))
            test_context_byte_values.append(get_random_byte_list(size=nbr_test_values, max_byte_len=max_byte_len))
            test_context_float_values.append(np.random.uniform(low=-128, high=128, size=nbr_test_values).tolist())
            test_context_int_values.append(np.random.randint(low=0, high=32, size=nbr_test_values).tolist())

            # Define random sample feature values
            for _ in range(nbr_test_samples):
                # Define random byte test values (the value of high is chosen completely arbitrary)
                nbr_test_values = int(np.random.randint(low=1, high=32, size=1))
                max_byte_len = int(np.random.randint(low=2, high=32, size=1))
                test_feat_byte_values[scene].append(get_random_byte_list(size=nbr_test_values, max_byte_len=max_byte_len))

                # Define random float test data (the values of low and high are chosen completely arbitrary)
                test_feat_float_values[scene].append(np.random.uniform(low=-128, high=128, size=nbr_test_values).tolist())

                # Define random int test data (the values of low and high for the context are chosen completely arbitrary)
                test_feat_int_values[scene].append(np.random.randint(low=0, high=nbr_of_classes[1], size=nbr_test_values).tolist())

            # Generate synthetic test data
            ref_seq_ex = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        test_feat_names[0]: tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(bytes_list=tf.train.BytesList(value=test_feat_byte_values[scene][0])),
                                tf.train.Feature(bytes_list=tf.train.BytesList(value=test_feat_byte_values[scene][1]))
                            ]
                        ),
                        test_feat_names[1]: tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(float_list=tf.train.FloatList(value=test_feat_float_values[scene][0])),
                                tf.train.Feature(float_list=tf.train.FloatList(value=test_feat_float_values[scene][1]))
                            ]
                        ),
                        test_feat_names[2]: tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(int64_list=tf.train.Int64List(value=test_feat_int_values[scene][0])),
                                tf.train.Feature(int64_list=tf.train.Int64List(value=test_feat_int_values[scene][1]))
                            ]
                        )
                    }
                ),
                context=tf.train.Features(
                    feature={
                        test_context_names[0]: tf.train.Feature(bytes_list=tf.train.BytesList(value=test_context_byte_values[scene])),
                        test_context_names[1]: tf.train.Feature(float_list=tf.train.FloatList(value=test_context_float_values[scene])),
                        test_context_names[2]: tf.train.Feature(int64_list=tf.train.Int64List(value=test_context_int_values[scene]))
                    }
                )
            )
            # Serialize synthetic test data
            ref_seq_ex = ref_seq_ex.SerializeToString()

            # Write synthetic test data to temporal test directory
            with tf.io.TFRecordWriter(self.TempLogDir + '/' + 'test_scene_' + str(scene) + '.tfrecord') as writer:
                writer.write(ref_seq_ex)

        # Get all possible combinations of sequence example generator settings (the order has to fit to self.keywords)
        settings = itertools.product(labels, batch_size, shuffle, buffer_size, seed, cache, one_hot, nbr_of_classes, [test_feat_names],
                                     [test_feature_types], [test_context_names], [test_context_types], label_major, dense)

        # Test all possible attribute setting combinations
        for setting in settings:
            # Generate initialization dictionary
            gen_init = copy.deepcopy(dict(zip(self.keywords, list(setting))))

            # Initialize sequence example generator
            if gen_init['labels'] and gen_init['label_major'] and gen_init['one_hot']:
                # This configuration leads to an AssertionError sice the first feature is a byte feature,
                # which can not be one hot encoded (however this is only true for this particular configuration).
                with self.assertRaises(AssertionError, msg=str(gen_init)):
                    self.seq_ex_generator = SequenceExampleGenerator(**gen_init)
            else:
                # Valid initialization
                self.seq_ex_generator = SequenceExampleGenerator(**gen_init)

                # Get test dataset
                if gen_init['nbr_of_classes'] < max(flatten(test_feat_int_values)) and gen_init['one_hot']:
                    # Error occurs only if the iterator (dataset) is called (graph execution), not during the generator initialization.
                    test_dataset = self.seq_ex_generator(self.TempLogDir)
                    # TODO: Change from unittest.TestCase to tf.test.TestCase is required, since unittest can not handle tf.errors.
                    # with self.assertRaises(tf.errors.InvalidArgumentError, msg=str(gen_init)):
                    #     # Arbitrary dataset iteration (iterator call)
                    #     element = tf.data.experimental.get_single_element(test_dataset.take(1))
                else:
                    test_dataset = self.seq_ex_generator(self.TempLogDir)

                    # Check number of dataset elements
                    self.assertEqual(len(list(test_dataset)), np.ceil(np.divide(nbr_test_scenes, gen_init['batch_size'])), msg=str(gen_init))

                    # Unbatch test dataset
                    test_dataset = test_dataset.unbatch()

                    # Determine maximum number of values per batch
                    if gen_init['batch_size'] >= nbr_test_scenes:
                        max_values_per_batch = [max([len(test_feat_byte_values[sc][sa]) for sc in range(nbr_test_scenes) for sa in range(nbr_test_samples)])] * nbr_test_scenes
                    else:
                        max_values_per_batch = [max([len(test_feat_byte_values[sc][sa]) for sa in range(nbr_test_samples)]) for sc in range(nbr_test_scenes)]

                    # Determine feature and label names:
                    if gen_init['labels'] and gen_init['label_major']:
                        # label_name = test_feat_names[0]
                        feat_0_name = test_feat_names[1]
                        feat_1_name = test_feat_names[2]

                    elif gen_init['labels']:
                        # label_name = test_feat_names[2]
                        feat_0_name = test_feat_names[0]
                        feat_1_name = test_feat_names[1]

                    else:
                        feat_0_name = test_feat_names[0]
                        feat_1_name = test_feat_names[1]
                        feat_2_name = test_feat_names[2]

                    # Determine reference data
                    if gen_init['labels'] and gen_init['label_major'] and gen_init['one_hot'] and gen_init['dense']:
                        # AssertionError (handled above).
                        pass

                    elif gen_init['labels'] and gen_init['label_major'] and gen_init['one_hot'] and not gen_init['dense']:
                        # AssertionError (handled above).
                        pass

                    elif gen_init['labels'] and gen_init['label_major'] and not gen_init['one_hot'] and gen_init['dense']:
                        label_ref = test_feat_byte_values
                        feat_0_ref = test_feat_float_values
                        feat_1_ref = test_feat_int_values

                        # Fill up reference values to the max length per batch (padding)
                        label_ref = [[pad_or_truncate_list(label_ref[sc][sa], max_values_per_batch[sc], b'') for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_0_ref = [[pad_or_truncate_list(feat_0_ref[sc][sa], max_values_per_batch[sc], 0.0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_1_ref = [[pad_or_truncate_list(feat_1_ref[sc][sa], max_values_per_batch[sc], 0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                    elif gen_init['labels'] and gen_init['label_major'] and not gen_init['one_hot'] and not gen_init['dense']:
                        label_ref = test_feat_byte_values
                        feat_0_ref = test_feat_float_values
                        feat_1_ref = test_feat_int_values

                    elif gen_init['labels'] and not gen_init['label_major'] and gen_init['one_hot'] and gen_init['dense']:
                        label_ref = test_feat_int_values
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values

                        # Fill up reference values to the max length per batch (padding)
                        label_ref = [[pad_or_truncate_list(label_ref[sc][sa], max_values_per_batch[sc], -1) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_0_ref = [[pad_or_truncate_list(feat_0_ref[sc][sa], max_values_per_batch[sc], b'') for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_1_ref = [[pad_or_truncate_list(feat_1_ref[sc][sa], max_values_per_batch[sc], 0.0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                        # Encode reference labels (one-hot encoding)
                        label_ref = [[tf.one_hot(label_ref[sc][sa], gen_init['nbr_of_classes']).numpy().tolist() for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                    elif gen_init['labels'] and not gen_init['label_major'] and gen_init['one_hot'] and not gen_init['dense']:
                        label_ref = test_feat_int_values
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values

                        # Encode reference labels (one-hot encoding)
                        label_ref = [[tf.one_hot(label_ref[sc][sa], gen_init['nbr_of_classes']).numpy().tolist() for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                    elif gen_init['labels'] and not gen_init['label_major'] and not gen_init['one_hot'] and gen_init['dense']:
                        label_ref = test_feat_int_values
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values

                        # Fill up reference values to the max length per batch (padding)
                        label_ref = [[pad_or_truncate_list(label_ref[sc][sa], max_values_per_batch[sc], 0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_0_ref = [[pad_or_truncate_list(feat_0_ref[sc][sa], max_values_per_batch[sc], b'') for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_1_ref = [[pad_or_truncate_list(feat_1_ref[sc][sa], max_values_per_batch[sc], 0.0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                    elif gen_init['labels'] and not gen_init['label_major'] and not gen_init['one_hot'] and not gen_init['dense']:
                        label_ref = test_feat_int_values
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values

                    elif not gen_init['labels'] and gen_init['dense']:
                        # Note: The values of 'label_major' and 'one_hot' are irrelevant if no labels exist.
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values
                        feat_2_ref = test_feat_int_values

                        # Fill up reference values to the max length per batch (padding)
                        feat_0_ref = [[pad_or_truncate_list(feat_0_ref[sc][sa], max_values_per_batch[sc], b'') for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_1_ref = [[pad_or_truncate_list(feat_1_ref[sc][sa], max_values_per_batch[sc], 0.0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]
                        feat_2_ref = [[pad_or_truncate_list(feat_2_ref[sc][sa], max_values_per_batch[sc], 0) for sa in range(nbr_test_samples)] for sc in range(nbr_test_scenes)]

                    elif not gen_init['labels'] and not gen_init['dense']:
                        # Note: The values of 'label_major' and 'one_hot' are irrelevant if no labels exist.
                        feat_0_ref = test_feat_byte_values
                        feat_1_ref = test_feat_float_values
                        feat_2_ref = test_feat_int_values

                    else:
                        raise Exception('Test case not handled!')

                    # Check if dataset elements are equal to the reference elements
                    if gen_init['labels']:
                        for scene, (element_features, element_labels) in zip(range(nbr_test_scenes), test_dataset):
                            for sample in range(nbr_test_samples):
                                if gen_init['label_major'] and gen_init['dense']:
                                    self.assertTrue(np.allclose(np.squeeze(element_features[feat_0_name].numpy())[sample].tolist(), feat_0_ref[scene][sample]), msg=str(gen_init))
                                    self.assertListEqual(np.squeeze(element_features[feat_1_name].numpy())[sample].tolist(), feat_1_ref[scene][sample], msg=str(gen_init))
                                    self.assertListEqual(np.squeeze(element_labels.numpy())[sample].tolist(), label_ref[scene][sample], msg=str(gen_init))

                                elif gen_init['label_major'] and not gen_init['dense']:
                                    self.assertSparseEqual(tf.sparse.split(element_features[feat_0_name], num_split=nbr_test_samples, axis=0)[sample], feat_0_ref[scene][sample], dtype='float')
                                    self.assertSparseEqual(tf.sparse.split(element_features[feat_1_name], num_split=nbr_test_samples, axis=0)[sample], feat_1_ref[scene][sample])
                                    self.assertSparseEqual(tf.sparse.split(element_labels, num_split=nbr_test_samples, axis=0)[sample], label_ref[scene][sample])

                                elif not gen_init['label_major'] and gen_init['dense']:
                                    self.assertListEqual(np.squeeze(element_features[feat_0_name].numpy())[sample].tolist(), feat_0_ref[scene][sample], msg=str(gen_init))
                                    self.assertTrue(np.allclose(np.squeeze(element_features[feat_1_name].numpy())[sample].tolist(), feat_1_ref[scene][sample]), msg=str(gen_init))
                                    self.assertListEqual(np.squeeze(element_labels.numpy())[sample].tolist(), label_ref[scene][sample], msg=str(gen_init))

                                elif not gen_init['label_major'] and not gen_init['dense']:
                                    self.assertSparseEqual(tf.sparse.split(element_features[feat_0_name], num_split=nbr_test_samples, axis=0)[sample], feat_0_ref[scene][sample])
                                    self.assertSparseEqual(tf.sparse.split(element_features[feat_1_name], num_split=nbr_test_samples, axis=0)[sample], feat_1_ref[scene][sample], dtype='float')
                                    self.assertSparseEqual(tf.sparse.split(element_labels, num_split=nbr_test_samples, axis=0)[sample], label_ref[scene][sample])

                                else:
                                    raise Exception('Test case not handled!')

                                # Test compute output shape method
                                if not (gen_init['batch_size'] > len(os.listdir(self.TempLogDir))):
                                    # Skip compute_output_shape test if batch size exceeds number of data files
                                    # Get output shapes
                                    output_features_shape, output_label_shape = self.seq_ex_generator.compute_output_shape(element_features[feat_0_name].shape)

                                    # Restore element batch size (due to unbatching at the beginning of the test)
                                    element_feature_shapes = tf.nest.map_structure(tf.shape, element_features)
                                    element_feature_shapes = {key: tf.concat([tf.expand_dims(gen_init['batch_size'], axis=0), value], axis=0) for key, value in element_feature_shapes.items()}
                                    element_label_shapes = element_labels.shape
                                    element_label_shapes = tf.concat([tf.expand_dims(gen_init['batch_size'], axis=0), element_label_shapes], axis=0)

                                    # Compare shapes
                                    self.assertDictEqual(output_features_shape, element_feature_shapes, msg=str(gen_init))
                                    self.assertEqual(output_label_shape, element_label_shapes, msg=str(gen_init))

                    else:
                        for scene, element in zip(range(nbr_test_scenes), test_dataset):
                            for sample in range(nbr_test_samples):
                                if gen_init['dense']:
                                    self.assertListEqual(np.squeeze(element[feat_0_name].numpy())[sample].tolist(), feat_0_ref[scene][sample], msg=str(gen_init))
                                    self.assertTrue(np.allclose(np.squeeze(element[feat_1_name].numpy())[sample].tolist(), feat_1_ref[scene][sample]), msg=str(gen_init))
                                    self.assertListEqual(np.squeeze(element[feat_2_name].numpy())[sample].tolist(), feat_2_ref[scene][sample], msg=str(gen_init))

                                elif not gen_init['dense']:
                                    self.assertSparseEqual(tf.sparse.split(element[feat_0_name], num_split=nbr_test_samples, axis=0)[sample], feat_0_ref[scene][sample])
                                    self.assertSparseEqual(tf.sparse.split(element[feat_1_name], num_split=nbr_test_samples, axis=0)[sample], feat_1_ref[scene][sample], dtype='float')
                                    self.assertSparseEqual(tf.sparse.split(element[feat_2_name], num_split=nbr_test_samples, axis=0)[sample], feat_2_ref[scene][sample])

                                else:
                                    raise Exception('Test case not handled!')

                                # Test compute output shape method
                                if not (gen_init['batch_size'] > len(os.listdir(self.TempLogDir))):
                                    # Skip compute_output_shape test if batch size exceeds number of data files
                                    # Get output shapes
                                    output_shape = self.seq_ex_generator.compute_output_shape(element[feat_0_name].shape)

                                    # Restore element batch size (due to unbatching at the beginning of the test)
                                    element_shapes = tf.nest.map_structure(tf.shape, element)
                                    element_shapes = {key: tf.concat([tf.expand_dims(gen_init['batch_size'], axis=0), value], axis=0) for key, value in element_shapes.items()}

                                    # Compare shapes
                                    self.assertDictEqual(output_shape[0], element_shapes, msg=str(gen_init))

        # Clear temporal test log (delete synthetic test data)
        os.remove(self.TempLogDir + '/' + 'test_scene_0.tfrecord')
        os.remove(self.TempLogDir + '/' + 'test_scene_1.tfrecord')

    def test_get_config(self):
        """
        Tests the correct behavior of the get_config function.

        All values of higth are set completely arbitrary, they could be any
        value greater than the values of low. The values of low are chosen in
        accordance to the specific boundary values of the particular attribute.
        """
        # Define number of test features
        nbr_test_feat = int(np.random.randint(low=1, high=32, size=1))

        # Define number of test context elements
        nbr_test_context = int(np.random.randint(low=0, high=32, size=1))

        # Define random list of test feature names
        string_length = int(np.random.randint(low=1, high=32, size=1))
        test_feat_names = [get_random_string(length=string_length) for _ in range(nbr_test_feat)]

        # Define random list of test context names
        test_context_names = [get_random_string(length=string_length) for _ in range(nbr_test_context)]

        # Define reference configuration
        ref_config = {
            'labels': False,
            'batch_size': int(np.random.randint(low=1, high=128, size=1)),
            'shuffle': bool(np.random.choice([True, False])),
            'buffer_size': int(np.random.choice([-1, int(np.random.randint(low=1, high=128, size=1))])),
            'seed': int(np.random.randint(low=0, high=128, size=1)),
            'cache': bool(np.random.choice([True, False])),
            'one_hot': bool(np.random.choice([True, False])),
            'nbr_of_classes': int(np.random.randint(low=1, high=8, size=1)),
            'feature_names': test_feat_names,
            'feature_types': np.random.choice(['byte', 'int', 'float'], size=(nbr_test_feat,)).tolist(),
            'context_names': test_context_names,
            'context_types': np.random.choice(['byte', 'int', 'float'], size=(nbr_test_context,)).tolist(),
            'label_major': bool(np.random.choice([True, False])),
            'dense': bool(np.random.choice([True, False]))
        }

        # Initialize sequence example generator configuration
        self.seq_ex_generator = SequenceExampleGenerator(**ref_config)

        # Test for equality of configurations
        self.assertDictEqual(self.seq_ex_generator.get_config(), ref_config)


if __name__ == "__main__":
    unittest.main()
