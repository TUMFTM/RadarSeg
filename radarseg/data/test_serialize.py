# Standard Libraries
import os
import shutil
import unittest

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.data.serialize import SequenceExampleSerializer, write_serialized_example


def gen_random(datatype: str = 'float', low: int = 0, high: int = 1, size: int = 1):
    """
    Returns a random value of the given data type.
    For lowercase ascii letters use low=97 and high=122.
    For uppercase ascii letters use low=65 and high=90.

    Arguments:
        datatype: Data type of the retrun value, <str>.
        low: Minimum value of the returned value, <int>.
        high: Maximum value of the returned value, <int>.
        size: Size of the returned value, <int>.
    """
    if datatype == 'float':
        return np.random.uniform(low=low, high=high, size=size)
    elif datatype == 'int':
        return np.random.randint(low=low, high=high, size=size)
    elif datatype == 'chr':
        if size == 1:
            return chr(np.random.randint(low=low, high=high, size=size))
        else:
            return [chr(value) for value in np.random.randint(low=low, high=high, size=size)]
    elif datatype == 'str':
        return ''.join([chr(value) for value in np.random.randint(low=low, high=high, size=size)])
    else:
        ValueError('No valid datatype: datatype has to be either float, int, chr or str.')


class TestSequenceExampleSerializer(unittest.TestCase):
    """
    Unit test for the SequenceExampleSerializer class and it's functions.
    """
    def setUp(self):
        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        self.seq_ex_serializer = SequenceExampleSerializer(feature_names=[], feature_types=[])

    def tearDown(self):
        del self.seq_ex_serializer

    def test_call_exceptions(self):
        # Define number of feature values (samples/frames)
        b = int(gen_random(datatype='int', low=1, high=10))
        # Define number of feature value elements (points)
        n = int(gen_random(datatype='int', low=1, high=100))
        # Define number of feature dimensions (channels)
        d = int(gen_random(datatype='int', low=1, high=10))

        # Test invalide feature type (not byte, float or int)
        self.seq_ex_serializer.feature_names = ['Name_' + str(i) for i in range(d)]
        self.seq_ex_serializer.feature_types = ['no_valid_feature_type'] * d
        feature_values = [[np.random.random_sample((n,)) for _ in range(b)] for _ in range(d)]
        with self.assertRaises(ValueError):
            self.seq_ex_serializer(feature_values)

    def test_call(self):
        """
        Test of the SequenceExampleSerializer call function.
        The overall test consists of 7 individual tests which compare the output of the call function with a defined reference output.
            - Test 1: test empty assignment.
            - Test 2: test byte value assignment.
            - Test 3: test int value assignment.
            - Test 4: test float value assignment.
            - Test 5: test byte context assignment.
            - Test 6: test int context assignment.
            - Test 7: test float context assignment.
        """
        # Number of test values within the base value list (the value of high is an arbitrary number, it just has to be > 0)
        num_values = int(gen_random(datatype='int', low=1, high=10))
        # Maximum length of a byte (str) value (the value an arbitrary number, the lenght just has to be > 0)
        max_len_byte_values = 10
        # Name (key) of the FeatureList (the value of size is an arbitrary number, it just has to be > 0)
        test_name = gen_random(datatype='str', low=97, high=122, size=gen_random(datatype='int', low=1, high=10))

        # Test 1: test if empty assignment leads to empty serialized SequenceExample
        # Further explanation, why b'\x12\x00' corresponds to an empty SequanceExample,
        # can be found here: https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details
        self.assertEqual(self.seq_ex_serializer(feature_values=[]), b'\x12\x00')

        # Test 2: test byte value serialization (single byte Feature and single FeatureList)
        # Define random byte values with random length (size)
        test_values = []
        for _ in range(num_values):
            size = gen_random(datatype='int', low=1, high=max_len_byte_values)
            test_values.append(bytes(gen_random(datatype='str', low=33, high=126, size=size), 'utf-8'))

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    test_name: tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(bytes_list=tf.train.BytesList(value=test_values))
                        ]
                    )
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = [test_name]
        self.seq_ex_serializer.feature_types = ['byte']
        test_input = np.chararray(shape=(num_values, ), itemsize=max_len_byte_values)
        test_input[:] = test_values

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[[test_input]]), ref_seq_ex)

        # Test 3: test int value serialization (single int64 Feature and single FeatureList)
        # The values of -100 and 100 are chose arbitrary (could be any int64 value)
        test_values = gen_random(datatype='int', low=-100, high=100, size=num_values)

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    test_name: tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(int64_list=tf.train.Int64List(value=test_values))
                        ]
                    )
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = [test_name]
        self.seq_ex_serializer.feature_types = ['int']

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[[test_values]]), ref_seq_ex)

        # Test 4: test float value serialization (single float Feature and single FeatureList)
        # The values of -100 and 100 are chose arbitrary (could be any float64 value)
        test_values = gen_random(datatype='float', low=-100, high=100, size=num_values)

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    test_name: tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(float_list=tf.train.FloatList(value=test_values))
                        ]
                    )
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = [test_name]
        self.seq_ex_serializer.feature_types = ['float']

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[[test_values]]), ref_seq_ex)

        # Test 5: test byte context value serialization (single byte Feature and empty FeatureLists)
        test_context = []
        for _ in range(num_values):
            size = gen_random(datatype='int', low=1, high=max_len_byte_values)
            test_context.append(bytes(gen_random(datatype='str', low=33, high=126, size=size), 'utf-8'))

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={}
            ),
            context=tf.train.Features(
                feature={
                    test_name: tf.train.Feature(bytes_list=tf.train.BytesList(value=test_context))
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = []
        self.seq_ex_serializer.feature_types = []
        self.seq_ex_serializer.context_names = [test_name]
        self.seq_ex_serializer.context_types = ['byte']
        test_input = np.chararray(shape=(num_values,), itemsize=max_len_byte_values)
        test_input[:] = test_context

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[], context_values=[test_input]), ref_seq_ex)

        # Test 6: test int context value serialization (single int Feature and empty FeatureLists)
        # The values of -100 and 100 are chose arbitrary (could be any int64 value)
        test_context = gen_random(datatype='int', low=-100, high=100, size=num_values)

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={}
            ),
            context=tf.train.Features(
                feature={
                    test_name: tf.train.Feature(int64_list=tf.train.Int64List(value=test_context.tolist()))
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = []
        self.seq_ex_serializer.feature_types = []
        self.seq_ex_serializer.context_names = [test_name]
        self.seq_ex_serializer.context_types = ['int']

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[], context_values=[test_context]), ref_seq_ex)

        # Test 7: test float context value serialization (single float Feature and empty FeatureLists)
        # The values of -100 and 100 are chose arbitrary (could be any float64 value)
        test_context = gen_random(datatype='float', low=-100, high=100, size=num_values)

        # Define reference sequence example
        ref_seq_ex = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={}
            ),
            context=tf.train.Features(
                feature={
                    test_name: tf.train.Feature(float_list=tf.train.FloatList(value=test_context.tolist()))
                }
            )
        )
        ref_seq_ex = ref_seq_ex.SerializeToString()

        # Define test serializer
        self.seq_ex_serializer.feature_names = []
        self.seq_ex_serializer.feature_types = []
        self.seq_ex_serializer.context_names = [test_name]
        self.seq_ex_serializer.context_types = ['float']

        # Test if serialized sequence example if equal to the reference sequence example
        self.assertEqual(self.seq_ex_serializer(feature_values=[], context_values=[test_context]), ref_seq_ex)


class TestWriteSerializedExample(unittest.TestCase):
    """
    Unit test for the write_serialized_example function.
    """
    @classmethod
    def setUpClass(cls):
        cls.TempLogDir = os.path.join(os.getcwd() + "/", "logs/temp_test_log/")
        os.makedirs(cls.TempLogDir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.TempLogDir != '/':
            shutil.rmtree(cls.TempLogDir, ignore_errors=True)

    def setUp(self):
        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)

        self.TempLogDir = os.path.join(os.getcwd() + "/", "logs/temp_test_log/")
        self.serialized = SequenceExampleSerializer(feature_names=[], feature_types=[])([])

    def tearDown(self):
        del self.serialized

    def test_write_serialized_example(self):
        # Test is function writes file
        file_name = os.path.join(self.TempLogDir, 'test_file.tfrecord')
        write_serialized_example(file_name, self.serialized)
        self.assertTrue(os.path.exists(file_name))


if __name__ == "__main__":
    unittest.main()
