# Standard Libraries
import os
from abc import ABC, abstractmethod

# 3rd Party Libraries
import tensorflow as tf

# Local imports


class Generator(ABC):
    """
    Abstract generator class to generate a tensorflow dataset.

    Performance advices:
    Caching can require a lot of system memory (RAM), why this attribute should be set to False
    for large datasets or little system memory.

    Note: A higher buffer size requires more system memory (can be a lot for large datasets).

    Arguments:
        labels: Whether the data contains labels (target values) or not (only features), <bool>.
        batch_size: Batch size of the dataset elements, <int>.
        shuffle: Whether to shuffle the dataset elements, <bool>.
        buffer_size: Size of the shuffle buffer (-1 indecates a buffer size with the same size as the dataset), <int>.
        seed: Seed for random operations, <int>.
        cache: Whether to cache the elements of the dataset in memory, <bool>.
        one_hot: Whether to use a one hot encoding to represent the labels, <bool>.
        nbr_of_classes: Number of classes (values of the label), has to be set if one_hot is enabled, <int>.
    """
    def __init__(self,
                 labels: bool = False,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 buffer_size: int = -1,
                 seed: int = 42,
                 cache: bool = False,
                 one_hot: bool = False,
                 nbr_of_classes: int = 0):
        # Initialize
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        self.cache = cache
        self.one_hot = one_hot
        self._nbr_of_classes = nbr_of_classes

        # Check if number_of_classes is set if one_hot is enabled
        assert (self._nbr_of_classes != 0 and self.one_hot) or (not self.one_hot)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError("The batch size has to be a positive integer value greater than one!")
        else:
            self._batch_size = value

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        if value < 1 and value != -1:
            raise ValueError("The buffer size has to be a positive integer value greater than one or negative one!")
        else:
            self._buffer_size = value

    @property
    def nbr_of_classes(self):
        return self._nbr_of_classes

    @nbr_of_classes.setter
    def nbr_of_classes(self, value):
        if value < 1:
            raise ValueError("Number of classes has to be a positive integer value greater than one!")
        else:
            self._nbr_of_classes = value

    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self):
        pass

    def get_config(self):
        """
        Returns generator configuration.
        """
        config = {
            'labels': self.labels,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'buffer_size': self.buffer_size,
            'seed': self.seed,
            'cache': self.cache,
            'one_hot': self.one_hot,
            'nbr_of_classes': self.nbr_of_classes
        }
        return config

    @abstractmethod
    def compute_output_shape(self, input_shape):
        """
        Abstractmethod that returns the output shape of a dataset element.
        """
        pass


class SequenceExampleGenerator(Generator):
    """
    Generator class to generate a tensorflow dataset from a serialized sequence example.

    Performance advices:
    The performance of the sequence example generator is highly dependet on the chosen settings.
    The best performance will be achived for sparse (dense=False) tensors which are not one hot
    encoded and high buffer as well as batch sizes. Both the batch size and the buffer size should
    be chosen as a true divisor of the dataset elements.

    Caching can require a lot of system memory (RAM), why this attribute should be set to False
    for large datasets or little system memory.

    Note: A higher buffer size requires more system memory (can be a lot for large datasets).

    More information on the performance of the input pipeline can be found here:
    https://www.tensorflow.org/guide/data_performance

    Arguments:
        feature_names: List of feature names, <List>.
        feature_types: List of feature types (data types), <List>.
        context_names: List of context names, <List>.
        context_types: List of context types (data types), <List>.
        label_major: Whether the class labels are the first (or last) element of feature_names, <bool>.
        dense: Whether to convert the sparse tensors to dense tensors (padded with zeros), <bool>.

    Returns:
        dataset: A tensorflow dataset with two elements (features and labels), <tf.data.Dataset>.
    """
    def __init__(self,
                 feature_names: list,
                 feature_types: list,
                 context_names: list = [],
                 context_types: list = [],
                 label_major: bool = False,
                 dense: bool = False,
                 **kwargs):
        # Initialize base instance
        super(SequenceExampleGenerator, self).__init__(**kwargs)

        # Initialize instance
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.context_names = context_names
        self.context_types = context_types
        self.label_major = label_major
        self.dense = dense

        # Check attribute dimensions
        assert len(self.feature_names) == len(self.feature_types)
        assert len(self.context_names) == len(self.context_types)

        # Check number of features, if the dataset should contain labels
        if self.labels:
            assert len(self.feature_names) > 1

        # Get label key (name of the labels)
        if self.labels and self.label_major:
            self.label_key = self.feature_names[0] if self.feature_names else ''
        elif self.labels:
            self.label_key = self.feature_names[-1] if self.feature_names else ''
        else:
            self.label_key = ''

        # Check label type for one-hot encoding (labels have to be type int)
        if self.one_hot and self.label_key:
            if self.label_major:
                assert self.feature_types[0] == 'int'
            else:
                assert self.feature_types[-1] == 'int'

    @tf.function
    def _parse_sequence_example(self, sequence_example):
        """
        Returns features and labels of a given sequence example.
        Arguments:
            sequence_example: Sequence example message, <SequenceExample>.
        """
        # Create feature dict
        if self.feature_names:
            feature_dict = {}
            for feature_name, feature_type in zip(self.feature_names, self.feature_types):
                if feature_type == 'byte':
                    feature_dict[feature_name] = tf.io.VarLenFeature(dtype=tf.string)
                elif feature_type == 'float':
                    feature_dict[feature_name] = tf.io.VarLenFeature(dtype=tf.float32)
                elif feature_type == 'int':
                    feature_dict[feature_name] = tf.io.VarLenFeature(dtype=tf.int64)
                else:
                    raise ValueError('No valid feature type: feature type has to be either byte, float or int.')
        else:
            feature_dict = None

        # Create context dict
        if self.context_names:
            context_dict = {}
            for context_name, context_type in zip(self.context_names, self.context_types):
                if context_type == 'byte':
                    context_dict[context_name] = tf.io.VarLenFeature(dtype=tf.string)
                elif context_type == 'float':
                    context_dict[context_name] = tf.io.VarLenFeature(dtype=tf.float32)
                elif context_type == 'int':
                    context_dict[context_name] = tf.io.VarLenFeature(dtype=tf.int64)
                else:
                    raise ValueError('No valid feature type: feature type has to be either byte, float or int.')
        else:
            context_dict = None

        # Parse sequence example
        if feature_dict is not None or context_dict is not None:
            _, feature_list, _ = tf.io.parse_sequence_example(sequence_example, context_features=context_dict, sequence_features=feature_dict)
        else:
            feature_list = tf.constant([])

        return feature_list

    def _sparse_to_dense(self, element):
        element.update((key, tf.sparse.to_dense(value)) for key, value in element.items())
        return element

    def _split_features_and_labels(self, element):
        labels = element.pop(self.label_key)
        return element, labels

    def _one_hot_encode_sparse_tensor(self, element):
        """
        Returns a dataset element with one-hot encoded labels.
        Arguments:
            element: Dataset element, <dict>.
        """
        # Check if the maximum label value is less or equal to the number of classes
        msg = "Error Message: nbr_of_classes has be greater or equal to the maximum label value!"
        assert_op = tf.debugging.assert_less_equal(tf.sparse.reduce_max(element[self.label_key]), tf.constant([self.nbr_of_classes], dtype=tf.int64), message=msg)

        with tf.control_dependencies([assert_op]):
            # Determine indeces of the one-hot encoding
            indices = tf.concat((element[self.label_key].indices, tf.expand_dims(element[self.label_key].values, axis=-1)), axis=-1)

            # Map values to either zero or one
            values = tf.ones_like(element[self.label_key].values, dtype=tf.int64)

            # Determine shape of the new sparse tensor
            dense_shape = tf.concat((element[self.label_key].dense_shape, tf.expand_dims(tf.cast(self.nbr_of_classes, tf.int64), axis=-1)), axis=-1)

            # Replace label values
            element[self.label_key] = tf.SparseTensor(indices, values, dense_shape)

        return element

    def call(self, data_path: str):
        """
        Retruns a tensorflow dataset from a serialized sequence example.
        Arguments:
            data_path: Directory of the serialized sequence example files, <str>.
        """
        # Extract files
        files = tf.data.Dataset.list_files(file_pattern=data_path + "/*.tfrecord", shuffle=False, seed=self.seed)
        dataset = files.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True)

        # Shuffle dataset
        if self.shuffle:
            if self.buffer_size == -1:
                self.buffer_size = tf.data.experimental.cardinality(files)
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)

        # Batch dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        # Map dataset (deserialize/decode)
        dataset = dataset.map(map_func=self._parse_sequence_example, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

        # One Hot encode the labels
        if self.one_hot and self.label_key:
            dataset = dataset.map(map_func=self._one_hot_encode_sparse_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

        # Convert dataset elements to dense tensors
        if self.dense and self.feature_names:
            dataset = dataset.unbatch()
            dataset = dataset.map(map_func=self._sparse_to_dense, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

            # Batch dense dataset
            if self.one_hot and self.label_key:
                if self.label_major:
                    # The label has a shape of [sample, point, nbr_of_classes] and every feature has a shape of [sample, point].
                    padded_shapes = dict(zip(self.feature_names, [[None, None, None]] + [[None, None]] * (len(self.feature_names) - 1)))
                else:
                    # Every feature has a shape of [sample, point] and the label has a shape of [sample, point, nbr_of_classes].
                    padded_shapes = dict(zip(self.feature_names, [[None, None]] * (len(self.feature_names) - 1) + [[None, None, None]]))
            else:
                # Every feature and the label has a shape of [sample, point].
                padded_shapes = dict(zip(self.feature_names, [[None, None]] * len(self.feature_names)))
            dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes, drop_remainder=False)

        # Split dataset elements into features and labels
        if self.label_key:
            dataset = dataset.map(map_func=self._split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

        # Cache dataset
        if self.cache:
            dataset = dataset.cache()

        # Prefetch the processing
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def get_config(self):
        """
        Returns the configuration of the sequence example generator.
        """
        config = {
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'context_names': self.context_names,
            'context_types': self.context_types,
            'label_major': self.label_major,
            'dense': self.dense
        }
        base_config = super(SequenceExampleGenerator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape=None):
        """
        Returns the output shape of a dataset element.

        Arguments:
            input_shape: Shape tuple or list of shape tuples, <tuple>.

        Returns:
            output_shape: An shape tuple, <tuple>.
        """
        if input_shape is None:
            input_shape = tf.TensorShape([None, None])
        else:
            input_shape = tf.keras.layers.Lambda(lambda x: x).compute_output_shape(input_shape)

        if self.labels and self.one_hot:
            feature_shapes = {name: tf.TensorShape([self.batch_size] + tf.TensorShape(input_shape)) for name in self.feature_names}
            feature_shapes.pop(self.label_key)
            element_shape = (feature_shapes, tf.TensorShape([self.batch_size] + tf.TensorShape(input_shape) + [self.nbr_of_classes]))
        elif self.labels:
            feature_shapes = {name: tf.TensorShape([self.batch_size] + tf.TensorShape(input_shape)) for name in self.feature_names}
            feature_shapes.pop(self.label_key)
            element_shape = (feature_shapes, tf.TensorShape([self.batch_size] + tf.TensorShape(input_shape)))
        else:
            element_shape = ({name: tf.TensorShape([self.batch_size] + tf.TensorShape(input_shape)) for name in self.feature_names},)

        return element_shape


def from_config(data_path: str, config: dict):
    """
    Retruns a tensorflow dataset from given data files.
    Arguments:
        data_path: Directory of the serialized sequence example files, <str>.
        config: Configuration file, <ConfigParser or dict>.
    """
    # Check if directory existes
    assert os.path.isdir(data_path)
    assert isinstance(config, dict)

    # Parse config
    labels = config['labels']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    buffer_size = config['buffer_size']
    seed = config['seed']
    cache = config['cache']
    one_hot = config['one_hot']
    nbr_of_classes = config['number_of_classes']
    feature_names = config['feature_names']
    feature_types = config['feature_types']
    context_names = config['context_names']
    context_types = config['context_types']
    label_major = config['label_major']
    dense = config['dense']

    # Get generator instance
    seq_ex_gen = SequenceExampleGenerator(labels=labels, batch_size=batch_size, shuffle=shuffle, buffer_size=buffer_size, seed=seed,
                                          cache=cache, one_hot=one_hot, nbr_of_classes=nbr_of_classes, feature_names=feature_names,
                                          feature_types=feature_types, context_names=context_names, context_types=context_types,
                                          label_major=label_major, dense=dense)

    return seq_ex_gen(data_path=data_path)
