# Standard Libraries
from abc import ABC, abstractmethod

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports


def write_serialized_example(out_path: str, serialized):
    """
    Writs the given serialized example to the defined location.
    Arguments:
        out_path: Directory to write the serialized example, <str>.
        serialized: Serialized example to be written, <tf.Example>.
    """
    with tf.io.TFRecordWriter(out_path) as writer:
        writer.write(serialized)


class Serializer(ABC):
    """
    The Serializer class provides basic functionalities for the feature creation and the wrapping of features.
    Arguments:
        feature_names: List of feature names, <str: (d,)>.
        feature_types: List of feature types (byte, float, int), <str: (d,)>.
        name: Name of the instance, <str>.
    """
    def __init__(self,
                 feature_names,
                 feature_types,
                 name=None):

        # Initialize features
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name

    def _bytes_list(self, values):
        """
        Returns a BytesList element from a list of string / byte.
        Arguments:
            values: List of values to be converted to a tf.Example-compatible BytesList, <List: str>.
        """
        return tf.train.BytesList(value=values)

    def _float_list(self, values):
        """
        Returns a FloatList element from a list of float / double.
        Arguments:
            values: List of values to be converted to a tf.Example-compatible FloatList, <List: float>.
        """
        return tf.train.FloatList(value=values)

    def _int64_list(self, values):
        """
        Returns an Int64List element from a list of bool / enum / int / uint.
        Arguments:
            values: List of values to be converted to a tf.Example-compatible Int64List, <List: int>.
        """
        return tf.train.Int64List(value=values)

    def _bytes_feature(self, bytes_list):
        """
        Returns a Feature element from a BytesList element.
        Arguments:
            bytes_list: BytesList to be converted to a tf.Example-compatible feature, <BytesList>.
        """
        return tf.train.Feature(bytes_list=bytes_list)

    def _float_feature(self, float_list):
        """
        Returns a Feature element from a FloatList element.
        Arguments:
            float_list: FloatList to be converted to a tf.Example-compatible feature, <FloatList>.
        """
        return tf.train.Feature(float_list=float_list)

    def _int64_feature(self, int64_list):
        """
        Returns an Feature element from a Int64List element.
        Arguments:
            int64_list: Int64List to be converted to a tf.Example-compatible feature, <Int64List>.
        """
        return tf.train.Feature(int64_list=int64_list)

    def wrap_features(self, feature_dict):
        """
        Returns a Features element form a dict of given Feature elements and their associated names.
        Arguments:
            feature_dict: Dictionary of Feature elements and their names, <dict>
        """
        return tf.train.Features(feature=feature_dict)

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
        """Returns the configuration of the serializer."""
        config = {
            'name': self.name,
            'feature_names': self.feature_names,
            'feature_types': self.feature_types
        }
        return config


class SequenceExampleSerializer(Serializer):
    """
    The SequenceExampleSerializer wraps and serializes a given sequence of feature values with additional context.
    Arguments:
        feature_names: List of feature names, <str: b>.
        feature_types: List of feature types (byte, float, int), <str: d>.
        context_names: List of context names, <str>.
        context_types: List of context types (byte, float, int), <str>.
        name: Name of the instance, <str>.
    """
    def __init__(self,
                 feature_names,
                 feature_types,
                 context_names=[],
                 context_types=[],
                 name='sequence_example_serializer',
                 **kwargs):

        super(SequenceExampleSerializer, self).__init__(feature_names=feature_names, feature_types=feature_types, name=name, **kwargs)

        # Initialize context
        self.context_names = context_names
        self.context_types = context_types

        assert len(self.context_names) == len(self.context_types)

    def wrap_feature_list(self, features):
        """
        Returns a Featurelist element form a list of given Feature elements.
        Arguments:
            features: List of Feature elements, <List>
        """
        return tf.train.FeatureList(feature=features)

    def wrap_feature_lists(self, featurelist_dict):
        """
        Returns a FeatureLists element form a dict of given FeatureList elements and their associated names.
        Arguments:
            featurelist_dict: Dictionary of FeatureList elements and their names, <dict>
        """
        return tf.train.FeatureLists(feature_list=featurelist_dict)

    def get_context(self, context_values):
        """
        Retruns a Features element containing the context of the serialized sequence example.
        Arguments:
            context_values: List of context values, <List: np.array>.
        """
        # Define
        context_dict = {}

        # Get Feature elements
        for context_name, context_type, context_value in zip(self.context_names, self.context_types, context_values):
            # Wrap features
            if context_type == 'byte':
                temp_byte_context_values = [bytes(value, 'utf-8') for value in context_value.astype(str).tolist()]
                context_dict[context_name] = self._bytes_feature(self._bytes_list(temp_byte_context_values))
            elif context_type == 'float':
                context_dict[context_name] = self._float_feature(self._float_list(context_value.astype(np.float64).tolist()))
            elif context_type == 'int':
                context_dict[context_name] = self._int64_feature(self._int64_list(context_value.astype(np.int64).tolist()))
            else:
                raise ValueError('No valid feature type: feature type has to be either byte, float or int.')

        return self.wrap_features(context_dict)

    def call(self, feature_values, context_values=[]):
        """
        Returns the serialized sequence tf.Example element of the given feature values.
        Arguments:
            feature_values: List of d Lists with s numpy arrays containing the feature values, <List(List(np.array))>
            context_values: List of numpy arrays containing context values, <List(np.array)>.
        """
        # Check if number of feature values is equal to the number of feature names
        assert len(feature_values) == len(self.feature_names)

        # Check if number of feature value dimensions is equal to the number of feature types
        assert len(feature_values) == len(self.feature_types)

        # Check if number of context values is equal to the number of context names (and context types)
        assert len(context_values) == len(self.context_names)

        # Define
        feature_dict = {}

        # Get feature lists
        for feature_value, feature_name, feature_type in zip(feature_values, self.feature_names, self.feature_types):
            # Reset feature list
            feature_list = []
            for values in feature_value:
                # Wrap features
                if feature_type == 'byte':
                    temp_byte_feature_values = [bytes(value, 'utf-8') for value in values.astype(str).tolist()]
                    feature_list.append(self._bytes_feature(self._bytes_list(temp_byte_feature_values)))
                elif feature_type == 'float':
                    feature_list.append(self._float_feature(self._float_list(values.astype(np.float64).tolist())))
                elif feature_type == 'int':
                    feature_list.append(self._int64_feature(self._int64_list(values.astype(np.int64).tolist())))
                else:
                    raise ValueError('No valid feature type: feature type has to be either byte, float or int.')

            # Wrap features
            feature_dict[feature_name] = self.wrap_feature_list(feature_list)

        # Wrap feature lists
        feature_lists = self.wrap_feature_lists(feature_dict)

        # Get context
        if context_values:
            context = self.get_context(context_values)
        else:
            context = None

        # Wrap feature lists and context to tf.Example
        example = tf.train.SequenceExample(feature_lists=feature_lists, context=context)

        return example.SerializeToString()

    def get_config(self):
        """Returns the configuration of the sequence example serializer."""
        config = {
            'context_names': self.context_names,
            'context_types': self.context_types,
        }
        base_config = super(SequenceExampleSerializer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
