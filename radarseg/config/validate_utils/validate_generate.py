"""
Configuration specifications of all generators handled by the ConfigFileHandler.

This validation file containes one class for each generator that is
handled (checkt) by the config file handler. Therefore, each class
contains the following functions: a static method that specifies the
attributes of the generator as well as the dataformat and several static
methods which specify the content (the elements) of complex generator
input values, like lists or tuples.

The config specifications of the generator attributes are defined as
a dict, whereas the dict keys correspond with the attribute or original
class. The value (of the key value pair) defines the datatype, boundary
values and default value of the attribute (key).

The specification of the list and tuple elements are provided as checks
of the validator module or custom validation functions - defined in the
config file handler.

Hint: All permissible values of the configspec dicts can be found in the
documentation of the validate module, except for the custom validation
functions, which are defined in the ConfigFileHandler class.

References:
ConfigObj documentation: https://configobj.readthedocs.io/en/latest/configobj.html#
Validate documentation: http://www.voidspace.org.uk/python/validate.html

Author: Felix Fent
Date: June 2020
"""


class validate_sequence_example():
    """
    Configuration specifications of the sequence example generator.
    """
    @staticmethod
    def validate_sequence_example():
        configspecs = {
            'labels': "boolean(default=True)",
            'batch_size': "integer(min=1, default=1)",
            'shuffle': "boolean(default=True)",
            'buffer_size': "integer(min=-1, default=-1)",
            'seed': "integer(min=0, default=42)",
            'cache': "boolean(default=True)",
            'one_hot': "boolean(default=True)",
            'number_of_classes': "integer(min=1)",
            'dataformat': "option(sequence_example)",
            'feature_names': "string_list(default=list())",
            'feature_types': "string_list(default=list())",
            'context_names': "string_list(default=list())",
            'context_types': "string_list(default=list())",
            'label_major': "boolean(default=False)",
            'dense': "boolean(default=True)",
        }
        return configspecs

    @staticmethod
    def validate_feature_types():
        return "option('byte', 'float', 'int')"

    @staticmethod
    def validate_context_types():
        return "option('byte', 'float', 'int')"
