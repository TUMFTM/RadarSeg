# Standard Libraries
import os
import re
import ast
import time
import copy
from abc import ABC, abstractmethod

# 3rd Party Libraries
from configobj import ConfigObj
from configobj import flatten_errors, get_extra_values
from validate import Validator
from validate import _is_num_param, is_list, is_integer, is_float, is_string
from validate import ValidateError, VdtValueError, VdtTypeError, VdtValueTooLongError, VdtValueTooShortError

# Local imports
from radarseg.config.validate_utils import validate_preprocess, validate_generate, validate_model, validate_train


class FileHandler(ABC):
    """
    Abstract base class to handle different configuration files.
    """
    def __init__(self):
        pass

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


class ConfigFileHandler(FileHandler):
    """
    File handler for config files (.cfg or .ini).

    Naming convention: Section (subsection and so on) names have to be capitalized, otherwise
    they will not be treated as sections but handled as dict values.

    Note: This file handler is based on the configobj libary and is not compatible with the
          configparser structure. Moreover, only config files with a Section/Subsection
          structure are supported. Config files with deeper structure (Sub-subsections ...)
          are not supported.

    Inputs:
        filename: File name of the config (.ini or .cfg) file, <str>.

    Arguments:
        configspec: File name of the configspec (.ini or .cfg) file or a dict, <str or dict>.
        encoding: Encoder to decode the provided config file (and encode it for writing).
        interpolation: Whether to use string interpolation or not, <bool>.
        raise_errors: Whether to raise errors immediately or after the  whole file is parsed, <bool>.
        list_values: Whether to allow list values to be parsed, <bool>.
        create_empty: Whether to create a file if the file specified in filename does not exist, <bool>.
        stringify: Whether to convert non-string values to strings when writing the config file, <bool>.
        indent_type: Indentation used to write the config file (default four spaces), <str>.
        default_encoding: Encoding used to decode byte strings in the ConfigObj before writing.
        write_empty_values: Whether to write empty strings as empty values, <bool>.
        allow_extra_values: Whether to allow extra (not defined) values or not, <int>.
                            Options: -1 raise exception for every undefined value
                                      0 print exception, delete undefined values and fall back to the default value
                                      1 allow extra (undefined) values

    Return:
        config: Validated config object, <ConfigObj>.
    """
    def __init__(self,
                 configspec=None,
                 encoding=None,
                 interpolation=True,
                 raise_errors=False,
                 list_values=True,
                 create_empty=False,
                 stringify=True,
                 indent_type=None,
                 default_encoding=None,
                 write_empty_values=False,
                 allow_extra_values=0,
                 **kwargs):
        # Initialize the file handler base class
        super(ConfigFileHandler, self).__init__(**kwargs)

        # Initialize the config file handler
        self.configspec = configspec
        self.encoding = encoding
        self.interpolation = interpolation
        self.raise_errors = raise_errors
        self.list_values = list_values
        self.create_empty = create_empty
        self.stringify = stringify
        self.indent_type = indent_type
        self.default_encoding = default_encoding
        self.write_empty_values = write_empty_values
        self.allow_extra_values = allow_extra_values
        self.valid = True

    def _is_dict(self, value, min=None, max=None):
        """
        Validation function for dict values.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Get boundary values
        (min_val, max_val) = _is_num_param(('min', 'max'), (min, max))

        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                if value == 'dict()' or value == 'dict':
                    value = dict()
                else:
                    raise ValidateError('A value "{}" could not be converted to a dict!'.format(value))
            except Exception as e:
                raise e

            # Validate whether the value is a dict
            if not isinstance(value, dict):
                raise ValidateError('A value of type {} was passed when a dict was expected!'.format(type(value)))

        elif isinstance(value, dict):
            pass

        else:
            raise VdtTypeError(value)

        # Validate whether the value is within the boundaries
        if (min_val is not None) and (len(value) < min_val):
            raise VdtValueTooShortError(value)

        if (max_val is not None) and (len(value) > max_val):
            raise VdtValueTooLongError(value)

        return value

    def _is_int_or_none(self, value, min=None, max=None):
        """
        Validation function for an integer values or None.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        if value in set((None, 'none', 'None')):
            return None
        else:
            return is_integer(value, min=min, max=max)

    def _is_float_or_none(self, value, min=None, max=None):
        """
        Validation function for an float values or None.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        if value in set((None, 'none', 'None')):
            return None
        else:
            return is_float(value, min=min, max=max)

    def _is_dict_list(self, value, min=None, max=None):
        """
        Validation function for a list of nested dicts.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted to a dict!'.format(value))
            except Exception as e:
                raise e

        return [self._is_dict(elem) for elem in is_list(value, min, max)]

    def _is_string_list(self, value, min=None, max=None):
        """
        Validation function for a list of string values.
        Replacement (extension) of the original 'is_string_list' function.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [is_string(elem) for elem in is_list(value, min, max)]
        else:
            raise ValidateError('A value of type {} was passed when a string list was expected!'.format(type(value)))

    def _is_int_list(self, value, min=None, max=None):
        """
        Validation function for a list of integer values.
        Replacement (extension) of the original 'is_int_list' function.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [is_integer(elem) for elem in is_list(value, min, max)]
        else:
            raise ValidateError('A value of type {} was passed when a integer list was expected!'.format(type(value)))

    def _is_float_list(self, value, min=None, max=None):
        """
        Validation function for a list of float values.
        Replacement (extension) of the original 'is_float_list' function.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [is_float(elem) for elem in is_list(value, min, max)]
        else:
            raise ValidateError('A value of type {} was passed when a float list was expected!'.format(type(value)))

    def _is_int_list_or_none(self, value, min=None, max=None):
        """
        Validation function for a list of integer values or None.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [self._is_int_or_none(elem) for elem in is_list(value, min, max)]

        elif value is None:
            return value

        else:
            raise ValidateError('A value of type {} was passed when a int list or None type was expected!'.format(type(value)))

    def _is_float_list_or_none(self, value, min=None, max=None):
        """
        Validation function for a list of float values or None.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [is_float(elem) for elem in is_list(value, min, max)]

        elif value is None:
            return value

        else:
            raise ValidateError('A value of type {} was passed when a float list or None type was expected!'.format(type(value)))

    def _is_string_list_or_none(self, value, min=None, max=None):
        """
        Validation function for a list of string values or None.

        Reference: https://github.com/DiffSK/configobj/blob/master/src/configobj/validate.py
        """
        # Check if value is string
        if isinstance(value, str):
            # Convert string value
            try:
                value = ast.literal_eval(value)
            except ValueError:
                raise ValidateError('The value "{}" could not be converted!'.format(value))
            except Exception as e:
                raise e

        if isinstance(value, list):
            return [is_string(elem) for elem in is_list(value, min, max)]

        elif value is None:
            return value

        else:
            raise ValidateError('A value of type {} was passed when a string list or None type was expected!'.format(type(value)))

    def _stringify_dict(self, config):
        """
        Converts config dict values to strings.

        Note: This method is required since a dict will be interpreted as section (subsection) by the
        configobj parser. However, a list of dicts does not have key for every dict within the list,
        why a parsing would fail due to the missing section name. For this reason and because sometimes
        a dict has to be passed as it is, a naming convention is made. All dict values with an associated
        capitalized key (name) will be interpreted as sections, everyting else will be converted to a
        string value to preserve the original (python) data type.
        """
        if not isinstance(config, dict):
            return config

        for key, value in config.items():
            if isinstance(value, list):
                for i, elem in enumerate(value):
                    value[i] = self._stringify_dict(elem)
                    if isinstance(value[i], dict):
                        value[i] = str(value[i])

            elif isinstance(value, dict):
                config[key] = self._stringify_dict(value)
            else:
                continue

            # Determine if item is a real subsection (convention: section keys are all upper case)
            if not key.isupper():
                config[key] = str(value)

        return config

    def _destringify_none(self, config):
        """
        Converts all string 'None' values of the config dict into None type values.
        """
        if not isinstance(config, dict):
            return config

        for key, value in config.items():
            if isinstance(value, dict):
                self._destringify_none(value)
            elif isinstance(value, (list, tuple)):
                for i, elem in enumerate(value):
                    value[i] = self._destringify_none(elem)

            if value == 'None':
                config[key] = None

        return config

    def _recursive_validation(self, config, validator, module, key: str, section_list: list):
        """
        Returns a recursively validated config object.

        Inputs:
            config: Config object, <dict or ConfigObj>.

        Arguments:
            validator: Validator instance used to validate the config object, <Validator>.
            module: Module to get the configspec from, <class>.
            key: Key/name of the current config object, <str>.
            section_list: Section list ('path') of the current config object, <list[str, ..., str]>.

        Return:
            config: Validated config object, <dict>.
        """
        # Recursive call of the validation
        def _recursive_call(self, value, validator, module, key, section_list):
            if isinstance(value, (dict, list)):
                return self._recursive_validation(value, validator, module, key, section_list)
            else:
                if hasattr(module, 'validate_' + key.lower()):
                    return self._recursive_validation(value, validator, module, key, section_list)
                else:
                    return value

        # Remove non alphanumeric characters, except underscores
        key = re.sub(r'\W+', '', key)

        # Check layer
        if isinstance(config, dict):
            # Get layer configspec
            try:
                configspec = getattr(module, 'validate_' + key.lower())()
            except AttributeError as a:
                self._print_errors([(section_list, key, str(a))])
                raise a
            except Exception as e:
                raise e
            else:
                # Set default section to allow string interpolation
                configspec['DEFAULT'] = self.configspec['DEFAULT']

            # Validate layer
            config = ConfigObj(config, configspec=configspec)
            results = config.validate(validator, copy=True, preserve_errors=True)

            # Remove default section
            del config['DEFAULT']

            # Report extra (not defined) value errors
            if self.allow_extra_values < 1:
                # Get extra values
                extra_values = get_extra_values(config)

                # Filter out wrongly assigned dict values which are interpreted as sections
                extra_values = list(filter(lambda x: x[1] not in configspec, extra_values))
                extra_values = list(filter(lambda x: len(x[0]) < 1, extra_values))

                # Set error message
                msg = 'Value is not defined in the model validation file!'

                # Print errors
                self._print_extra_val_results(extra_values, section_list, msg)

                # Remove extra values form config
                list(map(config.__delitem__, frozenset(map(lambda x: x[1], extra_values)) - frozenset(configspec)))

            # Report validation errors
            self._print_val_results(config, results, section_list=section_list)

            # Cast config
            config = config.dict()

            # Validate
            for key, value in config.items():
                section_list.append(key)
                config[key] = _recursive_call(self, value, validator, module, key, section_list)
                del section_list[-1]

        elif isinstance(config, (list, tuple)):
            for i, value in enumerate(config):
                if isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except ValueError:
                        pass
                    except Exception as e:
                        raise e

                    config[i] = _recursive_call(self, value, validator, module, key, section_list)

                else:
                    config[i] = _recursive_call(self, value, validator, module, key, section_list)

        elif hasattr(module, 'validate_' + key.lower()):
            # Get layer configspec
            check = getattr(module, 'validate_' + key.lower())()

            # Check configuration value
            try:
                return validator.check(check, config, missing=False)
            except ValidateError as v:
                self._print_val_results({}, {key: v}, section_list=section_list)
            except Exception as e:
                raise e

        return config

    def _print_errors(self, errors):
        for (section_list, key, error) in errors:
            # Print errors
            if not section_list:
                print('The section "{}" failed validation. Error: {}'.format(key, error))
            elif key is not None:
                print('The "{}" key in the section "{}" failed validation. Error: {}'.format(key, ', '.join(section_list), error))
            else:
                print('The following section was missing:{} '.format(', '.join(section_list)))

    def _print_extra_val_results(self, extra_values: list, section_list: list = None, msg: str = None):
        if extra_values and self.allow_extra_values < 0:
            # Set validation faild
            self.valid = False

        if section_list is not None:
            # Set section list of the error
            extra_values = [tuple([section_list if i == 0 else entry for i, entry in enumerate(value)]) for value in extra_values]

        if msg is not None:
            # Set error message of the error
            extra_values = [value + (msg, ) for value in extra_values]

        # Print errors
        self._print_errors(extra_values)

    def _print_val_results(self, config, results, section_list=None):
        if results is not True:
            # Set validation faild
            self.valid = False

            # Get errors
            flatt_errors = flatten_errors(config, results)

            # Set section list of the error
            if section_list is not None:
                flatt_errors = [tuple([section_list if i == 0 else entry for i, entry in enumerate(error)]) for error in flatt_errors]

            # Print errors
            self._print_errors(flatt_errors)

    def _check_preprocess(self, section, validator):
        # Get configured dataset
        dataset = section['dataset']

        # Validate subsections
        for subsection in section.sections:
            if dataset is not None and dataset != 'None':
                if subsection == 'PREPROCESSOR':
                    # Get validation module
                    module = getattr(validate_preprocess, 'validate_' + dataset)

                    # Valiate preprocessor
                    section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=dataset, section_list=['PREPROCESS', subsection])

                elif subsection == 'CLASSMATCHING':
                    section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=subsection, section_list=['PREPROCESS', subsection])

                else:
                    raise VdtValueError('Validation faild! Subsection {} in section "PREPROCESS" is no valid subsection!'.format(subsection))

            else:
                section[subsection].clear()

        return section.dict()

    def _check_generate(self, section, validator):
        # Get configured dataformat
        dataformat = section['dataformat']

        # Validate section
        if dataformat is not None and dataformat != 'None':
            # Get validation module
            module = getattr(validate_generate, 'validate_' + dataformat)

            # Valiate generator
            section = self._recursive_validation(config=section, validator=validator, module=module, key=dataformat, section_list=['GENERATE'])

            return section

        else:
            return {key: value for key, value in section.items() if key == 'dataformat'}

    def _check_model(self, section, validator):
        for subsection in section.sections:
            # Get configured model
            module = section[subsection]['module']
            key = section[subsection]['layer']

            if module is not None and module != 'None' and key is not None and key != 'None':
                # Get validation module
                module = getattr(validate_model, 'validate_' + module)

                # Valiate model layers
                section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=key, section_list=['MODEL', subsection])

            else:
                section[subsection] = {'module': None, 'layer': None}

        return section.dict()

    def _check_train(self, section, validator):
        # Validate subsections
        for subsection in section.sections:
            if subsection == 'OPTIMIZER':
                # Get validation module
                module = validate_train.validate_optimizer

                # Set default optimizer if subsection is empty
                if not section[subsection]:
                    section[subsection] = {'Adam': {}}

                # Validate subsection
                section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=subsection, section_list=['TRAIN', subsection])

            elif subsection == 'LOSSES':
                # Get validation module
                module = validate_train.validate_losses

                # Set default loss function if subsection is empty
                if not section[subsection]:
                    section[subsection] = {'CategoricalCrossentropy': {}}

                # Validate subsection
                section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=subsection, section_list=['TRAIN', subsection])

            elif subsection == 'METRICS':
                # Get validation module
                module = validate_train.validate_metrics

                # Validate subsection
                section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=subsection, section_list=['TRAIN', subsection])

            elif subsection == 'CALLBACKS':
                # Get validation module
                module = validate_train.validate_callbacks

                # Validate subsection
                section[subsection] = self._recursive_validation(config=section[subsection], validator=validator, module=module, key=subsection, section_list=['TRAIN', subsection])

            elif subsection.isupper():
                raise VdtValueError('Validation faild! Subsection {} in section "TRAIN" is no valid subsection!'.format(subsection))

        return section.dict()

    def read(self, filename):
        # Read configspec file
        self.configspec = ConfigObj(infile=self.configspec, interpolation=False, list_values=False, _inspec=True)

        # Read config file
        config = ConfigObj(infile=filename, configspec=self.configspec, encoding=self.encoding, interpolation=self.interpolation,
                           raise_errors=self.raise_errors, list_values=self.list_values, create_empty=self.create_empty, file_error=True,
                           stringify=self.stringify, indent_type=self.indent_type, default_encoding=self.default_encoding,
                           write_empty_values=self.write_empty_values)

        return config

    def write(self, config, filename):
        # Copy the config file to preserve to original object
        config = copy.deepcopy(config)

        # Set output filename (write path)
        config.filename = filename

        # Convert non-section dict elemtes to strings
        config = self._stringify_dict(config)

        # Write config file (.ini file)
        config.write()

    def call(self, filename):
        # Get config
        config = self.read(filename)

        # Validate top level config
        validator = Validator({'dict': self._is_dict, 'integer_or_none': self._is_int_or_none, 'float_or_none': self._is_float_or_none,
                               'dict_list': self._is_dict_list, 'string_list': self._is_string_list, 'int_list': self._is_int_list,
                               'float_list': self._is_float_list, 'int_list_or_none': self._is_int_list_or_none,
                               'float_list_or_none': self._is_float_list_or_none, 'string_list_or_none': self._is_string_list_or_none})
        results = config.validate(validator, preserve_errors=True)

        # Insert timestamp
        config['DEFAULT']['timestamp'] = config['DEFAULT'].get('timestamp', time.strftime("%Y%m%d-%H%M%S", time.gmtime()))

        # Get default values for the default section
        config['DEFAULT'] = {**self.configspec['DEFAULT'], **config['DEFAULT']}
        self.configspec['DEFAULT'] = config['DEFAULT']

        # Report extra (not defined) sections
        if self.allow_extra_values < 1:
            # Get extra values
            extra_values = get_extra_values(config)

            # Filter out non top level values
            extra_values = list(filter(lambda x: len(x[0]) < 2, extra_values))

            # Print errors
            msg = 'Section is not defined in the configspec file!'
            self._print_extra_val_results(extra_values, msg=msg)

            # Remove extra values form config
            for extra_value in extra_values:
                if not extra_value[0]:
                    del config[extra_value[-1]]
                else:
                    section_mane = extra_value[0][0]
                    del config.get(section_mane)[extra_value[-1]]

        # Report validation errors
        self._print_val_results(config, results)

        # Validate preprocessing
        config['PREPROCESS'] = self._check_preprocess(section=ConfigObj(config['PREPROCESS']), validator=validator)

        # Validate generate
        config['GENERATE'] = self._check_generate(section=ConfigObj(config['GENERATE']), validator=validator)

        # Validate model
        config['MODEL'] = self._check_model(section=ConfigObj(config['MODEL']), validator=validator)

        # Validate train
        config['TRAIN'] = self._check_train(section=ConfigObj(config['TRAIN']), validator=validator)

        # Parse None values (destringify)
        config = self._destringify_none(config)

        # Raise exception if validation failed
        if not self.valid:
            raise ValidateError('Validation faild! Please see validation error messages!')

        return config


if __name__ == "__main__":
    # Imports
    import argparse

    # Set working directory
    FILE_DIRECTORY = os.getcwd() + "/"

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=os.path.join(FILE_DIRECTORY, 'examples/02_KPConvCLSTM.ini'))
    parser.add_argument('--configspec', type=str, default=os.path.join(FILE_DIRECTORY, 'radarseg/config/validate_utils/configspecs.ini'))
    args = parser.parse_args()

    # Write config
    handler = ConfigFileHandler(configspec=args.configspec)
    config = handler(args.filename)
    handler.write(config, args.filename)
