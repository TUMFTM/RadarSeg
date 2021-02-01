# Config
Configuration module.

## Config file and handling
This module handles the configuration (config) of all other modules within the project. The config file holds the initialization values of all instances and defines the overall process pipeline. The config file handler is responsible for the processing (parsing and validating) of those instructions and the transformation to an usable data format.

## Validation
The validation specifications are defined within the *validate_utils* and grouped into modules. Therefore, the validation instructions are split into a configuration specification (configspec) file, which defines the top level structure of the config file and separate validation files for each module.

## Add a new module
To add a completly new module to the process pipeline three major changes have to be made to the configuration.
1. Add a module check function to the ConfigFileHandler
    ```
    def _check_newmodule(self, section, validator):
        module = getattr(validate_newmodule, 'validate_' + keyattribute)
        section = self._recursive_validation(config=section, validator=validator, module=module, key=keyattribute, section_list=['NEWMODULE'])
        return section.dict()
    ```
2. Add the module check to the ConfigFileHandler call function
    ```
    config['NEWMODULE'] = self._check_newmodule(section=ConfigObj(config['NEWMODULE']), validator=validator)
    ```
3. Add a new section and its attributes to the configspecs.ini
    ```
    [NEWMODULE]
    key = value
    ```
4. Add a new validation file
    ```
    validate_newmodule.py
    ```

## Add a new keyelement
The keyelement defines the content of a module and can be a new dataset or a new model architecture for example. The keyelement is part of a module and implemented as a new class within the module validation file. This class contains several static methods defining the elements of the keyelement (e.g. the layers of a model). Each static method returns a dictionary which defines the checks (data type, range and default value) for each attribute of the associated element. If an attribute defines a List or Dict object an additional static method has to be implemented which defines the check for the content of the List or Dict object. This additional static method is named after the atrribute name and returns the check of the List or Dict content.

To get familiar with this structure you should have a look at the [validation file](radarseg/config/validate_utils/validate_generate.py) of the data generator.

### Note
Do not use all upper case attribute names. This format is reserved to define config file sections. However, you can use CapitalizedWords as long as they are not all upper case.

## References
The config file handler as well as the validation are based on the ConfigObj library which can be found at the links below. \
[ConfigObj documentation](https://configobj.readthedocs.io/en/latest/configobj.html) \
[ConfigObj GitHub](https://github.com/DiffSK/configobj) \
[ConfigObj tutorial](http://www.voidspace.org.uk/python/articles/configobj.shtml) \
[Validate documentation](http://www.voidspace.org.uk/python/validate.html)
