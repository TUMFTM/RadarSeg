"""
Configuration specifications of the training handled by the ConfigFileHandler.

This validation file containes one class for each main attribute of
the model training. Theses attributes are given by the model optimizer,
the training loss functions and the evaluation metrics. Each of these
classes contains static methods which specify the attributes of the
particular object as well as static methods which specify the content
of complex attribute values, like lists or tuples.

The config specifications of the objects (optimizers, losses or metrics)
attributes and are defined as a dict, whereas the dict keys corresponds
to the attributes of the object and values (of the key value pair) defines
the datatype, boundary values and default value of the attributes (keys).

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


class validate_optimizer():
    """
    Configuration specifications of the optimizer.
    """
    @staticmethod
    def validate_optimizer():
        # Value of __many__ is not a string to represent it as section not scalar.
        configspecs = {
            '__many__': {'__many__': "pass"}
        }
        return configspecs

    @staticmethod
    def validate_adam():
        configspecs = {
            'learning_rate': "float(min=0.0, default=0.001)",
            'beta_1': "float(min=0.0, default=0.9)",
            'beta_2': "float(min=0.0, default=0.999)",
            'epsilon': "float(min=0.0, default=0.0000001)",
            'amsgrad': "boolean(default=False)",
            'name': "string(default='Adam')"
        }
        return configspecs


class validate_losses():
    """
    Configuration specifications of the loss functions.
    """
    @staticmethod
    def validate_losses():
        # Value of __many__ is not a string to represent it as section not scalar.
        configspecs = {
            '__many__': {'__many__': "pass"}
        }
        return configspecs

    @staticmethod
    def validate_customcategoricalfocalcrossentropy():
        configspecs = {
            'from_logits': "boolean(default=False)",
            'alpha': "float_list_or_none(default=None)",
            'gamma': "float(min=0.0, default=2.0)",
            'class_weight': "float_list_or_none(default=None)",
            'reduction': "option('auto', 'none', 'sum', 'sum_over_batch_size', default='auto')",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_alpha():
        return "float(min=0.0, max=1.0)"

    @staticmethod
    def validate_categoricalcrossentropy():
        configspecs = {
            'from_logits': "boolean(default=False)",
            'label_smoothing': "float(min=0.0, max=1.0, default=0)",
            'reduction': "option('auto', 'none', 'sum', 'sum_over_batch_size', default='auto')",
            'name': "string(default=None)"
        }
        return configspecs


class validate_metrics():
    """
    Configuration specifications of the metrics.
    """
    @staticmethod
    def validate_metrics():
        # Value of __many__ is not a string to represent it as section not scalar.
        configspecs = {
            '__many__': {'__many__': "pass"}
        }
        return configspecs

    @staticmethod
    def validate_customtopkcategoricaltruepositives():
        configspecs = {
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_customtopkcategoricaltruenegatives():
        configspecs = {
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_customtopkcategoricalfalsepositives():
        configspecs = {
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_customtopkcategoricalfalsenegatives():
        configspecs = {
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_customfbetascore():
        configspecs = {
            'num_classes': "integer(min=0)",
            'average': "option('None', 'micro', 'macro', 'weighted', default='macro')",
            'beta': "float(min=0.0, default=1.0)",
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_customf1score():
        configspecs = {
            'num_classes': "integer(min=0)",
            'average': "option('None', 'micro', 'macro', 'weighted', default='macro')",
            'thresholds': "float_list_or_none(default=None)",
            'top_k': "integer(min=0, default=None)",
            'class_id': "integer(min=0, default=None)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_thresholds():
        return "float(min=0.0, max=1.0)"

    @staticmethod
    def validate_customconfusionmatrix():
        configspecs = {
            'num_classes': "integer(min=0)",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_categoricalaccuracy():
        configspecs = {
            "name": "string(default=categorical_accuracy)"
        }
        return configspecs


class validate_callbacks():
    """
    Configuration specifications of the training callbacks.
    """
    @staticmethod
    def validate_callbacks():
        # Value of __many__ is not a string to represent it as section not scalar.
        configspecs = {
            '__many__': {'__many__': "pass"}
        }
        return configspecs

    @staticmethod
    def validate_confusionmatrixlogger():
        configspecs = {
            'log_dir': "string(default='%(logdir)s/%(timestamp)s-%(modelname)s')"
        }
        return configspecs

    @staticmethod
    def validate_csvlogger():
        configspecs = {
            'filename': "string(default='%(logdir)s/%(timestamp)s-%(modelname)s/logs.csv')",
            'separator': "string(default=',')",
            'append': "boolean(default=False)"
        }
        return configspecs

    @staticmethod
    def validate_earlystopping():
        configspecs = {
            'monitor': "string(default='val_loss')",
            'min_delta': "float(min=0, default=0)",
            'patience': "integer(min=0, default=0)",
            'verbose': "integer(min=0, max=1, default=0)",
            'mode': "option('auto', 'min', 'max', default='auto')",
            'baseline': "float_or_none(default=None)",
            'restore_best_weights': "boolean(default=False)"
        }
        return configspecs

    @staticmethod
    def validate_learningratescheduler():
        configspecs = {
            'schedule': "dict(min=1, default=dict())",
            'verbose': "integer(min=0, max=1, default=0)",
            'log_dir': "string(default='%(logdir)s/%(timestamp)s-%(modelname)s')"
        }
        return configspecs

    @staticmethod
    def validate_schedule():
        configspecs = {
            'class_name': "string(default=None)",
            'config': "dict()"
        }
        return configspecs

    @staticmethod
    def validate_config():
        configspecs = {
            '__many__': "pass"
        }
        return configspecs

    @staticmethod
    def validate_modelcheckpoint():
        configspecs = {
            'filepath': "string(default='%(logdir)s/%(timestamp)s-%(modelname)s/saved_models/checkpoint_{epoch:04d}')",
            'monitor': "string(default='val_loss')",
            'verbose': "integer(min=0, max=1, default=0)",
            'save_best_only': "boolean(default=False)",
            'save_weights_only': "boolean(default=False)",
            'mode': "option('auto', 'min', 'max', default='auto')"
        }
        return configspecs

    @staticmethod
    def validate_reducelronplateau():
        configspecs = {
            'monitor': "string(default='val_loss')",
            'factor': "float(min=0, default=0.1)",
            'patience': "integer(min=0, default=10)",
            'verbose': "integer(min=0, max=1, default=0)",
            'mode': "option('auto', 'min', 'max', default='auto')",
            'min_delta': "float(min=0, default=0.0001)",
            'cooldown': "integer(min=0, default=0)",
            'min_lr': "float(min=0, default=0)"
        }
        return configspecs

    @staticmethod
    def validate_remotemonitor():
        configspecs = {
            'root': "string(default='http://localhost:9000')",
            'path': "string(default='/publish/epoch/end/')",
            'field': "string(default='data')",
            'headers': "pass",
            'send_as_json': "boolean(default=False)"
        }
        return configspecs

    @staticmethod
    def validate_tensorboard():
        configspecs = {
            'log_dir': "string(default='%(logdir)s/%(timestamp)s-%(modelname)s')",
            'histogram_freq': "integer(min=0, default=1)",
            'write_graph': "boolean(default=False)",
            'write_images': "boolean(default=False)",
            'update_freq': "option('epoch', 'batch', default='epoch')",
            'profile_batch': "integer(min=1, default=2)",
            'embeddings_freq': "integer(min=0, default=0)"
        }
        return configspecs

    @staticmethod
    def validate_terminateonnan():
        return {}
