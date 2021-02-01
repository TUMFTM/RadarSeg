# Standard Libraries
import os
import random

# 3rd Party Libraries
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.data import generate
from radarseg.model import build
from radarseg.train import compile_model
from radarseg.train import callbacks


class Trainer():
    """
    Model Trainer

    Note: The differentiation between class instance attributes and inputs (attributes of the
    train function) is made with regards to an model optimization process. Therefore, all attributes
    that are subject to the optimization process are handled as inputs, whereas all attributes, which
    should be equal across multiple trainings are handled as instance attributes.

    Inputs:
        model: Compiled model instance to train, <tf.keras.Model>.
        training_data: Training dataset, <tf.data.Dataset>.
        validation_data: Validation dataset, <tf.data.Dataset>.
        shuffle: Whether to shuffle the training data before each epoch, <bool>.
        class_weight: Mapping of class indices (integers) to a weight (float) value, <dict>.
        sample_weight: Array to weight the training samples, <np.array>.
        initial_epoch: Index (value) of the initial epoch, <int>.
        steps_per_epoch: Total number of steps before declaring one epoch finished (optional), <int or None>.
        validation_steps: Total number of steps before stopping the validation at the end of every epoch (optional), <int or None>.

    Arguments:
        epochs: Total number of epoch before stopping the training, <int>.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch), <int>.
        callbacks: List of callbacks to apply during training, <list>.
        validation_freq: Specifies how many training epochs to run before a new validation run is performed, <int>.
        max_queue_size: Maximum size for the generator queue (used for generator input only), <int>.
        workers: Maximum number of processes to spin up when using process-based threading (used for generator input only), <int>.
        use_multiprocessing: Whether use process-based threading (used for generator input only), <bool>.

    Returns:
        history: A record of training loss values and metrics values at successive epochs,
                 as well as validation loss values and validation metrics values (if applicable), <tf.History>.
    """
    def __init__(self,
                 epochs: int = 1,
                 verbose: int = 2,
                 callbacks: list = None,
                 validation_freq: int = 1,
                 max_queue_size: int = 10,
                 workers: int = 1,
                 use_multiprocessing: bool = False):

        # Initialize Trainer instance
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks if callbacks is not None else list()
        self.validation_freq = validation_freq
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        if not isinstance(value, int):
            raise TypeError("The epochs attribute has to be of type int, but a value of type '{}' was given!".format(type(value)))
        elif value < 1:
            raise ValueError("The epochs attribute has to be greater or equal to one, but a value of '{}' was given!".format(value))
        else:
            self._epochs = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, int):
            raise TypeError("The verbose attribute has to be of type int, but a value of type '{}' was given!".format(type(value)))
        elif value < 0 or value > 2:
            raise ValueError("The verbose attribute has to be either 0, 1 or 2, but a value of '{}' was given!".format(value))
        else:
            self._verbose = value

    @property
    def validation_freq(self):
        return self._validation_freq

    @validation_freq.setter
    def validation_freq(self, value):
        if not isinstance(value, int):
            raise TypeError("The validation frequency attribute has to be of type int, but a value of type '{}' was given!".format(type(value)))
        elif value < 1:
            raise ValueError("The validation frequency attribute has to be greater than one, but a value of '{}' was given!".format(value))
        else:
            self._validation_freq = value

    @property
    def max_queue_size(self):
        return self._max_queue_size

    @max_queue_size.setter
    def max_queue_size(self, value):
        if not isinstance(value, int):
            raise TypeError("The max queue size attribute has to be of type int, but a value of type '{}' was given!".format(type(value)))
        elif value < 0:
            raise ValueError("The max queue size attribute has to be greater than zero, but a value of '{}' was given!".format(value))
        else:
            self._max_queue_size = value

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        if not isinstance(value, int):
            raise TypeError("The workers attribute has to be of type int, but a value of type '{}' was given!".format(type(value)))
        elif value < 0:
            raise ValueError("The workers attribute has to be greater than zero, but a value of '{}' was given!".format(value))
        else:
            self._workers = value

    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        return self.train(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def train(self,
              model,
              training_data,
              validation_data=None,
              shuffle=False,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_steps=None,
              **kwargs):
        # Map class weights to class indices (if a list is provided)
        if isinstance(class_weight, (list, tuple)):
            class_weight = dict(zip(list(range(len(class_weight))), class_weight))

        # Execute model training
        history = model.fit(training_data, epochs=self.epochs, verbose=self._verbose, callbacks=self.callbacks,
                            validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
                            sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                            validation_freq=self._validation_freq, max_queue_size=self._max_queue_size,
                            workers=self._workers, use_multiprocessing=self.use_multiprocessing, **kwargs)

        return history

    def get_config(self):
        config = {
            'epochs': self.epochs,
            'verbose': self.verbose,
            'callbacks': self.callbacks,
            'validation_freq': self.validation_freq,
            'max_queue_size': self.max_queue_size,
            'workers': self.workers,
            'use_multiprocessing': self.use_multiprocessing
        }

        return config


def main(data_path: str, log_path: str, config: dict):
    """
    Main function to start a training from a config file.

    Arguments:
        data_path: Folder path for the training and validation data, <str>.
        log_path: Folder path to store the training log files, <str>.
        config: Configuration specifications, <ConfigObj or dict>.
    """
    # Checks
    assert os.path.isdir(data_path)
    assert isinstance(config, dict)

    # Parse config
    seed = config['COMPUTING']['seed']
    modelname = config['DEFAULT']['modelname']
    timestamp = config['DEFAULT']['timestamp']
    logdir = config['DEFAULT']['logdir']
    saved_model = config['TRAIN']['saved_model']
    epochs = config['TRAIN']['epochs']
    verbose = config['TRAIN']['verbose']
    validation_freq = config['TRAIN']['validation_freq']
    max_queue_size = config['TRAIN']['max_queue_size']
    workers = config['TRAIN']['workers']
    use_multiprocessing = config['TRAIN']['use_multiprocessing']
    shuffle = config['TRAIN']['shuffle']
    class_weight = config['TRAIN']['class_weight']
    sample_weight = config['TRAIN']['sample_weight']
    initial_epoch = config['TRAIN']['initial_epoch']
    steps_per_epoch = config['TRAIN']['steps_per_epoch']
    validation_steps = config['TRAIN']['validation_steps']

    # Set and create log dir
    if log_path is None:
        log_path = os.path.join(logdir, timestamp + '-' + modelname)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/saved_models', exist_ok=True)

    # Save configuration
    ConfigFileHandler().write(config, log_path + '/config.ini')

    # Set global random seed
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Build model
    model = build.from_config(config['MODEL'])

    # Compile model
    compile_model.from_config(model, config['TRAIN'])

    # Load saved model weights
    if saved_model is not None:
        model.load_weights(saved_model)

    # Generate trainig data
    train_dataset = generate.from_config(data_path + '/train', config['GENERATE'])

    # Generate validation data
    val_dataset = generate.from_config(data_path + '/val', config['GENERATE'])

    # Get training callbacks
    callback_list = callbacks.from_config(config=config['TRAIN']['CALLBACKS'])

    # Initialize trainer instance
    trainer = Trainer(epochs=epochs, verbose=verbose, callbacks=callback_list, validation_freq=validation_freq,
                      max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    # Start training
    trainer(model=model, training_data=train_dataset, validation_data=val_dataset, shuffle=shuffle,
            class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)


if __name__ == "__main__":
    # Imports
    import argparse
    from radarseg.config.handle_config import ConfigFileHandler

    # Enable gpu memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Define working directory
    FILE_DIRECTORY = os.getcwd() + "/"

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY, "examples/02_KPConvCLSTM.ini"))
    parser.add_argument('--data', type=str, default="/preprocessed/dataset")
    parser.add_argument('--logs', type=str)
    args = parser.parse_args()

    # Get config from config file
    CONFIGSPEC = os.path.join(FILE_DIRECTORY, 'radarseg/config/validate_utils/configspecs.ini')
    config = ConfigFileHandler(configspec=CONFIGSPEC, allow_extra_values=-1)(args.config)

    # Call main function
    main(args.data, args.logs, config)
