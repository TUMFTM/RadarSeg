# Standard Libraries
import os
import csv
import random
from abc import ABC, abstractmethod

# 3rd Party Libraries
import numpy as np
import tensorflow as tf
import progressbar
from nuscenes.nuscenes import NuScenes

# Local imports
from radarseg.data import generate
from radarseg.model import build
from radarseg.train import metrics
from radarseg.train import compile_model
from radarseg.visu import render_sample
from radarseg.visu import utils
from radarseg.preprocess import NuScenesPreProcessor


class Evaluator(ABC):
    """
    Model Evaluator

    Arguments:
        logdir: Folder path to store the evaluation log files, <str>.
        batch_size: Batch size used for the model prediction, <int>.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch), <int>.
        steps: Total number of epoch before stopping the evaluation, <int>.
        callbacks: List of callbacks to apply during evaluation, <list>.
        max_queue_size: Maximum size for the generator queue (used for generator input only), <int>.
        workers: Maximum number of processes to spin up when using process-based threading (used for generator input only), <int>.
        use_multiprocessing: Whether use process-based threading (used for generator input only), <bool>.
    """
    def __init__(self,
                 logdir: str = None,
                 batch_size: int = 1,
                 verbose: int = 2,
                 steps: int = None,
                 callbacks: list = None,
                 max_queue_size: int = 10,
                 workers: int = 1,
                 use_multiprocessing: bool = False):

        # Initialize Trainer instance
        self.logdir = logdir
        self.batch_size = batch_size
        self.verbose = verbose
        self.steps = steps
        self.callbacks = callbacks if callbacks is not None else list()
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing

        # Define output
        self.confidence = []

    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        return self.evaluate(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @abstractmethod
    def evaluate(self, model, dataset):
        """
        Evaluates the model on the given dataset.

        Arguments:
            model: Model to evaluate, <tf.keras.Model>.
            dataset: Dataset for the model evaluation, <tf.data.Dataset>.
        """

    def _get_confidence(self, batch, logs):
        """
        Callback to log model outputs with variable sizes.

        Arguments:
            batch: Current batch, <int>.
            logs: Dictionary of model logs (metrics), <Dict>.
        """
        self.confidence.append(np.array(logs['outputs']))

    def predict(self, model, dataset):
        """
        Returns the model prediction for the given dataset.

        Arguments:
            model: Model to evaluate, <tf.keras.Model>.
            dataset: Dataset for the model evaluation, <tf.data.Dataset>.

        Returns:
            prediction: Predicted class labels, <List of np.array>.
            confidence: Confidance of the predicted labels, <List of np.array>.
        """
        # Set callback to log model outputs
        self.callbacks.append(tf.keras.callbacks.LambdaCallback(on_predict_batch_end=self._get_confidence))

        # Execute forward pass
        try:
            model.predict(dataset, batch_size=self.batch_size, verbose=self.verbose, steps=self.steps, callbacks=self.callbacks,
                          max_queue_size=self.max_queue_size, workers=self.workers, use_multiprocessing=self.use_multiprocessing)
        except tf.errors.InvalidArgumentError:
            # Exception is required since TensorFlow does not support outputs with variabel sizes. To get the model prediction anyway
            # the LambdaCallback is added to the prediction call. Therfore, the exception is handled by the callback function and can
            # be passed.
            pass

        # Compute prediction
        prediction = tf.nest.map_structure(lambda confidence: np.expand_dims(np.argmax(confidence, axis=-1), axis=-1), self.confidence)

        return prediction, self.confidence

    def get_config(self):
        # Get instance configuration
        config = {
            'logdir': self.logdir,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'steps': self.steps,
            'callbacks': self.callbacks,
            'max_queue_size': self.max_queue_size,
            'workers': self.workers,
            'use_multiprocessing': self.use_multiprocessing,
        }

        return config


class NuScenesEvaluator(Evaluator):
    """
    NuScenes Model Evaluator

    Evaluates models based on the nuScenes dataset. The evaluator logs per sample F1 scores as well as visualizations of the
    validation data samples.

    Arguments:
        logdir: Folder path to store the evaluation log files, <str>.
        batch_size: Batch size used for the model prediction, <int>.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch), <int>.
        steps: Total number of epoch before stopping the evaluation, <int>.
        callbacks: List of callbacks to apply during evaluation, <list>.
        max_queue_size: Maximum size for the generator queue (used for generator input only), <int>.
        workers: Maximum number of processes to spin up when using process-based threading (used for generator input only), <int>.
        use_multiprocessing: Whether use process-based threading (used for generator input only), <bool>.
        visualize: Whether to visualize the prediction or not, <bool>.
        velocities: Whether to visualize the velocity vectors or not, <bool>.
        version: Version of the nuScenes dataset, <str>.
        dataroot: Path to the raw nuScenes dataset files, <str>.
        class_matching_dict: Key-Value-Pairs of the original category names and the user configured class names, <dict>.
        preprocessor_attributes: Keyword attributes for the nuScenes pre-processor, <dict>.
    """
    def __init__(self,
                 logdir: str = None,
                 batch_size: int = 1,
                 verbose: int = 2,
                 steps: int = None,
                 callbacks: list = None,
                 max_queue_size: int = 10,
                 workers: int = 1,
                 use_multiprocessing: bool = False,
                 visualize: bool = True,
                 velocities: bool = True,
                 version: str = 'v1.0-mini',
                 dataroot: str = None,
                 class_matching_dict: dict = None,
                 preprocessor_attributes: dict = None):
        # Initialize base class
        super(NuScenesEvaluator, self).__init__(logdir=logdir,
                                                batch_size=batch_size,
                                                steps=steps,
                                                verbose=verbose,
                                                callbacks=callbacks,
                                                max_queue_size=max_queue_size,
                                                workers=workers,
                                                use_multiprocessing=use_multiprocessing)

        # Initialize NuScenesEvaluator
        self.visualize = visualize
        self.velocities = velocities
        self.version = version
        self.dataroot = dataroot
        self.class_matching_dict = class_matching_dict if class_matching_dict is not None else {}
        self.preprocessor_attributes = preprocessor_attributes if preprocessor_attributes is not None else {}

        # Load nuScenes dataset
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

        # Get nuScenes pre-processor
        self.nusc_preprocessor = NuScenesPreProcessor(self.class_matching_dict, **self.preprocessor_attributes)

    def _write_to_csv(self, filename, data):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Scene token"] + ["Val F1 score sample " + str(i) for i in range(41)])
            writer.writerows(data)

    def _get_bounding_boxes(self, files):
        """
        Returns the bounding boxes of the scenes specified in files.

        Hint: The filenames must correspond to the nuScenes scene token.

        Attributes:
            files: Dataset of filenames of the nuScenes scenes, <tf.data.Dataset>.

        Returns:
            boxes: Nested list of bounding boxes (scenes, samples, boxes), <List>.
        """
        # Define
        boxes = []

        # Get bounding boxes
        for filename in files:
            # Define
            sample_boxes = []

            # Get current scene
            scene_token = os.path.basename(os.path.splitext(filename.numpy())[0]).decode("utf-8")
            scene = self.nusc.get('scene', scene_token)
            sample = self.nusc.get('sample', scene['first_sample_token'])
            # Skip first sample since it has less radar sweeps available
            sample = self.nusc.get('sample', sample['next'])

            # Iterate over all samples within the scene
            while True:
                # Get bounding boxes of the sample
                sample_boxes.append(self.nusc_preprocessor.get_annotations(dataset=self.nusc, sample=sample))

                # Check if sample is the last sample in the current scene
                if sample['next']:
                    sample = self.nusc.get('sample', sample['next'])
                else:
                    break

            boxes.append(sample_boxes)

        return boxes

    def _get_f1_scores(self, confidence, dataset, num_classes):
        """
        Returns sample-wise F1 scores for all scenes.

        Arguments:
            confidence: Confidance of the predicted label, <tf.Tensor>.
            dataset: Dataset with features and labels, <tf.data.Dataset>.
            num_classes: Number of classes, <int>.

        Returns:
            F1: List of F1 scores per sample, <List>.
        """
        F1 = []
        # Iterate over scenes
        for (_, lables), score in zip(dataset, confidence):
            sample_F1 = []
            # Iterate over samples
            for sample_labels, sample_scores in zip(tf.unstack(lables, axis=1), tf.unstack(score, axis=1)):
                sample_F1.append(float(metrics.F1Score(num_classes=num_classes, average='macro', top_k=1)(sample_labels, sample_scores)))

            F1.append(sample_F1)

        return F1

    def log_f1_scores(self, F1, files):
        data = [[os.path.basename(os.path.splitext(token.numpy())[0]).decode("utf-8")] + scores for token, scores in zip(files, F1)]
        filename = os.path.join(self.logdir, 'F1Scores.csv')
        self._write_to_csv(filename, data)

    def visualize_results(self, dataset, prediction, confidence, boxes, files):
        """
        Visualizes the model predictions and the ground truth data per sample.

        Arguments:
            dataset: Dataset with features and labels, <tf.data.Dataset>.
            prediction: Predicted labels, <tf.Tensor>.
            confidence: Confidance of the predicted labels, <tf.Tensor>.
            boxes: Nested list of bounding boxes (scenes, samples, boxes), <List>.
            files: Dataset of filenames of the nuScenes scenes, <tf.data.Dataset>.
        """
        # Define
        colors = utils.get_colors('blues')
        i = 0

        # Visualize samples
        with progressbar.ProgressBar(max_value=len(prediction)) as bar:
            for filename, (features, _), scene_predictions, scene_confidence, scene_boxes in zip(files, dataset, prediction, confidence, boxes):
                n_samples_dataset = features["x"].shape[1]
                n_samples_script_loader = len(scene_boxes)
                assert n_samples_dataset == n_samples_script_loader, f"""Number of scene samples in dataset: {n_samples_dataset}
                    is different than that from nuScenes loader {n_samples_script_loader}. Make sure to skip first sample in dataset creation."""
                # Update progressbar
                bar.update(i)

                # Get current scene
                scene_token = os.path.basename(os.path.splitext(filename.numpy())[0]).decode("utf-8")
                scene = self.nusc.get('scene', scene_token)

                # Get map
                log = self.nusc.get('log', scene['log_token'])
                map_ = self.nusc.get('map', log['map_token'])

                # Get sample
                sample = self.nusc.get('sample', scene['first_sample_token'])
                # Skip first sample since it has less radar sweeps available
                sample = self.nusc.get('sample', sample['next'])

                for j, sample_boxes in enumerate(scene_boxes):
                    # Get sample data
                    ref_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    pose = self.nusc.get('ego_pose', ref_data['ego_pose_token'])
                    sample_points = tf.concat((features['x'][:, j, :], features['y'][:, j, :], features['z'][:, j, :]), axis=0).numpy()
                    sample_predictions = np.squeeze(scene_predictions[:, j, :])
                    sample_confidence = np.amax(np.squeeze(scene_confidence[:, j, :]), axis=-1)

                    # Get velocities
                    if self.velocities:
                        velocities = tf.concat((features['vx_comp'][:, j, :], features['vy_comp'][:, j, :]), axis=0).numpy()
                    else:
                        velocities = None

                    # Set image name
                    out_path = os.path.join(self.logdir, 'images', scene_token + '_' + str(sample['timestamp']) + '_' + sample['token'] + '.svg')

                    # Render sample
                    render_sample.render_sample(sample_points, sample_predictions, sample_boxes, score=sample_confidence,
                                                velocities=velocities, map_=map_, pose=pose, colors=colors, out_path=out_path, out_format='svg')

                    # Get next sample
                    if sample['next']:
                        sample = self.nusc.get('sample', sample['next'])

                i += 1

    def evaluate(self, model, dataset, files, num_classes):
        # Make prediction
        prediction, confidence = self.predict(model=model, dataset=dataset)

        # Get bounding boxes
        boxes = self._get_bounding_boxes(files)

        # Log F1 scores
        F1 = self._get_f1_scores(confidence, dataset, num_classes)
        self.log_f1_scores(F1, files)

        # Visualize prediction
        if self.visualize:
            self.visualize_results(dataset, prediction, confidence, boxes, files)

    def get_config(self):
        # Get instance configuration
        config = {
            'visualize': self.visualize,
            'version': self.version,
            'dataroot': self.dataroot,
            'class_matching_dict': self.class_matching_dict,
            'preprocessor_attributes': self.preprocessor_attributes
        }

        # Get base class configuration
        base_config = super(NuScenesEvaluator, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def main(dataroot: str, data_path: str, log_path: str, config: dict, model_checkpoint: str = None):
    """
    Main function to evaluate a model.

    The main function parses the configuration file, loads the required data and builds the model to
    pass it to an appropriate Evaluator based on the given dataset.

    Arguments:
        dataroot: Folder of the raw dataset, <str>.
        data_path: Folder path for the evaluation data, <str>.
        log_path: Folder path to store the evaluation log files, <str>.
        config: Configuration specifications, <ConfigObj or dict>.
    """
    # Checks
    assert os.path.isdir(data_path)
    assert isinstance(config, dict)

    # Parse config
    dataset = config['PREPROCESS']['dataset']
    seed = config['COMPUTING']['seed']
    modelname = config['DEFAULT']['modelname']
    timestamp = config['DEFAULT']['timestamp']
    logdir = config['DEFAULT']['logdir']
    saved_model = config['TRAIN']['saved_model']
    batch_size = 1
    steps = None
    verbose = config['TRAIN']['verbose']
    max_queue_size = config['TRAIN']['max_queue_size']
    workers = config['TRAIN']['workers']
    use_multiprocessing = config['TRAIN']['use_multiprocessing']

    # Set generator attributes
    config['GENERATE']['batch_size'] = batch_size
    config['GENERATE']['shuffle'] = False

    # Set callbacks
    callback_list = []

    # Set and create log dir
    if log_path is None:
        log_path = os.path.join(logdir, timestamp + '-' + modelname)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/images', exist_ok=True)

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
    if model_checkpoint is not None:
        model.load_weights(model_checkpoint)

    elif saved_model is not None:
        model.load_weights(saved_model)

    # Generate validation data
    val_dataset = generate.from_config(data_path + '/val', config['GENERATE'])

    # Select evaluator
    if dataset.lower() == 'nuscenes':
        ModelEvaluator = NuScenesEvaluator

        # Get additional instance attributes
        version = config['PREPROCESS']['PREPROCESSOR'].pop('version')
        kwargs = {'visualize': True, 'version': version, 'dataroot': dataroot, 'class_matching_dict': config['PREPROCESS']['CLASSMATCHING'],
                  'preprocessor_attributes': config['PREPROCESS']['PREPROCESSOR']}

        # Get additional input attributes
        files = tf.data.Dataset.list_files(file_pattern=data_path + '/val' + "/*.tfrecord", shuffle=False, seed=seed)
        num_classes = config['GENERATE']['number_of_classes']
        exkwargs = {'files': files, 'num_classes': num_classes}
    else:
        raise ValueError('The configured {} dataset has no evaluator assigned to it.'.format(dataset))

    # Initialize evaluator instance
    evaluator = ModelEvaluator(logdir=log_path, batch_size=batch_size, steps=steps, verbose=verbose, callbacks=callback_list,
                               max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, **kwargs)

    # Evaluate model
    evaluator(model=model, dataset=val_dataset, **exkwargs)


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
    parser.add_argument('--dataroot', type=str, default="/data/nuscenes")
    parser.add_argument('--data', type=str, default="/preprocessed/dataset")
    parser.add_argument('--logs', type=str)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    # Get config from config file
    CONFIGSPEC = os.path.join(FILE_DIRECTORY, 'radarseg/config/validate_utils/configspecs.ini')
    config = ConfigFileHandler(configspec=CONFIGSPEC, allow_extra_values=-1)(args.config)

    # Call main function
    main(args.dataroot, args.data, args.logs, config, args.checkpoint)
