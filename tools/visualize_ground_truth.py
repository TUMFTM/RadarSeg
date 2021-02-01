# Standard Libraries
import os
import random
import time
from abc import ABC

# 3rd Party Libraries
import numpy as np
import tensorflow as tf
import progressbar
import pickle
from nuscenes.nuscenes import NuScenes

# Local imports
from radarseg.data import generate
from radarseg.visu import render_sample
from radarseg.visu import utils
from radarseg.preprocess import NuScenesPreProcessor

"""
After creating a pre-processed data set, execute this script to visualize the ground truth classes of the radar
points based on the used input config file settings and the resulting classes stored in the preprocessed data set.
"""


class Visualizer(ABC):
    """
    Model Visualizer

    Arguments:
        logdir: Folder path to store the evaluation log files, <str>.
        batch_size: Batch size used for the model prediction, <int>.
    """
    def __init__(self, logdir: str = None, batch_size: int = 1):

        # Initialize Trainer instance
        self.logdir = logdir
        self.batch_size = batch_size

        # Define output
        self.confidence = []

    def __call__(self, *args, **kwargs):
        """
        Wraps `call`.
        Arguments:
            *args: Positional arguments to be passed to `self.call`.
            **kwargs: Keyword arguments to be passed to `self.call`.
        """
        return self.visualize(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        # Get instance configuration
        config = {
            'logdir': self.logdir,
            'batch_size': self.batch_size
        }

        return config


class NuscenesVisualizer(Visualizer):
    """
    NuScenes Model Visualizer

    Visualizes the data set with its ground truth labels.

    Arguments:
        logdir: Folder path to store the evaluation log files, <str>.
        batch_size: Batch size used for the model prediction, <int>.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch), <int>.
        velocities: Whether to visualize the velocity vectors or not, <bool>.
        version: Version of the nuScenes dataset, <str>.
        dataroot: Path to the raw nuScenes dataset files, <str>.
        class_matching_dict: Key-Value-Pairs of the original category names and the user configured class names, <dict>.
        preprocessor_attributes: Keyword attributes for the nuScenes pre-processor, <dict>.
    """
    def __init__(self,
                 logdir: str = None,
                 batch_size: int = 1,
                 velocities: bool = True,
                 version: str = 'v1.0-mini',
                 dataroot: str = None,
                 class_matching_dict: dict = None,
                 preprocessor_attributes: dict = None):
        # Initialize base class
        super(NuscenesVisualizer, self).__init__(logdir=logdir, batch_size=batch_size)

        # Initialize NuscenesVisualizer
        self.velocities = velocities
        self.version = version
        self.dataroot = dataroot
        self.class_matching_dict = class_matching_dict if class_matching_dict is not None else {}
        self.preprocessor_attributes = preprocessor_attributes if preprocessor_attributes is not None else {}

        # Load nuScenes dataset
        file_ending = os.path.splitext(self.dataroot)[-1]
        if file_ending == ".pkl":
            with open(self.dataroot, 'rb') as f:
                print("Loading NuScenes tables from pickle file...")
                timer_start = time.time()
                self.nusc = pickle.load(f)
                timer_end = time.time()
                print("Finished loading in %d seconds" % (timer_end - timer_start))
        else:
            print("Loading NuScenes...")
            timer_start = time.time()
            self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
            timer_end = time.time()
            print("Finished loading in %d seconds" % (timer_end - timer_start))

        # Get nuScenes pre-processor
        self.nusc_preprocessor = NuScenesPreProcessor(self.class_matching_dict, **self.preprocessor_attributes)

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

    def visualize_results(self, dataset, boxes, files):
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
        with progressbar.ProgressBar(max_value=len(boxes)) as bar:
            for filename, (features, labels), scene_boxes in zip(files, dataset, boxes):
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
                    sample_lls = labels[:, j, :, :].numpy()
                    sample_labels = np.squeeze(np.argmax(sample_lls, axis=2))

                    # Get velocities
                    if self.velocities and 'vx_comp' in features:
                        velocities = tf.concat((features['vx_comp'][:, j, :], features['vy_comp'][:, j, :]), axis=0).numpy()
                    else:
                        velocities = None

                    # Set image name
                    out_path = os.path.join(self.logdir, 'images', scene_token + '_' + str(sample['timestamp']) + '_' + sample['token'] + '.svg')

                    # Render sample
                    render_sample.render_sample(points=sample_points, labels=sample_labels, annotations=sample_boxes,
                                                velocities=velocities, map_=map_, pose=pose, colors=colors, out_path=out_path, out_format='svg')

                    # Get next sample
                    if sample['next']:
                        sample = self.nusc.get('sample', sample['next'])

                i += 1

    def visualize(self, dataset, files, num_classes):
        boxes = self._get_bounding_boxes(files)
        self.visualize_results(dataset, boxes, files)

    def get_config(self):
        # Get instance configuration
        config = {
            'version': self.version,
            'dataroot': self.dataroot,
            'class_matching_dict': self.class_matching_dict,
            'preprocessor_attributes': self.preprocessor_attributes
        }

        # Get base class configuration
        base_config = super(NuscenesVisualizer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def main(dataroot: str, data_path: str, log_path: str, config: dict):
    """
    Main function to visualize a model.

    The main function parses the configuration file, loads the required data and builds the model to
    pass it to an appropriate Visualizer based on the given dataset.

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
    logdir = config['DEFAULT']['logdir']

    # Set generator attributes
    config['GENERATE']['batch_size'] = 1
    config['GENERATE']['shuffle'] = False

    # Set and create log dir
    if log_path is None:
        log_path = os.path.join(logdir, modelname)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/images', exist_ok=True)

    # Set global random seed
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Generate validation data
    val_dataset = generate.from_config(data_path + '/val', config['GENERATE'])

    # Select visualizer
    if dataset.lower() == 'nuscenes':
        ModelVisualizer = NuscenesVisualizer

        # Get additional instance attributes
        version = config['PREPROCESS']['PREPROCESSOR'].pop('version')
        kwargs = {'version': version, 'dataroot': dataroot, 'class_matching_dict': config['PREPROCESS']['CLASSMATCHING'],
                  'preprocessor_attributes': config['PREPROCESS']['PREPROCESSOR']}

        # Get additional input attributes
        files = tf.data.Dataset.list_files(file_pattern=data_path + '/val' + "/*.tfrecord", shuffle=False, seed=seed)
        num_classes = config['GENERATE']['number_of_classes']
        exkwargs = {'files': files, 'num_classes': num_classes}
    else:
        raise ValueError('The configured {} dataset has no visualizer assigned to it.'.format(dataset))

    # Initialize visualizer instance
    visualizer = ModelVisualizer(logdir=log_path, **kwargs)

    # Evaluate model
    visualizer(dataset=val_dataset, **exkwargs)


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
    parser.add_argument('--dataroot', type=str, default="/data/nuscenes", help='Specify stored pickle file path instead of nuscenes path for speed up')
    parser.add_argument('--data', type=str, default="/preprocessed/dataset")
    parser.add_argument('--logs', type=str)
    args = parser.parse_args()

    # Get config from config file
    CONFIGSPEC = os.path.join(FILE_DIRECTORY, '../radarseg/config/validate_utils/configspecs.ini')
    config = ConfigFileHandler(configspec=CONFIGSPEC, allow_extra_values=-1)(args.config)

    # Call main function
    main(args.dataroot, args.data, args.logs, config)
