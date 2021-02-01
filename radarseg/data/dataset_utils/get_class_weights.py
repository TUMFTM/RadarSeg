# Standard Libraries
import os
import numpy as np
import tensorflow as tf

# Local imports
from radarseg.data import generate


def get_class_weights(dataset, nbr_of_classes, alpha=None):
    """

    Reference:
        https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

    Arguments:
        dataset: Dataset object, <tf.data.Dataset>.
        nbr_of_classes: Number of classes, <int>.
        alpha: List of alpha balance factors, <List>.

    Returns:
        nbr_of_points: List of points per class, <List>.
        class_weights: List of class weights, <List>.
        compensation_factor: Alpha compensation factor, <float>.
    """
    # Initialize
    class_weights = np.expand_dims(np.zeros(shape=(nbr_of_classes,), dtype=np.float), axis=0)
    nbr_of_points = np.zeros(shape=(nbr_of_classes,), dtype=np.float)
    alpha = alpha if alpha is not None else 1.0

    # Get number of points per class
    for _, labels in dataset:
        nbr_of_points += tf.reduce_sum(labels, axis=tf.range(0, tf.rank(labels) - 1))

    # Calculate class weights
    class_weights = np.multiply(np.divide(1.0, nbr_of_points, where=nbr_of_points != 0), np.divide(np.sum(nbr_of_points), nbr_of_classes, where=nbr_of_classes != 0))

    # Calculate loss factor
    compensation_factor = np.multiply(np.divide(1, alpha, where=alpha != 0), nbr_of_points)
    compensation_factor = np.multiply(np.sum(compensation_factor), np.divide(1, np.sum(nbr_of_points)))
    compensation_factor = np.asscalar(compensation_factor)

    return nbr_of_points, class_weights, compensation_factor


def main(data_path, config, verbose=1):
    """
    Main function of the class weight calculation.

    Arguments:
        data_path: Folder path for the training and validation data, <str>.
        config: Configuration specifications, <ConfigObj or dict>.
        verbose: Verbosity mode (0 = silent, 1 = print return values), <int>.
    """
    # Parse config
    num_classes = config['GENERATE']['number_of_classes']
    try:
        alpha = config['TRAIN']['LOSSES']['Custom>CategoricalFocalCrossentropy']['alpha']
    except KeyError:
        alpha = None

    # Generate trainig data
    train_dataset = generate.from_config(data_path + '/train', config['GENERATE'])

    # Generate validation data
    val_dataset = generate.from_config(data_path + '/val', config['GENERATE'])

    # Calculate training class weights
    nbr_of_points, class_weights, compensation_factor = get_class_weights(train_dataset, num_classes, alpha)
    if verbose:
        print('\nTraining Dataset')
        print('Number of points: {}'.format(nbr_of_points))
        print('Class weights: {}'.format(class_weights))
        print('Loss factor: {}\n'.format(compensation_factor))

    # Get validation data statistics
    nbr_of_points, _, _ = get_class_weights(val_dataset, num_classes, alpha)
    if verbose:
        print('Validation Dataset')
        print('Number of points: {}\n'.format(nbr_of_points))


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
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY, "radarseg/config/default.ini"))
    parser.add_argument('--data', type=str, default="/data/nuScenes/default")
    args = parser.parse_args()

    # Get config from config file
    CONFIGSPEC = os.path.join(FILE_DIRECTORY, 'radarseg/config/validate_utils/configspecs.ini')
    config = ConfigFileHandler(configspec=CONFIGSPEC, allow_extra_values=-1)(args.config)

    # Call main function
    main(args.data, config, verbose=1)
