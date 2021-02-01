# Standard Libraries
import os
from abc import ABC, abstractmethod
import warnings

# 3rd Party Libraries
import progressbar
import numpy as np
from nuscenes.nuscenes import NuScenes

# Local imports
from radarseg.config.handle_config import ConfigFileHandler
from radarseg.data.dataset_utils import nuscenes_utils
from radarseg.data.serialize import SequenceExampleSerializer, write_serialized_example


class PreProcessor(ABC):
    """
    Abstract pre-processor class.
    Arguments:
        class_matching_dict: Key-Value-Pairs of the original category names and the user configured class names, <dict: (str,str)>.
        point_major: Whether the output is point-major (n x d) or dim-major (d x n), <bool>.
        name: Name of the pre-processor instance, <str>.
    """
    def __init__(self,
                 class_matching_dict,
                 point_major=True,
                 name=None):
        # Initialize
        self.class_matching_dict = class_matching_dict
        self.point_major = point_major
        self.name = name

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

    @property
    @abstractmethod
    def class_matching_dict(self):
        return self._class_matching_dict

    @class_matching_dict.setter
    @abstractmethod
    def class_matching_dict(self, value):
        self._class_matching_dict = value

    def unify_class_matching_dict(self, user_set_classes: dict):
        """
        Assigns an integer to every unique value of the class_matching_dict.
        None values are associated with a zero value, while other values are assigned to an increasing integer value by ascending alphabetical order.
        Arguments:
            user_set_classes: Dictionary that mappes the category names of the dataset to specified calls names of the user, <dict>.
        """
        # Get class token
        self.unique_classes = list(set(user_set_classes.values()))
        self.unique_classes = sorted(self.unique_classes)

        # Add background class
        if 'None' in self.unique_classes:
            # Set 'None' to the 0th element
            self.unique_classes.insert(0, self.unique_classes.pop(self.unique_classes.index('None')))
        else:
            # Insert 'None' as the 0th element
            self.unique_classes.insert(0, 'None')

        # Get class matching dict
        for category in user_set_classes:
            user_set_classes[category] = self.unique_classes.index(user_set_classes[category])

        return user_set_classes

    def filter(self, values, boundaries=None):
        """
        Returns the filtered values by removing values outside the boundaries.

        Note: The first ("zeroth") dimension represents the elements to be filtered.

        Arguments:
            values: Array of values with d + 1 dimensions, <np.array>.
            boundaries: List of tuple pairs, mapping an indices (0, .., d) to two boundary values (min, max), <list>.

        Retruns:
            The filtered d + 1 dimensional values, <np.array>.
        """
        # Initialize boundary values
        lower_boundary = np.full(shape=values.shape[1:], fill_value=np.NINF)
        upper_boundary = np.full(shape=values.shape[1:], fill_value=np.Inf)

        # Set boundary values
        if boundaries is not None:
            ind = np.array(list(map(lambda x: list(x[0]), boundaries)), dtype=np.int)
            min_values = np.fromiter(map(lambda x: x[1][0], boundaries), dtype=np.float)
            max_values = np.fromiter(map(lambda x: x[1][1], boundaries), dtype=np.float)
            np.put(lower_boundary, ind=ind, v=min_values)
            np.put(upper_boundary, ind=ind, v=max_values)

        # Filter lower boundary (value > min)
        lower_mask = np.greater(values, lower_boundary)

        # Filter upper boundary (value < max)
        upper_mask = np.less(values, upper_boundary)

        # Combine masks
        mask = np.logical_and(lower_mask, upper_mask)
        mask = np.all(mask, axis=tuple(range(1, values.ndim)))

        return values[mask, ...]

    def put(self, elements, targets=None):
        """
        Maps the element values to the defined target values.

        Note: The first ("zeroth") dimension represents the elements to be mapped,
              all other dimensions represent the element values (channels).

        Arguments:
            elements: Array of values with d + 1 dimensions, <np.array>.
            targets: List of tuple pairs, mapping an indices (0, .., d) to a target value, <list>.

        Retruns:
            Mapped d + 1 dimensional value array, <np.array>.
        """
        if targets is not None:
            for target in targets:
                if elements.ndim > 1:
                    idx = [slice(None) for _ in range(elements.ndim)]
                    idx[1:] = target[0]
                    elements[tuple(idx)] = target[1]
                else:
                    elements[target[0]] = target[1]

        return elements

    def get_config(self):
        """
        Returns pre-processor configuration.
        """
        config = {
            'name': self.name,
            'class_matching_dict': self.class_matching_dict
        }
        return config

    @abstractmethod
    def compute_output_shape(self, input_shape):
        """
        Abstractmethod that returns the output shape of the pre-processed sample.
        """
        pass


class NuScenesPreProcessor(PreProcessor):
    """
    nuScenes Pre-Processor.
    Arguments:
        remove_background: Whether to remove all unlabeld points, <bool>.
        invalid_states: List of valid values for the invalide_state feature, <int list> or None.
        dynprop_states: List of valid dyn_prob values, <int list> or None.
        ambig_states: List of valid ambig_states values, <int list> or None.
        annotation_attributes: List of valid annotation attributes, <str list> or None.
        locations: List of valid scene locations, <str list> or None.
        sensor_names: List of considered sensors, <str list>.
        keep_channels: List of valid feature channel names, <str list>.
        wlh_factor: Scale factor for point in box determination, <float>.
        wlh_offset: Offset for point in box determination, <float>.
        use_multisweep: Whether to use multiple sweeps, <bool>.
        nsweeps: Number of sweeps to use (use_multisweep has to be True), <int>.
        channel_boundaries: Hash mapping of nuScenes channel names to a pair of boundary values (min, max), <dict>.
        channel_target_values: Hash mapping of nuScenes channel names to a scalar value, <dict>.
        attribut_matching_dict: Hash mapping of nuScenes annotation attributes to class names, <dict>.
    """
    def __init__(self,
                 class_matching_dict,
                 point_major=True,
                 remove_background=False,
                 invalid_states=None,
                 dynprop_states=None,
                 ambig_states=None,
                 annotation_attributes=['vehicle.moving', 'vehicle.stopped', 'vehicle.parked', 'cycle.with_rider', 'cycle.without_rider', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'pedestrian.moving'],
                 locations=['boston-seaport', 'singapore-queenstown', 'singapore-onenorth', 'singapore-hollandvillage'],
                 sensor_names=['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
                 keep_channels=None,
                 wlh_factor=1.0,
                 wlh_offset=0.0,
                 use_multisweep=False,
                 nsweeps=3,
                 channel_boundaries=None,
                 channel_target_values=None,
                 attribut_matching_dict=None,
                 name='nuscenes_preprocessor',
                 **kwargs):
        # Initialize base class
        super(NuScenesPreProcessor, self).__init__(class_matching_dict=class_matching_dict, point_major=point_major, name=name, **kwargs)

        # Initialize
        self.remove_background = remove_background
        self.invalid_states = invalid_states
        self.dynprop_states = dynprop_states
        self.ambig_states = ambig_states
        self.annotation_attributes = annotation_attributes
        self.locations = locations
        self.sensor_names = sensor_names
        self.keep_channels = keep_channels
        self.wlh_factor = wlh_factor
        self.wlh_offset = wlh_offset
        self.use_multisweep = use_multisweep
        self.nsweeps = nsweeps
        self.channel_boundaries = channel_boundaries
        self.channel_target_values = channel_target_values
        self.attribut_matching_dict = attribut_matching_dict

        # Display warning if wlh_factor is being used
        if self.wlh_factor != 1.0:
            warnings.warn("The 'wlh_factor' is deprecated. Consider using 'wlh_offset' instead.")

        # Reduce classes by one if remove_backgound is true
        if self.remove_background:
            del self.unique_classes[0]
            self._class_matching_dict = {key: value - 1 for key, value in self._class_matching_dict.items()}
            self._attribut_matching_dict = {key: value - 1 for key, value in self._attribut_matching_dict.items()}

    @property
    def class_matching_dict(self):
        return nuscenes_utils.category_name_mapping(self._class_matching_dict)

    @class_matching_dict.setter
    def class_matching_dict(self, value):
        value = {k: str(v) for k, v in value.items()}
        self._class_matching_dict = self.unify_class_matching_dict(nuscenes_utils.category_token_mapping(value))

    @property
    def annotation_attributes(self):
        return nuscenes_utils.attributes_name_mapping(self._annotation_attributes)

    @annotation_attributes.setter
    def annotation_attributes(self, attributes):
        self._annotation_attributes = nuscenes_utils.attributes_token_mapping(attributes)

    @property
    def sensor_names(self):
        return self._sensor_names

    @sensor_names.setter
    def sensor_names(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('The sensor_names attribute has to be of type list or tuple, '
                            'but a value of type {} was given!'.format(type(value)))
        elif 'LIDAR_TOP' in value and len(value) != 1:
            raise ValueError("Only lidar or radar data can be used, not both!")
        else:
            self._sensor_names = value
            self._sensor_selector()

    @property
    def keep_channels(self):
        return self._keep_channels

    @keep_channels.setter
    def keep_channels(self, value):
        if value is None:
            self._keep_channels = tuple(range(18))
        elif not isinstance(value, (list, tuple)):
            raise TypeError('The keep_channels attribute has to be of type list or tuple, '
                            'but a value of type {} was given!'.format(type(value)))
        else:
            self._keep_channels = tuple([nuscenes_utils.get_channel_index(name) for name in value])

    @property
    def wlh_factor(self):
        return self._wlh_factor

    @wlh_factor.setter
    def wlh_factor(self, value):
        if not isinstance(value, float):
            try:
                self._wlh_factor = float(value)
            except TypeError:
                raise TypeError("The wlh factor has to be a float value!")
        elif value < 0.0:
            raise ValueError("The wlh factor has to be a positive value!")
        else:
            self._wlh_factor = value

    @property
    def wlh_offset(self):
        return self._wlh_offset

    @wlh_offset.setter
    def wlh_offset(self, value):
        if not isinstance(value, float):
            try:
                self._wlh_offset = float(value)
            except TypeError:
                raise TypeError("The wlh offset has to be a float value!")
        elif value < 0.0:
            raise ValueError("The wlh offset has to be a positive value!")
        else:
            self._wlh_offset = value

    @property
    def nsweeps(self):
        return self._nsweeps

    @nsweeps.setter
    def nsweeps(self, value):
        if not isinstance(value, int):
            raise TypeError("The number of sweeps (nsweeps) has to be an integer value!")
        elif value < 0:
            raise ValueError("The number of sweeps (nsweeps) has to be a positive integer value!")
        else:
            self._nsweeps = value

    @property
    def channel_boundaries(self):
        if self._channel_boundaries is not None:
            # Map -inf or inf to None
            value = {k: (v[0], v[1]) if not np.isinf(v[0]) else (None, v[1]) for k, v in self._channel_boundaries}
            value = {k: (v[0], v[1]) if not np.isinf(v[1]) else (v[0], None) for k, v in value.items()}

            # Map channel indices to names
            return {str(self._get_channel_names(k)[0]): v for k, v in value.items()}
        else:
            return None

    @channel_boundaries.setter
    def channel_boundaries(self, value):
        if value is None:
            self._channel_boundaries = None
        elif not isinstance(value, dict):
            raise ValueError('The channel_boundaries attribute must be either '
                             'a dict or None, but a {} was given.'.format(type(value)))
        else:
            # Map None values to either -inf or inf
            value = {k: (v[0], v[1]) if v[0] is not None else (np.NINF, v[1]) for k, v in value.items()}
            value = {k: (v[0], v[1]) if v[1] is not None else (v[0], np.Inf) for k, v in value.items()}

            # Map channel names to indices
            value = {nuscenes_utils.get_channel_index(k): v for k, v in value.items()}

            # Convert to list of key value pairs
            self._channel_boundaries = [((int(k),), v) for k, v in value.items()]

    @property
    def channel_target_values(self):
        if self._channel_target_values is not None:
            # Map channel indices to names
            return {str(self._get_channel_names(k)[0]): v for k, v in self._channel_target_values}
        else:
            return None

    @channel_target_values.setter
    def channel_target_values(self, value):
        if value is None:
            self._channel_target_values = None
        elif not isinstance(value, dict):
            raise ValueError('The channel_boundaries attribute must be either '
                             'a dict or None, but a {} was given.'.format(type(value)))
        else:
            # Map channel names to indices
            value = {nuscenes_utils.get_channel_index(k): v for k, v in value.items()}

            # Convert to list of key value pairs
            self._channel_target_values = [((int(k),), v) for k, v in value.items()]

    @property
    def attribut_matching_dict(self):
        if self._attribut_matching_dict:
            return {nuscenes_utils.get_attribute_name(k): self.unique_classes[v] for k, v in self._attribut_matching_dict.items()}
        else:
            return None

    @attribut_matching_dict.setter
    def attribut_matching_dict(self, value):
        if value is None:
            self._attribut_matching_dict = {}
        elif not isinstance(value, dict):
            raise ValueError('The attribut_matching_dict attribute must be either '
                             'a dict or None, but a {} was given.'.format(type(value)))
        else:
            # Initialize attribut_matching_dict
            self._attribut_matching_dict = {}

            # Set attribut_matching_dict by exchanging attribute names with attribute token
            # and the target class name with the unique class index
            for k, v in value.items():
                if str(v) not in self.unique_classes:
                    self.unique_classes.append(str(v))
                self._attribut_matching_dict[nuscenes_utils.get_attribute_token(k)] = self.unique_classes.index(str(v))

    def _sensor_selector(self):
        if 'LIDAR_TOP' in self.sensor_names:
            self._get_channel_names = nuscenes_utils.get_lidar_channel_names
            self._get_number_of_pc_dimensions = nuscenes_utils.get_number_of_lidar_pc_dimensions
            self._get_pc_from_file = nuscenes_utils.get_lidar_pc_from_file
        else:
            self._get_channel_names = nuscenes_utils.get_radar_channel_names
            self._get_number_of_pc_dimensions = nuscenes_utils.get_number_of_radar_pc_dimensions
            self._get_pc_from_file = nuscenes_utils.get_radar_pc_from_file

    def _set_box_label(self, dataset, box):
        # Get box instance
        box_annotation = dataset.get('sample_annotation', box.token)
        box_attribute = box_annotation['attribute_tokens']
        box_instance = dataset.get('instance', box_annotation['instance_token'])

        # Assign box class label according to the following hierarchy.
        if not box_attribute:
            # 1. If the box has no attribute, the label is determined according to the corresponding class for the box instance
            box.label = self._class_matching_dict[box_instance['category_token']]
        elif any(elem in box_attribute for elem in self._attribut_matching_dict):
            # 2. If the box attribute is part of the attribute matching dict, the corresponding label is assigned
            box.label = self._attribut_matching_dict[box_attribute[0]]
        elif any(elem in box_attribute for elem in self._annotation_attributes):
            # 3. If the box attribute is included in the annotation attributes, the corresponding label for the box instance is assigned
            box.label = self._class_matching_dict[box_instance['category_token']]
        else:
            # 4. If the box attribute is not included in the annotation attributes, the points are marked as invalid
            box.label = -1

        return box

    def get_annotations(self, dataset, sample: dict):
        """
        Returns all annotations (bounding boxes) of the given sample and configured sensors.

        Arguments:
            dataset: nuScenes dataset item to be pre-processed, <NuScenes>.
            sample: Sample of the nuScenes dataset, <dict>.
        """
        # Initialize list of bounding boxes
        boxes = []
        sample_boxes = []

        # Get sensor bounding boxes
        for sensor_name in self.sensor_names:
            sensor_boxes = dataset.get_boxes(sample['data'][sensor_name])
            sensor_boxes = list(map(lambda box: self._set_box_label(dataset, box), sensor_boxes))
            sample_boxes += sensor_boxes

        # Remove duplicates (two boxes are equal if they have the same center and token)
        seen_boxes = set()
        for i, box in enumerate(sample_boxes):
            if tuple(box.center.tolist() + [box.token]) not in seen_boxes:
                seen_boxes.add(tuple(box.center.tolist() + [box.token]))
                boxes.append(box)

        return [box for box in boxes if box.label > -1]

    def get_points(self, dataset, sample: dict, sensor_name: str, min_distance: float = 1.0):
        """
        Extracts the points of a defined sensor of a given sample of the given nuScenes dataset.
        Arguments:
            dataset: nuScenes dataset item to be pre-processed, <NuScenes>.
            sample: Sample of the nuScenes dataset, <dict>.
            sensor_name: Name of the radar sensor, <str>.
            min_distance: Distance below which points are discarded during sweep aggregation, <float>.
        """
        # Get points
        if self.use_multisweep:
            # Load point cloud from multiple sweeps
            points, _ = nuscenes_utils.get_pc_from_file_multisweep(nusc=dataset, sample_rec=sample, chan=sensor_name, ref_chan=sensor_name,
                                                                   nsweeps=self._nsweeps, min_distance=min_distance, invalid_states=self.invalid_states,
                                                                   dynprop_states=self.dynprop_states, ambig_states=self.ambig_states)
        else:
            # Load sample point cloud (single sweep)
            points = self._get_pc_from_file(nusc=dataset, sample_rec=sample, chan=sensor_name, invalid_states=self.invalid_states,
                                            dynprop_states=self.dynprop_states, ambig_states=self.ambig_states)

        # Transform point cloud to vehicle frame
        sample_data = dataset.get('sample_data', sample['data'][sensor_name])
        calibrated_sensor = dataset.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        points = nuscenes_utils.transform_pc_from_sensor_to_vehicle_frame(calibrated_sensor, sensor_name, points.points)

        # Transpose point representation [d,n] -> [n,d]
        points = np.transpose(points)

        return points

    def get_labels(self, dataset, sample: dict, sensor_name: str, points: np.ndarray):
        """
        Determines the class lables of the radar points.
        Arguments:
            dataset: nuScenes dataset item to be pre-processed, <NuScenes>.
            sample: Sample of the nuScenes dataset, <dict>.
            sensor_name: Name of the radar sensor, <str>
            points: Point cloud, <np.float: n, d>.
        """
        # Transform point cloud to global frame
        sample_data = dataset.get('sample_data', sample['data'][sensor_name])
        ego_pose = dataset.get('ego_pose', sample_data['ego_pose_token'])
        points = np.transpose(points)
        points[:3, :] = nuscenes_utils.transform_pc_from_vehicle_to_global_frame(ego_pose, points[:3, :])
        points = np.transpose(points)

        # Define
        number_of_radar_points = points.shape[0]
        boxes = dataset.get_boxes(sample['data'][sensor_name])
        if self.remove_background:
            # Initialize all points as invalid points
            sensor_labels = np.ones(shape=[number_of_radar_points], dtype=int) * -1
        else:
            # Initialize all points as background (None class)
            sensor_labels = np.zeros(shape=[number_of_radar_points], dtype=int)

        # Set point z-coordinate values (second dimension) to zero to
        # compensate interpolation errors (this method only works for
        # quasi 2d radar data).
        # Format: targets = [target dimension, target value]
        points = self.put(elements=points, targets=[((2,), 0)])

        # Determine box labels
        boxes = list(map(lambda box: self._set_box_label(dataset, box), boxes))

        # Radar sensors usually don't have height information, only x and y. The z coordinate is set to 0 as default.
        # Nevertheless, bounding boxes can be annotated a few centimeters above zero, so using xyz might exclude points that are actually inside a box.
        # We therefore don't use the z coordinate for radar points (only x and y, as if we were looking only from the BEV).
        use_z_coordinate = not ('RADAR' in sensor_name)

        # Determine point labels
        for box in boxes:
            # Determine if points are within the bounding box and the box label
            points_in_box = nuscenes_utils.extended_points_in_box(box=box, points=np.transpose(points[:, 0:3]),
                                                                  wlh_factor=self._wlh_factor, wlh_offset=self._wlh_offset,
                                                                  use_z=use_z_coordinate)

            # Assign class label to all points within the bounding box.
            sensor_labels[points_in_box] = box.label

        return sensor_labels

    def call(self, dataset, sample):
        """
        Executes the pre-processing of a nuScenes dataset.
        Arguments:
            dataset: nuScenes dataset object, <NuScenes>
            sample: nuScenes keyframe/sample to be pre-processed, <dict>.
        """
        # Initialize sample points- and labels-list
        sample_points = np.empty(shape=[0, self._get_number_of_pc_dimensions()])
        sample_labels = np.empty(shape=(0,), dtype=int)

        for sensor_name in self.sensor_names:
            # Extract sensor point cloud
            sensor_points = self.get_points(dataset, sample, sensor_name)

            # Filer sensor point cloud
            sensor_points = self.filter(sensor_points, self._channel_boundaries)

            # Add to sample point cloud
            sample_points = np.append(sample_points, sensor_points, axis=0)

            # Determine class labels
            sensor_labels = self.get_labels(dataset, sample, sensor_name, sensor_points)
            sample_labels = np.append(sample_labels, sensor_labels, axis=0)

        # Map sample points
        sample_points = self.put(sample_points, self._channel_target_values)

        # Remove invalid points (invalid points are determined according to the 'get_labels' function)
        # For more details see additional comments in the 'get_labels' function
        invalide_indices = np.asarray(sample_labels < 0).nonzero()
        sample_points = np.delete(sample_points, invalide_indices, axis=0)
        sample_labels = np.delete(sample_labels, invalide_indices, axis=0)

        # Reject radar channels that are not meant to keep
        sample_points = sample_points[:, self.keep_channels]

        # Adjust output formate
        if not self.point_major:
            sample_points = sample_points.T

        return sample_points, sample_labels

    def get_config(self):
        """
        Returns the configuration of the nuScenes pre-processor.
        """
        config = {
            'remove_background': self.remove_background,
            'invalid_states': self.invalid_states,
            'dynprop_states': self.dynprop_states,
            'ambig_states': self.ambig_states,
            'annotation_attributes': self.annotation_attributes,
            'locations': self.locations,
            'sensor_names': self.sensor_names,
            'keep_channels': self.keep_channels,
            'wlh_factor': self.wlh_factor,
            'wlh_offset': self.wlh_offset,
            'use_multisweep': self.use_multisweep,
            'nsweeps': self.nsweeps,
            'channel_boundaries': self.channel_boundaries,
            'channel_target_values': self.channel_target_values,
            'attribut_matching_dict': self.attribut_matching_dict
        }
        base_config = super(NuScenesPreProcessor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the pre-processed sample.

        Arguments:
            input_shape: Shape tuple or shape list, <tuple or list>.

        Returns:
            output_shape: A shape tuple, <tuple>.
        """
        if input_shape is not None:
            output_shape = ((len(self.keep_channels),) + (tuple(input_shape)), (None,))
        else:
            output_shape = ((len(self.keep_channels), None), (None,))

        return output_shape


def write_dataframe(dataframe, datatype: str, filename: str):
    """
    Writes a given dataframe to a defined location.
    Arguments:
        dataframe: Dataframe (data) to write.
        datatype: Format of the written data, <str>.
        filename: Absolute path of the written data file, <str>.
    """
    # Create output directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Select writer based on the given data type
    if datatype == 'tfrecord':
        # Check data type and file format
        assert isinstance(dataframe, bytes)
        assert filename.endswith('.tfrecord')

        # Write dataframe
        write_serialized_example(filename, dataframe)
    else:
        raise ValueError('Datatype is not valid: datatype has to be tfrecord')


def main(data_path: str, out_path: str, config: dict):
    """
    Main function of the pre-processor.
    Arguments:
        data_path: Directory of the raw dataset, <str>.
        out_path: Directory to write the pre-processed dataset, <str>.
        config: Configuration file, <ConfigParser>.
    """
    # Checks
    assert os.path.isdir(data_path)
    assert isinstance(config, dict)

    # Get dataset name
    dataset = config['PREPROCESS']['dataset']

    # Select pre-processor
    if dataset.lower() == 'nuscenes':
        # Parse config
        version = config['PREPROCESS']['PREPROCESSOR']['version']
        remove_background = config['PREPROCESS']['PREPROCESSOR']['remove_background']
        invalid_states = config['PREPROCESS']['PREPROCESSOR']['invalid_states']
        dynprop_states = config['PREPROCESS']['PREPROCESSOR']['dynprop_states']
        ambig_states = config['PREPROCESS']['PREPROCESSOR']['ambig_states']
        annotation_attributes = config['PREPROCESS']['PREPROCESSOR']['annotation_attributes']
        locations = config['PREPROCESS']['PREPROCESSOR']['locations']
        sensor_names = config['PREPROCESS']['PREPROCESSOR']['sensor_names']
        keep_channels = config['PREPROCESS']['PREPROCESSOR']['keep_channels']
        wlh_factor = config['PREPROCESS']['PREPROCESSOR']['wlh_factor']
        wlh_offset = config['PREPROCESS']['PREPROCESSOR']['wlh_offset']
        use_multisweep = config['PREPROCESS']['PREPROCESSOR']['use_multisweep']
        nsweeps = config['PREPROCESS']['PREPROCESSOR']['nsweeps']
        channel_boundaries = config['PREPROCESS']['PREPROCESSOR']['channel_boundaries']
        channel_target_values = config['PREPROCESS']['PREPROCESSOR']['channel_target_values']
        attribut_matching_dict = config['PREPROCESS']['PREPROCESSOR']['attribut_matching_dict']
        name = config['PREPROCESS']['PREPROCESSOR']['name']
        class_matching_dict = config['PREPROCESS']['CLASSMATCHING']

        # Load nuScenes dataset
        print("Loading NuScenes dataset...")
        nusc = NuScenes(version=version, dataroot=data_path, verbose=False)

        # Create nuScenes pre-processor instance
        nuscenes_preprocessor = NuScenesPreProcessor(class_matching_dict=class_matching_dict, point_major=False, remove_background=remove_background,
                                                     invalid_states=invalid_states, dynprop_states=dynprop_states, ambig_states=ambig_states,
                                                     annotation_attributes=annotation_attributes, locations=locations, sensor_names=sensor_names,
                                                     keep_channels=keep_channels, wlh_factor=wlh_factor, wlh_offset=wlh_offset,
                                                     use_multisweep=use_multisweep, nsweeps=nsweeps,
                                                     channel_boundaries=channel_boundaries, channel_target_values=channel_target_values,
                                                     attribut_matching_dict=attribut_matching_dict, name=name)

        # Create nuScenes serializer instance
        feature_names = keep_channels + ['label']  # Add label feature name
        feature_types = [nuscenes_utils.get_channel_types(channel) for channel in keep_channels]
        feature_types.append('int')  # Add feature type of the labels
        nuscenes_serializer = SequenceExampleSerializer(feature_names=feature_names, feature_types=feature_types)

        # Execute nuScenes pre-processing loop
        # TODO: execute in parallel.
        print("Entering pre-processing loop...")
        scene_splits = nuscenes_utils.get_scenes(version=nusc.version)
        number_of_scenes = nuscenes_utils.get_number_of_scenes(version=nusc.version)
        i = 0
        with progressbar.ProgressBar(max_value=number_of_scenes) as bar:
            for split in scene_splits.keys():
                for scene_token in scene_splits[split]:
                    # Update progressbar
                    bar.update(i)

                    # Get scene
                    scene = nusc.get('scene', scene_token)
                    # Get sample
                    sample = nusc.get('sample', scene['first_sample_token'])
                    # Skip first sample since it has less radar sweeps available
                    sample = nusc.get('sample', sample['next'])

                    # Reset point and label list of the scene
                    scene_points = [[] for _ in keep_channels]
                    scene_lables = [[]]

                    while True:
                        # Get points and labels (Extract, Determine, Filter and Manipulate)
                        sample_points, sample_labels = nuscenes_preprocessor(nusc, sample)
                        [scene_points[dim].append(points) for (dim, points) in enumerate(sample_points)]
                        scene_lables[0].append(sample_labels)

                        # Check if sample is the last sample in the current scene
                        if sample['next']:
                            sample = nusc.get('sample', sample['next'])
                        else:
                            break

                    # Transform dataframe
                    feature_values = scene_points + scene_lables
                    serialized_example = nuscenes_serializer(feature_values)

                    # Write dataframe
                    filename = out_path + '/' + split + '/' + scene_token + '.tfrecord'
                    write_dataframe(serialized_example, datatype='tfrecord', filename=filename)

                    i += 1
    else:
        raise ValueError('Dataset is not valid')

    # Save configuration
    ConfigFileHandler().write(config, out_path + '/config.ini')


if __name__ == "__main__":
    # Imports
    import argparse

    # Parse input arguments
    FILE_DIRECTORY = os.getcwd() + "/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY, "examples/02_KPConvCLSTM.ini"))
    parser.add_argument('--data', type=str, default="/data/nuscenes")
    parser.add_argument('--out', type=str, default="/preprocessed/dataset")
    args = parser.parse_args()

    # Get config from config file
    CONFIGSPEC = os.path.join(FILE_DIRECTORY, 'radarseg/config/validate_utils/configspecs.ini')
    config = ConfigFileHandler(configspec=CONFIGSPEC, allow_extra_values=-1)(args.config)

    # Call main function
    main(args.data, args.out, config)
