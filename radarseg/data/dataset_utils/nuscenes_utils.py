# Standard Libraries
import os
from functools import reduce
from typing import Dict, List

# 3rd Party Libraries
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# Local imports
from radarseg.data.dataset_utils import nuscenes_splits


# Utility functions
def get_scenes(version: str = None, split: str = None) -> Dict[str, List[str]]:
    """
    Returns the scene tokens of a given nuScenes version and split.
    Arguments:
        version: Version of the nuScenes dataset, <str>.
        split: Name of the dataset split (either train or val), <str>.
    """
    if version == 'v1.0-mini':
        if split == 'train':
            scene_splits = {'train': nuscenes_splits.mini_train}
        elif split == 'val':
            scene_splits = {'val': nuscenes_splits.mini_val}
        else:
            scene_splits = {'train': nuscenes_splits.mini_train, 'val': nuscenes_splits.mini_val}
    elif version == 'v1.0-trainval':
        if split == 'train':
            scene_splits = {'train': nuscenes_splits.train}
        elif split == 'val':
            scene_splits = {'val': nuscenes_splits.val}
        else:
            scene_splits = {'train': nuscenes_splits.train, 'val': nuscenes_splits.val}
    elif version is None and split is None:
        scene_splits = {'train': nuscenes_splits.train, 'val': nuscenes_splits.val, 'test': nuscenes_splits.test,
                        'mini_train': nuscenes_splits.mini_train, 'mini_val': nuscenes_splits.mini_val,
                        'train_detect': nuscenes_splits.train_detect, 'train_track': nuscenes_splits.train_track}
    else:
        raise ValueError('Version has to be a valid nuScenes version and split has to be either train or val')

    return scene_splits


def get_number_of_scenes(version: str = None, split: str = None) -> int:
    """
    Returns the number of scenes of the given nuScenes version and split.
    Arguments:
        version: Version of the nuScenes dataset, <str>.
        split: Name of the dataset split (either train or val), <str>.
    """
    if version is None and split is None:
        # The number of all nuScenes (version v1.0) scenes is 1000 (see https://www.nuscenes.org/overview).
        return 1000
    else:
        scene_splits = get_scenes(version, split)
        return sum([len(split) for split in scene_splits.values()])


def get_number_of_radar_pc_dimensions() -> int:
    """
    Returns the number of dimensions of the nuScenes radar point cloud data.
    The number of dimensions of the radar point cloud is equal to the number of radar sensor channels.
    """
    return RadarPointCloud.nbr_dims()


def get_number_of_lidar_pc_dimensions() -> int:
    """
    Returns the number of dimensions of the nuScenes lidar point cloud data.
    The number of dimensions of the lidar point cloud is equal to the number of lidar sensor channels.
    """
    return LidarPointCloud.nbr_dims()


def get_max_samples_per_scene() -> int:
    """
    Returns the maximum number of keyframes/samples per scene of the nuScenes dataset.
    Since keyframes are provided at a sample frequence of 2Hz and a Scene is a 20s long sequence of consecutive frames,
    the maximum number of samples per scene is 41 samples (see https://www.nuscenes.org/data-format).
    """
    return 41


def get_radar_channel_names(channels: tuple) -> List[str]:
    """
    Returns the channel names of the nuScenes radar channels.
    Arguments:
        channels: Indices of the required radar channels, <tuple>.
    """
    channel_names = ['x', 'y', 'z', 'dyn_prob', 'id',
                     'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp',
                     'is_quality_valid', 'ambig_state', 'x_rms',
                     'y_rms', 'invalide_state', 'pdh0', 'vx_rms', 'vy_rms']

    return [channel_names[channel] for channel in channels]


def get_lidar_channel_names(channels: tuple) -> List[str]:
    """
    Returns the channel names of the nuScenes lidar channels.
    Arguments:
        channels: Indices of the required lidar channels, <tuple>.
    """
    channel_names = ['x', 'y', 'z', 'intensity']

    return [channel_names[channel] for channel in channels]


def get_channel_index(channel_name: str) -> int:
    """
    Returns the channel index of a nuScenes radar or lidar channel name.

    Hint: The x,y,z channels are included in both lidar and radar data.
          The 'intensity' channel is included only in lidar data. All other
          channels are only included in radar data.

    Arguments:
        channel_name: Channel name of the requested radar or lidar channel, <str>.
    """
    channel_name_index_mapping = {
        'x': 0, 'y': 1, 'z': 2,
        'dyn_prob': 3, 'id': 4, 'rcs': 5, 'vx': 6,
        'vy': 7, 'vx_comp': 8, 'vy_comp': 9, 'is_quality_valid': 10,
        'ambig_state': 11, 'x_rms': 12, 'y_rms': 13,
        'invalide_state': 14, 'pdh0': 15, 'vx_rms': 16, 'vy_rms': 17,
        'intensity': 3
    }
    return channel_name_index_mapping[channel_name]


def get_channel_types(channel_name: str) -> str:
    """
    Returns the data type of a given nuScenes radar channel name.
    Arguments:
        channel_name: Channel name of the requested radar channel type, <list>.
    """
    channel_name_type_mapping = {
        'x': 'float', 'y': 'float', 'z': 'float', 'dyn_prob': 'int', 'id': 'int',
        'rcs': 'float', 'vx': 'float', 'vy': 'float', 'vx_comp': 'float', 'vy_comp': 'float',
        'is_quality_valid': 'int', 'ambig_state': 'int', 'x_rms': 'int', 'y_rms': 'int',
        'invalide_state': 'int', 'pdh0': 'int', 'vx_rms': 'int', 'vy_rms': 'int',
        'intensity': 'int'
    }

    return channel_name_type_mapping[channel_name]


def get_attribute_token(attribute_name: str) -> str:
    """
    Returns the attribute token of a given attribute name.
    Arguments:
        attribute_name: Name of NuScenes annotation attribute, <str>.
    """
    attribute_token_dict = {
        'vehicle.moving': 'cb5118da1ab342aa947717dc53544259',
        'vehicle.stopped': 'c3246a1e22a14fcb878aa61e69ae3329',
        'vehicle.parked': '58aa28b1c2a54dc88e169808c07331e3',
        'cycle.with_rider': 'a14936d865eb4216b396adae8cb3939c',
        'cycle.without_rider': '5a655f9751944309a277276b8f473452',
        'pedestrian.sitting_lying_down': '03aa62109bf043afafdea7d875dd4f43',
        'pedestrian.standing': '4d8821270b4a47e3a8a300cbec48188e',
        'pedestrian.moving': 'ab83627ff28b465b85c427162dec722f'
    }

    return attribute_token_dict[attribute_name]


def get_attribute_name(attribute_token: str) -> str:
    """
    Returns the attribute name of a given attribute token.
    Arguments:
        attribute_token: Token of the NuScenes annotation attribute, <str>.
    """
    attribute_names_dict = {
        'cb5118da1ab342aa947717dc53544259': 'vehicle.moving',
        'c3246a1e22a14fcb878aa61e69ae3329': 'vehicle.stopped',
        '58aa28b1c2a54dc88e169808c07331e3': 'vehicle.parked',
        'a14936d865eb4216b396adae8cb3939c': 'cycle.with_rider',
        '5a655f9751944309a277276b8f473452': 'cycle.without_rider',
        '03aa62109bf043afafdea7d875dd4f43': 'pedestrian.sitting_lying_down',
        '4d8821270b4a47e3a8a300cbec48188e': 'pedestrian.standing',
        'ab83627ff28b465b85c427162dec722f': 'pedestrian.moving'
    }

    return attribute_names_dict[attribute_token]


def attributes_token_mapping(attribut_names: list):
    """
    Replaces all nuScenes attribute names in a list with their token.
    Arguments:
        attribut_names: List with NuScenes attribute names, <list>.
    """
    try:
        return [get_attribute_token(name) for name in attribut_names]
    except KeyError:
        raise ValueError("Attribute name is no valid nuScenes annotation attribute name!")


def attributes_name_mapping(attribute_token: list):
    """
    Replaces all nuScenes attribute token in a list with their names.
    Arguments:
        attribute_token: List with NuScenes attribute token, <list>.
    """
    try:
        return [get_attribute_name(token) for token in attribute_token]
    except KeyError:
        raise ValueError("Attribute name is no valid nuScenes annotation attribute name!")


def get_category_token(category_name: str) -> str:
    """
    Returns the category token of a given category name.
    Arguments:
        category_name: Name of NuScenes category, <str>.
    """
    category_token_dict = {
        'human.pedestrian.adult': '1fa93b757fc74fb197cdd60001ad8abf',
        'human.pedestrian.child': 'b1c6de4c57f14a5383d9f963fbdcb5cb',
        'human.pedestrian.wheelchair': 'b2d7c6c701254928a9e4d6aac9446d79',
        'human.pedestrian.stroller': '6a5888777ca14867a8aee3fe539b56c4',
        'human.pedestrian.personal_mobility': '403fede16c88426885dd73366f16c34a',
        'human.pedestrian.police_officer': 'bb867e2064014279863c71a29b1eb381',
        'human.pedestrian.construction_worker': '909f1237d34a49d6bdd27c2fe4581d79',
        'animal': '63a94dfa99bb47529567cd90d3b58384',
        'vehicle.car': 'fd69059b62a3469fbaef25340c0eab7f',
        'vehicle.motorcycle': 'dfd26f200ade4d24b540184e16050022',
        'vehicle.bicycle': 'fc95c87b806f48f8a1faea2dcc2222a4',
        'vehicle.bus.bendy': '003edbfb9ca849ee8a7496e9af3025d4',
        'vehicle.bus.rigid': 'fedb11688db84088883945752e480c2c',
        'vehicle.truck': '6021b5187b924d64be64a702e5570edf',
        'vehicle.construction': '5b3cd6f2bca64b83aa3d0008df87d0e4',
        'vehicle.emergency.ambulance': '732cce86872640628788ff1bb81006d4',
        'vehicle.emergency.police': '7b2ff083a64e4d53809ae5d9be563504',
        'vehicle.trailer': '90d0f6f8e7c749149b1b6c3a029841a8',
        'movable_object.barrier': '653f7efbb9514ce7b81d44070d6208c1',
        'movable_object.trafficcone': '85abebdccd4d46c7be428af5a6173947',
        'movable_object.pushable_pullable': 'd772e4bae20f493f98e15a76518b31d7',
        'movable_object.debris': '063c5e7f638343d3a7230bc3641caf97',
        'static_object.bicycle_rack': '0a30519ee16a4619b4f4acfe2d78fb55'
    }

    return category_token_dict[category_name]


def get_category_name(category_token: str) -> str:
    """
    Returns the category name of a given category token.
    Arguments:
        category_token: Token of the NuScenes category, <str>.
    """
    category_name_dict = {
        '1fa93b757fc74fb197cdd60001ad8abf': 'human.pedestrian.adult',
        'b1c6de4c57f14a5383d9f963fbdcb5cb': 'human.pedestrian.child',
        'b2d7c6c701254928a9e4d6aac9446d79': 'human.pedestrian.wheelchair',
        '6a5888777ca14867a8aee3fe539b56c4': 'human.pedestrian.stroller',
        '403fede16c88426885dd73366f16c34a': 'human.pedestrian.personal_mobility',
        'bb867e2064014279863c71a29b1eb381': 'human.pedestrian.police_officer',
        '909f1237d34a49d6bdd27c2fe4581d79': 'human.pedestrian.construction_worker',
        '63a94dfa99bb47529567cd90d3b58384': 'animal',
        'fd69059b62a3469fbaef25340c0eab7f': 'vehicle.car',
        'dfd26f200ade4d24b540184e16050022': 'vehicle.motorcycle',
        'fc95c87b806f48f8a1faea2dcc2222a4': 'vehicle.bicycle',
        '003edbfb9ca849ee8a7496e9af3025d4': 'vehicle.bus.bendy',
        'fedb11688db84088883945752e480c2c': 'vehicle.bus.rigid',
        '6021b5187b924d64be64a702e5570edf': 'vehicle.truck',
        '5b3cd6f2bca64b83aa3d0008df87d0e4': 'vehicle.construction',
        '732cce86872640628788ff1bb81006d4': 'vehicle.emergency.ambulance',
        '7b2ff083a64e4d53809ae5d9be563504': 'vehicle.emergency.police',
        '90d0f6f8e7c749149b1b6c3a029841a8': 'vehicle.trailer',
        '653f7efbb9514ce7b81d44070d6208c1': 'movable_object.barrier',
        '85abebdccd4d46c7be428af5a6173947': 'movable_object.trafficcone',
        'd772e4bae20f493f98e15a76518b31d7': 'movable_object.pushable_pullable',
        '063c5e7f638343d3a7230bc3641caf97': 'movable_object.debris',
        '0a30519ee16a4619b4f4acfe2d78fb55': 'static_object.bicycle_rack'
    }

    return category_name_dict[category_token]


def category_token_mapping(categories: dict):
    """
    Replaces all nuScenes category names as values of a dict with their token.
    Arguments:
        categories: Dictionary with NuScenes category names as keys, <dict>.
    """
    category_token_mapping = {get_category_token(category_name): value for (category_name, value) in categories.items()}

    return category_token_mapping


def category_name_mapping(categories: dict):
    """
    Replaces all nuScenes category token as values of a dict with their names.
    Arguments:
        categories: Dictionary with NuScenes category token as keys, <dict>.
    """
    category_name_mapping = {get_category_name(category_token): value for (category_token, value) in categories.items()}

    return category_name_mapping


def get_pc_from_file_multisweep(nusc,
                                sample_rec,
                                chan: str,
                                ref_chan: str,
                                nsweeps: int = 3,
                                min_distance: float = 1.0,
                                invalid_states=None,
                                dynprop_states=None,
                                ambig_states=None):
    """
    Returns a point cloud of multiple aggregated sweeps.
    Original function can be found at: https://github.com/nutonomy/nuscenes-devkit/blob/ae022aba3b37f07202ea402f8278979097738369/python-sdk/nuscenes/utils/data_classes.py#L56.
    Function is modified to accept additional input parametes, just like the get_pc_from_file function. This enables the filtering by invalid_states, dynprop_states and ambig_states.

    Arguments:
        nusc: A NuScenes instance, <NuScenes>.
        sample_rec: The current sample, <dict>.
        chan: The radar/lidar channel from which we track back n sweeps to aggregate the point cloud, <str>.
        ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to, <str>.
        nsweeps: Number of sweeps to aggregated, <int>.
        min_distance: Distance below which points are discarded, <float>.
        invalid_states: Radar states to be kept, <int List>.
        dynprop_states: Radar states to be kept, <int List>. Use [0, 2, 6] for moving objects only.
        ambig_states: Radar states to be kept, <int List>.
    """
    # Define
    if chan == 'LIDAR_TOP':
        points = np.zeros((get_number_of_lidar_pc_dimensions(), 0))
        all_pc = LidarPointCloud(points)
    else:
        points = np.zeros((get_number_of_radar_pc_dimensions(), 0))
        all_pc = RadarPointCloud(points)

    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud.
        if chan == 'LIDAR_TOP':
            current_pc = LidarPointCloud.from_file(file_name=os.path.join(nusc.dataroot, current_sd_rec['filename']))
        else:
            current_pc = RadarPointCloud.from_file(file_name=os.path.join(nusc.dataroot, current_sd_rec['filename']),
                                                   invalid_states=invalid_states, dynprop_states=dynprop_states,
                                                   ambig_states=ambig_states)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Remove close points and add timevector.
        current_pc.remove_close(min_distance)
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return all_pc, all_times


def get_radar_pc_from_file(nusc,
                           sample_rec,
                           chan: str,
                           invalid_states=None,
                           dynprop_states=None,
                           ambig_states=None):
    """
    Returns the radar point cloud of a single keyframe.
    Original function can be found at: https://github.com/nutonomy/nuscenes-devkit/blob/ae022aba3b37f07202ea402f8278979097738369/python-sdk/nuscenes/utils/data_classes.py#L296.

    Arguments:
        nusc: A NuScenes instance, <NuScenes>.
        sample_rec: The current sample, <dict>.
        chan: The radar channel from which the point cloud is extracted, <str>.
        invalid_states: Radar states to be kept, <int List>.
        dynprop_states: Radar states to be kept, <int List>. Use [0, 2, 6] for moving objects only.
        ambig_states: Radar states to be kept, <int List>.
    """
    # Get filename
    sample_data = nusc.get('sample_data', sample_rec['data'][chan])
    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    # Extract point cloud
    pc = RadarPointCloud.from_file(pcl_path, invalid_states=invalid_states, dynprop_states=dynprop_states, ambig_states=ambig_states)
    return pc


def get_lidar_pc_from_file(nusc,
                           sample_rec,
                           chan: str):
    """
    Returns the lidar point cloud of a single keyframe.

    Arguments:
        nusc: A NuScenes instance, <NuScenes>.
        sample_rec: The current sample, <dict>.
        chan: The lidar channel from which the point cloud is extracted, <str>.
    """
    # Get filename
    sample_data = nusc.get('sample_data', sample_rec['data'][chan])
    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])

    # Extract point cloud
    pc = LidarPointCloud.from_file(pcl_path)
    return pc


def translate_pc(points: np.ndarray, trans_matrix: np.ndarray):
    """
    Applies a translation to the point cloud.
    Arguments:
        points: d-dimensional input point cloud matrix, <np.float: d, n>.
        trans_matrix: Translation in x, y and z, <np.float: 3, 1>.
    """
    points[:3, :] = np.add(points[:3, :], np.expand_dims(trans_matrix, axis=-1))
    return points


def rotate_pc(points: np.ndarray, rot_matrix: np.ndarray):
    """
    Applies a rotation to the point cloud.
    Arguments:
        points: d-dimensional input point cloud matrix, <np.float: d, n>.
        rot_matrix: Rotation matrix, <np.float: 3, 3>.
    """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points


def rotate_velocities(points: np.ndarray, rot_matrix: np.ndarray):
    """
    Radar velocities are rotated to car frame. This is not automatically done in the point cloud loading.
    Arguments:
    points: d-dimensional input point cloud matrix, <np.float: d, n>.
    rot_matrix: Rotation matrix, <np.float: 2, 2>.
    """
    points[8:10, :] = np.dot(rot_matrix, points[8:10, :])
    return points


def transform_pc_from_sensor_to_vehicle_frame(calibrated_sensor: dict, sensor_name: str, pc: np.ndarray):
    """
    Transforms the point cloud from the sensor frame to the ego vehicle frame for the timestamp of the sweep.
    Arguments:
        calibrated_sensor: Calibration parameter of the sensor, <dict>.
        sensor_name: Name of the sensor which is loaded, <str>.
        pc: Point cloud, <np.float: 3, n>.
    """
    pc_rotation_matrix = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    pc = rotate_pc(pc, pc_rotation_matrix)
    pc = translate_pc(pc, np.array(calibrated_sensor['translation']))

    if "RADAR" in sensor_name:  # rotate radar velocities to correct frame
        velocity_rotation_matrix = pc_rotation_matrix[:2, :2]
        pc = rotate_velocities(pc, velocity_rotation_matrix)

    return pc


def transform_pc_from_vehicle_to_global_frame(ego_pose: dict, pc: np.ndarray):
    """
    Transforms the point cloud from the vehicle frame to the global frame.
    Arguments:
        ego_pose: Global position of the ego vehicle, <dict>.
        pc: Point cloud, <np.float: 3, n>.
    """
    pc = rotate_pc(pc, Quaternion(ego_pose['rotation']).rotation_matrix)
    pc = translate_pc(pc, np.array(ego_pose['translation']))
    return pc


def extended_points_in_box(box, points: np.ndarray, wlh_factor: float = 1.0, wlh_offset: float = 0.0, use_z: bool = True):
    """
    Returns a mask indicating whether or not each point is inside the bounding box, <np.bool: n, >.
    Inspired by NuScenes points_in_box, with an additional wlh_offset (which is often more meaningful than the wlh_factor)
    Arguments:
        box: bounding box, <Box>.
        points: point cloud, <np.float: 3, n>.
        wlh_factor: factor to inflate or deflate the box (1.1 makes it 10% larger in all dimensions), <float>.
        wlh_offset: offset to inflate or deflate the box (1.0 makes it 1 m larger in all dimensions, on both sides), <float>.
        use_z: if False, only x and y coordinates are taken into account, <bool>.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v) / np.linalg.norm(i)
    jv = np.dot(j, v) / np.linalg.norm(j)
    kv = np.dot(k, v) / np.linalg.norm(k)

    mask_x = np.logical_and(0 - wlh_offset <= iv, iv <= np.linalg.norm(i) + wlh_offset)
    mask_y = np.logical_and(0 - wlh_offset <= jv, jv <= np.linalg.norm(j) + wlh_offset)

    if use_z:
        mask_z = np.logical_and(0 - wlh_offset <= kv, kv <= np.linalg.norm(k) + wlh_offset)
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    else:
        mask = np.logical_and(mask_x, mask_y)

    return mask
