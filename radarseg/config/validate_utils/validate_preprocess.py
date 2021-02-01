"""
Configuration specifications of the pre-processors handled by the ConfigFileHandler.

This validation file containes one class for each preprocessor that is
handled (checkt) by the config file handler. Therefore, each class
contains the following functions: a static method that specifies the
attributes of the pre-processor and the dataset, a static method that
specifies the content of the class matching (dictionary) as well as
several static methods which specify the content (the elements) of
complex pre-processor input values, like lists or tuples.

The config specifications of the pre-processor attributes and the class
matching dict are defined as a dict, whereas the dict keys correspond
with the attribute or original class. The value (of the key value pair)
defines the datatype, boundary values and default value of the attribute
(key).

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


class validate_nuscenes():
    """
    Configuration specifications of the nuScenes Preprocessor.
    """
    @staticmethod
    def validate_nuscenes():
        configspecs = {
            'version': "option('v1.0-mini', 'v1.0-trainval', 'v1.0-test', default='v1.0-mini')",
            'point_major': "boolean(default=False)",
            'remove_background': "boolean(default=False)",
            'invalid_states': "int_list_or_none(default=None)",
            'dynprop_states': "int_list_or_none(default=None)",
            'ambig_states': "int_list_or_none(default=None)",
            'annotation_attributes': "string_list(default=list('vehicle.moving', 'vehicle.stopped', 'vehicle.parked', 'cycle.with_rider', 'cycle.without_rider', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'pedestrian.moving'))",
            'locations': "string_list(default=list('boston-seaport', 'singapore-queenstown', 'singapore-onenorth', 'singapore-hollandvillage'))",
            'sensor_names': "string_list(default=list('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'))",
            'keep_channels': "string_list(default=list('x', 'y', 'z', 'dyn_prob', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp', 'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 'invalide_state', 'pdh0', 'vx_rms', 'vy_rms'))",
            'wlh_factor': "float(min=0.0, default=1.0)",
            'wlh_offset': "float(min=0.0, default=0.0)",
            'use_multisweep': "boolean(default=False)",
            'nsweeps': "integer(min=0, default=0)",
            'channel_boundaries': "dict(default=dict())",
            'channel_target_values': "dict(default=dict())",
            'attribut_matching_dict': "dict(default=dict())",
            'name': "string(default=None)"
        }
        return configspecs

    @staticmethod
    def validate_invalid_states():
        return "integer(min=0, max=17)"

    @staticmethod
    def validate_dynprop_states():
        return "integer(min=0, max=7)"

    @staticmethod
    def validate_ambig_states():
        return "integer(min=0, max=4)"

    @staticmethod
    def validate_annotation_attributes():
        return "option('vehicle.moving', 'vehicle.stopped', 'vehicle.parked', 'cycle.with_rider', 'cycle.without_rider', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'pedestrian.moving')"

    @staticmethod
    def validate_locations():
        return "option('boston-seaport', 'singapore-queenstown', 'singapore-onenorth', 'singapore-hollandvillage')"

    @staticmethod
    def validate_sensor_names():
        return "option('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP')"

    @staticmethod
    def validate_keep_channels():
        return "string()"

    @staticmethod
    def validate_channel_boundaries():
        configspecs = {
            '__many__': "list()"
        }
        return configspecs

    @staticmethod
    def validate_channel_target_values():
        configspecs = {
            '__many__': "float()"
        }
        return configspecs

    @staticmethod
    def validate_attribut_matching_dict():
        configspecs = {
            '__many__': "string()"
        }
        return configspecs

    @staticmethod
    def validate_classmatching():
        configspecs = {
            'human.pedestrian.adult': "string(min=1, default='human.pedestrian.adult')",
            'human.pedestrian.child': "string(min=1, default='human.pedestrian.child')",
            'human.pedestrian.wheelchair': "string(min=1, default='human.pedestrian.wheelchair')",
            'human.pedestrian.stroller': "string(min=1, default='human.pedestrian.stroller')",
            'human.pedestrian.personal_mobility': "string(min=1, default='human.pedestrian.personal_mobility')",
            'human.pedestrian.police_officer': "string(min=1, default='human.pedestrian.police_officer')",
            'human.pedestrian.construction_worker': "string(min=1, default='human.pedestrian.construction_worker')",
            'animal': "string(min=1, default='animal')",
            'vehicle.car': "string(min=1, default='vehicle.car')",
            'vehicle.motorcycle': "string(min=1, default='vehicle.motorcycle')",
            'vehicle.bicycle': "string(min=1, default='vehicle.bicycle')",
            'vehicle.bus.bendy': "string(min=1, default='vehicle.bus.bendy')",
            'vehicle.bus.rigid': "string(min=1, default='vehicle.bus.rigid')",
            'vehicle.truck': "string(min=1, default='vehicle.truck')",
            'vehicle.construction': "string(min=1, default='vehicle.construction')",
            'vehicle.emergency.ambulance': "string(min=1, default='vehicle.emergency.ambulance')",
            'vehicle.emergency.police': "string(min=1, default='vehicle.emergency.police')",
            'vehicle.trailer': "string(min=1, default='vehicle.trailer')",
            'movable_object.barrier': "string(min=1, default='movable_object.barrier')",
            'movable_object.trafficcone': "string(min=1, default='movable_object.trafficcone')",
            'movable_object.pushable_pullable': "string(min=1, default='movable_object.pushable_pullable')",
            'movable_object.debris': "string(min=1, default='movable_object.debris')",
            'static_object.bicycle_rack': "string(min=1, default='static_object.bicycle_rack')"
        }
        return configspecs
