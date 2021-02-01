# Standard Libraries
import unittest

# 3rd Party Libraries

# Local imports
from radarseg.preprocess import NuScenesPreProcessor


class TestNuScenesPreProcessor(unittest.TestCase):
    """
    Unit test of the NuScenesPreProcessor class and it's functions.
    """
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # Create new NuScenesPreProcessor instance before every test
        self.nusc_pre = NuScenesPreProcessor(class_matching_dict={})

    def tearDown(self):
        # Delete NuScenesPreProcessor instance after every test
        del self.nusc_pre

    def test_instance_creation(self):
        """
        Tests the creation and deletion of a NuScenesPreProcessor instance to ensure
        that no reference is remaining after deletion.
        """
        # Change any attribute of the NuScenesPreProcessor instance
        self.nusc_pre.remove_background = True
        # Delete existing instance
        del self.nusc_pre
        # Create new instance with default attribute values
        self.nusc_pre = NuScenesPreProcessor(class_matching_dict={})
        # Test if attribute values are as expacted (default value)
        self.assertNotEqual(self.nusc_pre.remove_background, True)

    def test_class_matching_dict_property(self):
        """
        Tests the functionality of the class_matching_dict property.
        """
        # Test empty dict assignment
        self.nusc_pre.class_matching_dict = {}
        self.assertFalse(self.nusc_pre.class_matching_dict)

        # Test set property / internal (private) state
        self.nusc_pre.class_matching_dict = {'human.pedestrian.adult': 1}
        self.assertDictEqual(self.nusc_pre._class_matching_dict, {'1fa93b757fc74fb197cdd60001ad8abf': 1})

        # Test get property / (public) state
        self.assertDictEqual(self.nusc_pre.class_matching_dict, {'human.pedestrian.adult': 1})

        # Test invalid property value (no valid nuScenes category name)
        with self.assertRaises(KeyError):
            self.nusc_pre.class_matching_dict = {'not_a_nuScenes_category_name': 1}

    def test_annotation_attributes_property(self):
        """
        Tests the functionality of the annotation_attributes property.
        """
        # Test empty list assignment
        self.nusc_pre.annotation_attributes = []
        self.assertFalse(self.nusc_pre.annotation_attributes)

        # Test set property / internal (private) state
        self.nusc_pre.annotation_attributes = ['vehicle.moving']
        self.assertListEqual(self.nusc_pre._annotation_attributes, ['cb5118da1ab342aa947717dc53544259'])

        # Test get property / (public) state
        self.assertListEqual(self.nusc_pre.annotation_attributes, ['vehicle.moving'])

        # Test invalid property value (no valid nuScenes annotation attribute name)
        with self.assertRaises(ValueError):
            self.nusc_pre.annotation_attributes = ['not_a_nuScenes_annotation_attribute_name']

    def test_wlh_factor_property(self):
        """
        Tests the functionality of the wlh_factor property.
        """
        # Test zero assignment
        self.nusc_pre.wlh_factor = 0
        self.assertFalse(self.nusc_pre.wlh_factor)

        # Test set property / internal (private) state
        self.nusc_pre.wlh_factor = 1.1
        self.assertEqual(self.nusc_pre._wlh_factor, 1.1)

        # Test get property / (public) state
        self.assertEqual(self.nusc_pre.wlh_factor, 1.1)

        # Test invalid property type (no "castable" float value)
        with self.assertRaises(ValueError):
            self.nusc_pre.wlh_factor = "no_float_value"

        # Test invalid property type str ("castable" float value)
        self.nusc_pre.wlh_factor = str("1.1")
        self.assertEqual(self.nusc_pre.wlh_factor, 1.1)

        # Test invalid property type int ("castable" float value)
        self.nusc_pre.wlh_factor = int(1)
        self.assertEqual(self.nusc_pre.wlh_factor, 1.0)

        # Test invalid property value (negative value)
        with self.assertRaises(ValueError):
            self.nusc_pre.wlh_factor = -1.0

    def test_wlh_offset_property(self):
        """
        Tests the functionality of the wlh_offset property.
        """
        # Test zero assignment
        self.nusc_pre.wlh_offset = 0
        self.assertFalse(self.nusc_pre.wlh_offset)

        # Test set property / internal (private) state
        self.nusc_pre.wlh_offset = 1.1
        self.assertEqual(self.nusc_pre._wlh_offset, 1.1)

        # Test get property / (public) state
        self.assertEqual(self.nusc_pre.wlh_offset, 1.1)

        # Test invalid property type (no "castable" float value)
        with self.assertRaises(ValueError):
            self.nusc_pre.wlh_offset = "no_float_value"

        # Test invalid property type str ("castable" float value)
        self.nusc_pre.wlh_offset = str("1.1")
        self.assertEqual(self.nusc_pre.wlh_offset, 1.1)

        # Test invalid property type int ("castable" float value)
        self.nusc_pre.wlh_offset = int(1)
        self.assertEqual(self.nusc_pre.wlh_offset, 1.0)

        # Test invalid property value (negative value)
        with self.assertRaises(ValueError):
            self.nusc_pre.wlh_offset = -1.0

    def test_nsweeps_property(self):
        """
        Tests the functionality of the nsweeps property.
        """
        # Test zero assignment
        self.nusc_pre.nsweeps = 0
        self.assertFalse(self.nusc_pre.nsweeps)

        # Test set property / internal (private) state
        self.nusc_pre.nsweeps = 1
        self.assertEqual(self.nusc_pre._nsweeps, 1)

        # Test get property / (public) state
        self.assertEqual(self.nusc_pre.nsweeps, 1)

        # Test invalid property type (no int value)
        with self.assertRaises(TypeError):
            self.nusc_pre.nsweeps = 1.1

        # Test invalid property value (negative value)
        with self.assertRaises(ValueError):
            self.nusc_pre.nsweeps = -1

    def test_unify_class_matching_dict(self):
        """
        Tests the correct behavior of the unify_class_matching_dict function.
        """
        # Test single, unique class assignment
        unified_class_matching_dict = self.nusc_pre.unify_class_matching_dict({'human.pedestrian.adult': 'Class_1'})
        self.assertDictEqual(unified_class_matching_dict, {'human.pedestrian.adult': 1})

        # Test multiple, equal class assignments
        unified_class_matching_dict = self.nusc_pre.unify_class_matching_dict({'human.pedestrian.adult': 'Class_1', 'human.pedestrian.child': 'Class_1'})
        self.assertDictEqual(unified_class_matching_dict, {'human.pedestrian.adult': 1, 'human.pedestrian.child': 1})

        # Test multiple, unequal class assignments
        unified_class_matching_dict = self.nusc_pre.unify_class_matching_dict({'human.pedestrian.adult': 'Class_1', 'human.pedestrian.child': 'Class_2'})
        self.assertDictEqual(unified_class_matching_dict, {'human.pedestrian.adult': 1, 'human.pedestrian.child': 2})

        # Test single, 'None' class assignment
        unified_class_matching_dict = self.nusc_pre.unify_class_matching_dict({'human.pedestrian.adult': 'None'})
        self.assertDictEqual(unified_class_matching_dict, {'human.pedestrian.adult': 0})

        # Test multiple class assignments ('None' class and not 'None' class)
        unified_class_matching_dict = self.nusc_pre.unify_class_matching_dict({'human.pedestrian.adult': 'None', 'human.pedestrian.child': 'Class_1'})
        self.assertDictEqual(unified_class_matching_dict, {'human.pedestrian.adult': 0, 'human.pedestrian.child': 1})

    def test_get_config(self):
        # Define reference configuration
        reference_config = {
            'name': '',
            'class_matching_dict': {},
            'remove_background': False,
            'invalid_states': [],
            'dynprop_states': [],
            'ambig_states': [],
            'annotation_attributes': [],
            'locations': [],
            'sensor_names': [],
            'keep_channels': (),
            'wlh_factor': 1.1,
            'wlh_offset': 0.0,
            'use_multisweep': False,
            'nsweeps': 1,
            'channel_boundaries': None,
            'channel_target_values': None,
            'attribut_matching_dict': None
        }

        # Create instance with reference configuration
        self.nusc_pre = NuScenesPreProcessor(**reference_config)

        # Test if config of the instance is equal to the reference config
        self.assertDictEqual(self.nusc_pre.get_config(), reference_config)


if __name__ == "__main__":
    unittest.main()
