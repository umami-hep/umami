import tempfile
import unittest

import h5py
import numpy as np

from umami.configuration import logger, set_log_level
from umami.preprocessing_tools import (
    add_data,
    check_keys,
    check_shapes,
    check_size,
    create_datasets,
    get_size,
)

set_log_level(logger, "DEBUG")


class Check_SizeTestCase(unittest.TestCase):
    """
    Test the check_size function.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_h5_file_name = f"{self.test_dir.name}/test_h5.h5"
        self.test_h5 = h5py.File(self.test_h5_file_name, "w")

    def test_equal_size_case(self):
        self.test_h5.create_dataset("test_1", data=np.ones(10))
        self.test_h5.create_dataset("test_2", data=np.ones(10))
        self.assertEqual(check_size(self.test_h5), 10)

    def test_different_size_case(self):
        self.test_h5.create_dataset("test_1", data=np.ones(10))
        self.test_h5.create_dataset("test_2", data=np.ones(15))
        with self.assertRaises(ValueError):
            check_size(self.test_h5)


class Check_KeysTestCase(unittest.TestCase):
    """
    Test the check_keys function.
    """

    def setUp(self):
        """
        Create default datasets for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_h5_file_name_1 = f"{self.test_dir.name}/test_h5_1.h5"
        self.test_h5_file_name_2 = f"{self.test_dir.name}/test_h5_2.h5"
        self.test_h5_file_name_3 = f"{self.test_dir.name}/test_h5_3.h5"

        self.test_h5_1 = h5py.File(self.test_h5_file_name_1, "w")
        self.test_h5_2 = h5py.File(self.test_h5_file_name_2, "w")
        self.test_h5_3 = h5py.File(self.test_h5_file_name_3, "w")

        self.test_h5_1.create_dataset("test", data=np.ones(10))
        self.test_h5_2.create_dataset("test", data=np.ones(10))
        self.test_h5_3.create_dataset("something_different", data=np.ones(10))

    def test_equal_keys_case(self):
        self.assertTrue(check_keys(self.test_h5_1, self.test_h5_2))

    def test_different_keys_case(self):
        with self.assertRaises(ValueError):
            check_keys(self.test_h5_1, self.test_h5_3)


class Check_ShapesTestCase(unittest.TestCase):
    """
    Test the check_shapes function.
    """

    def setUp(self):
        """
        Create default datasets for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_h5_file_name_1 = f"{self.test_dir.name}/test_h5_1.h5"
        self.test_h5_file_name_2 = f"{self.test_dir.name}/test_h5_2.h5"
        self.test_h5_file_name_3 = f"{self.test_dir.name}/test_h5_3.h5"

        self.test_h5_1 = h5py.File(self.test_h5_file_name_1, "w")
        self.test_h5_2 = h5py.File(self.test_h5_file_name_2, "w")
        self.test_h5_3 = h5py.File(self.test_h5_file_name_3, "w")

        self.test_h5_1.create_dataset("test", data=np.ones(10))
        self.test_h5_2.create_dataset("test", data=np.ones(10))
        self.test_h5_3.create_dataset("test", data=np.array([np.ones(10), np.ones(10)]))

    def test_equal_keys_case(self):
        self.assertTrue(check_shapes(self.test_h5_1, self.test_h5_2))

    def test_different_keys_case(self):
        with self.assertRaises(ValueError):
            check_shapes(self.test_h5_1, self.test_h5_3)


class Get_SizeTestCase(unittest.TestCase):
    """
    Test the get_size function.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_h5_file_name_1 = f"{self.test_dir.name}/test_h5_1.h5"
        self.test_h5_file_name_2 = f"{self.test_dir.name}/test_h5_2.h5"
        self.test_h5_file_name_3 = f"{self.test_dir.name}/test_h5_3.h5"
        self.test_h5_file_list = [
            self.test_h5_file_name_1,
            self.test_h5_file_name_2,
            self.test_h5_file_name_3,
        ]
        self.test_h5_1 = h5py.File(self.test_h5_file_name_1, "w")
        self.test_h5_2 = h5py.File(self.test_h5_file_name_2, "w")
        self.test_h5_3 = h5py.File(self.test_h5_file_name_3, "w")

    def test_total_size(self):
        self.test_h5_1.create_dataset("test_1", data=np.ones(10))
        self.test_h5_2.create_dataset("test_1", data=np.ones(10))
        self.test_h5_3.create_dataset("test_1", data=np.ones(10))
        total_size, ranges = get_size(self.test_h5_file_list)
        self.assertEqual(total_size, 30)


class Create_DatasetsTestCase(unittest.TestCase):
    """
    Test the create_datasets function.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_reference_h5_file_name = f"{self.test_dir.name}/test_h5_reference.h5"
        self.test_reference_h5 = h5py.File(self.test_reference_h5_file_name, "w")
        self.test_reference_h5.create_dataset("test", data=np.ones(10))

    def test_create_from_dict(self):
        output = h5py.File(f"{self.test_dir.name}/test_h5.h5", "w")
        source = {"test": np.ones(10)}
        size = 10
        create_datasets(output, source, size)
        output.close()

    def test_create_from_dataset(self):
        output = h5py.File(f"{self.test_dir.name}/test_h5.h5", "w")
        source = self.test_reference_h5
        size = check_size(source)
        create_datasets(output, source, size)
        output.close()


class Add_DataTestCase(unittest.TestCase):
    """
    Test the add_data function.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_h5_file_name = f"{self.test_dir.name}/test_h5.h5"
        self.test_h5 = h5py.File(self.test_h5_file_name, "w")
        self.test_h5.create_dataset("test", data=np.ones(10))

    def test_add_data(self):
        output = h5py.File(f"{self.test_dir.name}/test_h5_new.h5", "w")
        input_files = [self.test_h5_file_name]
        size, ranges = get_size(input_files)
        create_datasets(output, input_files[0], size)
        add_data(input_files[0], output, ranges[input_files[0]])
        output.close()
