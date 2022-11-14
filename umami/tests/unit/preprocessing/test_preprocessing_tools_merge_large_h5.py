"""Unit tests for merge_large_h5 in preprocessing tools."""
# pylint: disable=consider-using-with
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


class CheckSizeTestCase(unittest.TestCase):
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
        """Test equal size."""
        self.test_h5.create_dataset("test_1", data=np.ones(10))
        self.test_h5.create_dataset("test_2", data=np.ones(10))
        self.assertEqual(check_size(self.test_h5), 10)

    def test_different_size_case(self):
        """Test different dataset sizes."""
        self.test_h5.create_dataset("test_1", data=np.ones(10))
        self.test_h5.create_dataset("test_2", data=np.ones(15))
        with self.assertRaises(ValueError):
            check_size(self.test_h5)


class CheckKeysTestCase(unittest.TestCase):
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
        """Test scenario if equal keys."""
        self.assertTrue(check_keys(self.test_h5_1, self.test_h5_2))

    def test_different_keys_case(self):
        """Test scenario with different keys."""
        with self.assertRaises(ValueError):
            check_keys(self.test_h5_1, self.test_h5_3)


class CheckShapesTestCase(unittest.TestCase):
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
        """Test equal key case."""
        self.assertTrue(check_shapes(self.test_h5_1, self.test_h5_2))

    def test_different_keys_case(self):
        """Test scenario with different keys."""
        with self.assertRaises(ValueError):
            check_shapes(self.test_h5_1, self.test_h5_3)


class GetSizeTestCase(unittest.TestCase):
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
        """Test total size."""
        self.test_h5_1.create_dataset("test_1", data=np.ones(10))
        self.test_h5_2.create_dataset("test_1", data=np.ones(10))
        self.test_h5_3.create_dataset("test_1", data=np.ones(10))
        total_size, _ = get_size(self.test_h5_file_list)
        self.assertEqual(total_size, 30)


class CreateDatasetsTestCase(unittest.TestCase):
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
        """Test creation from dictionary."""
        with h5py.File(f"{self.test_dir.name}/test_h5.h5", "w") as output:
            source = {"test": np.ones(10)}
            size = 10
            create_datasets(output, source, size)

    def test_create_from_dataset(self):
        """Test creation from dataset."""
        with h5py.File(f"{self.test_dir.name}/test_h5.h5", "w") as output:
            source = self.test_reference_h5
            size = check_size(source)
            create_datasets(output, source, size)


class AddDataTestCase(unittest.TestCase):
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
        """Test adding data."""
        with h5py.File(f"{self.test_dir.name}/test_h5_new.h5", "w") as output:
            input_files = [self.test_h5_file_name]
            size, ranges = get_size(input_files)
            create_datasets(output, input_files[0], size)
            add_data(input_files[0], output, ranges[input_files[0]])
