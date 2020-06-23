import unittest
import numpy as np
import h5py
import tempfile
from umami.preprocessing_tools import check_size, get_size, create_datasets


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
        self.test_h5 = h5py.File(self.test_h5_file_name, 'w')

    def test_equal_size_case(self):
        self.test_h5.create_dataset('test_1', data=np.ones(10))
        self.test_h5.create_dataset('test_2', data=np.ones(10))
        self.assertEqual(check_size(self.test_h5), 10)

    def test_different_size_case(self):
        self.test_h5.create_dataset('test_1', data=np.ones(10))
        self.test_h5.create_dataset('test_2', data=np.ones(15))
        with self.assertRaises(ValueError):
            check_size(self.test_h5)


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
        self.test_h5_file_list = [self.test_h5_file_name_1,
                                  self.test_h5_file_name_2,
                                  self.test_h5_file_name_3]
        self.test_h5_1 = h5py.File(self.test_h5_file_name_1, 'w')
        self.test_h5_2 = h5py.File(self.test_h5_file_name_2, 'w')
        self.test_h5_3 = h5py.File(self.test_h5_file_name_3, 'w')

    def test_total_size(self):
        self.test_h5_1.create_dataset('test_1', data=np.ones(10))
        self.test_h5_2.create_dataset('test_1', data=np.ones(10))
        self.test_h5_3.create_dataset('test_1', data=np.ones(10))
        total_size, ranges = get_size(self.test_h5_file_list)
        self.assertEqual(total_size, 30)
