import tempfile
import unittest

import h5py
import pandas as pd

from umami.data_tools import compare_h5_files_variables


class compare_h5_files_variablesTestCase(unittest.TestCase):
    """
    Test the implementation of the compare_h5_files_variables function.
    """

    def setUp(self) -> None:
        # creating dummy h5 files
        df_1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4], "d": [3, 4]})
        df_2 = pd.DataFrame({"a": [5, 6], "b": [7, 8], "e": [9, 10], "f": [11, 12]})
        df_3 = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4], "c": [3, 4], "d": [3, 4], "g": [13, 14]}
        )
        # creating temporary files
        self.f_h5_1 = tempfile.NamedTemporaryFile(suffix="dummy_file_1.h5")
        self.f_h5_2 = tempfile.NamedTemporaryFile(suffix="dummy_file_2.h5")
        self.f_h5_3 = tempfile.NamedTemporaryFile(suffix="dummy_file_3.h5")
        # TODO: using new with syntax of python 3.10
        with h5py.File(self.f_h5_1, "w") as out_file_1, h5py.File(
            self.f_h5_2, "w"
        ) as out_file_2, h5py.File(self.f_h5_3, "w") as out_file_3:
            out_file_1.create_dataset("jets", data=df_1.to_records(index=False))
            out_file_2.create_dataset("jets", data=df_2.to_records(index=False))
            out_file_3.create_dataset("jets", data=df_3.to_records(index=False))

    def test_no_inputs(self):
        with self.assertRaises(ValueError):
            compare_h5_files_variables(key="jets")

    def test_common_vars(self):
        results, _ = compare_h5_files_variables(
            self.f_h5_1.name,
            self.f_h5_2.name,
            key="jets",
        )
        self.assertEqual(set(["a", "b"]), set(results))

    def test_common_vars_3_files(self):
        results, _ = compare_h5_files_variables(
            self.f_h5_1.name,
            self.f_h5_2.name,
            self.f_h5_3.name,
            key="jets",
        )
        self.assertEqual(set(["a", "b"]), set(results))

    def test_different_vars(self):
        _, results = compare_h5_files_variables(
            self.f_h5_1.name,
            self.f_h5_2.name,
            key="jets",
        )
        self.assertEqual(set(["c", "d", "e", "f"]), set(results))

    def test_different_vars_3_files(self):
        _, results = compare_h5_files_variables(
            self.f_h5_1.name,
            self.f_h5_2.name,
            self.f_h5_3.name,
            key="jets",
        )
        self.assertEqual(set(["c", "d", "e", "f", "g"]), set(results))
