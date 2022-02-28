import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from umami.configuration import logger, set_log_level
from umami.preprocessing_tools import Configuration
from umami.tf_tools import Convert_to_Record

set_log_level(logger, "DEBUG")


class ConvertTest(unittest.TestCase):
    """
    Unit test the functions inside the Convert_to_Record class.
    """

    def setUp(self):
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
        )
        self.faulty_config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_faulty_config.yaml"
        )
        self.config = Configuration(self.config_file)
        tracks_name = self.config.sampling["options"]["tracks_names"][0]
        self.faulty_config = Configuration(self.faulty_config_file)
        # create dummy data
        x_train = np.ones(shape=(3, 41))
        x_trks_train = np.ones(shape=(3, 40, 5))
        y_train = np.ones(shape=(3, 3))
        # save dummy data to temporary file
        self.tfh5 = tempfile.NamedTemporaryFile(suffix="-resampled_scaled_shuffled.h5")
        with h5py.File(self.tfh5, "w") as out_file:
            out_file.create_dataset("X_train", data=x_train)
            out_file.create_dataset(f"X_{tracks_name}_train", data=x_trks_train)
            out_file.create_dataset("Y_train", data=y_train)
        self.config.outfile_name = self.tfh5.name.replace(
            "-resampled_scaled_shuffled.h5", ".h5"
        )

    def test_save_parameters(self):
        cv = Convert_to_Record.h5toTFRecordConverter(self.config)
        # create temporary directory where data should be saved
        record_dir = tempfile.TemporaryDirectory()
        cv.save_parameters(record_dir.name)
        parameters = {
            "n_jets": 3,
            "n_jet_features": 41,
            "n_dim": 3,
            "n_trks": {"tracks": 40},
            "n_trk_features": {"tracks": 5},
        }
        metadata_file = os.path.join(record_dir.name, "metadata.json")
        with open(metadata_file, "r") as metadata:
            parameters_saved = json.load(metadata)
        self.assertEqual(parameters, parameters_saved)

    def test_save_parameters_nadd_vars(self):
        cv = Convert_to_Record.h5toTFRecordConverter(self.config)
        # create temporary directory where data should be saved
        record_dir = tempfile.TemporaryDirectory()
        cv.n_add_vars = 4
        cv.save_parameters(record_dir.name)
        parameters = {
            "n_jets": 3,
            "n_jet_features": 41,
            "n_dim": 3,
            "n_trks": {"tracks": 40},
            "n_trk_features": {"tracks": 5},
            "n_add_vars": 4,
        }
        metadata_file = os.path.join(record_dir.name, "metadata.json")
        with open(metadata_file, "r") as metadata:
            parameters_saved = json.load(metadata)
        self.assertEqual(parameters, parameters_saved)

    def test_faulty_setup(self):
        cv = Convert_to_Record.h5toTFRecordConverter(self.faulty_config)
        default_chunk_size = 5_000
        self.assertEqual(cv.chunk_size, default_chunk_size)
