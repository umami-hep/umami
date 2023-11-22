"""Unit test script for the convert to tfrecord functions of tf_tools."""
import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from umami.configuration import logger, set_log_level
from umami.preprocessing_tools import PreprocessConfiguration
from umami.tf_tools import convert_to_record

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
        self.config = PreprocessConfiguration(self.config_file)
        tracks_name = self.config.sampling.options.tracks_names[0]
        self.faulty_config = PreprocessConfiguration(self.faulty_config_file)
        # create dummy data
        x_train = np.ones(shape=(3, 41))
        x_trks_train = np.ones(shape=(3, 40, 5))
        y_train = np.ones(shape=(3, 3))
        y_trks_train = np.ones(shape=(3, 40))
        # save dummy data to temporary file
        self.tfh5 = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
            suffix="-resampled_scaled_shuffled.h5"
        )
        with h5py.File(self.tfh5, "w") as out_file:
            out_file.create_dataset("jets/inputs", data=x_train)
            out_file.create_dataset(f"{tracks_name}/inputs", data=x_trks_train)
            out_file.create_dataset("jets/labels_one_hot", data=y_train)
            out_file.create_dataset(
                f"{tracks_name}/labels/ftagTruthOriginLabel", data=y_trks_train
            )
        self.config.general.outfile_name = self.tfh5.name.replace(
            "-resampled_scaled_shuffled.h5", ".h5"
        )

    def test_save_parameters(self):
        """Test the saving of the parameters for the tfrecord conversion."""
        conv = convert_to_record.H5ToTFRecords(self.config)
        # create temporary directory where data should be saved
        record_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        conv.save_parameters(record_dir.name)
        parameters = {
            "n_jets": 3,
            "n_jet_features": 41,
            "n_dim": 3,
            "n_trks": {"tracks": 40},
            "n_trk_features": {"tracks": 5},
            "n_add_vars": None,
            "n_trks_classes": {"tracks": 8},
            "n_trks_labels": {"tracks": 1},
        }
        metadata_file = os.path.join(record_dir.name, "metadata.json")
        with open(metadata_file, "r") as metadata:
            parameters_saved = json.load(metadata)
        self.assertEqual(parameters, parameters_saved)

    def test_save_parameters_nadd_vars(self):
        """Test the saving of the parameters for the tfrecord conversion
        with conditional info added."""
        conv = convert_to_record.H5ToTFRecords(self.config)
        # create temporary directory where data should be saved
        record_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        conv.n_add_vars = 4
        conv.save_parameters(record_dir.name)
        parameters = {
            "n_jets": 3,
            "n_jet_features": 41,
            "n_dim": 3,
            "n_trks": {"tracks": 40},
            "n_trk_features": {"tracks": 5},
            "n_add_vars": 4,
            "n_trks_classes": {"tracks": 8},
            "n_trks_labels": {"tracks": 1},
        }
        metadata_file = os.path.join(record_dir.name, "metadata.json")
        with open(metadata_file, "r") as metadata:
            parameters_saved = json.load(metadata)
        self.assertEqual(parameters, parameters_saved)

    def test_faulty_setup(self):
        """Test raising of an error for a faulty setup."""
        with self.assertRaises(ValueError):
            _ = convert_to_record.H5ToTFRecords(self.faulty_config)
