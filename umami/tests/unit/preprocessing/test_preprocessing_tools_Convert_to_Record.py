import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from umami.preprocessing_tools import Configuration, Convert_to_Record


class ConvertTest(unittest.TestCase):
    """
    Unit test the functions inside the Scaling class.
    """

    def setUp(self):
        self.config_file = os.path.join(
            os.path.dirname(__file__), "test_preprocess_config.yaml"
        )
        self.config = Configuration(self.config_file)
        # create dummy data
        x_train = np.ones(shape=(3, 41))
        x_trks_train = np.ones(shape=(3, 40, 5))
        y_train = np.ones(shape=(3, 3))
        # save dummy data to temporary file
        self.tfh5 = tempfile.NamedTemporaryFile(
            suffix="-resampled_scaled_shuffled.h5"
        )
        with h5py.File(self.tfh5, "w") as out_file:
            out_file.create_dataset("X_train", data=x_train)
            out_file.create_dataset("X_trk_train", data=x_trks_train)
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
            "nJets": 3,
            "njet_features": 41,
            "nTrks": 40,
            "nFeatures": 5,
            "nDim": 3,
        }
        metadata_file = os.path.join(record_dir.name, "metadata.json")
        with open(metadata_file, "r") as metadata:
            parameters_saved = json.load(metadata)
        self.assertEqual(parameters, parameters_saved)
