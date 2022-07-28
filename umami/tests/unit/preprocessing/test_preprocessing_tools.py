"""Unit tests for preprocessing_tools."""
import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config, logger, set_log_level
from umami.preprocessing_tools import (
    Configuration,
    PrepareSamples,
    binarise_jet_labels,
    get_variable_dict,
)

set_log_level(logger, "DEBUG")


class ConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
        )

    def test_missing_key_error(self):
        """Test missing key error."""
        config = Configuration(self.config_file)
        del config.config["outfile_name"]
        with self.assertRaises(KeyError):
            config.get_configuration()

    # this functionality is still there but is not used for now
    # so will keep the test here in case this is used again
    # def test_missing_key_warning(self):
    #     config = Configuration(self.config_file)
    #     del config.config["outfile_name"]
    #     with self.assertWarns(Warning):
    #         config.get_configuration()

    def test_get_file_name_no_input(self):
        """Test filename without input."""
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(out_file, config.outfile_name)

    def test_get_file_name_no_iterations(self):
        """Test no iterations"""
        config = Configuration(self.config_file)
        with self.subTest():
            self.assertNotIn("test", config.outfile_name)
        out_file = config.GetFileName(option="test")
        with self.subTest():
            self.assertIn("test", out_file)

    def test_get_file_name_no_iterations_no_input(self):
        """Test no iterations and no input."""
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(config.outfile_name, out_file)


class GetVariableDictTestCase(unittest.TestCase):
    """
    Test the implementation of GetVariableDict function.
    """

    def setUp(self):
        self.var_file = os.path.join(
            os.path.dirname(__file__), "fixtures/dummy_var_file_short.yaml"
        )
        self.label = "HadronConeExclTruthLabelID"
        self.train_variables = {
            "JetKinematics": ["absEta_btagJes", "pt_btagJes"],
            "JetFitter": ["JetFitter_isDefaults"],
        }
        self.tracks = {
            "noNormVars": ["IP3D_signed_d0_significance"],
            "logNormVars": ["ptfrac", "dr"],
            "jointNormVars": ["numberOfPixelHits", "numberOfSCTHits", "btagIp_d0"],
        }
        self.def_vars = 0
        self.test_dict = get_variable_dict(self.var_file)

    def test_nested_structure(self):
        """Test nested structure."""
        self.assertIn("tracks", self.test_dict["track_train_variables"])

    def test_label_reading(self):
        """Test label reading."""
        self.assertEqual(self.test_dict["label"], self.label)

    def test_train_variables(self):
        """Test train variables."""
        self.assertEqual(self.test_dict["train_variables"], self.train_variables)

    def test_trk_train_variables(self):
        """Test track training variables."""
        self.assertEqual(self.test_dict["track_train_variables"]["tracks"], self.tracks)

    def test_defaults_vars(self):
        """Test defaults variables."""
        self.assertEqual(
            self.test_dict["custom_defaults_vars"]["JetFitter_energyFraction"],
            self.def_vars,
        )


class GetBinaryLabelsTestCase(unittest.TestCase):
    """Test the implementation of the GetBinaryLabels function."""

    def setUp(self):
        """Create a default dataset for testing."""
        self.y = np.concatenate(
            [np.zeros(12), 4 * np.ones(35), 6 * np.ones(5), 15 * np.ones(35)]
        )
        np.random.seed(42)
        np.random.shuffle(self.y)
        self.df = pd.DataFrame({"label": self.y})

    def test_zero_length(self):
        """Test zero case."""
        df_0 = pd.DataFrame({"label": []})
        with self.assertRaises(ValueError):
            binarise_jet_labels(df_0)

    def test_shape(self):
        """Test shape."""
        y_categ = binarise_jet_labels(self.df, "label")
        self.assertEqual(y_categ.shape, (len(self.y), 4))


class PrepareSamplesTestCase(unittest.TestCase):
    """
    Test the implementation of the PrepareSamples class.
    """

    class c_args:
        """Helper class replacing command line arguments."""

        def __init__(self) -> None:
            """Initialise class with preset settings."""
            self.sample = "ttbar"
            self.config_file = os.path.join(
                os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
            )
            self.shuffle_array = True

    def setUp(self):
        self.args = self.c_args()
        self.config_file = os.path.join(
            os.path.dirname(__file__), self.args.config_file
        )
        self.config = Configuration(self.config_file)
        # create temporary h5 file
        jets = pd.DataFrame(
            {
                "eventNumber": np.array([0, 1, 2], dtype=np.uint8),
                "JetFitterSecondaryVertex_mass": np.array([0, 0, 0], dtype=np.uint8),
                "JetFitterSecondaryVertex_energy": np.array([0, 0, 0], dtype=np.uint8),
                "HadronConeExclTruthLabelID": np.array([5, 4, 0], dtype=np.uint8),
                "GhostBHadronsFinalPt": 5e5 * np.ones(3),
                global_config.pTvariable: 5e3 * np.ones(3),
            },
        )
        tracks = np.ones(shape=(3, 5, 40))
        self.tf = tempfile.NamedTemporaryFile()  # pylint: disable=R1732
        with h5py.File(self.tf, "w") as out_file:
            out_file.create_dataset("jets", data=jets.to_records())
            out_file.create_dataset("tracks", data=tracks)
            out_file.create_dataset("fs_tracks", data=tracks)
        self.output_file = tempfile.NamedTemporaryFile()  # pylint: disable=R1732

    def test_no_sample_provided(self):
        """Test no provided sample."""
        self.args.sample = None
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_wrong_sample_provided(self):
        """Test wrong sample provided."""
        self.args.sample = "None"
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_wrong_category_provided(self):
        """Test wrong category provided."""
        self.config.preparation["samples"]["ttbar"]["category"] = "None"
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_get_batches_per_file(self):
        """Test batches per file."""
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        for batch_tuple in list(files_in_batches):
            # first entry of tuples is the filename
            with self.subTest():
                self.assertEqual(type(batch_tuple[0]), str)
            # second entry of tuples is a list of tuples with the batch indices
            for batch in batch_tuple[1]:
                with self.subTest():
                    self.assertEqual(type(batch), tuple)

    def test_jets_generator_fullcuts_wotracks(self):
        """Test jet generator without tracks."""
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = None
        expected_jets = np.array([])
        for num, (jets, tracks) in enumerate(ps.jets_generator(files_in_batches)):
            with self.subTest("sub test track", i=num):
                self.assertEqual(tracks, None)
            with self.subTest("sub test jets", i=num):
                self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_fullcuts(self):
        """Test jet generator including tracks with full cuts."""
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = True
        expected_jets = np.array([])
        expected_tracks = np.array([])
        for jets, tracks in ps.jets_generator(files_in_batches):
            for tracks_name in ps.tracks_names:
                with self.subTest(f"sub test track {tracks_name}"):
                    self.assertEqual(len(tracks[tracks_name]), len(expected_tracks))
            with self.subTest():
                self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_lightcut(self):
        """Test jet generator including tracks with cut on light jets."""
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = True
        ps.cuts = [{"eventNumber": {"operator": "==", "condition": 0}}]
        expected_jets_len = expected_tracks_len = 1
        for jets, tracks in ps.jets_generator(files_in_batches):
            for tracks_name in ps.tracks_names:
                with self.subTest(f"sub test track {tracks_name}"):
                    self.assertEqual(len(tracks[tracks_name]), expected_tracks_len)
            with self.subTest():
                self.assertEqual(len(jets), expected_jets_len)

    def test_run(self):
        """Test the run function."""
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        ps.output_file = self.output_file.name
        ps.Run()
        assert os.path.exists(self.output_file.name) == 1
