"""Unit tests for preprocessing_tools."""
import os
import tempfile
import unittest
from subprocess import CalledProcessError, run

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config, logger, set_log_level
from umami.preprocessing_tools import (
    PrepareSamples,
    PreprocessConfiguration,
    binarise_jet_labels,
    get_variable_dict,
)

set_log_level(logger, "DEBUG")


class CreateSamplesTestCase(unittest.TestCase):
    """Testing the creation of the samples for the preprocessing config."""

    def setUp(self) -> None:
        """Setting up needed paths."""
        self.test_dir_path = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = f"{self.test_dir_path.name}"
        self.output_path = os.path.join(self.test_dir, "test_samples.yaml")
        self.config_file = os.path.join(
            "examples/preprocessing",
            "PFlow-Preprocessing.yaml",
        )

    def test_create_samples_dict(self):
        """Test nominal behaviour.

        Raises
        ------
        AssertionError
            If the process returns an error.
        """
        run_script = run(
            [
                "python",
                "scripts/create_preprocessing_samples.py",
                "-c",
                f"{self.config_file}",
                "-o",
                f"{self.output_path}",
            ],
            check=True,
        )

        try:
            run_script.check_returncode()

        except CalledProcessError as error:
            raise AssertionError(
                "Test failed: run_script for create_samples_dict."
            ) from error


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


class BinariseJetLabelsTestCase(unittest.TestCase):
    """Test the implementation of the binarise_jet_labels function."""

    def setUp(self):
        """Create a default dataset for testing."""
        self.y = np.concatenate(
            [np.zeros(12), np.ones(35), 2 * np.ones(5), 3 * np.ones(35)]
        )
        np.random.seed(42)
        np.random.shuffle(self.y)
        self.df = pd.DataFrame({"label": self.y})
        self.two_classes_y = np.asarray([0, 1, 1, 0, 1])
        self.four_classes_y = np.asarray([0, 1, 2, 3, 2])

    def test_correct_output(self):
        """Testing correct behaviour."""
        y_correct = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )

        y_categ = binarise_jet_labels(
            labels=self.four_classes_y,
            internal_labels=list(range(4)),
        )
        np.testing.assert_array_equal(y_correct, y_categ)

    def test_2_classes_correct_output(self):
        """Testing correct behaviour."""
        y_correct = np.asarray(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
            ]
        )

        y_categ = binarise_jet_labels(
            labels=self.two_classes_y,
            internal_labels=list(range(2)),
        )
        np.testing.assert_array_equal(y_correct, y_categ)

    def test_zero_length(self):
        """Test zero case."""
        df_0 = pd.DataFrame({"label": []})
        with self.assertRaises(ValueError):
            binarise_jet_labels(
                labels=df_0,
                internal_labels=list(range(4)),
            )

    def test_wrong_type(self):
        """Test wrong input type case."""
        df_0 = "I am a wrong type"
        with self.assertRaises(TypeError):
            binarise_jet_labels(
                labels=df_0,
                internal_labels=list(range(4)),
            )

    def test_missing_internal_label(self):
        """Test wrong input type case."""
        with self.assertRaises(ValueError):
            binarise_jet_labels(
                labels=self.y,
                internal_labels=list(range(3)),
            )

    def test_shape_array(self):
        """Test shape for array input."""
        y_categ = binarise_jet_labels(
            labels=self.y,
            internal_labels=list(range(4)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(self.y), 4))

    def test_shape_DataFrame(self):
        """Test shape for DataFrame input."""
        y_categ = binarise_jet_labels(
            labels=self.df,
            internal_labels=list(range(4)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(self.y), 4))

    def test_shape_2_classes(self):
        """Test 2 classes"""
        y = np.concatenate([np.zeros(12), np.ones(35)])
        np.random.seed(42)
        np.random.shuffle(y)
        df_two_classes = pd.DataFrame({"label": y})

        y_categ = binarise_jet_labels(
            labels=df_two_classes,
            internal_labels=list(range(2)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(y), 2))


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
        self.config = PreprocessConfiguration(self.config_file)
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
        self.config.preparation.samples["ttbar"].category = "None"
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_get_batches_per_file(self):
        """Test batches per file."""
        self.config.preparation.input_files[self.args.sample] = [self.tf.name]
        ps = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            ps.get_batches_per_file,
            ps.config.preparation.get_input_files(self.args.sample),
        )
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
        self.config.preparation.input_files[self.args.sample] = [self.tf.name]
        ps = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            ps.get_batches_per_file,
            ps.config.preparation.get_input_files(self.args.sample),
        )
        ps.save_tracks = None
        expected_jets = np.array([])
        for num, (jets, tracks) in enumerate(ps.jets_generator(files_in_batches)):
            with self.subTest("sub test track", i=num):
                self.assertEqual(tracks, None)
            with self.subTest("sub test jets", i=num):
                self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_fullcuts(self):
        """Test jet generator including tracks with full cuts."""
        self.config.preparation.input_files[self.args.sample] = [self.tf.name]
        ps = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            ps.get_batches_per_file,
            ps.config.preparation.get_input_files(self.args.sample),
        )
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
        self.config.preparation.input_files[self.args.sample] = [self.tf.name]
        ps = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            ps.get_batches_per_file,
            ps.config.preparation.get_input_files(self.args.sample),
        )
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
        self.config.preparation.input_files[self.args.sample] = [self.tf.name]
        ps = PrepareSamples(self.args, self.config)
        ps.sample.output_name = self.output_file.name
        ps.run()
        assert os.path.exists(self.output_file.name) == 1
