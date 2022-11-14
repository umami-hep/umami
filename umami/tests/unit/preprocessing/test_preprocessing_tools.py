"""Unit tests for preprocessing_tools."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from subprocess import CalledProcessError, run

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config, logger, set_log_level
from umami.preprocessing_tools import (
    MergeConfig,
    PrepareSamples,
    PreprocessConfiguration,
    TTbarMerge,
    binarise_jet_labels,
    event_indices,
    event_list,
    get_scale_dict,
    get_variable_dict,
)

set_log_level(logger, "DEBUG")


class GetDictTestCase(unittest.TestCase):
    """Testing the get_scale_dict and the get_variable_dict functions."""

    def setUp(self) -> None:
        """Setting up needed paths."""
        self.var_file = os.path.join(
            os.path.dirname(__file__), "fixtures/dummy_var_file_short.yaml"
        )
        self.scale_file = os.path.join(
            os.path.dirname(__file__), "fixtures/dummy_scale_file_short.json"
        )
        self.control_var_dict = {
            "custom_defaults_vars": {"JetFitter_energyFraction": 0},
            "label": "HadronConeExclTruthLabelID",
            "track_train_variables": {
                "tracks": {
                    "jointNormVars": [
                        "numberOfPixelHits",
                        "numberOfSCTHits",
                        "btagIp_d0",
                    ],
                    "logNormVars": ["ptfrac", "dr"],
                    "noNormVars": ["IP3D_signed_d0_significance"],
                }
            },
            "train_variables": {
                "JetFitter": ["JetFitter_isDefaults"],
                "JetKinematics": ["absEta_btagJes", "pt_btagJes"],
            },
        }
        self.control_scale_dict = {
            "jets": {
                "eta_btagJes": {"shift": 1, "scale": 2, "default": 0},
                "pt_btagJes": {"shift": 5, "scale": 2, "default": 1},
            }
        }
        self.control_scale_dict_jets = {
            "eta_btagJes": {"shift": 1, "scale": 2, "default": 0},
            "pt_btagJes": {"shift": 5, "scale": 2, "default": 1},
        }

    def test_get_scale_dict(self):
        """Testing default behaviour of get_scale_dict."""

        with self.subTest("Loading test"):
            scale_dict = get_scale_dict(file_path=self.scale_file, dict_key="jets")
            self.assertEqual(scale_dict, self.control_scale_dict_jets)

        with self.subTest("Already loaded test"):
            scale_dict = get_scale_dict(
                file_path=self.control_scale_dict, dict_key="jets"
            )
            self.assertEqual(scale_dict, self.control_scale_dict_jets)

        with self.subTest("Already loaded but without key"):
            scale_dict = get_scale_dict(
                file_path={"jets": self.control_scale_dict},
                dict_key="jets",
            )
            self.assertEqual(scale_dict, self.control_scale_dict)

        with self.subTest("Not compatible input type test"):
            with self.assertRaises(ValueError):
                get_scale_dict(file_path=int(5), dict_key="jets")

    def test_get_variable_dict(self):
        """Testing default behaviour of get_variable_dict."""

        with self.subTest("Loading test"):
            var_dict = get_variable_dict(file_path=self.var_file)
            self.assertEqual(var_dict, self.control_var_dict)

        with self.subTest("Already loaded test"):
            var_dict = get_variable_dict(file_path=self.control_var_dict)
            self.assertEqual(var_dict, self.control_var_dict)

        with self.subTest("Not compatible input type test"):
            with self.assertRaises(ValueError):
                get_variable_dict(file_path=int(5))


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
        self.labels = np.concatenate(
            [np.zeros(12), np.ones(35), 2 * np.ones(5), 3 * np.ones(35)]
        )
        np.random.seed(42)
        np.random.shuffle(self.labels)
        self.df_label = pd.DataFrame({"label": self.labels})
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
                labels=self.labels,
                internal_labels=list(range(3)),
            )

    def test_shape_array(self):
        """Test shape for array input."""
        y_categ = binarise_jet_labels(
            labels=self.labels,
            internal_labels=list(range(4)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(self.labels), 4))

    def test_shape_data_frame(self):
        """Test shape for DataFrame input."""
        y_categ = binarise_jet_labels(
            labels=self.df_label,
            internal_labels=list(range(4)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(self.labels), 4))

    def test_shape_2_classes(self):
        """Test 2 classes"""
        labels = np.concatenate([np.zeros(12), np.ones(35)])
        np.random.seed(42)
        np.random.shuffle(labels)
        df_two_classes = pd.DataFrame({"label": labels})

        y_categ = binarise_jet_labels(
            labels=df_two_classes,
            internal_labels=list(range(2)),
            column="label",
        )
        self.assertEqual(y_categ.shape, (len(labels), 2))


class PrepareSamplesTestCase(unittest.TestCase):
    """
    Test the implementation of the PrepareSamples class.
    """

    class CArgs:
        """Helper class replacing command line arguments."""

        def __init__(self) -> None:
            """Initialise class with preset settings."""
            self.sample = "ttbar"
            self.config_file = os.path.join(
                os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
            )
            self.shuffle_array = True

    def setUp(self):
        self.args = self.CArgs()
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
        self.f_tmp = tempfile.NamedTemporaryFile()  # pylint: disable=R1732
        with h5py.File(self.f_tmp, "w") as out_file:
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
        self.config.preparation.input_files[self.args.sample] = [self.f_tmp.name]
        prep_s = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            prep_s.get_batches_per_file,
            prep_s.config.preparation.get_input_files(self.args.sample),
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
        self.config.preparation.input_files[self.args.sample] = [self.f_tmp.name]
        prep_s = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            prep_s.get_batches_per_file,
            prep_s.config.preparation.get_input_files(self.args.sample),
        )
        prep_s.save_tracks = None
        expected_jets = np.array([])
        for num, (jets, tracks) in enumerate(prep_s.jets_generator(files_in_batches)):
            with self.subTest("sub test track", i=num):
                self.assertEqual(tracks, None)
            with self.subTest("sub test jets", i=num):
                self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_fullcuts(self):
        """Test jet generator including tracks with full cuts."""
        self.config.preparation.input_files[self.args.sample] = [self.f_tmp.name]
        prep_s = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            prep_s.get_batches_per_file,
            prep_s.config.preparation.get_input_files(self.args.sample),
        )
        prep_s.save_tracks = True
        expected_jets = np.array([])
        expected_tracks = np.array([])
        for jets, tracks in prep_s.jets_generator(files_in_batches):
            for tracks_name in prep_s.tracks_names:
                with self.subTest(f"sub test track {tracks_name}"):
                    self.assertEqual(len(tracks[tracks_name]), len(expected_tracks))
            with self.subTest():
                self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_lightcut(self):
        """Test jet generator including tracks with cut on light jets."""
        self.config.preparation.input_files[self.args.sample] = [self.f_tmp.name]
        prep_s = PrepareSamples(self.args, self.config)
        files_in_batches = map(
            prep_s.get_batches_per_file,
            prep_s.config.preparation.get_input_files(self.args.sample),
        )
        prep_s.save_tracks = True
        prep_s.cuts = [{"eventNumber": {"operator": "==", "condition": 0}}]
        expected_jets_len = expected_tracks_len = 1
        for jets, tracks in prep_s.jets_generator(files_in_batches):
            for tracks_name in prep_s.tracks_names:
                with self.subTest(f"sub test track {tracks_name}"):
                    self.assertEqual(len(tracks[tracks_name]), expected_tracks_len)
            with self.subTest():
                self.assertEqual(len(jets), expected_jets_len)

    def test_run(self):
        """Test the run function."""
        self.config.preparation.input_files[self.args.sample] = [self.f_tmp.name]
        prep_s = PrepareSamples(self.args, self.config)
        prep_s.sample.output_name = self.output_file.name
        prep_s.run()
        assert os.path.exists(self.output_file.name) == 1


class TTbarMergeTestCase(unittest.TestCase):
    """
    Test the implementation of the TTbarMerge class
    """

    @classmethod
    def setUpClass(self):  # pylint: disable=C0202
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_merge_config.yaml"
        )

        self.config = MergeConfig(self.config_file)

        # create temporary h5 files
        jets_single = pd.DataFrame(
            {
                "eventNumber": np.array([0, 1, 2, 3, 4, 4], dtype=np.uint8),
                global_config.pTvariable: 5e3 * np.random.uniform(size=6),
            },
        )

        jets_double = pd.DataFrame(
            {
                "eventNumber": np.array([0, 1, 2, 3, 4, 4], dtype=np.uint8),
                global_config.pTvariable: 5e3 * np.random.uniform(size=6),
            },
        )

        tracks = np.ones(shape=(6, 5, 40))

        # pylint: disable=R1732
        self.tf_single = tempfile.NamedTemporaryFile(
            suffix="single_lepton.h5", dir=".", delete=False
        )
        # pylint: disable=R1732
        self.tf_double = tempfile.NamedTemporaryFile(
            suffix="dilepton.h5", dir=".", delete=False
        )

        with h5py.File(self.tf_single, "w") as out_file:
            out_file.create_dataset("jets", data=jets_single.to_records())
            out_file.create_dataset("tracks", data=tracks)
            out_file.create_dataset("fs_tracks", data=tracks)

        with h5py.File(self.tf_double, "w") as out_file:
            out_file.create_dataset("jets", data=jets_double.to_records())
            out_file.create_dataset("tracks", data=tracks)
            out_file.create_dataset("fs_tracks", data=tracks)

        self.config.config["single_lepton"]["file_patten"] = self.tf_single.name
        self.config.config["dilepton"]["file_patten"] = self.tf_double.name

    @classmethod
    def tearDownClass(self):  # pylint: disable=C0202
        os.remove(self.tf_single.name)
        os.remove(self.tf_double.name)
        shutil.rmtree(self.config.config["index_dir"])
        shutil.rmtree(self.config.config["out_dir"])

    def test_get_input_files(self):
        """Test get_input_files function."""
        ttm = TTbarMerge(self.config)

        with self.subTest("Test single lepton"):
            self.assertEqual(
                ttm.get_input_files("single_lepton")[0].stem,
                Path(self.tf_single.name).stem,
            )

        with self.subTest("Test dilepton"):
            self.assertEqual(
                ttm.get_input_files("dilepton")[0].stem, Path(self.tf_double.name).stem
            )

        with self.subTest("Test no sample found"):
            self.assertRaises(ValueError, ttm.get_input_files, "no_sample")

        with self.subTest("Test sample channel not found"):
            self.assertRaises(KeyError, ttm.get_input_files, "no_channel")

    def test_event_list(self):
        """Test event_list function."""
        ttm = TTbarMerge(self.config)
        expected = np.array([0, 1, 2, 3, 4])
        events_single, _ = event_list(ttm.get_input_files("single_lepton"))

        self.assertTrue(np.array_equal(events_single.astype(int), expected))

    def test_event_indices(self):
        """Test event_indices function."""

        events_subsample = np.array([0, 4])

        ttm = TTbarMerge(self.config)
        indices_single = event_indices(
            ttm.get_input_files("single_lepton"), events_subsample
        )

        # 1 jet with event number 0, 2 jets with event number 4
        self.assertEqual(len(indices_single[0]), 3)

    def test_load_events_generator(self):
        """Test load_events_generator function."""

        ttm = TTbarMerge(self.config)
        ttm.tracks_names = ["tracks", "fs_tracks"]

        with self.subTest("Load all jets"):
            input_file = ttm.get_input_files("single_lepton")[0]
            for jets, tracks in ttm.load_jets_generator(input_file):
                self.assertEqual(len(jets), 6)

        with self.subTest("Don't load tracks"):
            input_file = ttm.get_input_files("single_lepton")[0]
            for jets, tracks in ttm.load_jets_generator(input_file, save_tracks=False):
                self.assertEqual(tracks, None)

        with self.subTest("Load multiple track collections"):
            input_file = ttm.get_input_files("single_lepton")[0]

            for jets, tracks in ttm.load_jets_generator(input_file, save_tracks=True):
                # Expect two track collections - tracks and fs_tracks
                self.assertEqual(len(tracks), 2)

        with self.subTest("Smaller chunk size"):
            input_file = ttm.get_input_files("single_lepton")[0]
            # check with a chunk size smaller than the number of events and
            # does not divide into the number of events
            chunk_count = 0
            for jets, tracks in ttm.load_jets_generator(input_file, chunk_size=4):
                chunk_count += 1

            self.assertEqual(chunk_count, 2)

        with self.subTest("Indices passed"):
            input_file = ttm.get_input_files("dilepton")[0]
            indices = np.array([0, 5])
            for jets, tracks in ttm.load_jets_generator(input_file, indices=indices):
                # only 1 jet with event number 0 and 1 jet with event number 4
                self.assertTrue(np.array_equal(jets["eventNumber"], np.array([0, 4])))

    def test_get_indices(self):
        """Test get_indices function."""

        ttm = TTbarMerge(self.config)

        with self.subTest("Test not enough events for desired ratio"):
            ttm.ratio = 0.5
            self.assertRaises(ValueError, ttm.get_indices)

        with self.subTest("Test ratio 1"):
            ttm.ratio = 1
            ttm.get_indices()

            yaml_file = Path(ttm.index_dir) / "ttbar_merge_0.yaml"
            self.assertTrue(os.path.exists(yaml_file))
