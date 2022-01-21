import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config
from umami.preprocessing_tools import Configuration, GetBinaryLabels, PrepareSamples


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
        config = Configuration(self.config_file)
        del config.config["outfile_name"]
        with self.assertRaises(KeyError):
            config.GetConfiguration()

    # this functionality is still there but is not used for now
    # so will keep the test here in case this is used again
    # def test_missing_key_warning(self):
    #     config = Configuration(self.config_file)
    #     del config.config["outfile_name"]
    #     with self.assertWarns(Warning):
    #         config.GetConfiguration()

    def test_GetFileName_no_input(self):
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(out_file, config.outfile_name)

    def test_GetFileName_no_iterations(self):
        config = Configuration(self.config_file)
        self.assertNotIn("test", config.outfile_name)
        out_file = config.GetFileName(option="test")
        self.assertIn("test", out_file)

    def test_GetFileName_no_iterations_no_input(self):
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(config.outfile_name, out_file)


class GetBinaryLabelsTestCase(unittest.TestCase):
    """
    Test the implementation of the GetBinaryLabels function.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        self.y = np.concatenate(
            [np.zeros(12), 4 * np.ones(35), 6 * np.ones(5), 15 * np.ones(35)]
        )
        np.random.seed(42)
        np.random.shuffle(self.y)
        self.df = pd.DataFrame({"label": self.y})

    def testZeroLength(self):
        df_0 = pd.DataFrame({"label": []})
        with self.assertRaises(ValueError):
            GetBinaryLabels(df_0)

    def testShape(self):
        y_categ = GetBinaryLabels(self.df, "label")
        self.assertEqual(y_categ.shape, (len(self.y), 4))


class PrepareSamplesTestCase(unittest.TestCase):
    """
    Test the implementation of the PrepareSamples class.
    """

    class c_args:
        def __init__(self) -> None:
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
        self.tf = tempfile.NamedTemporaryFile()
        with h5py.File(self.tf, "w") as out_file:
            out_file.create_dataset("jets", data=jets.to_records())
            out_file.create_dataset("tracks", data=tracks)
        self.output_file = tempfile.NamedTemporaryFile()

    def test_NoSampleProvided(self):
        self.args.sample = None
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_WrongSampleProvided(self):
        self.args.sample = "None"
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_WrongCategoryProvided(self):
        self.config.preparation["samples"]["ttbar"]["category"] = "None"
        with self.assertRaises(KeyError):
            PrepareSamples(self.args, self.config)

    def test_GetBatchesPerFile(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        for batch_tuple in list(files_in_batches):
            # first entry of tuples is the filename
            self.assertEqual(type(batch_tuple[0]), str)
            # second entry of tuples is a list of tuples with the batch indices
            for batch in batch_tuple[1]:
                self.assertEqual(type(batch), tuple)

    def test_jets_generator_fullcuts_wotracks(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = None
        expected_jets = np.array([])
        for jets, tracks in ps.jets_generator(files_in_batches):
            self.assertEqual(tracks, None)
            self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_fullcuts(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = True
        expected_jets = np.array([])
        expected_tracks = np.array([])
        for jets, track in ps.jets_generator(files_in_batches):
            self.assertEqual(len(track), len(expected_tracks))
            self.assertEqual(len(jets), len(expected_jets))

    def test_jets_generator_lightcut(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        files_in_batches = map(ps.GetBatchesPerFile, ps.ntuples)
        ps.save_tracks = True
        ps.cuts = [{"eventNumber": {"operator": "==", "condition": 0}}]
        expected_jets_len = expected_tracks_len = 1
        for jets, tracks in ps.jets_generator(files_in_batches):
            self.assertEqual(len(tracks), expected_tracks_len)
            self.assertEqual(len(jets), expected_jets_len)

    def test_Run(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        ps.output_file = self.output_file.name
        ps.Run()
        assert os.path.exists(self.output_file.name) == 1
