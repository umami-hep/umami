import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config
from umami.preprocessing_tools import (
    Configuration,
    GetBinaryLabels,
    GetCategoryCuts,
    GetSampleCuts,
    PrepareSamples,
)


class ConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = os.path.join(
            os.path.dirname(__file__), "test_preprocess_config.yaml"
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


class PreprocessingTestSampleCuts(unittest.TestCase):
    """
    Test the implementation of the Preprocessing sample cut application.
    """

    def setUp(self):
        self.extended_cuts = [
            {"eventNumber": {"operator": "mod_2_==", "condition": 0}},
            {"pt_btagJes": {"operator": "<=", "condition": 250000.0}},
            {
                "HadronConeExclExtendedTruthLabelID": {
                    "operator": "==",
                    "condition": [5, 54],
                }
            },
        ]
        self.cuts = [
            {"eventNumber": {"operator": "mod_2_==", "condition": 0}},
            {"pt_btagJes": {"operator": "<=", "condition": 250000.0}},
            {"HadronConeExclTruthLabelID": {"operator": "==", "condition": 5}},
        ]
        self.jets = pd.DataFrame(
            {
                "GhostBHadronsFinalPt": [
                    2e3,
                    2.6e4,
                    np.nan,
                    2.6e4,
                    2e3,
                    2.6e4,
                ],
                "pt_btagJes": [
                    2e3,
                    2.6e4,
                    np.nan,
                    2.6e4,
                    2e3,
                    2.6e4,
                ],
                "HadronConeExclTruthLabelID": [5, 5, 4, 4, 0, 15],
                "HadronConeExclExtendedTruthLabelID": [5, 54, 4, 44, 0, 15],
                "eventNumber": [1, 2, 3, 4, 5, 6],
            }
        )
        self.pass_ttbar = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_cuts_passing_ttbar(self):
        indices_to_remove = GetSampleCuts(
            self.jets.to_records(index=False),
            self.cuts,
        )
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))

    def test_cuts_passing_ttbar_extended_labelling(self):
        indices_to_remove = GetSampleCuts(
            self.jets.to_records(index=False),
            self.extended_cuts,
        )
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))


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
            self.config_file = "test_preprocess_config.yaml"
            self.tracks = False
            self.tracks_name = "tracks"
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
                "JetFitterSecondaryVertex_mass": np.array(
                    [0, 0, 0], dtype=np.uint8
                ),
                "JetFitterSecondaryVertex_energy": np.array(
                    [0, 0, 0], dtype=np.uint8
                ),
                "HadronConeExclTruthLabelID": np.array(
                    [5, 4, 0], dtype=np.uint8
                ),
                "GhostBHadronsFinalPt": 5e5 * np.ones(3),
                global_config.pTvariable: 5e3 * np.ones(3),
            },
        )
        self.tf = tempfile.NamedTemporaryFile()
        with h5py.File(self.tf, "w") as out_file:
            out_file.create_dataset(
                "jets",
                data=jets.to_records(),
            )
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

    def test_GetJets(self):
        ps = PrepareSamples(self.args, self.config)
        jets, tracks = ps.get_jets(self.tf.name)
        expected_jets = np.array([])
        self.assertEqual(len(jets), len(expected_jets))
        self.assertIsInstance(tracks, type(None))

    def test_Run(self):
        ps = PrepareSamples(self.args, self.config)
        ps.ntuples = [self.tf.name]
        ps.output_file = self.output_file.name
        ps.Run()
        assert os.path.exists(self.output_file.name) == 1


class GetCategoryCutsTestCase(unittest.TestCase):
    """
    Test the implementation of the GetCategoryCuts function.
    """

    def setUp(self) -> None:
        self.label_var = "HadronConeExclTruthLabelID"
        self.label_value = 5

    def test_WrongTypeProvided(self):
        self.label_value = "5"
        with self.assertRaises(ValueError):
            GetCategoryCuts(self.label_var, self.label_value)

    def test_IntegerCase(self):
        cuts = GetCategoryCuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)

    def test_FloatCase(self):
        self.label_value = 5.0
        cuts = GetCategoryCuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)

    def test_ListCase(self):
        self.label_value = [5, 55]
        cuts = GetCategoryCuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)
