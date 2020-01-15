import unittest
from unittest import mock
import argparse
import os
from umami.preprocessing import GetParser


class PreprocessingTestParser(unittest.TestCase):
    """
    Test the implementation of the Prerocessing command line parser.
    """
    config_file = os.path.join(os.path.dirname(__file__),
                               "test_preprocess_config.yaml")

    def setUp(self):
        self.config_file = os.path.join(os.path.dirname(__file__),
                                        "test_preprocess_config.yaml")

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(config_file=config_file))
    def test_Parser(self, mock_args):
        parser = GetParser()
        self.assertEqual(parser.config_file, self.config_file)
        self.assertFalse(parser.tracks)
        self.assertNone(parser.cut_config_file)

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(config_file=config_file,
                                                tracks=True))
    def test_Parser(self, mock_args):
        self.parser = GetParser()
        self.assertTrue(self.parser.tracks)



    # def test_zero_case(self):
    #     jets = pd.DataFrame({"secondaryVtx_m": [2e3, 2.6e4, 2.7e4, 2.4e4,
    #                                             np.nan, np.nan, 25, 30e4,
    #                                             np.nan, 0],
    #                          "secondaryVtx_E": [2001, 26001, 1e9, 1.5e8,
    #                                             5, np.nan, np.nan, np.nan,
    #                                             4e8, 0],
    #                          "HadronConeExclTruthLabelID": 5 * np.ones(10),
    #                          "GhostBHadronsFinalPt": 5e3 * np.ones(10),
    #                          "pt_uncalib": 5.2e3 * np.ones(10),
    #                          }
    #                         ).to_records(index=False)
    #     # print(list(jets.dtype.names))
    #     # indices_to_remove = GetPtCuts(jets, self.config)
    #     indices_to_remove = GetCuts(jets, self.config)
    #     print(indices_to_remove)
    #     jets_cut = np.delete(jets, indices_to_remove, 0)
    #     print(jets)
    #     print(jets_cut)
    #     print(pd.DataFrame(jets_cut).head())

#
# class PreprocessingTestCases(unittest.TestCase):
#     """
#     Test the implementation of the Prerocessing cut application.
#     """
#     config_file = os.path.join(os.path.dirname(__file__),
#                                "test_preprocess_config.yaml")
#     @mock.patch('argparse.ArgumentParser.parse_args',
#                 return_value=argparse.Namespace(config_file=config_file,
#                                                 tracks=True))
#     def setUp(self, mock_args):
#         self.parser = GetParser()
#         self.config = upt.Configuration(self.parser.config_file)
