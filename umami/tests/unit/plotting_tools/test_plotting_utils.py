"""Unit test script for utils functions of the plotting functions."""

import unittest

from umami.configuration import logger, set_log_level
from umami.data_tools import retrieve_cut_string

set_log_level(logger, "DEBUG")


class RetrieveTruthLabelVarValueTestCase(unittest.TestCase):
    """Test class for the retrieve_truth_label_var_value function."""

    def setUp(self) -> None:
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.class_labels_extended = ["singlebjets", "cjets", "ujets", "bbjets"]

        self.control_cut_strings = {
            "bjets": "HadronConeExclTruthLabelID in [5]",
            "cjets": "HadronConeExclTruthLabelID in [4]",
            "ujets": "HadronConeExclTruthLabelID in [0]",
        }
        self.control_cut_strings_extended = {
            "singlebjets": "HadronConeExclExtendedTruthLabelID in [5]",
            "cjets": "HadronConeExclTruthLabelID in [4]",
            "ujets": "HadronConeExclTruthLabelID in [0]",
            "bbjets": "HadronConeExclExtendedTruthLabelID in [55]",
        }

    def test_3_classes(self):
        """Test nomimal behaviour for 3 classes without extended labeling"""

        string_cuts = retrieve_cut_string(self.class_labels)

        with self.subTest("Test label variable dict"):
            self.assertEqual(string_cuts, self.control_cut_strings)

    def test_4_classes(self):
        """Test nomimal behaviour for 4 classes without extended labeling"""

        string_cuts = retrieve_cut_string(self.class_labels_extended)

        with self.subTest("Test Label dict"):
            self.assertEqual(string_cuts, self.control_cut_strings_extended)
