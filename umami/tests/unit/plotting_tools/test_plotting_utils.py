"""Unit test script for utils functions of the plotting functions."""

import unittest

from umami.configuration import logger, set_log_level
from umami.plotting_tools.utils import retrieve_truth_label_var_value

set_log_level(logger, "DEBUG")


class RetrieveTruthLabelVarValueTestCase(unittest.TestCase):
    """Test class for the retrieve_truth_label_var_value function."""

    def setUp(self) -> None:
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.class_labels_extended = ["singlebjets", "cjets", "ujets", "bbjets"]

        self.control_label_dict = {
            "bjets": 5,
            "cjets": 4,
            "ujets": 0,
        }
        self.control_label_dict_extended = {
            "singlebjets": [5, 54],
            "cjets": 4,
            "ujets": 0,
            "bbjets": 55,
        }
        self.control_label_var_dict = {
            "bjets": "HadronConeExclTruthLabelID",
            "cjets": "HadronConeExclTruthLabelID",
            "ujets": "HadronConeExclTruthLabelID",
        }
        self.control_label_var_dict_extended = {
            "singlebjets": "HadronConeExclExtendedTruthLabelID",
            "cjets": "HadronConeExclTruthLabelID",
            "ujets": "HadronConeExclTruthLabelID",
            "bbjets": "HadronConeExclExtendedTruthLabelID",
        }

    def test_3_classes(self):
        """Test nomimal behaviour for 3 classes without extended labeling"""

        label_dict, label_var_dict = retrieve_truth_label_var_value(self.class_labels)

        with self.subTest("Test Label dict"):
            self.assertEqual(label_dict, self.control_label_dict)

        with self.subTest("Test label variable dict"):
            self.assertEqual(label_var_dict, self.control_label_var_dict)

    def test_4_classes(self):
        """Test nomimal behaviour for 4 classes without extended labeling"""

        label_dict, label_var_dict = retrieve_truth_label_var_value(
            self.class_labels_extended
        )

        with self.subTest("Test Label dict"):
            self.assertEqual(label_dict, self.control_label_dict_extended)

        with self.subTest("Test label variable dict"):
            self.assertEqual(label_var_dict, self.control_label_var_dict_extended)
