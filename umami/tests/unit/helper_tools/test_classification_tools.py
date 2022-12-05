"""Unit test script for the classification helper functions."""
import unittest

from umami.configuration import logger, set_log_level
from umami.data_tools import get_cut_list
from umami.helper_tools import get_class_label_variables, get_class_prob_var_names

set_log_level(logger, "DEBUG")


class GetClassTestCase(unittest.TestCase):
    """Test class for the get_class_labels method."""

    def setUp(self):
        self.class_labels_3 = ["bjets", "cjets", "ujets"]
        self.tagger_prob = "rnnip"
        self.class_prob_names_3 = ["rnnip_pb", "rnnip_pc", "rnnip_pu"]
        self.label_var_list_3 = [
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
        ]
        self.flatten_class_labels_3 = ["bjets", "cjets", "ujets"]
        self.cuts_labels_3 = {
            "bjets": ["HadronConeExclTruthLabelID == 5"],
            "cjets": ["HadronConeExclTruthLabelID == 4"],
            "ujets": ["HadronConeExclTruthLabelID == 0"],
        }
        self.class_labels_4 = ["bjets", "cjets", "ujets", "singlebjets"]
        self.class_prob_names_4 = [
            "rnnip_pb",
            "rnnip_pc",
            "rnnip_pu",
            "rnnip_pb",
        ]
        self.label_var_list_4 = [
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclExtendedTruthLabelID",
        ]
        self.cuts_labels_4 = {
            "bjets": ["HadronConeExclTruthLabelID == 5"],
            "cjets": ["HadronConeExclTruthLabelID == 4"],
            "ujets": ["HadronConeExclTruthLabelID == 0"],
            "singlebjets": ["HadronConeExclExtendedTruthLabelID == [5, 54]"],
        }

    def test_get_class_label_variables_3_classes(self):
        """Get the class labels for 3 classes."""
        labels_var_list_3 = get_class_label_variables(class_labels=self.class_labels_3)

        self.assertEqual(labels_var_list_3, self.label_var_list_3)

    def test_get_class_label_variables_4_classes(self):
        """Get the class labels for 4 classes."""
        labels_var_list_4 = get_class_label_variables(class_labels=self.class_labels_4)

        self.assertEqual(labels_var_list_4, self.label_var_list_4)

    def test_get_cut_list_4_classes(self):
        """Get the class labels for 4 classes."""
        cut_list_4_classes = get_cut_list(class_labels=self.class_labels_4)

        self.assertEqual(cut_list_4_classes, self.cuts_labels_4)

    def test_get_cut_list_3_classes(self):
        """Get the class labels for 4 classes."""
        cut_list_3_classes = get_cut_list(class_labels=self.class_labels_3)

        self.assertEqual(cut_list_3_classes, self.cuts_labels_3)

    def test_get_class_prob_var_names_3_classes(self):
        """Test the class prob var names for 3 classes."""
        class_prob_names = get_class_prob_var_names(
            tagger_name=self.tagger_prob,
            class_labels=self.class_labels_3,
        )

        self.assertEqual(class_prob_names, self.class_prob_names_3)

    def test_get_class_prob_var_names_4_classes(self):
        """Test the class prob var names for 4 classes."""
        class_prob_names = get_class_prob_var_names(
            tagger_name=self.tagger_prob,
            class_labels=self.class_labels_4,
        )

        self.assertEqual(class_prob_names, self.class_prob_names_4)
