"""Unit test script for the classification helper functions."""
import unittest

from umami.configuration import logger, set_log_level
from umami.helper_tools import (
    get_class_label_ids,
    get_class_label_ops,
    get_class_label_variables,
    get_class_prob_var_names,
)

set_log_level(logger, "DEBUG")


class GetClassTestCase(unittest.TestCase):
    """Test class for the get_class_labels method."""

    def setUp(self):
        self.class_labels_3 = ["bjets", "cjets", "ujets"]
        self.tagger_prob = "rnnip"
        self.class_prob_names_3 = ["rnnip_pb", "rnnip_pc", "rnnip_pu"]
        self.class_id_3 = [5, 4, 0]
        self.label_var_list_3 = [
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
        ]
        self.flatten_class_labels_3 = ["bjets", "cjets", "ujets"]
        self.class_ops_3 = ["==", "==", "=="]
        self.class_labels_4 = ["bjets", "cjets", "ujets", "singlebjets"]
        self.class_prob_names_4 = [
            "rnnip_pb",
            "rnnip_pc",
            "rnnip_pu",
            "rnnip_pb",
        ]
        self.class_id_4 = [5, 4, 0, 5, 54]
        self.label_var_list_4 = [
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclTruthLabelID",
            "HadronConeExclExtendedTruthLabelID",
            "HadronConeExclExtendedTruthLabelID",
        ]
        self.flatten_class_labels_4 = [
            "bjets",
            "cjets",
            "ujets",
            "singlebjets",
            "singlebjets",
        ]
        self.class_ops_4 = ["==", "==", "==", "==", "=="]
        self.class_labels_diff_ops = ["bjets", "cjets", "ujets", "lepcbjets"]
        self.class_ops_diff = ["==", "==", "==", "!=", "!="]

    def test_get_class_label_ids_3_classes(self):
        """Get the class label ids for 3 classes."""
        class_ids = get_class_label_ids(class_labels=self.class_labels_3)

        self.assertEqual(class_ids, self.class_id_3)

    def test_get_class_label_ids_4_classes(self):
        """Get the class label ids for 4 classes."""
        class_ids = get_class_label_ids(class_labels=self.class_labels_4)

        self.assertEqual(class_ids, self.class_id_4)

    def test_get_class_label_variables_3_classes(self):
        """Get the class labels for 3 classes."""
        label_var_list_3, flatten_class_labels_3 = get_class_label_variables(
            class_labels=self.class_labels_3
        )

        with self.subTest("Test label variable list"):
            self.assertEqual(label_var_list_3, self.label_var_list_3)

        with self.subTest("Test flatten label variable list"):
            self.assertEqual(flatten_class_labels_3, self.flatten_class_labels_3)

    def test_get_class_label_variables_4_classes(self):
        """Get the class labels for 4 classes."""
        label_var_list_4, flatten_class_labels_4 = get_class_label_variables(
            class_labels=self.class_labels_4
        )

        with self.subTest("Test label variable list"):
            self.assertEqual(label_var_list_4, self.label_var_list_4)

        with self.subTest("Test flatten label variable list"):
            self.assertEqual(flatten_class_labels_4, self.flatten_class_labels_4)

    def test_get_class_label_ops_3(self):
        """Get the class operators for 3 classes."""
        class_ops_3 = get_class_label_ops(self.class_labels_3)

        self.assertEqual(class_ops_3, self.class_ops_3)

    def test_get_class_label_ops_4(self):
        """Get the class operators for 4 classes."""
        class_ops_4 = get_class_label_ops(self.class_labels_4)

        self.assertEqual(class_ops_4, self.class_ops_4)

    def test_get_class_label_ops_diff(self):
        """Get the class operators for classes with different operators."""
        class_ops_diff_ops = get_class_label_ops(self.class_labels_diff_ops)

        self.assertEqual(class_ops_diff_ops, self.class_ops_diff)

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
