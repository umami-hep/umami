"""Test scripts for the data tools cuts."""
import unittest

import numpy as np
import pandas as pd

from umami.configuration import logger, set_log_level
from umami.data_tools import get_category_cuts, get_sample_cuts

set_log_level(logger, "DEBUG")


class GetSampleCutsTestCase(unittest.TestCase):
    """
    Test the implementation of the get_sample_cuts function.
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
        self.wrong_format_cuts = [
            {
                "eventNumber": {"operator": "mod_2_==", "condition": 0},
                "pt_btagJes": {"operator": "<=", "condition": 250000.0},
            },
        ]
        self.wrong_modulo_cuts = [
            {"eventNumber": {"operator": "mod_N_==", "condition": 0}},
            {"pt_btagJes": {"operator": "mod_5_is", "condition": 0}},
        ]
        self.wrong_operator_cuts = [
            {"eventNumber": {"operator": "is", "condition": 0}},
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
        """Testing cuts for ttbar pass."""
        indices_to_remove = get_sample_cuts(
            self.jets.to_records(index=False),
            self.cuts,
        )
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))

    def test_cuts_passing_ttbar_extended_labelling(self):
        """Testing cuts for ttbar pass with the extended labelling."""
        indices_to_remove = get_sample_cuts(
            self.jets.to_records(index=False),
            self.extended_cuts,
        )
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))

    def test_cuts_wrong_format(self):
        """Testing raise of KeyError for wrongly formatted cuts."""
        with self.assertRaises(KeyError):
            get_sample_cuts(
                self.jets.to_records(index=False),
                self.wrong_format_cuts,
            )

    def test_cuts_wrong_modulo(self):
        """Testing raise of KeyError for wrong modulo cuts."""
        with self.assertRaises(RuntimeError):
            get_sample_cuts(
                self.jets.to_records(index=False),
                self.wrong_modulo_cuts,
            )

    def test_cuts_wrong_operator(self):
        """Testing raise of KeyError for wrong operator cuts."""
        with self.assertRaises(KeyError):
            get_sample_cuts(
                self.jets.to_records(index=False),
                self.wrong_operator_cuts,
            )


class GetCategoryCutsTestCase(unittest.TestCase):
    """
    Test the implementation of the get_category_cuts function.
    """

    def setUp(self) -> None:
        self.label_var = "HadronConeExclTruthLabelID"
        self.label_value = 5

    def test_wrong_type_provided(self):
        """Testing wrong type provided."""
        self.label_value = "5"
        with self.assertRaises(ValueError):
            get_category_cuts(self.label_var, self.label_value)

    def test_integer_case(self):
        """Testing ints."""
        cuts = get_category_cuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)

    def test_float_case(self):
        """Testing floats."""
        self.label_value = 5.0
        cuts = get_category_cuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)

    def test_list_case(self):
        """Testing lists."""
        self.label_value = [5, 55]
        cuts = get_category_cuts(self.label_var, self.label_value)
        expected_cuts = [
            {self.label_var: {"operator": "==", "condition": self.label_value}}
        ]
        self.assertEqual(cuts, expected_cuts)
