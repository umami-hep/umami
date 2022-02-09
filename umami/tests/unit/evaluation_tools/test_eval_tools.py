#!/usr/bin/env python

"""
Unit test script for the functions in eval_tools.py
"""

import unittest

from umami.evaluation_tools.eval_tools import calculate_fraction_dict


class small_functions_TestCase(unittest.TestCase):
    """Test class for small functions in eval_tools.py"""

    def setUp(self):
        self.class_labels_wo_main = ["cjets", "ujets"]
        self.frac_min = 0.1
        self.frac_max = 1
        self.step = 0.1
        self.control_dict_list = [
            {"cjets": 0.1, "ujets": 0.9},
            {"cjets": 0.2, "ujets": 0.8},
            {"cjets": 0.3, "ujets": 0.7},
            {"cjets": 0.4, "ujets": 0.6},
            {"cjets": 0.6, "ujets": 0.4},
            {"cjets": 0.7, "ujets": 0.3},
            {"cjets": 0.8, "ujets": 0.2},
            {"cjets": 0.9, "ujets": 0.1},
        ]

    def test_calculate_fraction_dict(self):
        """Test the correct behaviour of calculate_fraction_dict."""
        dict_list = calculate_fraction_dict(
            class_labels_wo_main=self.class_labels_wo_main,
            frac_min=self.frac_min,
            frac_max=self.frac_max,
            step=self.step,
        )
        print(dict_list)

        self.assertEqual(dict_list, self.control_dict_list)

    def test_calculate_fraction_dict_ValueError(self):
        """Test the ValueError behaviour of calculate_fraction_dict."""
        with self.assertRaises(ValueError):
            _ = calculate_fraction_dict(
                class_labels_wo_main=self.class_labels_wo_main,
                frac_min=0.01,
                frac_max=1,
                step=0.1,
            )
