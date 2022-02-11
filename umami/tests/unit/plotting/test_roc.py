#!/usr/bin/env python

"""
Unit test script for the functions in metrics.py
"""

import unittest

import numpy as np

from umami.plotting.roc import roc


class roc_TestCase(unittest.TestCase):
    """Test class for the umami.plotting.roc functions."""

    def setUp(self):
        self.sig_eff = np.linspace(0.4, 1, 100)
        self.bkg_rej = np.exp(-self.sig_eff) * 10e3

    def test_roc_init(self):
        """Test roc init."""
        with self.assertRaises(ValueError):
            roc(np.ones(4), np.ones(5))

    def test_ratio_same_object(self):
        """Test roc divide function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = roc(self.sig_eff, self.bkg_rej)
        ratio = roc_curve.divide(roc_curve_ref)

        np.testing.assert_array_almost_equal(ratio, np.ones(len(self.bkg_rej)))

    def test_ratio_factor_two(self):
        """Test roc divide function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = roc(self.sig_eff, self.bkg_rej * 2)
        ratio = roc_curve.divide(roc_curve_ref)

        np.testing.assert_array_almost_equal(ratio, 1 / 2 * np.ones(len(self.bkg_rej)))

    def test_ratio_factor_two_inverse(self):
        """Test roc divide function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = roc(self.sig_eff, self.bkg_rej * 2)
        ratio = roc_curve.divide(roc_curve_ref, inverse=True)

        np.testing.assert_array_almost_equal(ratio, 2 * np.ones(len(self.bkg_rej)))

    def test_binomial_error_no_ntest(self):
        """Test roc binomial_error function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        with self.assertRaises(ValueError):
            roc_curve.binomial_error()

    def test_binomial_error_only_zeros(self):
        """Test roc binomial_error function."""
        roc_curve = roc(self.sig_eff, np.zeros(len(self.sig_eff)), n_test=10e5)
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(), [])

    def test_binomial_error_example(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744])
        roc_curve = roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]), n_test=100)
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(), error_rej)

    def test_binomial_error_example_norm(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744]) / np.array([20, 50, 100])
        roc_curve = roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]), n_test=100)
        np.testing.assert_array_almost_equal(
            roc_curve.binomial_error(norm=True), error_rej
        )

    def test_binomial_error_example_pass_ntest(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744])
        roc_curve = roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]))
        np.testing.assert_array_almost_equal(
            roc_curve.binomial_error(n_test=100), error_rej
        )

    def test_fct_inter(self):
        """Test roc fct_inter function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        np.testing.assert_array_almost_equal(
            roc_curve.fct_inter(self.sig_eff), self.bkg_rej
        )


class roc_mask_TestCase(unittest.TestCase):
    """Test class for the umami.plotting.roc non_zero_mask function."""

    def setUp(self):
        self.sig_eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.bkg_rej = np.array([0, 0.2, 0, 0.4, 0.5, 0, 0.7])

    def test_non_zero_mask(self):
        """Test roc non_zero_mask function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, True, False, True, True, False, True]
        )

    def test_non_zero_mask_xmin(self):
        """Test roc non_zero_mask function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmin=0.4)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, False, False, True, True, False, True]
        )

    def test_non_zero_mask_xmax(self):
        """Test roc non_zero_mask function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmax=0.6)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, True, False, True, True, False, False]
        )

    def test_non_zero_mask_xmin_xmax(self):
        """Test roc non_zero_mask function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmax=0.6, xmin=0.4)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, False, False, True, True, False, False]
        )

    def test_non_zero(self):
        """Test roc non_zero function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej)
        result_bkg_rej = self.bkg_rej[[False, True, False, True, True, False, True]]
        result_sig_eff = self.sig_eff[[False, True, False, True, True, False, True]]
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero, (result_bkg_rej, result_sig_eff)
        )

    def test_non_zero_xmin(self):
        """Test roc non_zero function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmin=0.4)
        result_bkg_rej = self.bkg_rej[[False, False, False, True, True, False, True]]
        result_sig_eff = self.sig_eff[[False, False, False, True, True, False, True]]
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero, (result_bkg_rej, result_sig_eff)
        )

    def test_non_zero_xmax(self):
        """Test roc non_zero function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmax=0.6)
        result_bkg_rej = self.bkg_rej[[False, True, False, True, True, False, False]]
        result_sig_eff = self.sig_eff[[False, True, False, True, True, False, False]]
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero, (result_bkg_rej, result_sig_eff)
        )

    def test_non_zero_xmin_xmax(self):
        """Test roc non_zero function."""
        roc_curve = roc(self.sig_eff, self.bkg_rej, xmax=0.6, xmin=0.4)
        result_bkg_rej = self.bkg_rej[[False, False, False, True, True, False, False]]
        result_sig_eff = self.sig_eff[[False, False, False, True, True, False, False]]
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero, (result_bkg_rej, result_sig_eff)
        )


# what happens when dividing two rocs which are defined in different intervals?
