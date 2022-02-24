#!/usr/bin/env python

"""
Unit test script for the functions in metrics.py
"""

import unittest

import numpy as np

from umami.configuration import logger, set_log_level
from umami.plotting.var_vs_eff import var_vs_eff

set_log_level(logger, "DEBUG")


class var_vs_eff_TestCase(unittest.TestCase):
    """Test class for the umami.plotting.var_vs_eff functions."""

    def setUp(self):
        self.wp = 0.77
        self.disc_sig = np.linspace(-6, +6, 100)
        self.x_var_sig = np.exp(-self.disc_sig) * 10e3
        self.disc_bkg = np.linspace(-5.5, +6.6, 120)
        self.x_var_bkg = np.exp(-self.disc_bkg * 0.8) * 10e3 + 30

    def test_var_vs_eff_init_wrong_sig_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(4), np.ones(5))

    def test_var_vs_eff_init_wrong_bkg_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(6), np.ones(6), np.ones(4), np.ones(5))

    def test_var_vs_eff_init_fixed_eff_disc_cut(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(
                np.ones(6), np.ones(6), fixed_eff_bin=True, disc_cut=1.0, wp=0.77
            )

    def test_var_vs_eff_init_fixed_eff_no_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(6), np.ones(6), fixed_eff_bin=True)

    def test_var_vs_eff_init_disc_cut_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(6), np.ones(6), disc_cut=1.0, wp=0.77)

    def test_var_vs_eff_init_no_disc_cut_no_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(6), np.ones(6))

    def test_var_vs_eff_init_disc_cut_wrong_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            var_vs_eff(np.ones(6), np.ones(6), disc_cut=[1.0, 2.0])

    def test_var_vs_eff_set_bin_edges_list(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = var_vs_eff(
            x_var_sig=[0, 1, 2], disc_sig=[3, 4, 5], bins=[0, 1, 2], wp=0.7
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2])

    def test_var_vs_eff_set_bin_edges_only_signal(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = var_vs_eff(x_var_sig=[0, 1, 2], disc_sig=[3, 4, 5], bins=2, wp=0.7)
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2], decimal=4)

    def test_var_vs_eff_set_bin_edges(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = var_vs_eff(
            x_var_sig=[0, 1, 2],
            disc_sig=[3, 4, 5],
            x_var_bkg=[-1, 1, 3],
            disc_bkg=[3, 4, 5],
            bins=2,
            wp=0.7,
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [-1, 1, 3], decimal=4)

    def test_var_vs_eff_fixed_eff_sig_eff(self):
        """Test var_vs_eff sig_eff."""
        n_bins = 4
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(
            var_plot.sig_eff[0], [self.wp] * n_bins, decimal=2
        )

    def test_var_vs_eff_fixed_eff_sig_rej(self):
        """Test var_vs_eff sig_rej."""
        n_bins = 4
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(
            var_plot.sig_rej[0], [1 / self.wp] * n_bins, decimal=2
        )

    def test_var_vs_eff_one_bin(self):
        """Test var_vs_eff."""
        n_bins = 1
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=n_bins,
        )
        var_plot_comp = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(var_plot.sig_eff, var_plot_comp.sig_eff)

    def test_var_vs_eff_divide_same(self):
        """Test var_vs_eff divide."""
        n_bins = 1
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(
            var_plot.divide(var_plot, mode="sig_eff")[0], np.ones(1)
        )

    def test_var_vs_eff_divide_wrong_mode(self):
        """Test var_vs_eff divide."""
        n_bins = 1
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=n_bins,
        )
        with self.assertRaises(ValueError):
            var_plot.divide(var_plot, mode="test")[0], np.ones(1)

    def test_var_vs_eff_divide_different_binning(self):
        """Test var_vs_eff divide."""
        var_plot = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=1,
        )
        var_plot_comp = var_vs_eff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            wp=self.wp,
            fixed_eff_bin=True,
            bins=2,
        )
        with self.assertRaises(ValueError):
            var_plot.divide(var_plot_comp, mode="sig_eff")
