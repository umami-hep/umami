#!/usr/bin/env python

"""
Unit test script for the functions in histogram.py
"""

import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
from umami.plotting import histogram, histogram_plot

set_log_level(logger, "DEBUG")


class histogram_TestCase(unittest.TestCase):
    """Test class for the umami.plotting.histogram functions."""

    def test_empty_histogram(self):
        """test if providing wrong input type to histogram raises ValueError"""
        with self.assertRaises(ValueError):
            histogram(values=5)

    def test_divide_before_plotting(self):
        """test if ValueError is raised when dividing before plotting the histograms"""
        hist_1 = histogram([1, 1, 1, 2, 2])
        hist_2 = histogram([1, 2, 2, 2])
        with self.assertRaises(ValueError):
            hist_1.divide(hist_2)

    def test_divide_after_plotting_no_norm(self):
        """test if ratio is calculated correctly after plotting (without norm)"""
        hist_1 = histogram([1, 1, 1, 2, 2])
        hist_2 = histogram([1, 2, 2, 2])
        binning = np.array([1, 2, 3])
        hist_plot = histogram_plot(binning=binning, norm=False)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.array([3, 3, 2 / 3])
        expected_ratio_unc = np.array([3.46410162, 3.46410162, 0.60858062])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_after_plotting_norm(self):
        """test if ratio is calculated correctly after plotting (with norm)"""
        hist_1 = histogram([1, 1, 1, 2, 2])
        hist_2 = histogram([1, 2, 2, 2])
        binning = np.array([1, 2, 3])
        hist_plot = histogram_plot(binning=binning, norm=True)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.array([2.4, 2.4, 0.53333333])
        expected_ratio_unc = np.array([2.77128129, 2.77128129, 0.4868645])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_ratio_same_histogram(self):
        """test if ratio is 1 for equal histograms (with norm)"""
        hist_1 = histogram([1, 1, 1, 2, 2])
        hist_2 = histogram([1, 1, 1, 2, 2])
        binning = np.array([1, 2, 3])
        hist_plot = histogram_plot(binning=binning, norm=True)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.ones(3)
        expected_ratio_unc = np.array([0.81649658, 0.81649658, 1])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])


class histogram_plot_TestCase(unittest.TestCase):
    """Test class for umami.plotting.histogram_plot"""

    def setUp(self):

        np.random.seed(42)
        n_random = 10_000
        self.hist_1 = histogram(
            np.random.normal(size=n_random), label=f"N={n_random:_}"
        )
        self.hist_2 = histogram(
            np.random.normal(size=2 * n_random), label=f"N={2*n_random:_}"
        )

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )

    def test_output_ratio(self):
        """check with a plot if the ratio is the expected value"""
        hist_plot = histogram_plot(norm=False, ymax_ratio_1=4, figsize=(6.5, 5))
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.add(self.hist_2)
        hist_plot.draw()
        hist_plot.axis_ratio_1.axhline(2, color="r", label="Expected ratio")
        hist_plot.axis_ratio_1.legend(frameon=False)

        plotname = "test_histogram_ratio_value.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_empty_histogram_norm(self):
        hist_plot = histogram_plot(
            norm=True,
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty\n"
                "(+ normalised)"
            ),
        )
        hist_plot.add(histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        plotname = "test_histogram_empty_reference_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_empty_histogram_no_norm(self):
        hist_plot = histogram_plot(
            norm=False,
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty"
            ),
        )
        hist_plot.add(histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        plotname = "test_histogram_empty_reference_no_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_different_range_histogram(self):
        hist_plot = histogram_plot(
            atlas_second_tag=(
                "Test ratio for the case of different histogram ranges. \n"
            ),
            xlabel="x",
            figsize=(7, 6),
            leg_loc="upper right",
            y_scale=1.5,
        )
        np.random.seed(42)
        n_random = 10_000
        x1 = np.concatenate(
            (
                np.random.uniform(-2, 0, n_random),
                np.random.uniform(0.5, 0.99, int(0.5 * n_random)),
            )
        )
        x2 = np.random.uniform(0, 2, n_random)
        x3 = np.random.uniform(-1, 1, n_random)
        hist_plot.add(
            histogram(x1, label="uniform [-2, 0] and uniform [0.5, 1] \n(reference)"),
            reference=True,
        )
        hist_plot.add(histogram(x2, label="uniform [0, 2]"))
        hist_plot.add(histogram(x3, label="uniform [-1, 1]"))
        hist_plot.draw()

        plotname = "test_histogram_different_ranges.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
