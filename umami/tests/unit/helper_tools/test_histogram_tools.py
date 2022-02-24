import unittest

import numpy as np

from umami.configuration import logger, set_log_level
from umami.helper_tools import hist_ratio, hist_w_unc, save_divide

set_log_level(logger, "DEBUG")


class hist_w_unc_TestCase(unittest.TestCase):
    def setUp(self):
        self.bin_edges = np.array([0, 1, 2, 3, 4, 5])
        self.input = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        self.hist_normed = np.array([0, 0.25, 0.25, 0.25, 0.25])
        self.hist = np.array([0, 2, 2, 2, 2])
        self.unc_normed = np.array([0, 0.1767767, 0.1767767, 0.1767767, 0.1767767])
        self.unc = np.array([0.0, 1.4142136, 1.4142136, 1.4142136, 1.4142136])
        self.band_normed = np.array([0, 0.0732233, 0.0732233, 0.0732233, 0.0732233])
        self.band = np.array([0.0, 0.5857864, 0.5857864, 0.5857864, 0.5857864])

    def test_hist_w_unc_zero_case(self):
        bins, hist, unc, band = hist_w_unc(
            a=[],
            bins=[],
        )

        np.testing.assert_almost_equal(bins, [])
        np.testing.assert_almost_equal(hist, [])
        np.testing.assert_almost_equal(unc, [])
        np.testing.assert_almost_equal(band, [])

    def test_hist_w_unc_normed(self):
        bins, hist, unc, band = hist_w_unc(
            a=self.input,
            bins=self.bin_edges,
        )

        np.testing.assert_almost_equal(bins, self.bin_edges)
        np.testing.assert_almost_equal(hist, self.hist_normed)
        np.testing.assert_almost_equal(unc, self.unc_normed)
        np.testing.assert_almost_equal(band, self.band_normed)

    def test_hist_w_unc_not_normed(self):
        bins, hist, unc, band = hist_w_unc(
            a=self.input,
            bins=self.bin_edges,
            normed=False,
        )

        np.testing.assert_almost_equal(bins, self.bin_edges)
        np.testing.assert_almost_equal(hist, self.hist)
        np.testing.assert_almost_equal(unc, self.unc)
        np.testing.assert_almost_equal(band, self.band)


class save_divide_TestCase(unittest.TestCase):
    def test_zero_case(self):
        steps = save_divide(np.zeros(2), np.zeros(2))
        np.testing.assert_equal(steps, np.ones(2))

    def test_ones_case(self):
        steps = save_divide(np.ones(2), np.ones(2))
        np.testing.assert_equal(steps, np.ones(2))

    def test_half_case(self):
        steps = save_divide(np.ones(2), 2 * np.ones(2))
        np.testing.assert_equal(steps, 0.5 * np.ones(2))

    def test_denominator_float(self):
        steps = save_divide(np.ones(2), 2)
        np.testing.assert_equal(steps, 0.5 * np.ones(2))

    def test_numerator_float(self):
        steps = save_divide(1, np.ones(2) * 2)
        np.testing.assert_equal(steps, 0.5 * np.ones(2))


class hist_ratio_TestCase(unittest.TestCase):
    def setUp(self):
        self.numerator = np.array([5, 3, 2, 5, 6, 2])
        self.denominator = np.array([3, 6, 2, 7, 10, 12])
        self.numerator_unc = np.array([0.5, 1, 0.3, 0.2, 0.5, 0.3])
        self.denominator_unc = np.array([1, 0.3, 2, 1, 5, 3])
        self.step = np.array([1.6666667, 1.6666667, 0.5, 1, 0.7142857, 0.6, 0.1666667])
        self.step_unc = np.array(
            [
                0.580017,
                0.580017,
                0.1685312,
                1.0111874,
                0.1059653,
                0.3041381,
                0.0485913,
            ]
        )

    def test_hist_ratio(self):
        step, step_unc = hist_ratio(
            numerator=self.numerator,
            denominator=self.denominator,
            numerator_unc=self.numerator_unc,
            denominator_unc=self.denominator_unc,
        )

        np.testing.assert_almost_equal(step, self.step)
        np.testing.assert_almost_equal(step_unc, self.step_unc)

    def test_hist_not_same_length_nominator_denominator(self):
        with self.assertRaises(AssertionError):
            _, _ = hist_ratio(
                numerator=np.ones(2),
                denominator=np.ones(3),
                numerator_unc=np.ones(3),
                denominator_unc=np.ones(3),
            )

    def test_hist_not_same_length_nomiantor_and_unc(self):
        with self.assertRaises(AssertionError):
            _, _ = hist_ratio(
                numerator=np.ones(3),
                denominator=np.ones(3),
                numerator_unc=np.ones(2),
                denominator_unc=np.ones(3),
            )

    def test_hist_not_same_length_denomiantor_and_unc(self):
        with self.assertRaises(AssertionError):
            _, _ = hist_ratio(
                numerator=np.ones(3),
                denominator=np.ones(3),
                numerator_unc=np.ones(3),
                denominator_unc=np.ones(2),
            )
