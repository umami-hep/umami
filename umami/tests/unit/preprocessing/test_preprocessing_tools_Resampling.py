import os
import unittest

import numpy as np
import pandas as pd

from umami.configuration import global_config
from umami.preprocessing_tools import (  # UndersamplingGenerator,
    CalculateBinning,
    Configuration,
    CorrectFractions,
    UnderSampling,
    UnderSamplingTemplate,
)


class CorrectFractionsTestCase(unittest.TestCase):
    """
    Test the implementation of the CorrectFractions function.
    """

    def test_CorrectFractions_zero_length(self):
        with self.assertRaises(ValueError):
            CorrectFractions([], [])

    def test_CorrectFractions_different_input_lengths(self):
        with self.assertRaises(AssertionError):
            CorrectFractions([1, 2, 3, 4], [0.2, 0.8, 0.0])

    def test_CorrectFractions_not_fraction_sum_one(self):
        with self.assertRaises(ValueError):
            CorrectFractions([1, 2, 3, 4], [0.2, 0.5, 0.2, 0.2])

    def test_CorrectFractions_different_input_length_class_names(self):
        with self.assertRaises(AssertionError):
            CorrectFractions(
                [5000, 6000, 3000], [0.2, 0.6, 0.2], ["Zjets", "ttbar"]
            )

    def test_CorrectFractions_zero_Njets(self):
        with self.assertRaises(ValueError):
            CorrectFractions([0, 6000, 3000], [0.2, 0.6, 0.2])

    def test_CorrectFractions_twice_same_fractions(self):
        self.assertListEqual(
            list(CorrectFractions([1000, 6000, 3000], [0.2, 0.6, 0.2])),
            [1000, 3000, 1000],
        )

    def test_CorrectFractions_input_correct_fractions(self):
        N_jets = [2000, 6000, 2000]
        self.assertListEqual(
            list(CorrectFractions(N_jets, [0.2, 0.6, 0.2])), N_jets
        )

    def test_CorrectFractions_scaling_down_largest(self):
        self.assertListEqual(
            list(CorrectFractions([3000, 6000, 3000], [0.3, 0.5, 0.2])),
            [3000, 5000, 2000],
        )

    def test_CorrectFractions_scaling_down_small(self):
        self.assertListEqual(
            list(CorrectFractions([10000, 6000, 7000], [0.4, 0.5, 0.1])),
            [4800, 6000, 1200],
        )


class CalculateBinningTestCase(unittest.TestCase):
    """
    Test the implementation of the CalculateBinning function.
    """

    def test_NonListCase(self):
        with self.assertRaises(TypeError):
            CalculateBinning(1)

    def test_SingleListCase(self):
        np.testing.assert_array_equal(
            CalculateBinning([1, 2, 3]), np.linspace(1, 2, 3)
        )

    def test_NestedListCase(self):
        bins = [[1, 2, 3], [3, 4, 5]]
        expected_outcome = np.concatenate(
            [np.linspace(*elem) for elem in bins]
        )
        np.testing.assert_array_equal(CalculateBinning(bins), expected_outcome)


class UndersamplingGeneratorTestCase(unittest.TestCase):
    """
    Test the implementation of the UndersamplingGenerator function.
    """

    pass


class ResamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the Resampling base class.
    """

    pass


class UnderSamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the UnderSampling class.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        self.config_file = os.path.join(
            os.path.dirname(__file__), "test_preprocess_config.yaml"
        )
        self.config = Configuration(self.config_file)
        self.sampling_config = self.config.sampling
        self.samples_config = (self.config.preparation).get("samples")

        self.df_bjets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(300000, 30000, 10000)
                ),
                global_config.etavariable: abs(
                    np.random.normal(1.25, 1, 10000)
                ),
            }
        )
        self.df_cjets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(280000, 28000, 10000)
                ),
                global_config.etavariable: abs(
                    np.random.normal(1.4, 1, 10000)
                ),
            }
        )
        self.df_ujets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(250000, 25000, 10000)
                ),
                global_config.etavariable: abs(
                    np.random.normal(1.0, 1, 10000)
                ),
            }
        )

    def test_CountNoSamplesDefined(self):
        del self.sampling_config["options"]["samples"]
        us = UnderSampling(self.config)
        with self.assertRaises(KeyError):
            us.InitialiseSamples()

    def test_DifferentSamplesPerCategory(self):
        del self.sampling_config["options"]["samples"]["zprime"][1]
        us = UnderSampling(self.config)
        with self.assertRaises(RuntimeError):
            us.InitialiseSamples()


# TODO: this can be used to extend the UnderSamplingTestCase
# class UnderSamplingOldTestCase(unittest.TestCase):
#     """
#     Test the implementation of the UnderSampling class.
#     """

#     def setUp(self):
#         """
#         Create a default dataset for testing.
#         """
#         self.df_bjets = pd.DataFrame(
#             {
#                 global_config.pTvariable: abs(
#                     np.random.normal(300000, 30000, 10000)
#                 ),
#                 global_config.etavariable: abs(
#                     np.random.normal(1.25, 1, 10000)
#                 ),
#             }
#         )
#         self.df_cjets = pd.DataFrame(
#             {
#                 global_config.pTvariable: abs(
#                     np.random.normal(280000, 28000, 10000)
#                 ),
#                 global_config.etavariable: abs(
#                     np.random.normal(1.4, 1, 10000)
#                 ),
#             }
#         )
#         self.df_ujets = pd.DataFrame(
#             {
#                 global_config.pTvariable: abs(
#                     np.random.normal(250000, 25000, 10000)
#                 ),
#                 global_config.etavariable: abs(
#                     np.random.normal(1.0, 1, 10000)
#                 ),
#             }
#         )

#     def test_zero_case(self):
#         df_zeros = pd.DataFrame(
#             np.zeros((1000, 2)),
#             columns=[global_config.pTvariable, global_config.etavariable],
#         )
#         down_s = UnderSampling(df_zeros, df_zeros, df_zeros)
#         b_ind, c_ind, u_ind, _ = down_s.GetIndices()
#         self.assertEqual(len(b_ind), len(df_zeros))

#     def test_underflow(self):
#         df_minus_ones = pd.DataFrame(
#             -1 * np.ones((1000, 2)),
#             columns=[global_config.pTvariable, global_config.etavariable],
#         )
#         down_s = UnderSampling(df_minus_ones, df_minus_ones, df_minus_ones)
#         b_ind, c_ind, u_ind, _ = down_s.GetIndices()
#         self.assertEqual(b_ind.size, 0)
#         self.assertEqual(c_ind.size, 0)
#         self.assertEqual(u_ind.size, 0)

#     def test_overflow(self):
#         df_large = pd.DataFrame(
#             1e10 * np.ones((1000, 2)),
#             columns=[global_config.pTvariable, global_config.etavariable],
#         )
#         down_s = UnderSampling(df_large, df_large, df_large)
#         b_ind, c_ind, u_ind, _ = down_s.GetIndices()
#         self.assertEqual(b_ind.size, 0)
#         self.assertEqual(c_ind.size, 0)
#         self.assertEqual(u_ind.size, 0)

#     def test_equal_length(self):
#         down_s = UnderSampling(self.df_bjets, self.df_cjets, self.df_ujets)
#         b_ind, c_ind, u_ind, _ = down_s.GetIndices()
#         self.assertEqual(len(b_ind), len(c_ind))
#         self.assertEqual(len(b_ind), len(u_ind))


class PDFResamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the PDFResampling base class.
    """

    pass


class UnderSamplingTemplateTestCase(unittest.TestCase):
    """
    Test the implementation of the UnderSamplingTemplate class.
    """

    def setUp(self):
        """
        Create a default dataset for testing with c-jets being the lowest distribution
        """
        self.df_bjets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(300000, 30000, 10000)
                ),
                global_config.etavariable: abs(
                    np.random.normal(1.25, 1, 10000)
                ),
            }
        )
        self.df_cjets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(180000, 18000, 5000)
                ),
                global_config.etavariable: abs(np.random.normal(1.4, 1, 5000)),
            }
        )
        self.df_ujets = pd.DataFrame(
            {
                global_config.pTvariable: abs(
                    np.random.normal(250000, 25000, 10000)
                ),
                global_config.etavariable: abs(
                    np.random.normal(1.0, 1, 10000)
                ),
            }
        )

    def test_equal_length(self):
        down_s = UnderSamplingTemplate(
            self.df_bjets, self.df_cjets, self.df_ujets, count=True
        )
        b_indices, c_indices, u_indices, _ = down_s.GetIndices()
        self.assertEqual(len(b_indices), len(c_indices))
        self.assertEqual(len(b_indices), len(u_indices))

    def test_zero_case(self):
        df_zeros = pd.DataFrame(
            np.zeros((1000, 2)),
            columns=[global_config.pTvariable, global_config.etavariable],
        )
        down_s = UnderSamplingTemplate(
            df_zeros, df_zeros, df_zeros, count=True
        )
        b_ind, c_ind, u_ind, _ = down_s.GetIndices()
        self.assertEqual(len(b_ind), len(df_zeros))

    def test_overflow(self):
        df_large = pd.DataFrame(
            1e10 * np.ones((1000, 2)),
            columns=[global_config.pTvariable, global_config.etavariable],
        )
        down_s = UnderSamplingTemplate(
            df_large, df_large, df_large, count=True
        )
        b_ind, c_ind, u_ind, _ = down_s.GetIndices()
        self.assertEqual(b_ind.size, 0)
        self.assertEqual(c_ind.size, 0)
        self.assertEqual(u_ind.size, 0)

    def test_underflow(self):
        df_minus_ones = pd.DataFrame(
            -1 * np.ones((1000, 2)),
            columns=[global_config.pTvariable, global_config.etavariable],
        )
        down_s = UnderSamplingTemplate(
            df_minus_ones, df_minus_ones, df_minus_ones, count=True
        )
        b_ind, c_ind, u_ind, _ = down_s.GetIndices()
        self.assertEqual(b_ind.size, 0)
        self.assertEqual(c_ind.size, 0)
        self.assertEqual(u_ind.size, 0)
