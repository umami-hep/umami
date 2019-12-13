import unittest
import numpy as np
import pandas as pd
import os
from umami.preprocessing import DownSampling, Configuration


class DownSamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the DownSampling class.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        self.df_bjets = pd.DataFrame(
            {"pt_uncalib": abs(np.random.normal(300000, 30000, 10000)),
             "abs_eta_uncalib": abs(np.random.normal(1.25, 1, 10000))})
        self.df_cjets = pd.DataFrame(
            {"pt_uncalib": abs(np.random.normal(280000, 28000, 10000)),
             "abs_eta_uncalib": abs(np.random.normal(1.4, 1, 10000))})
        self.df_ujets = pd.DataFrame(
            {"pt_uncalib": abs(np.random.normal(250000, 25000, 10000)),
             "abs_eta_uncalib": abs(np.random.normal(1.0, 1, 10000))})

    def test_zero_case(self):
        df_zeros = pd.DataFrame(np.zeros((1000, 2)),
                                columns=["pt_uncalib", "abs_eta_uncalib"])
        down_s = DownSampling(df_zeros, df_zeros, df_zeros)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        assert len(b_ind) == len(df_zeros)

    def test_underflow(self):
        df_minus_ones = pd.DataFrame(-1 * np.ones((1000, 2)),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        down_s = DownSampling(df_minus_ones, df_minus_ones, df_minus_ones)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        assert b_ind.size == 0
        assert c_ind.size == 0
        assert u_ind.size == 0

    def test_overflow(self):
        df_minus_ones = pd.DataFrame(1e10 * np.ones((1000, 2)),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        down_s = DownSampling(df_minus_ones, df_minus_ones, df_minus_ones)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        assert b_ind.size == 0
        assert c_ind.size == 0
        assert u_ind.size == 0

    def test_equal_length(self):
        down_s = DownSampling(self.df_bjets, self.df_cjets, self.df_ujets)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        assert len(b_ind) == len(c_ind)
        assert len(b_ind) == len(u_ind)


class ConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def test_missing_key_error(self):
        config = Configuration(os.path.join(os.path.dirname(__file__),
                                            "test_preprocess_config.yaml"))
        del config.config["outfile_name"]
        with self.assertRaises(KeyError):
            config.GetConfiguration()

    def test_missing_key_warning(self):
        config = Configuration(os.path.join(os.path.dirname(__file__),
                                            "test_preprocess_config.yaml"))
        del config.config["pT_max"]
        with self.assertWarns(Warning):
            config.GetConfiguration()
