import unittest
import numpy as np
import pandas as pd
from umami.preprocessing import DownSampling


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
        b_ind, c_ind, u_ind = down_s.Apply()
        assert len(b_ind) == len(df_zeros)
