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
        self.df_bjets = pd.DataFrame(np.random.normal(300000, 30000, 10000),
                                     np.random.normal(1.25, 1, 10000),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        self.df_cjets = pd.DataFrame(np.random.normal(280000, 30000, 10000),
                                     np.random.normal(1.4, 1, 10000),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        self.df_ujets = pd.DataFrame(np.random.normal(250000, 30000, 10000),
                                     np.random.normal(1.0, 1, 10000),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])

    def test_zero_case(self):
        down_s = DownSampling(self.df_bjets, self.df_cjets, self.df_ujets)
        b_ind, c_ind, u_ind = down_s.Apply()
