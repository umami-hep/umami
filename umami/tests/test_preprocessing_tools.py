import unittest
import numpy as np
import pandas as pd
import os
from umami.preprocessing_tools import UnderSampling, Configuration
from umami.preprocessing_tools import GetNJetsPerIteration, GetCuts, GetScales


class UnderSamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the UnderSampling class.
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
        down_s = UnderSampling(df_zeros, df_zeros, df_zeros)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        self.assertEqual(len(b_ind), len(df_zeros))

    def test_underflow(self):
        df_minus_ones = pd.DataFrame(-1 * np.ones((1000, 2)),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        down_s = UnderSampling(df_minus_ones, df_minus_ones, df_minus_ones)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        self.assertEqual(b_ind.size, 0)
        self.assertEqual(c_ind.size, 0)
        self.assertEqual(u_ind.size, 0)

    def test_overflow(self):
        df_minus_ones = pd.DataFrame(1e10 * np.ones((1000, 2)),
                                     columns=["pt_uncalib", "abs_eta_uncalib"])
        down_s = UnderSampling(df_minus_ones, df_minus_ones, df_minus_ones)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        self.assertEqual(b_ind.size, 0)
        self.assertEqual(c_ind.size, 0)
        self.assertEqual(u_ind.size, 0)

    def test_equal_length(self):
        down_s = UnderSampling(self.df_bjets, self.df_cjets, self.df_ujets)
        b_ind, c_ind, u_ind = down_s.GetIndices()
        self.assertEqual(len(b_ind), len(c_ind))
        self.assertEqual(len(b_ind), len(u_ind))


class ConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = os.path.join(os.path.dirname(__file__),
                                        "test_preprocess_config.yaml")

    def test_missing_key_error(self):
        config = Configuration(self.config_file)
        del config.config["outfile_name"]
        with self.assertRaises(KeyError):
            config.GetConfiguration()

    def test_missing_key_warning(self):
        config = Configuration(self.config_file)
        del config.config["pT_max"]
        with self.assertWarns(Warning):
            config.GetConfiguration()

    def test_GetFileName_no_input(self):
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(out_file, config.outfile_name)

    def test_GetFileName_no_iterations(self):
        config = Configuration(self.config_file)
        self.assertNotIn("test", config.outfile_name)
        out_file = config.GetFileName(option="test")
        self.assertIn("test", out_file)

    def test_GetFileName_no_iterations_no_input(self):
        config = Configuration(self.config_file)
        out_file = config.GetFileName()
        self.assertEqual(config.outfile_name, out_file)


class GetNJetsPerIterationTestCase(unittest.TestCase):
    """
    Test the implementation of the GetNJetsPerIteration function.
    """
    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = os.path.join(os.path.dirname(__file__),
                                        "test_preprocess_config.yaml")

    def test_zero_case(self):
        config = Configuration(self.config_file)
        config.ttbar_frac = 0
        N_list = GetNJetsPerIteration(config)
        self.assertEqual(N_list[-1]['nbjets'], 0)
        self.assertEqual(N_list[-1]['ncjets'], 0)
        self.assertEqual(N_list[-1]['nujets'], 0)

    def test_one_case(self):
        config = Configuration(self.config_file)
        config.ttbar_frac = 1
        N_list = GetNJetsPerIteration(config)
        self.assertEqual(N_list[-1]['nZ'], 0)
        self.assertGreater(N_list[-1]['nbjets'], config.njets * 0.9)

    def test_one_iteration(self):
        config = Configuration(self.config_file)
        config.iterations = 1
        config.njets = 1e6
        N_list = GetNJetsPerIteration(config)
        self.assertEqual(N_list[-1]['nbjets'], 1e6)

    def test_no_iteration(self):
        config = Configuration(self.config_file)
        config.iterations = 0
        config.njets = 1e6
        with self.assertRaises(ValueError):
            GetNJetsPerIteration(config)


class PreprocessingTestCuts(unittest.TestCase):
    """
    Test the implementation of the Prerocessing cut application.
    """

    def setUp(self):
        self.config_file = os.path.join(os.path.dirname(__file__),
                                        "test_preprocess_config.yaml")
        self.config = Configuration(self.config_file)
        self.jets = pd.DataFrame({"secondaryVtx_m": [2e3, 2.6e4, 2.7e4, 2.4e4,
                                                     np.nan, np.nan, 25, 30e4,
                                                     np.nan, 0],
                                  "secondaryVtx_E": [2001, 26001, 1e9, 1.5e8,
                                                     5, np.nan, np.nan, np.nan,
                                                     4e8, 0],
                                  "HadronConeExclTruthLabelID": [5, 5, 5, 5,
                                                                 4, 4, 4,
                                                                 0, 0, 0],
                                  "GhostBHadronsFinalPt": 5e3 * np.ones(10),
                                  "pt_uncalib": 5.2e3 * np.ones(10),
                                  })
        self.pass_ttbar = np.array([1., 0., 0., 0., 1., 1., 1., 0., 0., 1.])

    def test_cuts_passing_ttbar(self):
        indices_to_remove = GetCuts(self.jets.to_records(index=False),
                                    self.config)
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))

    def test_cuts_passing_Zprime_all_false(self):
        pass_Zprime = np.zeros(len(self.jets))
        indices_to_remove = GetCuts(self.jets.to_records(index=False),
                                    self.config, sample='Zprime')
        cut_result = np.ones(len(self.jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, pass_Zprime))

    def test_cuts_passing_Zprime_inverted_pt(self):
        jets = self.jets.copy()
        jets["GhostBHadronsFinalPt"] *= 1e2
        jets["pt_uncalib"] *= 1e2
        indices_to_remove = GetCuts(jets.to_records(index=False),
                                    self.config, sample='Zprime')
        cut_result = np.ones(len(jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, self.pass_ttbar))

    def test_cuts_exceed_pTmax_Zprime(self):
        jets = self.jets.copy()
        jets["GhostBHadronsFinalPt"] *= 1e5
        jets["pt_uncalib"] *= 1e5
        indices_to_remove = GetCuts(jets.to_records(index=False),
                                    self.config, sample='Zprime')
        cut_result = np.ones(len(jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, np.zeros(len(jets))))

    def test_cuts_exceed_pTmax_ttbar(self):
        jets = self.jets.copy()
        jets["GhostBHadronsFinalPt"] *= 1e5
        jets["pt_uncalib"] *= 1e5
        indices_to_remove = GetCuts(jets.to_records(index=False),
                                    self.config, sample='ttbar')
        cut_result = np.ones(len(jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, np.zeros(len(jets))))

    def test_cuts_bjets_exceed_pt_ttbar(self):
        jets = pd.DataFrame({"secondaryVtx_m": [0, 0, 0],
                             "secondaryVtx_E": [0, 0, 0],
                             "HadronConeExclTruthLabelID": [5, 4, 0],
                             "GhostBHadronsFinalPt": 5e5 * np.ones(3),
                             "pt_uncalib": 5e3 * np.ones(3)
                             })
        indices_to_remove = GetCuts(jets.to_records(index=False),
                                    self.config, sample='ttbar')
        cut_result = np.ones(len(jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, np.array([0, 1, 1])))

    def test_cuts_bjets_pass_pt_Zprime(self):
        jets = pd.DataFrame({"secondaryVtx_m": [0, 0, 0],
                             "secondaryVtx_E": [0, 0, 0],
                             "HadronConeExclTruthLabelID": [5, 4, 0],
                             "GhostBHadronsFinalPt": 5e5 * np.ones(3),
                             "pt_uncalib": 5e3 * np.ones(3)
                             })
        indices_to_remove = GetCuts(jets.to_records(index=False),
                                    self.config, sample='Zprime')
        cut_result = np.ones(len(jets))
        np.put(cut_result, indices_to_remove, 0)
        self.assertTrue(np.array_equal(cut_result, np.array([1, 0, 0])))


class GetScalesTestCase(unittest.TestCase):
    """
    Test the implementation of the GetScales class.
    """

    def setUp(self):
        self.arr_0 = np.zeros(500)
        self.arr_1 = np.ones(500)

    def test_ZeroCase(self):
        varname, average, std, default = GetScales(
            self.arr_0, self.arr_1, "zeros", {})
        self.assertEqual(average, 0)
        self.assertEqual(std, 0)
        self.assertEqual(default, 0)

    def test_ReturnVarname(self):
        varname, _, _, _ = GetScales(self.arr_0, self.arr_1, "zeros", {})
        self.assertEqual(varname, "zeros")

    def test_WeightZero(self):
        with self.assertRaises(ValueError):
            varname, average, std, default = GetScales(
                self.arr_1,  self.arr_0, "zeros", {})

    def test_OneCase(self):
        varname, average, std, default = GetScales(
            self.arr_1, self.arr_1, "ones", {})
        self.assertEqual(average, 1)
        self.assertEqual(std, 0)
        self.assertEqual(default, 1)

