import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images

import umami.preprocessing_tools as upt
from umami.configuration import logger, set_log_level

set_log_level(logger, "DEBUG")


class PreprocessPlotting_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")

        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_plot_dir = f"{self.tmp_dir.name}/"
        self.control_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")

        # Load test preprocessing config file from current directory
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
        )
        self.config = upt.Configuration(self.config_file)

    def test_generate_process_tag(self):
        """
        Test the function which creates the second tag for plots (including
        the used processes (like ttbar, Z', Z+jets, ...)
        """
        expected_label_ttbar_only = r"$\sqrt{s}$ = 13 TeV, $t\bar{t}$ PFlow Jets"
        expected_label_ttbar_zprime = (
            r"$\sqrt{s}$ = 13 TeV, Combined $t\bar{t}$ + $Z'$ PFlow Jets"
        )
        expected_label_ttbar_zprime_zjets = (
            r"$\sqrt{s}$ = 13 TeV, Combined $t\bar{t}$ + $Z'$ + $Z$+jets PFlow Jets"
        )

        dict_ttbar_zprime = {"ttbar": "_", "zprime": "_"}
        dict_ttbar_zprime_zjets = {"ttbar": "_", "zprime": "_", "zjets": "_"}

        # Test the case where we have only one entry preparation.ntuples
        # in the preprocessing config.
        # This test is done using the test config.
        self.assertEqual(
            expected_label_ttbar_only,
            upt.utils.generate_process_tag(self.config.preparation["ntuples"].keys()),
        )
        # Test the case with two processes (using a dict defined above)
        self.assertEqual(
            expected_label_ttbar_zprime,
            upt.utils.generate_process_tag(dict_ttbar_zprime.keys()),
        )
        # Test the case with three processes (using a dict defined above)
        self.assertEqual(
            expected_label_ttbar_zprime_zjets,
            upt.utils.generate_process_tag(dict_ttbar_zprime_zjets.keys()),
        )

    def test_resampling_plot(self):
        """
        Testing the ResamplingPlots function
        """

        # Set random seed for reproducibility
        np.random.seed(1)

        # Number of bins in the two variables (eta and pT usually)
        nbins_pT = 20
        nbins_eta = 20

        # Intialise some random data
        histo_dict = {
            "ujets": np.abs(np.random.normal(loc=1, size=(nbins_pT, nbins_eta))),
            "cjets": np.abs(np.random.normal(loc=1, size=(nbins_pT, nbins_eta))),
            "bjets": np.abs(np.random.normal(loc=1, size=(nbins_pT, nbins_eta))),
        }

        # Produce the pT and eta plots
        upt.utils.ResamplingPlots(
            concat_samples=histo_dict,
            positions_x_y=[0, 1],
            variable_names=["pT", "eta"],
            plot_base_name=self.tmp_plot_dir,
            # plot_base_name=self.control_plots_dir,
            binning={
                "pT": np.linspace(-2, 2, nbins_pT + 1),
                "eta": np.linspace(-2, 2, nbins_eta + 1),
            },
            Log=True,
            hist_input=True,
            second_tag=upt.utils.generate_process_tag(
                self.config.preparation["ntuples"].keys()
            ),
            fileformat="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.control_plots_dir + "pT.png",
                self.tmp_plot_dir + "pT.png",
                tol=1,
            ),
        )
