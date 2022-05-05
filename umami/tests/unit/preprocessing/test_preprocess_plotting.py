import os
import tempfile
import unittest
from subprocess import run

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
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")

        # Load test preprocessing config file from current directory
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
        )
        self.config = upt.Configuration(self.config_file)

        # Set random seed for reproducibility
        np.random.seed(1)

        # Number of bins in the two variables (eta and pT usually)
        self.nbins_pT = 199
        self.nbins_eta = 25

        # Intialise some random data
        self.histo_dict = {
            "ujets": np.abs(
                np.random.normal(loc=1, size=(self.nbins_pT, self.nbins_eta))
            ),
            "cjets": np.abs(
                np.random.normal(loc=1, size=(self.nbins_pT, self.nbins_eta))
            ),
            "bjets": np.abs(
                np.random.normal(loc=1, size=(self.nbins_pT, self.nbins_eta))
            ),
        }

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
        dict_nothing = {"not_in_dict": "_"}

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

        # Test the case with a key which is not in the global_config
        self.assertRaises(KeyError, upt.utils.generate_process_tag, dict_nothing.keys())

    def test_resampling_plot_default(self):
        """
        Testing the ResamplingPlots function with default parameters
        """

        # TODO: This is not the default value, write unit test for hist_input=False
        # Produce the pT and eta plots
        upt.utils.ResamplingPlots(
            concat_samples=self.histo_dict,
            plot_base_name=self.actual_plots_dir,
            hist_input=True,
            second_tag=upt.utils.generate_process_tag(
                self.config.preparation["ntuples"].keys()
            ),
            fileformat="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "default_pT.png",
                self.actual_plots_dir + "pT.png",
                tol=1,
            ),
        )

    def test_resampling_plot(self):
        """
        Testing the ResamplingPlots function given specific parameters
        """

        # Produce the pT and eta plots
        upt.utils.ResamplingPlots(
            concat_samples=self.histo_dict,
            positions_x_y=[0, 1],
            variable_names=["pT", "eta"],
            plot_base_name=self.actual_plots_dir,
            binning={
                "pT": np.linspace(-2, 2, self.nbins_pT + 1),
                "eta": np.linspace(-2, 2, self.nbins_eta + 1),
            },
            Log=False,
            after_sampling=True,
            normalised=True,
            hist_input=True,
            second_tag=upt.utils.generate_process_tag(
                self.config.preparation["ntuples"].keys()
            ),
            fileformat="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "pT.png",
                self.actual_plots_dir + "pT.png",
                tol=1,
            ),
        )


class preprocessing_plots_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Get dataset for testing.
        """
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")

        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")

        run(
            [
                "wget",
                os.path.join(
                    "https://umami-ci-provider.web.cern.ch/",
                    "preprocessing",
                    "ci_preprocessing_plotting.h5",
                ),
                "--directory-prefix",
                self.actual_plots_dir,
            ],
            check=True,
        )

    def test_preprocessing_plots(self):
        upt.preprocessing_plots(
            sample=os.path.join(
                self.actual_plots_dir,
                "ci_preprocessing_plotting.h5",
            ),
            var_dict={
                "train_variables": {"JetKinematics": ["absEta_btagJes", "pt_btagJes"]},
                "track_train_variables": {
                    "tracks": {
                        "noNormVars": [],
                        "logNormVars": [],
                        "jointNormVars": ["d0", "z0SinTheta"],
                    },
                    "tracks_loose": {
                        "noNormVars": [],
                        "logNormVars": [],
                        "jointNormVars": ["d0", "z0SinTheta"],
                    },
                },
            },
            class_labels=["ujets", "cjets", "bjets"],
            plots_dir=self.actual_plots_dir,
            track_collection_list=["tracks", "tracks_loose"],
            fileformat="png",
        )

        # Check plots
        for var in [
            "absEta_btagJes",
            "pt_btagJes",
            "tracks/d0",
            "tracks/z0SinTheta",
            "tracks_loose/d0",
            "tracks_loose/z0SinTheta",
        ]:
            self.assertEqual(
                None,
                compare_images(
                    self.expected_plots_dir + f"{var}.png",
                    self.actual_plots_dir + f"{var}.png",
                    tol=1,
                ),
            )
