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
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow jets,\nResampled $t\\bar{t}$ training sample"
            ),
            logy=True,
            ylabel="Normalised number of jets",
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


class plot_resampling_variables_TestCase(unittest.TestCase):
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

        # Create dummy data
        self.concat_samples = {
            "bjets": {
                "jets": np.c_[
                    np.linspace(1000, 10000, 3000),
                    np.linspace(0, 2.5, 3000),
                    np.arange(0, 3000, 1),
                    np.hstack((np.zeros(1500), np.ones(1500))),
                    np.hstack((np.zeros(1500), np.ones(1500))),
                ]
            },
            "cjets": {
                "jets": np.c_[
                    np.linspace(1000, 10000, 2000),
                    np.linspace(0, 2.5, 2000),
                    np.arange(0, 2000, 1),
                    np.hstack((np.zeros(1000), np.ones(1000))),
                    np.hstack((np.zeros(1000), np.ones(1000))),
                ]
            },
            "ujets": {
                "jets": np.c_[
                    np.linspace(1000, 10000, 1000),
                    np.linspace(0, 2.5, 1000),
                    np.arange(0, 1000, 1),
                    np.hstack((np.zeros(500), np.ones(500))),
                    np.hstack((np.zeros(500), np.ones(500))),
                ]
            },
        }

        self.variables = ["pT", "eta"]

    def test_plot_resampling_variables(self):
        """Test the plot_resampling_variables"""

        upt.plot_resampling_variables(
            concat_samples=self.concat_samples,
            var_positions=[0, 1],
            variable_names=self.variables,
            sample_categories=["ttbar", "zprime"],
            output_dir=self.actual_plots_dir,
            bins_dict={
                "pT": 200,
                "eta": 20,
            },
            fileformat="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            logy=False,
            ylabel="Normalised number of jets",
        )

        for var in self.variables:
            self.assertEqual(
                None,
                compare_images(
                    self.expected_plots_dir + f"{var}_before_resampling.png",
                    self.actual_plots_dir + f"{var}_before_resampling.png",
                    tol=1,
                ),
            )

    def test_plot_resampling_variables_wrong_binning(self):
        """Test the plot_resampling_variables wrong binning error"""

        with self.assertRaises(ValueError):
            upt.plot_resampling_variables(
                concat_samples=self.concat_samples,
                var_positions=[0, 1],
                variable_names=self.variables,
                sample_categories=["ttbar", "zprime"],
                output_dir=self.actual_plots_dir,
                bins_dict={
                    "pT": [1, 2],
                    "eta": 20,
                },
                fileformat="png",
                atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
                logy=False,
                ylabel="Normalised number of jets",
            )
