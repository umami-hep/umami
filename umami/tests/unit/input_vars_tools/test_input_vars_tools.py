import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import yaml
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
from umami.input_vars_tools.plotting_functions import (
    plot_input_vars_jets,
    plot_input_vars_trks,
    plot_n_tracks_per_jet,
)
from umami.tools import yaml_loader

set_log_level(logger, "DEBUG")


class JetPlotting_TestCase(unittest.TestCase):
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
        self.data_url = "https://umami-ci-provider.web.cern.ch/plot_input_vars/"

        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")

        self.yaml_file = os.path.join(
            os.path.dirname(__file__), "fixtures/plot_input_variables.yaml"
        )

        self.r21_url = os.path.join(self.data_url, "plot_input_vars_r21_check.h5")

        self.r22_url = os.path.join(self.data_url, "plot_input_vars_r22_check.h5")

        self.r21_test_file = os.path.join(
            self.actual_plots_dir, "plot_input_vars_r21_check.h5"
        )

        self.r22_test_file = os.path.join(
            self.actual_plots_dir, "plot_input_vars_r22_check.h5"
        )

        with open(self.yaml_file) as yaml_config:
            self.plot_config = yaml.load(yaml_config, Loader=yaml_loader)

        run(["wget", self.r21_url, "--directory-prefix", self.actual_plots_dir])
        run(["wget", self.r22_url, "--directory-prefix", self.actual_plots_dir])

    def test_plot_input_vars_jets_wrong_type(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["binning"]["IP2D_bu"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                class_labels=plotting_config["class_labels"],
                var_dict=self.plot_config["Eval_parameters"]["var_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                binning=plotting_config["binning"],
                output_directory=f"{self.actual_plots_dir}"
                + plotting_config["folder_to_save"],
                plot_type="png",
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_jets(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]

        plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=f"{self.actual_plots_dir}"
            + plotting_config["folder_to_save"],
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "jets_input_vars/" + "IP2D_bu.png",
                self.actual_plots_dir + "jets_input_vars/" + "IP2D_bu.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "jets_input_vars/" + "IP2D_cu.png",
                self.actual_plots_dir + "jets_input_vars/" + "IP2D_cu.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_jets_comparison_wrong_type(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["binning"]["IP2D_bu"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                class_labels=plotting_config["class_labels"],
                var_dict=self.plot_config["Eval_parameters"]["var_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                binning=plotting_config["binning"],
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_jets_comparison(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=os.path.join(self.actual_plots_dir, "comp/"),
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "comp/" + "IP2D_bu.png",
                self.actual_plots_dir + "comp/" + "IP2D_bu.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "comp/" + "IP2D_cu.png",
                self.actual_plots_dir + "comp/" + "IP2D_cu.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_trks_wrong_type(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["binning"]["dr"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_trks(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_track_names=tracks_name_list,
                class_labels=plotting_config["class_labels"],
                var_dict=self.plot_config["Eval_parameters"]["var_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                binning=plotting_config["binning"],
                output_directory=f"{self.actual_plots_dir}",
                plot_type="png",
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_trks(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_track_names=tracks_name_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=f"{self.actual_plots_dir}",
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "ptfrac/All/" + "dr_None_All.png",
                self.actual_plots_dir + "ptfrac/All/" + "dr_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                self.actual_plots_dir
                + "ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None_All.png",
                self.actual_plots_dir
                + "ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "ptfrac/0/" + "dr_0_All.png",
                self.actual_plots_dir + "ptfrac/0/" + "dr_0_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "ptfrac/0/"
                + "IP3D_signed_d0_significance_0_All.png",
                self.actual_plots_dir
                + "ptfrac/0/"
                + "IP3D_signed_d0_significance_0_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0_All.png",
                self.actual_plots_dir
                + "ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0_All.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_trks_comparison_wrong_type(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["binning"]["dr"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_trks(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_track_names=tracks_name_list,
                class_labels=plotting_config["class_labels"],
                var_dict=self.plot_config["Eval_parameters"]["var_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                binning=plotting_config["binning"],
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_trks_comparison_not_normalised(self):
        plotting_config = self.plot_config["tracks_test_not_normalised"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        tracks_name_list = ["tracks", "tracks"]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_track_names=tracks_name_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=os.path.join(self.actual_plots_dir, "comp_no_norm/"),
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "comp_no_norm/ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                self.actual_plots_dir
                + "comp_no_norm/ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_trks_comparison(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        tracks_name_list = ["tracks", "tracks"]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_track_names=tracks_name_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=os.path.join(self.actual_plots_dir, "comp/"),
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "comp/ptfrac/All/" + "dr_None_All.png",
                self.actual_plots_dir + "comp/ptfrac/All/" + "dr_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "comp/ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                self.actual_plots_dir
                + "comp/ptfrac/All/"
                + "IP3D_signed_d0_significance_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "comp/ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None_All.png",
                self.actual_plots_dir
                + "comp/ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "comp/ptfrac/0/" + "dr_0_All.png",
                self.actual_plots_dir + "comp/ptfrac/0/" + "dr_0_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "comp/ptfrac/0/"
                + "IP3D_signed_d0_significance_0_All.png",
                self.actual_plots_dir
                + "comp/ptfrac/0/"
                + "IP3D_signed_d0_significance_0_All.png",
                tol=1,
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir
                + "comp/ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0_All.png",
                self.actual_plots_dir
                + "comp/ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0_All.png",
                tol=1,
            ),
        )

    def test_plot_n_tracks_per_jet(self):
        plotting_config = self.plot_config["nTracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        tracks_name_list = ["tracks", "tracks"]
        labels_list = ["R21 Test", "R22 Test"]

        plot_n_tracks_per_jet(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_track_names=tracks_name_list,
            class_labels=plotting_config["class_labels"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            output_directory=f"{self.actual_plots_dir}",
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "nTracks_per_Jet_All.png",
                self.actual_plots_dir + "nTracks_per_Jet_All.png",
                tol=1,
            ),
        )
