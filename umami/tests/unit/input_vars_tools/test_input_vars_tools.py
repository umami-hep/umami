"""Unit tests for input variable plots."""
import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import yaml
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
from umami.input_vars_tools.plotting_functions import (
    get_datasets_configuration,
    plot_input_vars_jets,
    plot_input_vars_trks,
    plot_n_tracks_per_jet,
)
from umami.tools import yaml_loader

set_log_level(logger, "DEBUG")


class HelperFunctionTestCase(unittest.TestCase):
    """Test class for helper functions."""

    def setUp(self):
        self.plotting_config = {
            "class_labels": ["ujets", "cjets", "bjets"],
            "Datasets_to_plot": {
                "ds_1": {
                    "files": "dummy_path_1",
                    "label": "dummy_label_1",
                },
                "ds_2": {
                    "files": "dummy_path_2",
                    "label": "dummy_label_2",
                },
            },
        }

    def test_get_datasets_configuration_all_default(self):
        """Test the helper function for the default case (same class labels for both
        datasets)"""
        exp_filepath_list = ["dummy_path_1", "dummy_path_2"]
        exp_labels_list = ["dummy_label_1", "dummy_label_2"]
        exp_class_labels_list = [
            ["ujets", "cjets", "bjets"],
            ["ujets", "cjets", "bjets"],
        ]
        (  # pylint: disable=unbalanced-tuple-unpacking
            filepath_list,
            labels_list,
            class_labels_list,
        ) = get_datasets_configuration(self.plotting_config)

        with self.subTest():
            self.assertEqual(exp_filepath_list, filepath_list)
        with self.subTest():
            self.assertEqual(exp_labels_list, labels_list)
        with self.subTest():
            self.assertEqual(exp_class_labels_list, class_labels_list)

    def test_get_datasets_configuration_specific_class_labels(self):
        """Test the helper function for the case of specifying specific class labels
        for one of the datasets"""

        # modify the config for this test
        plotting_config = self.plotting_config
        plotting_config["Datasets_to_plot"]["ds_2"]["class_labels"] = ["bjets"]
        plotting_config["Datasets_to_plot"]["ds_1"]["tracks_name"] = "tracks_loose"
        plotting_config["Datasets_to_plot"]["ds_2"]["tracks_name"] = "tracks"

        # define expected outcome
        exp_filepath_list = ["dummy_path_1", "dummy_path_2"]
        exp_labels_list = ["dummy_label_1", "dummy_label_2"]
        exp_class_labels_list = [
            ["ujets", "cjets", "bjets"],
            ["bjets"],
        ]
        exp_tracks_name_list = ["tracks_loose", "tracks"]
        (
            filepath_list,
            labels_list,
            class_labels_list,
            tracks_name_list,
        ) = get_datasets_configuration(self.plotting_config, tracks=True)

        with self.subTest():
            self.assertEqual(exp_filepath_list, filepath_list)
        with self.subTest():
            self.assertEqual(exp_labels_list, labels_list)
        with self.subTest():
            self.assertEqual(exp_class_labels_list, class_labels_list)
        with self.subTest():
            self.assertEqual(exp_tracks_name_list, tracks_name_list)


class JetPlottingTestCase(unittest.TestCase):
    """Test class for jet plotting functions."""

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")
        # Create a temporary directory
        self.tmp_dir = (
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )
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

        run(
            ["wget", self.r21_url, "--directory-prefix", self.actual_plots_dir],
            check=True,
        )
        run(
            ["wget", self.r22_url, "--directory-prefix", self.actual_plots_dir],
            check=True,
        )

    def test_plot_input_vars_jets_wrong_type(self):
        """Test jet input variable plots with wrong type."""
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        class_labels_list = [["ujets", "cjets"]]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["SV1_NGTinSvx"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_class_labels=class_labels_list,
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                output_directory=f"{self.actual_plots_dir}"
                + plotting_config["folder_to_save"],
                plot_type="png",
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_jets(self):
        """Test jet input variable plots."""
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]
        class_labels_list = [["bjets", "cjets", "ujets", "taujets"]]

        plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_class_labels=class_labels_list,
            var_dict=plotting_config["var_dict"],
            cut_vars_dict=plotting_config["cut_vars_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            output_directory=f"{self.actual_plots_dir}"
            # output_directory=f"{self.expected_plots_dir}"
            + plotting_config["folder_to_save"],
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "jets_input_vars/" + "SV1_NGTinSvx.png",
                    self.actual_plots_dir + "jets_input_vars/" + "SV1_NGTinSvx.png",
                    tol=1,
                ),
            )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir
                    + "jets_input_vars/"
                    + "JetFitterSecondaryVertex_nTracks.png",
                    self.actual_plots_dir
                    + "jets_input_vars/"
                    + "JetFitterSecondaryVertex_nTracks.png",
                    tol=1,
                ),
            )

    def test_plot_input_vars_jets_comparison_wrong_type(self):
        """Test jet input plots with comparison and wrong type."""
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]
        class_labels_list = [["ujets", "cjets"]]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["SV1_NGTinSvx"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_class_labels=class_labels_list,
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_jets_log_value_error(self):
        """Test jet input plots error raising for log."""
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]
        class_labels_list = [["ujets", "cjets"]]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["pt_btagJes_log"]["variables"] = [
            "rnnip_pc",
            "rnnip_pu",
        ]

        with self.assertRaises(ValueError):
            plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_class_labels=class_labels_list,
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_jets_comparison(self):
        """Test jet input variable plot with comparison."""
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        class_labels_list = [
            ["bjets", "cjets", "ujets", "taujets"],
            ["bjets", "cjets", "ujets", "taujets"],
        ]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_class_labels=class_labels_list,
            var_dict=plotting_config["var_dict"],
            cut_vars_dict=plotting_config["cut_vars_dict"],
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            output_directory=os.path.join(self.actual_plots_dir, "comp/"),
            # output_directory=os.path.join(self.expected_plots_dir, "comp/"),
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "comp/" + "SV1_NGTinSvx.png",
                    self.actual_plots_dir + "comp/" + "SV1_NGTinSvx.png",
                    tol=1,
                ),
            )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir
                    + "comp/"
                    + "JetFitterSecondaryVertex_nTracks.png",
                    self.actual_plots_dir
                    + "comp/"
                    + "JetFitterSecondaryVertex_nTracks.png",
                    tol=1,
                ),
            )

    def test_plot_input_vars_trks_wrong_type(self):
        """Test track input variable plots with wrong type."""
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["dr"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_trks(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_track_names=tracks_name_list,
                datasets_class_labels=class_labels_list,
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                output_directory=f"{self.actual_plots_dir}",
                plot_type="png",
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_trks(self):
        """Test track input variables with wrong type."""
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_class_labels=class_labels_list,
            datasets_track_names=tracks_name_list,
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            var_dict=plotting_config["var_dict"],
            cut_vars_dict=plotting_config["cut_vars_dict"],
            output_directory=f"{self.actual_plots_dir}",
            # output_directory=f"{self.expected_plots_dir}",
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "ptfrac/All/" + "dr_None_All.png",
                    self.actual_plots_dir + "ptfrac/All/" + "dr_None_All.png",
                    tol=1,
                ),
            )

        with self.subTest():
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

        with self.subTest():
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

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "ptfrac/0/" + "dr_0_All.png",
                    self.actual_plots_dir + "ptfrac/0/" + "dr_0_All.png",
                    tol=1,
                ),
            )

        with self.subTest():
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

        with self.subTest():
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
        """Test track input variables with comparison and wrong type."""
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["dr"] = "test"

        with self.assertRaises(ValueError):
            plot_input_vars_trks(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_track_names=tracks_name_list,
                datasets_class_labels=class_labels_list,
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_trks_log_error_raise(self):
        """Test track input variables log ValueError raise."""
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks"]
        labels_list = ["R21 Test"]

        # Change type in plotting_config to string to produce error
        plotting_config["var_dict"]["pt_frac_log"]["variables"] = [
            "IP3D_signed_d0_significance",
            "numberOfInnermostPixelLayerHits",
        ]

        with self.assertRaises(ValueError):
            plot_input_vars_trks(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                datasets_track_names=tracks_name_list,
                datasets_class_labels=class_labels_list,
                n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
                var_dict=plotting_config["var_dict"],
                cut_vars_dict=plotting_config["cut_vars_dict"],
                output_directory=os.path.join(self.actual_plots_dir, "comp/"),
                plot_type="png",
                **plotting_config["plot_settings"],
            )

    def test_plot_input_vars_trks_comparison_not_normalised(self):
        """Test track variable plots without normalisation."""
        plotting_config = self.plot_config["tracks_test_not_normalised"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"], ["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks", "tracks_loose"]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_class_labels=class_labels_list,
            datasets_track_names=tracks_name_list,
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            var_dict=plotting_config["var_dict"],
            cut_vars_dict=plotting_config["cut_vars_dict"],
            output_directory=os.path.join(self.actual_plots_dir, "comp_no_norm/"),
            # output_directory=os.path.join(self.expected_plots_dir, "comp_no_norm/"),
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
        """Test plotting track input variables with comparison."""
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        class_labels_list = [["bjets", "cjets", "ujets"], ["bjets", "cjets", "ujets"]]
        tracks_name_list = ["tracks", "tracks_loose"]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_track_names=tracks_name_list,
            datasets_class_labels=class_labels_list,
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            var_dict=plotting_config["var_dict"],
            cut_vars_dict=plotting_config["cut_vars_dict"],
            output_directory=os.path.join(self.actual_plots_dir, "comp/"),
            # output_directory=os.path.join(self.expected_plots_dir, "comp/"),
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "comp/ptfrac/All/" + "dr_None_All.png",
                    self.actual_plots_dir + "comp/ptfrac/All/" + "dr_None_All.png",
                    tol=1,
                ),
            )

        with self.subTest():
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

        with self.subTest():
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

        with self.subTest():
            self.assertIsNone(
                compare_images(
                    self.expected_plots_dir + "comp/ptfrac/0/" + "dr_0_All.png",
                    self.actual_plots_dir + "comp/ptfrac/0/" + "dr_0_All.png",
                    tol=1,
                ),
            )

        with self.subTest():
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

        with self.subTest():
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
        """Test plotting n tracks per jet."""
        plotting_config = self.plot_config["nTracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        tracks_name_list = ["tracks", "tracks_loose"]
        class_labels_list = [["bjets", "cjets", "ujets"], ["bjets", "cjets", "ujets"]]
        labels_list = ["R21 Test", "R22 Test"]

        plot_n_tracks_per_jet(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            datasets_class_labels=class_labels_list,
            datasets_track_names=tracks_name_list,
            n_jets=int(self.plot_config["Eval_parameters"]["n_jets"]),
            cut_vars_dict=plotting_config["cut_vars_dict"],
            output_directory=f"{self.actual_plots_dir}",
            # output_directory=f"{self.expected_plots_dir}",
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
