import os
import tempfile
import unittest
from subprocess import run

import yaml
from matplotlib.testing.compare import compare_images

from umami.input_vars_tools.PlottingFunctions import (
    plot_input_vars_jets,
    plot_input_vars_jets_comparison,
    plot_input_vars_trks,
    plot_input_vars_trks_comparison,
    plot_nTracks_per_Jet,
)
from umami.tools import yaml_loader


class JetPlotting_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_plot_dir = f"{self.tmp_dir.name}/"
        self.data_url = (
            "https://umami-ci-provider.web.cern.ch/plot_input_vars/"
        )

        self.Control_plots_dir = os.path.join(
            os.path.dirname(__file__), "plots/"
        )

        self.yaml_file = os.path.join(
            os.path.dirname(__file__), "fixtures/plot_input_variables.yaml"
        )

        self.r21_url = os.path.join(
            self.data_url, "plot_input_vars_r21_check.h5"
        )

        self.r22_url = os.path.join(
            self.data_url, "plot_input_vars_r22_check.h5"
        )

        self.r21_test_file = os.path.join(
            self.tmp_plot_dir, "plot_input_vars_r21_check.h5"
        )

        self.r22_test_file = os.path.join(
            self.tmp_plot_dir, "plot_input_vars_r22_check.h5"
        )

        with open(self.yaml_file) as yaml_config:
            self.plot_config = yaml.load(yaml_config, Loader=yaml_loader)

        run(["wget", self.r21_url, "--directory-prefix", self.tmp_plot_dir])
        run(["wget", self.r22_url, "--directory-prefix", self.tmp_plot_dir])

    def test_plot_input_vars_jets(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]

        plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            nJets=int(self.plot_config["Eval_parameters"]["nJets"]),
            binning=plotting_config["binning"],
            output_directory=f"{self.tmp_plot_dir}"
            + plotting_config["folder_to_save"],
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "jets_input_vars/" + "IP2D_bu.png",
                self.tmp_plot_dir + "jets_input_vars/" + "IP2D_bu.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "jets_input_vars/" + "IP2D_cu.png",
                self.tmp_plot_dir + "jets_input_vars/" + "IP2D_cu.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_jets_comparison(self):
        plotting_config = self.plot_config["jets_input_vars"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_jets_comparison(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            nJets=int(self.plot_config["Eval_parameters"]["nJets"]),
            binning=plotting_config["binning"],
            output_directory=os.path.join(self.tmp_plot_dir, "comp/"),
            plot_type="png",
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "comp/" + "IP2D_bu.png",
                self.tmp_plot_dir + "comp/" + "IP2D_bu.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "comp/" + "IP2D_cu.png",
                self.tmp_plot_dir + "comp/" + "IP2D_cu.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_trks(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file]
        labels_list = ["R21 Test"]

        del plotting_config["plot_settings"]["Ratio_Cut"]

        plot_input_vars_trks(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            nJets=int(self.plot_config["Eval_parameters"]["nJets"]),
            binning=plotting_config["binning"],
            output_directory=f"{self.tmp_plot_dir}",
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "ptfrac/All/" + "dr_None.png",
                self.tmp_plot_dir + "ptfrac/All/" + "dr_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "ptfrac/All/"
                + "IP3D_signed_d0_significance_None.png",
                self.tmp_plot_dir
                + "ptfrac/All/"
                + "IP3D_signed_d0_significance_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None.png",
                self.tmp_plot_dir
                + "ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "ptfrac/0/" + "dr_0.png",
                self.tmp_plot_dir + "ptfrac/0/" + "dr_0.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "ptfrac/0/"
                + "IP3D_signed_d0_significance_0.png",
                self.tmp_plot_dir
                + "ptfrac/0/"
                + "IP3D_signed_d0_significance_0.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0.png",
                self.tmp_plot_dir
                + "ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0.png",
                tol=1,
            ),
        )

    def test_plot_input_vars_trks_comparison(self):
        plotting_config = self.plot_config["Tracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        labels_list = ["R21 Test", "R22 Test"]

        plot_input_vars_trks_comparison(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            nJets=int(self.plot_config["Eval_parameters"]["nJets"]),
            binning=plotting_config["binning"],
            output_directory=os.path.join(self.tmp_plot_dir, "comp/"),
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "comp/ptfrac/All/" + "dr_None.png",
                self.tmp_plot_dir + "comp/ptfrac/All/" + "dr_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "comp/ptfrac/All/"
                + "IP3D_signed_d0_significance_None.png",
                self.tmp_plot_dir
                + "comp/ptfrac/All/"
                + "IP3D_signed_d0_significance_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "comp/ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None.png",
                self.tmp_plot_dir
                + "comp/ptfrac/All/"
                + "numberOfInnermostPixelLayerHits_None.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "comp/ptfrac/0/" + "dr_0.png",
                self.tmp_plot_dir + "comp/ptfrac/0/" + "dr_0.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "comp/ptfrac/0/"
                + "IP3D_signed_d0_significance_0.png",
                self.tmp_plot_dir
                + "comp/ptfrac/0/"
                + "IP3D_signed_d0_significance_0.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir
                + "comp/ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0.png",
                self.tmp_plot_dir
                + "comp/ptfrac/0/"
                + "numberOfInnermostPixelLayerHits_0.png",
                tol=1,
            ),
        )

    def test_plot_nTracks_per_Jet(self):
        plotting_config = self.plot_config["nTracks_Test"]
        filepath_list = [self.r21_test_file, self.r22_test_file]
        labels_list = ["R21 Test", "R22 Test"]

        plot_nTracks_per_Jet(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=self.plot_config["Eval_parameters"]["var_dict"],
            nJets=int(self.plot_config["Eval_parameters"]["nJets"]),
            output_directory=f"{self.tmp_plot_dir}",
            plot_type="png",
            **plotting_config["plot_settings"],
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "nTracks_per_Jet.png",
                self.tmp_plot_dir + "nTracks_per_Jet.png",
                tol=1,
            ),
        )
