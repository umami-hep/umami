import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images

from umami.train_tools.Plotting import (
    CompTaggerRejectionDict,
    PlotAccuracies,
    PlotAccuraciesUmami,
    PlotDiscCutPerEpoch,
    PlotDiscCutPerEpochUmami,
    PlotLosses,
    PlotLossesUmami,
    PlotRejPerEpoch,
    PlotRejPerEpochComparison,
)


class GetRejection_TestCase(unittest.TestCase):
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
        self.Control_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")
        list_of_keys = [
            "cjets_rej",
            "ujets_rej",
            "epoch",
            "loss",
            "val_loss",
            "val_loss_add",
            "accuracy",
            "val_acc",
            "val_acc_add",
            "umami_loss",
            "umami_val_loss",
            "umami_val_loss_add",
            "umami_accuracy",
            "umami_val_acc",
            "umami_val_acc_add",
            "dips_loss",
            "dips_val_loss",
            "dips_val_loss_add",
            "dips_accuracy",
            "dips_val_acc",
            "dips_val_acc_add",
            "disc_cut",
            "disc_cut_add",
            "disc_cut_dips",
            "disc_cut_umami",
            "disc_cut_dips_add",
            "disc_cut_umami_add",
        ]
        self.df_results = dict(
            zip(
                list_of_keys,
                [np.arange(0, 10, 1) for x in range(len(list_of_keys))],
            )
        )
        self.train_history_dict = {
            key: self.df_results[key]
            for key in [
                "epoch",
                "loss",
                "accuracy",
                "umami_loss",
                "umami_accuracy",
                "dips_accuracy",
                "dips_loss",
            ]
        }
        self.comp_tagger_frac_dict = {"RNNIP": {"cjets": 0.018, "ujets": 0.982}}
        self.frac_dict = {"cjets": 0.018, "ujets": 0.982}
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.comp_tagger_rej_dict = {"RNNIP": {"cjets_rej": 2, "ujets_rej": 1}}
        self.label_extension = r"$t\bar{t}$"
        self.frac_class = "cjets"

    def test_PlotRejPerEpochComparison(self):
        PlotRejPerEpochComparison(
            df_results=self.df_results,
            tagger_label="dips",
            frac_dict=self.frac_dict,
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            comp_tagger_frac_dict=self.comp_tagger_frac_dict,
            plot_name=self.tmp_plot_dir + "PlotRejPerEpochComparison",
            class_labels=self.class_labels,
            main_class=self.main_class,
            label_extension=self.label_extension,
            rej_string="rej",
            plot_datatype="png",
            legend_loc="upper right",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotRejPerEpochComparison.png",
                self.tmp_plot_dir + "PlotRejPerEpochComparison.png",
                tol=1,
            ),
        )

    def test_PlotRejPerEpoch(self):
        PlotRejPerEpoch(
            df_results=self.df_results,
            tagger_label="dips",
            frac_dict=self.frac_dict,
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            comp_tagger_frac_dict=self.comp_tagger_frac_dict,
            plot_name=self.tmp_plot_dir + "PlotRejPerEpoch",
            class_labels=self.class_labels,
            main_class=self.main_class,
            label_extension=self.label_extension,
            rej_string="rej",
            plot_datatype="png",
            legend_loc="upper right",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotRejPerEpoch_cjets_rejection.png",
                self.tmp_plot_dir + "PlotRejPerEpoch_cjets_rejection.png",
                tol=1,
            ),
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotRejPerEpoch_ujets_rejection.png",
                self.tmp_plot_dir + "PlotRejPerEpoch_ujets_rejection.png",
                tol=1,
            ),
        )

    def test_PlotLosses(self):
        PlotLosses(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotLosses",
            train_history_dict=self.train_history_dict,
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotLosses.png",
                self.tmp_plot_dir + "PlotLosses.png",
                tol=1,
            ),
        )

    def test_PlotAccuracies(self):
        PlotAccuracies(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotAccuracies",
            train_history_dict=self.train_history_dict,
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotAccuracies.png",
                self.tmp_plot_dir + "PlotAccuracies.png",
                tol=1,
            ),
        )

    def test_PlotDiscCutPerEpoch(self):
        PlotDiscCutPerEpoch(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotDiscCutPerEpoch",
            frac_class="cjets",
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotDiscCutPerEpoch.png",
                self.tmp_plot_dir + "PlotDiscCutPerEpoch.png",
                tol=1,
            ),
        )

    def test_PlotDiscCutPerEpochUmami(self):
        PlotDiscCutPerEpochUmami(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotDiscCutPerEpochUmami",
            frac_class="cjets",
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotDiscCutPerEpochUmami.png",
                self.tmp_plot_dir + "PlotDiscCutPerEpochUmami.png",
                tol=1,
            ),
        )

    def test_PlotLossesUmami(self):
        PlotLossesUmami(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotLossesUmami",
            train_history_dict=self.train_history_dict,
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotLossesUmami.png",
                self.tmp_plot_dir + "PlotLossesUmami.png",
                tol=1,
            ),
        )

    def test_PlotAccuraciesUmami(self):
        PlotAccuraciesUmami(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotAccuraciesUmami",
            train_history_dict=self.train_history_dict,
            plot_datatype="png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "PlotAccuraciesUmami.png",
                self.tmp_plot_dir + "PlotAccuraciesUmami.png",
                tol=1,
            ),
        )


class CompTaggerRejectionDict_TestCase(unittest.TestCase):
    def setUp(self):
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")
        self.test_dir = tempfile.TemporaryDirectory()
        self.validation_file = (
            f"{self.test_dir.name}/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5"
        )
        self.add_validation_file = (
            f"{self.test_dir.name}/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5"
        )
        self.tagger_comp_var_rnnip = ["rnnip_pb", "rnnip_pc", "rnnip_pu"]
        self.tagger_comp_var_dl1r = ["DL1r_pb", "DL1r_pc", "DL1r_pu"]
        self.recommended_frac_dict = {"cjets": 0.018, "ujets": 0.982}
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.WP = 0.77
        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/umami/"
                "MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5",
                "--directory-prefix",
                self.test_dir.name,
            ]
        )

    def test_CompTaggerRejectionDict_umami(self):
        comp_rej_dict = CompTaggerRejectionDict(
            file=self.validation_file,
            tagger_comp_var=self.tagger_comp_var_rnnip,
            recommended_frac_dict=self.recommended_frac_dict,
            WP=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        self.assertTrue("cjets_rej" in comp_rej_dict and "ujets_rej" in comp_rej_dict)

        self.assertFalse("bjets_rej" in comp_rej_dict)

    def test_CompTaggerRejectionDict_dips_dl1(self):
        comp_rej_dict = CompTaggerRejectionDict(
            file=self.validation_file,
            tagger_comp_var=self.tagger_comp_var_dl1r,
            recommended_frac_dict=self.recommended_frac_dict,
            WP=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        self.assertTrue("cjets_rej" in comp_rej_dict and "ujets_rej" in comp_rej_dict)

        self.assertFalse("bjets_rej" in comp_rej_dict)
