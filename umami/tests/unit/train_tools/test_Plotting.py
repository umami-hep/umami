import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
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

set_log_level(logger, "DEBUG")


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
            "epoch",
            "loss",
            "accuracy",
            "cjets_rej",
            "ujets_rej",
            "val_loss_ttbar_r21_val",
            "val_acc_ttbar_r21_val",
            "disc_cut_ttbar_r21_val",
            "ujets_rej_ttbar_r21_val",
            "cjets_rej_ttbar_r21_val",
            "val_loss_zprime_r21_val",
            "val_acc_zprime_r21_val",
            "disc_cut_zprime_r21_val",
            "ujets_rej_zprime_r21_val",
            "cjets_rej_zprime_r21_val",
            # umami related stuff
            "loss_umami",
            "loss_dips",
            "accuracy_umami",
            "accuracy_dips",
            "val_loss_umami_ttbar_r21_val",
            "val_acc_umami_ttbar_r21_val",
            "disc_cut_umami_ttbar_r21_val",
            "ujets_rej_umami_ttbar_r21_val",
            "cjets_rej_umami_ttbar_r21_val",
            "val_loss_umami_zprime_r21_val",
            "val_acc_umami_zprime_r21_val",
            "disc_cut_umami_zprime_r21_val",
            "ujets_rej_umami_zprime_r21_val",
            "cjets_rej_umami_zprime_r22_val",
            "val_loss_dips_ttbar_r21_val",
            "val_acc_dips_ttbar_r21_val",
            "disc_cut_dips_ttbar_r21_val",
            "ujets_rej_dips_ttbar_r21_val",
            "cjets_rej_dips_ttbar_r21_val",
            "val_loss_dips_zprime_r21_val",
            "val_acc_dips_zprime_r21_val",
            "disc_cut_dips_zprime_r21_val",
            "ujets_rej_dips_zprime_r21_val",
            "cjets_rej_dips_zprime_r21_val",
        ]

        self.val_files = {
            "ttbar_r21_val": {
                "path": "dummy",
                "label": "$t\\bar{t}$ Release 21",
                "variable_cuts": [
                    {"pt_btagJes": {"operator": "<=", "condition": 250_000}}
                ],
            },
            "zprime_r21_val": {
                "path": "dummy",
                "label": "$Z'$ Release 21",
                "variable_cuts": [
                    {"pt_btagJes": {"operator": ">", "condition": 250_000}}
                ],
            },
        }
        self.validation_unique_identifiers = self.val_files.keys()

        # TODO: Change the plotted data for each key? Atm we plot only straight lines
        # on top of each other, which maybe makes the test less robust?

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
                "loss_umami",
                "loss_dips",
                "accuracy_umami",
                "accuracy_dips",
            ]
        }
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.comp_tagger_rej_dict = {
            "rnnip": {"cjets_rej_ttbar_r21_val": 2, "ujets_rej_ttbar_r21_val": 1}
        }
        self.label_extension = r"$t\bar{t}$"
        self.frac_class = "cjets"

    def test_PlotRejPerEpochComparison(self):
        PlotRejPerEpochComparison(
            df_results=self.df_results,
            tagger_label="dips",
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            taggers_from_file={"rnnip": "Recomm. RNNIP"},
            unique_identifier="ttbar_r21_val",
            plot_name=self.tmp_plot_dir + "PlotRejPerEpochComparison",
            class_labels=self.class_labels,
            main_class=self.main_class,
            label_extension=self.label_extension,
            rej_string="rej",
            plot_datatype="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            figsize=(7.7, 5),
            y_scale=1.1,
            label_fontsize=10,
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
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            plot_name=self.tmp_plot_dir + "PlotRejPerEpoch",
            taggers_from_file={"rnnip": "Recomm. RNNIP"},
            unique_identifier="ttbar_r21_val",
            class_labels=self.class_labels,
            main_class=self.main_class,
            label_extension=self.label_extension,
            rej_string="rej",
            plot_datatype="png",
            leg_loc="upper right",
            figsize=(6, 4.5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
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
            val_files=self.val_files,
            plot_datatype="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            figsize=(6, 5),
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
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(7, 5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.2,
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
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(6, 4.5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
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
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(6.3, 4.5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
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
            val_files=self.val_files,
            plot_datatype="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.4,
            figsize=(6.5, 5),
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
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(7, 5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.4,
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
        self.validation_files = {
            "ttbar_r21_val": {
                "path": f"{self.test_dir.name}/ci_ttbar_testing.h5",
                "label": "dummylabel",
            }
        }
        self.validation_unique_identifiers = list(self.validation_files.keys())
        self.tagger_comp_var_rnnip = ["rnnip_pb", "rnnip_pc", "rnnip_pu"]
        self.tagger_comp_var_dl1r = ["DL1r_pb", "DL1r_pc", "DL1r_pu"]
        self.recommended_frac_dict = {"cjets": 0.018, "ujets": 0.982}
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.WP = 0.77
        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/preprocessing/"
                "ci_ttbar_testing.h5",
                "--directory-prefix",
                self.test_dir.name,
            ]
        )

    def test_CompTaggerRejectionDict_umami(self):
        comp_rej_dict = CompTaggerRejectionDict(
            file=self.validation_files[self.validation_unique_identifiers[0]]["path"],
            unique_identifier=self.validation_unique_identifiers[0],
            tagger_comp_var=self.tagger_comp_var_rnnip,
            recommended_frac_dict=self.recommended_frac_dict,
            nJets=5000,
            WP=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        self.assertTrue(
            f"cjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
            and f"ujets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
        )

        self.assertFalse(
            f"bjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
        )

    def test_CompTaggerRejectionDict_dips_dl1(self):
        comp_rej_dict = CompTaggerRejectionDict(
            file=self.validation_files[self.validation_unique_identifiers[0]]["path"],
            unique_identifier=self.validation_unique_identifiers[0],
            tagger_comp_var=self.tagger_comp_var_dl1r,
            recommended_frac_dict=self.recommended_frac_dict,
            nJets=5000,
            WP=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        self.assertTrue(
            f"cjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
            and f"ujets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
        )

        self.assertFalse(
            f"bjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
        )
