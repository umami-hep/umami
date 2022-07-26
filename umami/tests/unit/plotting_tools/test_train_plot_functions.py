"""Unit test script for the train plot functions."""

import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
from umami.plotting_tools.train_plotting_functions import (
    get_comp_tagger_rej_dict,
    plot_accuracies,
    plot_accuracies_umami,
    plot_disc_cut_per_epoch,
    plot_disc_cut_per_epoch_umami,
    plot_losses,
    plot_losses_umami,
    plot_rej_per_epoch,
    plot_rej_per_epoch_comp,
)

set_log_level(logger, "DEBUG")


class Train_Plots_TestCase(unittest.TestCase):
    """Test class for the train plot function."""

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")
        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")
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
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.comp_tagger_rej_dict = {
            "rnnip": {"cjets_rej_ttbar_r21_val": 2, "ujets_rej_ttbar_r21_val": 1}
        }
        self.label_extension = r"$t\bar{t}$"
        self.frac_class = "cjets"

    def test_plot_rej_per_epoch_comp(self):
        """Test the rejection per epoch comparison plot."""
        plot_rej_per_epoch_comp(
            df_results=self.df_results,
            tagger_label="dips",
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            taggers_from_file={"rnnip": "Recomm. RNNIP"},
            unique_identifier="ttbar_r21_val",
            plot_name=self.actual_plots_dir + "plot_rej_per_epoch_comp",
            # plot_name=self.expected_plots_dir + "plot_rej_per_epoch_comp",
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
                self.expected_plots_dir + "plot_rej_per_epoch_comp.png",
                self.actual_plots_dir + "plot_rej_per_epoch_comp.png",
                tol=1,
            ),
        )

    def test_plot_rej_per_epoch(self):
        """Test the rejection per epoch plot."""
        plot_rej_per_epoch(
            df_results=self.df_results,
            tagger_label="dips",
            comp_tagger_rej_dict=self.comp_tagger_rej_dict,
            plot_name=self.actual_plots_dir + "plot_rej_per_epoch",
            # plot_name=self.expected_plots_dir + "plot_rej_per_epoch",
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

        with self.subTest("Testing cjets rejection"):
            self.assertEqual(
                None,
                compare_images(
                    self.expected_plots_dir + "plot_rej_per_epoch_cjets_rejection.png",
                    self.actual_plots_dir + "plot_rej_per_epoch_cjets_rejection.png",
                    tol=1,
                ),
            )

        with self.subTest("Testing ujets rejection"):
            self.assertEqual(
                None,
                compare_images(
                    self.expected_plots_dir + "plot_rej_per_epoch_ujets_rejection.png",
                    self.actual_plots_dir + "plot_rej_per_epoch_ujets_rejection.png",
                    tol=1,
                ),
            )

    def test_plot_losses(self):
        """Test the loss plot."""
        plot_losses(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_losses",
            # plot_name=self.expected_plots_dir + "plot_losses",
            val_files=self.val_files,
            plot_datatype="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            figsize=(6, 5),
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_losses.png",
                self.actual_plots_dir + "plot_losses.png",
                tol=1,
            ),
        )

    def test_plot_accuracies(self):
        """Test the accuracy plot."""
        plot_accuracies(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_accuracies",
            # plot_name=self.expected_plots_dir + "plot_accuracies",
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(7, 5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.2,
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_accuracies.png",
                self.actual_plots_dir + "plot_accuracies.png",
                tol=1,
            ),
        )

    def test_plot_disc_cut_per_epoch(self):
        """Test the discriminant cut per epoch plot."""
        plot_disc_cut_per_epoch(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_disc_cut_per_epoch",
            # plot_name=self.expected_plots_dir + "plot_disc_cut_per_epoch",
            frac_class="cjets",
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(6, 4.5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_disc_cut_per_epoch.png",
                self.actual_plots_dir + "plot_disc_cut_per_epoch.png",
                tol=1,
            ),
        )

    def test_plot_disc_cut_per_epoch_umami(self):
        """Test the discriminant cut per epoch plot with two taggers."""
        plot_disc_cut_per_epoch_umami(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_disc_cut_per_epoch_umami",
            # plot_name=self.expected_plots_dir + "plot_disc_cut_per_epoch_umami",
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(6.3, 4.5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_disc_cut_per_epoch_umami.png",
                self.actual_plots_dir + "plot_disc_cut_per_epoch_umami.png",
                tol=1,
            ),
        )

    def test_plot_losses_umami(self):
        """Test the loss plot with two taggers."""
        plot_losses_umami(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_losses_umami",
            # plot_name=self.expected_plots_dir + "plot_losses_umami",
            val_files=self.val_files,
            plot_datatype="png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.4,
            figsize=(6.5, 5),
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_losses_umami.png",
                self.actual_plots_dir + "plot_losses_umami.png",
                tol=1,
            ),
        )

    def test_plot_accuracies_umami(self):
        """Test the accuracy plot with two taggers."""
        plot_accuracies_umami(
            df_results=self.df_results,
            plot_name=self.actual_plots_dir + "plot_accuracies_umami",
            # plot_name=self.expected_plots_dir + "plot_accuracies_umami",
            val_files=self.val_files,
            plot_datatype="png",
            figsize=(7, 5),
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets",
            y_scale=1.4,
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_accuracies_umami.png",
                self.actual_plots_dir + "plot_accuracies_umami.png",
                tol=1,
            ),
        )


class get_comp_tagger_rej_dict_TestCase(unittest.TestCase):
    """Test the rejection dict calculation."""

    def setUp(self):
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
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
            ],
            check=True,
        )

    def test_get_comp_tagger_rej_dict_umami(self):
        """Test get_comp_tagger_rej_dict for 2 taggers."""
        comp_rej_dict = get_comp_tagger_rej_dict(
            file=self.validation_files[self.validation_unique_identifiers[0]]["path"],
            unique_identifier=self.validation_unique_identifiers[0],
            tagger_comp_var=self.tagger_comp_var_rnnip,
            recommended_frac_dict=self.recommended_frac_dict,
            n_jets=5000,
            working_point=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        with self.subTest("Test both rejections"):
            self.assertTrue(
                f"cjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
                and f"ujets_rej_{self.validation_unique_identifiers[0]}"
                in comp_rej_dict
            )

        with self.subTest("Test signal"):
            self.assertFalse(
                f"bjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
            )

    def test_get_comp_tagger_rej_dict_dips_dl1(self):
        """Test get_comp_tagger_rej_dict for 1 taggers."""
        comp_rej_dict = get_comp_tagger_rej_dict(
            file=self.validation_files[self.validation_unique_identifiers[0]]["path"],
            unique_identifier=self.validation_unique_identifiers[0],
            tagger_comp_var=self.tagger_comp_var_dl1r,
            recommended_frac_dict=self.recommended_frac_dict,
            n_jets=5000,
            working_point=self.WP,
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        with self.subTest("Test both rejections"):
            self.assertTrue(
                f"cjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
                and f"ujets_rej_{self.validation_unique_identifiers[0]}"
                in comp_rej_dict
            )

        with self.subTest("Test signal"):
            self.assertFalse(
                f"bjets_rej_{self.validation_unique_identifiers[0]}" in comp_rej_dict
            )
