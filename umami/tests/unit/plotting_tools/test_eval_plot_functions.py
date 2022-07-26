#!/usr/bin/env python

"""
Unit test script for eval plot functions.
"""

import os
import tempfile
import unittest
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images

from umami.configuration import logger, set_log_level
from umami.plotting_tools.eval_plotting_functions import (
    plot_fraction_contour,
    plot_prob,
    plot_pt_dependence,
    plot_roc,
    plot_saliency,
    plot_score,
)

set_log_level(logger, "DEBUG")


class Eval_plots_TestCase(unittest.TestCase):
    """Test class for the eval plot functions."""

    def setUp(self):
        # reset matplotlib parameters
        plt.rcdefaults()
        plt.close("all")
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "plots/")
        self.plot_config = {
            "evaluation_file": None,
            "data_set_name": "ttbar",
            "prediction_label": "dips_pb",
            "prediction_labels": ["dips_pb", "dips_pc", "dips_pu"],
            "discriminant": "b",
        }
        self.eval_params = {
            "epoch": 1,
            "bool_use_taus": False,
        }
        self.data_url = "https://umami-ci-provider.web.cern.ch/eval_files/"
        self.results_url = self.data_url + "results-1_new.h5"
        self.rej_url = self.data_url + "results-rej_per_eff-1_new.h5"
        self.frac_url = self.data_url + "results-rej_per_fractions-1.h5"
        self.dips_df_key = "dips_ujets_rej"
        self.rnnip_df_key = "rnnip_ujets_rej"
        self.class_labels = ["ujets", "cjets", "bjets"]
        self.main_class = "bjets"

        run(
            ["wget", self.results_url, "--directory-prefix", self.actual_plots_dir],
            check=True,
        )
        run(
            ["wget", self.rej_url, "--directory-prefix", self.actual_plots_dir],
            check=True,
        )
        run(
            [
                "wget",
                self.frac_url,
                "--directory-prefix",
                self.actual_plots_dir,
            ],
            check=True,
        )

    def test_plot_score(self):
        """Test nominal behaviour for plot_score without ratio."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_score(
            df_list=[df_results_ttbar],
            model_labels=["DIPS"],
            tagger_list=["dips"],
            class_labels_list=[
                self.class_labels,
            ],
            main_class=self.main_class,
            # plot_name=self.expected_plots_dir + "plot_score.png",
            plot_name=self.actual_plots_dir + "plot_score.png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
            ylabel="Normalised number of jets",
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "plot_score.png",
                self.actual_plots_dir + "plot_score.png",
                tol=1,
            )
        )

    def test_plot_score_comparison(self):
        """Test nominal behaviour for plot_score with ratio."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_score(
            df_list=[df_results_ttbar, df_results_ttbar],
            model_labels=["DIPS ttbar", "DIPS ttbar 2"],
            tagger_list=["dips", "dips"],
            class_labels_list=[
                self.class_labels,
                self.class_labels,
            ],
            main_class=self.main_class,
            # plot_name=self.expected_plots_dir + "plot_score_comparison.png",
            plot_name=self.actual_plots_dir + "plot_score_comparison.png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
            ylabel="Normalised number of jets",
            figsize=(6, 5),
            y_scale=1.5,
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "plot_score_comparison.png",
                self.actual_plots_dir + "plot_score_comparison.png",
                tol=1,
            )
        )

    def test_plot_roc(self):
        """Test nominal for roc with one rejection."""
        df_results_eff_rej_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-rej_per_eff-1_new.h5",
            "ttbar",
        )

        plot_roc(
            df_results_list=[
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
            ],
            tagger_list=["rnnip", "dips"],
            rej_class_list=["ujets", "ujets"],
            labels=["RNNIP ttbar", "DIPS ttbar"],
            # plot_name=self.expected_plots_dir + "ROC_Test.png",
            plot_name=self.actual_plots_dir + "ROC_Test.png",
            n_test=[100_000, 100_000],
            working_points=[0.60, 0.70, 0.77, 0.85],
            main_class="bjets",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets,\n"
                "$t\\bar{t}$ Test Sample, $f_{c}=0.018$"
            ),
            figsize=(5.5, 5),
            y_scale=1.6,
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "ROC_Test.png",
                self.actual_plots_dir + "ROC_Test.png",
                tol=1,
            )
        )

    def test_plot_roc_comparison(self):
        """Test nominal for roc with two rejection."""
        df_results_eff_rej_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-rej_per_eff-1_new.h5",
            "ttbar",
        )

        plot_roc(
            df_results_list=[
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
            ],
            tagger_list=["rnnip", "dips", "rnnip", "dips"],
            rej_class_list=["ujets", "ujets", "cjets", "cjets"],
            labels=["RNNIP", "DIPS", "RNNIP", "DIPS"],
            # plot_name=self.expected_plots_dir + "ROC_Comparison_Test.png",
            plot_name=self.actual_plots_dir + "ROC_Comparison_Test.png",
            n_test=[100_000, 100_000, 100_000, 100_000],
            working_points=[0.60, 0.70, 0.77, 0.85],
            reference_ratio=[True, False, True, False],
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets,\n"
                "$t\\bar{t}$ Test Sample, $f_{c}=0.018$"
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "ROC_Comparison_Test.png",
                self.actual_plots_dir + "ROC_Comparison_Test.png",
                tol=1,
            )
        )

    def test_plot_pt_dependence(self):
        """Test nominal for pT vs rejection."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_pt_dependence(
            df_list=[df_results_ttbar, df_results_ttbar],
            tagger_list=["dips", "dips"],
            model_labels=["DIPS ttbar", "DIPS 2"],
            # plot_name=self.expected_plots_dir + "pT_vs_Test.png",
            plot_name=self.actual_plots_dir + "pT_vs_Test.png",
            class_labels=self.class_labels,
            main_class=self.main_class,
            flavour="cjets",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets,\n"
                "$t\\bar{t}$ Test Sample, $f_{c}=0.018$"
            ),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "pT_vs_Test.png",
                self.actual_plots_dir + "pT_vs_Test.png",
                tol=1,
            )
        )

    def test_plot_saliency(self):
        """Test nominal behaviour for the saliency plots."""
        rng = np.random.default_rng(42)
        maps_dict = {
            "Variables_list": [f"p{i}" for i in range(15)],
            "77_bjets_True": rng.integers(1, 5, size=(15, 8)),
        }

        plot_saliency(
            maps_dict=maps_dict,
            plot_name=self.actual_plots_dir + "plot_saliency.png",
            target_eff=0.77,
            jet_flavour="bjets",
            PassBool=True,
            nFixedTrks=8,
            title="Test Saliency",
        )

        self.assertEqual(
            None,
            compare_images(
                self.expected_plots_dir + "plot_saliency.png",
                self.actual_plots_dir + "plot_saliency.png",
                tol=1,
            ),
        )

    def test_plot_prob(self):
        """Test nominal behaviour for the probability plots without ratio."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_prob(
            df_list=[df_results_ttbar],
            model_labels=["DIPS"],
            tagger_list=["dips"],
            class_labels_list=[self.class_labels],
            flavour="bjets",
            # plot_name=self.expected_plots_dir + "plot_prob.png",
            plot_name=self.actual_plots_dir + "plot_prob.png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
            ylabel="Normalised number of jets",
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "plot_prob.png",
                self.actual_plots_dir + "plot_prob.png",
                tol=1,
            )
        )

    def test_plot_prob_comparison(self):
        """Test nominal behaviour for the probability plots with ratio."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_prob(
            df_list=[df_results_ttbar, df_results_ttbar],
            model_labels=["DIPS ttbar", "DIPS ttbar 2"],
            tagger_list=["dips", "dips"],
            class_labels_list=[self.class_labels, self.class_labels],
            flavour="bjets",
            # plot_name=self.expected_plots_dir + "plot_prob_comparison.png",
            plot_name=self.actual_plots_dir + "plot_prob_comparison.png",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
            ylabel="Normalised number of jets",
            figsize=(6, 5),
            y_scale=1.5,
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "plot_prob_comparison.png",
                self.actual_plots_dir + "plot_prob_comparison.png",
                tol=1,
            )
        )

    def test_plot_fraction_contour(self):
        """Test the plot_fraction_contour function."""
        df_results_ttbar = pd.read_hdf(
            self.actual_plots_dir + "/results-rej_per_fractions-1.h5",
            "ttbar_r21",
        )

        plot_fraction_contour(
            df_results_list=[df_results_ttbar, df_results_ttbar],
            tagger_list=["dips", "rnnip"],
            label_list=["DIPS", "RNNIP"],
            rejections_to_plot=["ujets", "cjets"],
            marked_points_list=[
                {"cjets": 0.1, "ujets": 0.9},
                {"cjets": 0.1, "ujets": 0.9},
            ],
            plot_name=self.actual_plots_dir + "plot_fraction_contour.png",
            rejections_to_fix_list=[None, None],
            atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
            figsize=(5.5, 4),
        )

        self.assertIsNone(
            compare_images(
                self.expected_plots_dir + "plot_fraction_contour.png",
                self.actual_plots_dir + "plot_fraction_contour.png",
                tol=1,
            )
        )
