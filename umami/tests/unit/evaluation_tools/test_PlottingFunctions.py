import os
import pickle
import tempfile
import unittest
from subprocess import run

import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images

from umami.evaluation_tools.PlottingFunctions import (
    calc_bins,
    calc_ratio,
    eff_err,
    plot_prob,
    plot_prob_comparison,
    plot_score,
    plot_score_comparison,
    plotPtDependence,
    plotROCRatio,
    plotROCRatioComparison,
    plotSaliency,
    rej_err,
)


class calc_bins_TestCase(unittest.TestCase):
    """
    Test the calc bins function
    """

    def setUp(self):
        self.bool_use_taus = True
        self.Binning = np.array([0, 1, 2, 3, 4, 5])
        self.tracks = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        self.bins = np.array([0, 1, 2, 3, 4, 5])
        self.weights = np.array([0, 0.25, 0.25, 0.25, 0.25])
        self.unc = np.array([0, 0.1767767, 0.1767767, 0.1767767, 0.1767767])
        self.band = np.array([0, 0.0732233, 0.0732233, 0.0732233, 0.0732233])
        self.counter = np.array([5, 3, 2, 5, 6, 2])
        self.denominator = np.array([3, 6, 2, 7, 10, 12])
        self.counter_unc = np.array([0.5, 1, 0.3, 0.2, 0.5, 0.3])
        self.denominator_unc = np.array([1, 0.3, 2, 1, 5, 3])
        self.step = np.array(
            [1.6666667, 1.6666667, 0.5, 1, 0.7142857, 0.6, 0.1666667]
        )
        self.step_unc = np.array(
            [
                0.580017,
                0.580017,
                0.1685312,
                1.0111874,
                0.1059653,
                0.3041381,
                0.0485913,
            ]
        )

    def test_calc_bins(self):
        bins, weights, unc, band = calc_bins(
            input_array=self.tracks,
            Binning=self.Binning,
        )

        np.testing.assert_almost_equal(bins, self.bins)
        np.testing.assert_almost_equal(weights, self.weights)
        np.testing.assert_almost_equal(unc, self.unc)
        np.testing.assert_almost_equal(band, self.band)

    def test_calc_ratio(self):
        step, step_unc = calc_ratio(
            counter=self.counter,
            denominator=self.denominator,
            counter_unc=self.counter_unc,
            denominator_unc=self.denominator_unc,
        )

        np.testing.assert_almost_equal(step, self.step)
        np.testing.assert_almost_equal(step_unc, self.step_unc)


class Small_funcs_TestCase(unittest.TestCase):
    """
    Test eff_err, rej_err
    """

    def setUp(self):
        self.x_eff = np.array([0.25, 0.5, 0.75])
        self.x_rej = np.array([20, 50, 100])
        self.error_eff = np.array([0.043301, 0.05, 0.043301])
        self.error_rej = np.array([0.021794, 0.014, 0.00995])
        self.N = 100

    def test_eff_err(self):
        error = eff_err(
            x=self.x_eff,
            N=self.N,
        )

        self.assertEqual(len(error), len(self.x_eff))
        np.testing.assert_array_almost_equal(error, self.error_eff)

    def test_rej_err(self):
        error = rej_err(
            x=self.x_rej,
            N=self.N,
        )

        self.assertEqual(len(error), len(self.x_rej))
        np.testing.assert_array_almost_equal(error, self.error_rej)


class plot_score_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_plot_dir = f"{self.tmp_dir.name}/"
        self.plots_dir = os.path.join(os.path.dirname(__file__), "plots/")
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
        self.data_url = (
            "https://umami-docs.web.cern.ch/umami-docs/ci/eval_files/"
        )
        self.results_url = self.data_url + "results-1_new.h5"
        self.rej_url = self.data_url + "results-rej_per_eff-1_new.h5"
        self.saliency_url = self.data_url + "saliency_1_ttbar_new.pkl"
        self.dips_df_key = "dips_ujets_rej"
        self.rnnip_df_key = "rnnip_ujets_rej"
        self.class_labels = ["ujets", "cjets", "bjets"]
        self.main_class = "bjets"

        run(
            ["wget", self.results_url, "--directory-prefix", self.tmp_plot_dir]
        )
        run(["wget", self.rej_url, "--directory-prefix", self.tmp_plot_dir])
        run(
            [
                "wget",
                self.saliency_url,
                "--directory-prefix",
                self.tmp_plot_dir,
            ]
        )

    def test_plot_score(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_score(
            df_results=df_results_ttbar,
            plot_name=self.tmp_plot_dir + "plot_score.png",
            tagger_name="dips",
            class_labels=self.class_labels,
            main_class=self.main_class,
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_score.png",
                self.tmp_plot_dir + "plot_score.png",
                tol=1,
            ),
        )

    def test_plot_score_comparison(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1_new.h5",
            "ttbar",
        )

        self.model_labels = ["DIPS ttbar", "DIPS ttbar 2"]

        plot_score_comparison(
            df_list=[df_results_ttbar, df_results_ttbar],
            model_labels=self.model_labels,
            tagger_list=["dips", "dips"],
            class_labels_list=[
                self.class_labels,
                self.class_labels,
            ],
            main_class=self.main_class,
            plot_name=self.tmp_plot_dir + "plot_score_comparison.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_score_comparison.png",
                self.tmp_plot_dir + "plot_score_comparison.png",
                tol=1,
            ),
        )

    def test_plotROCRatio(self):
        df_results_eff_rej_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-rej_per_eff-1_new.h5",
            "ttbar",
        )

        plotROCRatio(
            df_results_list=[
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
            ],
            tagger_list=["rnnip", "dips"],
            rej_class_list=["ujets", "ujets"],
            labels=["RNNIP ttbar", "DIPS ttbar"],
            plot_name=self.tmp_plot_dir + "ROC_Test.png",
            nTest=[100000, 100000],
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "ROC_Test.png",
                self.tmp_plot_dir + "ROC_Test.png",
                tol=1,
            ),
        )

    def test_plotROCRatioComparison(self):
        df_results_eff_rej_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-rej_per_eff-1_new.h5",
            "ttbar",
        )

        plotROCRatioComparison(
            df_results_list=[
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
                df_results_eff_rej_ttbar,
            ],
            tagger_list=["rnnip", "dips", "rnnip", "dips"],
            rej_class_list=["ujets", "ujets", "cjets", "cjets"],
            labels=["RNNIP", "DIPS", "RNNIP", "DIPS"],
            plot_name=self.tmp_plot_dir + "ROC_Comparison_Test.png",
            nTest=[100000, 100000, 100000, 100000],
            WorkingPoints=[0.60, 0.70, 0.77, 0.85],
            ratio_id=[0, 0, 1, 1],
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "ROC_Comparison_Test.png",
                self.tmp_plot_dir + "ROC_Comparison_Test.png",
                tol=1,
            ),
        )

    def test_plotPtDependence(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1_new.h5",
            "ttbar",
        )

        self.df_list = [df_results_ttbar, df_results_ttbar]
        self.tagger_list = ["dips", "dips"]
        self.model_labels = ["DIPS ttbar", "DIPS 2"]

        plotPtDependence(
            df_list=self.df_list,
            tagger_list=self.tagger_list,
            model_labels=self.model_labels,
            plot_name=self.tmp_plot_dir + "pT_vs_Test.png",
            class_labels=self.class_labels,
            main_class=self.main_class,
            flavour="cjets",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "pT_vs_Test.png",
                self.tmp_plot_dir + "pT_vs_Test.png",
                tol=1,
            ),
        )

    def test_plotSaliency(self):
        with open(self.tmp_plot_dir + "saliency_1_ttbar_new.pkl", "rb") as f:
            maps_dict = pickle.load(f)

        plotSaliency(
            maps_dict=maps_dict,
            plot_name=self.tmp_plot_dir + "plotSaliency.png",
            title="Test Saliency",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plotSaliency.png",
                self.tmp_plot_dir + "plotSaliency.png",
                tol=1,
            ),
        )

    def test_plot_prob(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_prob(
            df_results=df_results_ttbar,
            plot_name=self.tmp_plot_dir + "plot_prob.png",
            tagger_name="dips",
            class_labels=self.class_labels,
            flavour="bjets",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_prob.png",
                self.tmp_plot_dir + "plot_prob.png",
                tol=1,
            ),
        )

    def test_plot_prob_comparison(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1_new.h5",
            "ttbar",
        )

        plot_prob_comparison(
            df_list=[df_results_ttbar, df_results_ttbar],
            model_labels=["DIPS ttbar", "DIPS ttbar 2"],
            tagger_list=["dips", "dips"],
            class_labels_list=[self.class_labels, self.class_labels],
            flavour="bjets",
            plot_name=self.tmp_plot_dir + "plot_prob_comparison.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_prob_comparison.png",
                self.tmp_plot_dir + "plot_prob_comparison.png",
                tol=1,
            ),
        )
