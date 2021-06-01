import os
import tempfile
import unittest
from subprocess import run

import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images

from umami.evaluation_tools.PlottingFunctions import (
    GetCutDiscriminant,
    GetScore,
    GetScoreC,
    discriminant_output_shape,
    eff_err,
    getDiscriminant,
    plot_score,
    plot_score_comparison,
    plotPtDependence,
    plotROCRatio,
    plotSaliency,
)


class Small_funcs_TestCase(unittest.TestCase):
    """
    Test eff_err, discriminant_output_shape, getDiscriminant
    """

    def setUp(self):
        self.x = 0.5
        self.N = 1
        self.result = 0.5
        self.x_keras = np.array([[0.20, 0.20, 0.60], [0.20, 0.40, 0.20]])
        self.fc = 0.018

    def test_eff_err(self):
        score = eff_err(x=self.x, N=self.N)

        self.assertEqual(score, self.result)

    def test_discriminant_output_shape(self):
        shape = discriminant_output_shape((2, 4))

        self.assertEqual(shape, (2,))

    def test_getDiscriminant(self):
        discriminant = getDiscriminant(x=self.x_keras, fc=self.fc)

        self.assertEqual(discriminant.shape, (len(self.x_keras),))

        results = np.array([1.0986122886681096, -0.01783991812833112])
        for i in range(len(results)):
            self.assertEqual(discriminant.numpy()[i], results[i])


class GetScore_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create numbers for testing
        """
        # Create a temporary directory
        self.pb = np.array([0.33, 0.33, 0.33])
        self.pc = np.array([0.33, 0.33, 0.33])
        self.pu = np.array([0.33, 0.33, 0.33])
        self.ptau = np.array([0.01, 0.01, 0.01])
        self.fc = 0.018
        self.ftau = 0.018
        self.WP = 0.77
        self.add_small = 1e-10

    def test_GetScore_no_taus(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=None,
            fc=self.fc,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.0)

        self.assertEqual(scores.shape, (3,))

    def test_GetScore_taus_no_ftau(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == -0.029323411188329818)

        self.assertEqual(scores.shape, (3,))

    def test_GetScore_taus_with_ftau(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=self.ftau,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.017608672135089463)

        self.assertEqual(scores.shape, (3,))

    def test_GetCutDiscriminant(self):
        cut_value = GetCutDiscriminant(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=self.ftau,
            wp=self.WP,
        )

        self.assertEqual(cut_value, 0.017608672135089463)
        self.assertEqual(cut_value.shape, ())


class GetScoreC_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create numbers for testing
        """
        # Create a temporary directory
        self.pb = np.array([0.33, 0.33, 0.33])
        self.pc = np.array([0.33, 0.33, 0.33])
        self.pu = np.array([0.33, 0.33, 0.33])
        self.ptau = np.array([0.01, 0.01, 0.01])
        self.fb = 0.018
        self.ftau = 0.018
        self.add_small = 1e-10

    def test_GetScoreC_no_taus(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=None,
            fb=self.fb,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.0)

        self.assertEqual(scores.shape, (3,))

    def test_GetScoreC_taus_no_ftau(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fb=self.fb,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == -0.029323411188329818)

        self.assertEqual(scores.shape, (3,))

    def test_GetScoreC_taus_with_ftau(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fb=self.fb,
            ftau=self.ftau,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.017608672135089463)

        self.assertEqual(scores.shape, (3,))


class plot_score_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_plot_dir = f"{self.tmp_dir.name}/"
        self.plots_dir = os.path.join(os.path.dirname(__file__), "plots/")
        self.plot_config = {
            "evaluation_file": None,
            "data_set_name": "ttbar",
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
        self.results_url = self.data_url + "results-1.h5"
        self.rej_url = self.data_url + "results-rej_per_eff-1.h5"
        self.saliency_url = self.data_url + "saliency_1_ttbar.pkl"
        self.df_key = "dips_urej"

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
        plot_score(
            plot_name=self.tmp_plot_dir + "plot_score.png",
            plot_config=self.plot_config,
            eval_params=self.eval_params,
            eval_file_dir=self.tmp_plot_dir,
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_score.png",
                self.tmp_plot_dir + "plot_score.png",
                tol=1,
            ),
        )

        self.plot_config["evaluation_file"] = (
            self.tmp_plot_dir + "results-1.h5"
        )
        plot_score(
            plot_name=self.tmp_plot_dir + "plot_score.png",
            plot_config=self.plot_config,
            eval_params=self.eval_params,
            eval_file_dir=self.tmp_plot_dir,
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "plot_score.png",
                self.tmp_plot_dir + "plot_score.png",
                tol=1,
            ),
        )

        del self.plot_config["evaluation_file"]
        plot_score(
            plot_name=self.tmp_plot_dir + "plot_score.png",
            plot_config=self.plot_config,
            eval_params=self.eval_params,
            eval_file_dir=self.tmp_plot_dir,
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
            self.tmp_plot_dir + "/results-1.h5",
            "ttbar",
        )

        df_results_zpext = pd.read_hdf(
            self.tmp_plot_dir + "/results-1.h5",
            "zpext",
        )

        self.df_list = [df_results_ttbar, df_results_zpext]
        self.prediction_labels_list = [
            ["dips_pb", "dips_pc", "dips_pu"],
            ["dips_pb", "dips_pc", "dips_pu"],
        ]
        self.model_labels = ["DIPS ttbar", "DIPS zpext"]

        plot_score_comparison(
            df_list=self.df_list,
            prediction_labels_list=self.prediction_labels_list,
            model_labels=self.model_labels,
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
            self.tmp_plot_dir + "/results-rej_per_eff-1.h5",
            "ttbar",
        )

        rej_rates_ttbar = 1.0 / df_results_eff_rej_ttbar[self.df_key]

        df_results_eff_rej_zpext = pd.read_hdf(
            self.tmp_plot_dir + "/results-rej_per_eff-1.h5",
            "zpext",
        )

        rej_rates_zpext = 1.0 / df_results_eff_rej_zpext[self.df_key]

        teffs = [
            df_results_eff_rej_ttbar["beff"],
            df_results_eff_rej_zpext["beff"],
        ]
        beffs = [rej_rates_ttbar, rej_rates_zpext]
        labels = ["ttbar Test", "zpext Test"]

        plotROCRatio(
            teffs=teffs,
            beffs=beffs,
            labels=labels,
            plot_name=self.tmp_plot_dir + "ROC_Test.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.plots_dir + "ROC_Test.png",
                self.tmp_plot_dir + "ROC_Test.png",
                tol=1,
            ),
        )

    def test_plotPtDependence(self):
        df_results_ttbar = pd.read_hdf(
            self.tmp_plot_dir + "/results-1.h5",
            "ttbar",
        )

        df_results_zpext = pd.read_hdf(
            self.tmp_plot_dir + "/results-1.h5",
            "zpext",
        )

        self.df_list = [df_results_ttbar, df_results_zpext]
        self.prediction_labels_list = [
            ["dips_pb", "dips_pc", "dips_pu"],
            ["dips_pb", "dips_pc", "dips_pu"],
        ]
        self.model_labels = ["DIPS ttbar", "DIPS zpext"]

        plotPtDependence(
            df_list=self.df_list,
            prediction_labels=self.prediction_labels_list,
            model_labels=self.model_labels,
            plot_name=self.tmp_plot_dir + "pT_vs_Test.png",
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
        plotSaliency(
            plot_name=self.tmp_plot_dir + "plotSaliency.png",
            FileDir=self.tmp_plot_dir,
            epoch=self.eval_params["epoch"],
            data_set_name=self.plot_config["data_set_name"],
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
