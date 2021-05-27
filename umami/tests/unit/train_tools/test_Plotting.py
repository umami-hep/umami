import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from umami.train_tools.Plotting import (
    PlotAccuracies,
    PlotAccuraciesUmami,
    PlotLosses,
    PlotLossesUmami,
    PlotRejPerEpoch,
)


class GetRejection_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_plot_dir = f"{self.tmp_dir.name}"
        self.Control_plots_dir = os.path.join(
            os.path.dirname(__file__), "plots/"
        )
        list_of_keys = [
            "epoch",
            "b_rej",
            "c_rej",
            "u_rej",
            "tau_rej",
            "loss",
            "val_loss",
            "val_loss_add",
            "acc",
            "val_acc",
            "val_acc_add",
            "umami_loss",
            "umami_val_loss",
            "umami_val_loss_add",
            "umami_acc",
            "umami_val_acc",
            "umami_val_acc_add",
            "dips_loss",
            "dips_val_loss",
            "dips_val_loss_add",
            "dips_acc",
            "dips_val_acc",
            "dips_val_acc_add",
        ]
        self.df_results = dict(
            zip(
                list_of_keys,
                [np.arange(0, 10, 1) for x in range(len(list_of_keys))],
            )
        )

    def test_PlotRejPerEpoch(self):
        self.b_rej = 2
        self.c_rej = 1
        self.u_rej = 5
        self.tau_rej = 3
        PlotRejPerEpoch(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotRejPerEpoch.png",
            b_rej=self.b_rej,
            c_rej=self.c_rej,
            u_rej=self.u_rej,
            tau_rej=self.tau_rej,
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "Control_PlotRejPerEpoch.png",
                self.tmp_plot_dir + "PlotRejPerEpoch.png",
                tol=1,
            ),
        )

    def test_PlotLosses(self):
        PlotLosses(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotLosses.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "Control_PlotLosses.png",
                self.tmp_plot_dir + "PlotLosses.png",
                tol=1,
            ),
        )

    def test_PlotAccuracies(self):
        PlotAccuracies(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotAccuracies.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "Control_PlotAccuracies.png",
                self.tmp_plot_dir + "PlotAccuracies.png",
                tol=1,
            ),
        )

    def test_PlotLossesUmami(self):
        PlotLossesUmami(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotLossesUmami.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "Control_PlotLossesUmami.png",
                self.tmp_plot_dir + "PlotLossesUmami.png",
                tol=1,
            ),
        )

    def test_PlotAccuraciesUmami(self):
        PlotAccuraciesUmami(
            df_results=self.df_results,
            plot_name=self.tmp_plot_dir + "PlotAccuraciesUmami.png",
        )

        self.assertEqual(
            None,
            compare_images(
                self.Control_plots_dir + "Control_PlotAccuraciesUmami.png",
                self.tmp_plot_dir + "PlotAccuraciesUmami.png",
                tol=1,
            ),
        )
