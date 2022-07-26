"""Unit test script for the functions in metrics.py."""

import unittest

import numpy as np

from umami.configuration import logger, set_log_level
from umami.metrics.metrics import (
    calc_disc_values,
    discriminant_output_shape,
    get_rejection,
    get_score,
)

set_log_level(logger, "DEBUG")


class Metrics_Small_TestCase(unittest.TestCase):
    """Test class for the umami.metrics functions."""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_discriminant_output_shape(self):
        """Test the discriminant output shape layer."""
        out = discriminant_output_shape(self.array)

        np.testing.assert_array_almost_equal(out, [[1, 2, 3]])


class CalcDiscValues_TestCase(unittest.TestCase):
    """Test class for the CalcDiscValues function."""

    def setUp(self):
        self.jets_dict = {
            "bjets": np.random.uniform(0, 1, size=(100, 3)),
            "cjets": np.random.uniform(0, 1, size=(100, 3)),
            "ujets": np.random.uniform(0, 1, size=(100, 3)),
        }
        self.index_dict = {
            "bjets": 0,
            "cjets": 1,
            "ujets": 2,
        }
        self.main_class = "bjets"
        self.rej_class = "cjets"
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

        self.jets_dict_4_Classes = {
            "bjets": np.random.uniform(0, 1, size=(100, 4)),
            "cjets": np.random.uniform(0, 1, size=(100, 4)),
            "ujets": np.random.uniform(0, 1, size=(100, 4)),
            "taujets": np.random.uniform(0, 1, size=(100, 4)),
        }
        self.index_dict_4_Classes = {
            "bjets": 0,
            "cjets": 1,
            "ujets": 2,
            "taujets": 3,
        }
        self.main_class_4_Classes = ["bjets", "taujets"]
        self.rej_class = "cjets"
        self.frac_dict_4_Classes = {
            "cjets": 0.018,
            "ujets": 0.782,
            "taujets": 0.2,
        }

    def test_CalcDiscValues(self):
        """Test CalcDiscValues for three classes and bjets (main)."""
        disc_score = calc_disc_values(
            jets_dict=self.jets_dict,
            index_dict=self.index_dict,
            main_class=self.main_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_score), len(self.jets_dict["bjets"]))

    def test_CalcDiscValues_Rejection(self):
        """Test CalcDiscValues for three classes and cjets (not-main class)."""
        disc_score = calc_disc_values(
            jets_dict=self.jets_dict,
            index_dict=self.index_dict,
            main_class=self.main_class,
            rej_class=self.rej_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_score), len(self.jets_dict["bjets"]))

    def test_CalcDiscValues_2Singal_Classes(self):
        """Test CalcDiscValues for three classes and cjets (not-main class)."""

        disc_score = calc_disc_values(
            jets_dict=self.jets_dict_4_Classes,
            index_dict=self.index_dict_4_Classes,
            main_class=self.main_class_4_Classes,
            frac_dict=self.frac_dict_4_Classes,
        )

        self.assertEqual(
            len(disc_score),
            len(self.jets_dict_4_Classes["bjets"])
            + len(self.jets_dict_4_Classes["taujets"]),
        )

    def test_CalcDiscValues_2Signal_Classes_bkg_jets(self):
        """
        Test CalcDiscValues for four classes and 2 signal classes
        and calculation of scores for background class (not-main class).
        """

        disc_score = calc_disc_values(
            jets_dict=self.jets_dict_4_Classes,
            index_dict=self.index_dict_4_Classes,
            main_class=self.main_class_4_Classes,
            frac_dict=self.frac_dict_4_Classes,
            rej_class=self.rej_class,
        )

        self.assertEqual(
            len(disc_score),
            len(self.jets_dict_4_Classes["cjets"]),
        )


class GetRejection_TestCase(unittest.TestCase):
    """Test class for the GetRejection function."""

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        rng = np.random.default_rng(42)
        self.y_pred = rng.random(size=(10000, 3))
        self.y_true = rng.random(size=(10000, 3))
        self.y_pred_tau = rng.random(size=(10000, 4))
        self.y_true_tau = rng.random(size=(10000, 4))
        self.y_true = (self.y_true == self.y_true.max(axis=1)[:, None]).astype(int)
        self.y_true_tau = (
            self.y_true_tau == self.y_true_tau.max(axis=1)[:, None]
        ).astype(int)
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.class_labels_tau = ["bjets", "cjets", "ujets", "taujets"]
        self.main_class = "bjets"
        self.target_eff = 0.77
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }
        self.frac_dict_tau = {
            "cjets": 0.018,
            "ujets": 0.782,
            "taujets": 0.2,
        }

    def test_GetRejection(self):
        """Test GetRejection for three classes."""

        rej_dict, _ = get_rejection(
            y_pred=self.y_pred,
            y_true=self.y_true,
            class_labels=self.class_labels,
            main_class=self.main_class,
            frac_dict=self.frac_dict,
            target_eff=self.target_eff,
        )

        self.assertTrue(("cjets_rej" in rej_dict and "ujets_rej" in rej_dict))

    def test_GetRejection_4_classes(self):
        """Test GetRejection for four classes."""

        rej_dict, _ = get_rejection(
            y_pred=self.y_pred_tau,
            y_true=self.y_true_tau,
            class_labels=self.class_labels_tau,
            main_class=self.main_class,
            frac_dict=self.frac_dict_tau,
            target_eff=self.target_eff,
        )

        self.assertTrue(
            "cjets_rej" in rej_dict
            and "ujets_rej" in rej_dict
            and "taujets_rej" in rej_dict
        )

    def test_GetRejection_wrong_shapes(self):
        """
        Test GetRejection with incorrect shapes in y_pred and y_true.
        Also checking if incorrect shapes between y_* and class_labels.
        """
        with self.assertRaises(ValueError):
            get_rejection(
                y_pred=self.y_pred,
                y_true=self.y_true_tau,
                class_labels=self.class_labels_tau,
                main_class=self.main_class,
                frac_dict=self.frac_dict_tau,
                target_eff=self.target_eff,
            )

        with self.assertRaises(ValueError):
            get_rejection(
                y_pred=self.y_pred_tau,
                y_true=self.y_true_tau,
                class_labels=self.class_labels,
                main_class=self.main_class,
                frac_dict=self.frac_dict_tau,
                target_eff=self.target_eff,
            )


class GetScore_TestCase(unittest.TestCase):
    """Test class for the GetScore function."""

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        rng = np.random.default_rng(42)
        self.y_pred = rng.random(size=(10000, 3))
        self.y_true = rng.random(size=(10000, 3))
        self.y_pred_tau = rng.random(size=(10000, 4))
        self.y_true_tau = rng.random(size=(10000, 4))
        self.y_true = (self.y_true == self.y_true.max(axis=1)[:, None]).astype(int)
        self.y_true_tau = (
            self.y_true_tau == self.y_true_tau.max(axis=1)[:, None]
        ).astype(int)
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.class_labels_tau = ["bjets", "cjets", "ujets", "taujets"]
        self.main_class = "bjets"
        self.target_eff = 0.77
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }
        self.frac_dict_tau = {
            "cjets": 0.018,
            "ujets": 0.782,
            "taujets": 0.2,
        }

    def test_GetScore(self):
        """Test GetScore for three classes."""
        disc_scores = get_score(
            y_pred=self.y_pred,
            class_labels=self.class_labels,
            main_class=self.main_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_scores), len(self.y_pred))
        self.assertAlmostEqual(disc_scores[0], -0.09494753279842187)

    def test_GetScore4Classes(self):
        """Test GetScore for four classes."""
        disc_scores = get_score(
            y_pred=self.y_pred_tau,
            class_labels=self.class_labels_tau,
            main_class=self.main_class,
            frac_dict=self.frac_dict_tau,
        )

        self.assertEqual(len(disc_scores), len(self.y_pred))
        self.assertAlmostEqual(disc_scores[0], -0.0597642740794453)

    def test_GetScore_wrong_shapes(self):
        """Test GetScore for incorrect shapes in y_pred and class_labels."""
        with self.assertRaises(AssertionError):
            _ = get_score(
                y_pred=self.y_pred,
                class_labels=self.class_labels_tau,
                main_class=self.main_class,
                frac_dict=self.frac_dict,
            )
