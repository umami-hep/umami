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


class MetricsSmallTestCase(unittest.TestCase):
    """Test class for the umami.metrics functions."""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_discriminant_output_shape(self):
        """Test the discriminant output shape layer."""
        out = discriminant_output_shape(self.array)

        np.testing.assert_array_almost_equal(out, [[1, 2, 3]])


class CalcDiscValuesTestCase(unittest.TestCase):
    """Test class for the calc_disc_values function."""

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
        self.faulty_frac_dict = {
            "cjets": 0.018,
        }

        self.jets_dict_4_classes = {
            "bjets": np.random.uniform(0, 1, size=(100, 4)),
            "cjets": np.random.uniform(0, 1, size=(100, 4)),
            "ujets": np.random.uniform(0, 1, size=(100, 4)),
            "taujets": np.random.uniform(0, 1, size=(100, 4)),
        }
        self.index_dict_4_classes = {
            "bjets": 0,
            "cjets": 1,
            "ujets": 2,
            "taujets": 3,
        }
        self.main_class_4_classes = ["bjets", "taujets"]
        self.rej_class = "cjets"
        self.frac_dict_4_classes = {
            "cjets": 0.018,
            "ujets": 0.782,
            "taujets": 0.2,
        }

    def test_calc_disc_values(self):
        """Test calc_disc_values for three classes and bjets (main)."""
        disc_score = calc_disc_values(
            jets_dict=self.jets_dict,
            index_dict=self.index_dict,
            main_class=self.main_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_score), len(self.jets_dict["bjets"]))

    def test_calc_disc_values_rejection(self):
        """Test calc_disc_values for three classes and cjets (not-main class)."""
        disc_score = calc_disc_values(
            jets_dict=self.jets_dict,
            index_dict=self.index_dict,
            main_class=self.main_class,
            rej_class=self.rej_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_score), len(self.jets_dict["bjets"]))

    def test_calc_disc_values_2singal_classes(self):
        """Test calc_disc_values for three classes and cjets (not-main class)."""

        disc_score = calc_disc_values(
            jets_dict=self.jets_dict_4_classes,
            index_dict=self.index_dict_4_classes,
            main_class=self.main_class_4_classes,
            frac_dict=self.frac_dict_4_classes,
        )

        self.assertEqual(
            len(disc_score),
            len(self.jets_dict_4_classes["bjets"])
            + len(self.jets_dict_4_classes["taujets"]),
        )

    def test_calc_disc_values_2signal_classes_bkg_jets(self):
        """
        Test calc_disc_values for four classes and 2 signal classes
        and calculation of scores for background class (not-main class).
        """

        disc_score = calc_disc_values(
            jets_dict=self.jets_dict_4_classes,
            index_dict=self.index_dict_4_classes,
            main_class=self.main_class_4_classes,
            frac_dict=self.frac_dict_4_classes,
            rej_class=self.rej_class,
        )

        self.assertEqual(
            len(disc_score),
            len(self.jets_dict_4_classes["cjets"]),
        )

    def test_calc_disc_values_error(self):
        """Testing raising of errors."""

        with self.assertRaises(KeyError):
            calc_disc_values(
                jets_dict=self.jets_dict,
                index_dict=self.index_dict,
                main_class=self.main_class,
                rej_class=self.rej_class,
                frac_dict=self.faulty_frac_dict,
            )


class GetRejectionTestCase(unittest.TestCase):
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

    def test_get_rejection(self):
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

    def test_get_rejection_4_classes(self):
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

    def test_get_rejection_wrong_shapes(self):
        """
        Test GetRejection with incorrect shapes in y_pred and y_true.
        Also checking if incorrect shapes between y_* and class_labels.
        """

        with self.assertRaises(ValueError):
            get_rejection(
                y_pred=self.y_pred_tau,
                y_true=self.y_true_tau,
                class_labels=self.class_labels,
                main_class=self.main_class,
                frac_dict=self.frac_dict_tau,
                target_eff=self.target_eff,
            )


class GetScoreTestCase(unittest.TestCase):
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

    def test_get_score(self):
        """Test GetScore for three classes."""
        disc_scores = get_score(
            y_pred=self.y_pred,
            class_labels=self.class_labels,
            main_class=self.main_class,
            frac_dict=self.frac_dict,
        )

        self.assertEqual(len(disc_scores), len(self.y_pred))
        self.assertAlmostEqual(disc_scores[0], -0.09494753279842187)

    def test_get_score_4classes(self):
        """Test GetScore for four classes."""
        disc_scores = get_score(
            y_pred=self.y_pred_tau,
            class_labels=self.class_labels_tau,
            main_class=self.main_class,
            frac_dict=self.frac_dict_tau,
        )

        self.assertEqual(len(disc_scores), len(self.y_pred))
        self.assertAlmostEqual(disc_scores[0], -0.0597642740794453)

    def test_get_score_errors(self):
        """Test error raising of get_score"""

        with self.subTest("Assertion Error"):
            with self.assertRaises(AssertionError):
                get_score(
                    y_pred=self.y_pred,
                    class_labels=self.class_labels_tau,
                    main_class=self.main_class,
                    frac_dict=self.frac_dict,
                )

        with self.subTest("Key Error"):
            with self.assertRaises(KeyError):
                get_score(
                    y_pred=self.y_pred_tau,
                    class_labels=self.class_labels_tau,
                    main_class=self.main_class,
                    frac_dict=self.frac_dict,
                )
