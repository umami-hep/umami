"""Unit test script for the functions of tf_tools."""

import os
import tempfile
import unittest
from subprocess import run

import numpy as np
import tensorflow as tf

from umami.configuration import logger, set_log_level
from umami.tf_tools import (
    Attention,
    ConditionalDeepSet,
    DeepSet,
    DenseNet,
    prepare_model,
)

set_log_level(logger, "DEBUG")


class ConfigObject:
    """Object class with attributes."""


class AttentionTestCase(tf.test.TestCase):
    """Test class for the attention layer."""

    def setUp(self):
        """
        Setting up the Attention network
        """

        self.nodes = [3, 3, 3]
        self.activation = "relu"
        self.mask_zero = True
        self.apply_softmax = True
        super().setUp()

        self.my_attention = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=self.mask_zero,
            apply_softmax=self.apply_softmax,
        )

    def test_get_config(self):
        """Test the returning of the config values."""
        # Get configs from Dense Net
        configs = self.my_attention.get_config()

        # Test configs
        with self.subTest("Test architecture values which are returned"):
            self.assertEqual(self.nodes, configs["nodes"])
            self.assertEqual(self.activation, configs["activation"])
            self.assertEqual(self.mask_zero, configs["mask_zero"])
            self.assertEqual(self.apply_softmax, configs["apply_softmax"])

    def test_call(self):
        """Test the call function."""
        inputs = np.array(
            [
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
            ]
        )
        expected_output = np.array(
            [
                [0.3318959, 0.335652, 0.3324521],
                [0.3318959, 0.335652, 0.3324521],
            ]
        )

        # Get net output
        out = self.my_attention(inputs=inputs, mask=None)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_no_softmax(self):
        """Test the call function with no softmax applied."""
        attention = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=self.mask_zero,
            apply_softmax=False,
        )

        inputs = np.array(
            [
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
            ]
        )
        expected_output = np.array(
            [
                [0.4821225, 0.493376, 0.4837971],
                [0.4821225, 0.493376, 0.4837971],
            ]
        )

        # Get net output
        out = attention(inputs=inputs, mask=None)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_with_mask(self):
        """The the call function with given masking."""
        attention = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=self.mask_zero,
            apply_softmax=self.apply_softmax,
        )

        inputs = np.array(
            [
                [[0, 1, 2], [1, 2, 0], [1, 2, 1]],
                [[0, 1, 2], [1, 2, 0], [1, 2, 1]],
            ]
        )
        expected_output = np.array(
            [
                [0.3407553, 0.3291148, 0.33013],
                [0.3407553, 0.3291148, 0.33013],
            ]
        )

        # Get net output
        out = attention(inputs=inputs, mask=2)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_assertion_error(self):
        """Test call functions assertion error."""
        inputs = np.array([[0, 1, 1], [1, 1, 0]])

        # Get net output
        with self.assertRaises(AssertionError):
            _ = self.my_attention(inputs=inputs)

    def test_compute_mask(self):
        """Test computing of the masking."""
        inputs = np.array(
            [
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
            ]
        )
        expected_output = np.array([[True, False, False], [True, False, False]])

        mask = self.my_attention.compute_mask(
            inputs=inputs,
            mask=None,
        )

        self.assertAllEqual(mask, expected_output)

    def test_compute_mask_errors(self):
        """Test error of the masking."""
        _ = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=True,
            apply_softmax=self.apply_softmax,
        )

        _ = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=False,
            apply_softmax=self.apply_softmax,
        )


class DeepSetTestCase(tf.test.TestCase):
    """Test class for the DeepSet."""

    def setUp(self):
        """
        Setting up the DeepSet network
        """

        self.nodes = [2, 2, 2]
        self.activation = "relu"
        self.mask_zero = True
        self.batch_norm = True
        super().setUp()

        self.my_deepset = DeepSet(
            nodes=self.nodes,
            activation=self.activation,
            batch_norm=self.batch_norm,
            mask_zero=self.mask_zero,
        )

    def test_get_config(self):
        """Test the returning of the config values."""
        # Get configs from Dense Net
        configs = self.my_deepset.get_config()

        # Test configs
        with self.subTest("Test architecture values which are returned"):
            self.assertEqual(self.nodes, configs["nodes"])
            self.assertEqual(self.activation, configs["activation"])
            self.assertEqual(self.batch_norm, configs["batch_norm"])
            self.assertEqual(self.mask_zero, configs["mask_zero"])

    def test_call(self):
        """Test call function."""
        # Define an input
        inputs = np.array(
            [
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
            ]
        )

        # Get the control output
        expected_output = np.array(
            [
                [
                    [0.3135153, 0.4023385],
                    [0.2763494, 0.354643],
                    [0.7137673, 0.9159874],
                ],
                [
                    [0.3135153, 0.4023385],
                    [0.2763494, 0.354643],
                    [0.7137673, 0.9159874],
                ],
            ]
        )

        # Get network output
        out = self.my_deepset(inputs=inputs, mask=None)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_no_batch_norm(self):
        """Test call function with no batch normalisation."""
        deepset = DeepSet(
            nodes=self.nodes,
            activation=self.activation,
            batch_norm=False,
            mask_zero=self.mask_zero,
        )

        # Define an input
        inputs = np.array(
            [
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
            ]
        )

        # Get the control output
        expected_output = np.array(
            [
                [
                    [0.31382877, 0.4027408],
                    [0.27662575, 0.35499766],
                    [0.714481, 0.9169034],
                ],
                [
                    [0.31382877, 0.4027408],
                    [0.27662575, 0.35499766],
                    [0.714481, 0.9169034],
                ],
            ]
        )

        # Get net output
        out = deepset(inputs=inputs, mask=None)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_with_mask(self):
        """Test call function with given mask."""
        deepset = DeepSet(
            nodes=self.nodes,
            activation=self.activation,
            batch_norm=self.batch_norm,
            mask_zero=self.mask_zero,
        )

        inputs = np.array(
            [
                [[0, 1, 2], [1, 2, 0], [1, 2, 1]],
                [[0, 1, 2], [1, 2, 0], [1, 2, 1]],
            ]
        )
        expected_output = np.array(
            [
                [
                    [0.7509331, 0.9636828],
                    [0.1524468, 0.1956371],
                    [0.5898647, 0.7569815],
                ],
                [
                    [0.7509331, 0.9636828],
                    [0.1524468, 0.1956371],
                    [0.5898647, 0.7569815],
                ],
            ]
        )

        # Get net output
        out = deepset(inputs=inputs, mask=2)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_assertion_error(self):
        """Test call functions assertion error."""
        inputs = np.array([[0, 1, 1], [1, 1, 0]])

        # Get net output
        with self.assertRaises(AssertionError):
            _ = self.my_deepset(inputs=inputs)

    def test_compute_mask(self):
        """Test compute_mask function."""
        inputs = np.array(
            [
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
            ]
        )
        expected_output = np.array([[True, False, False], [True, False, False]])

        mask = self.my_deepset.compute_mask(
            inputs=inputs,
            mask=None,
        )

        self.assertAllEqual(mask, expected_output)


class DenseNetTestCase(tf.test.TestCase):
    """The class for the DenseNet"""

    def setUp(self):
        """
        Setting up the DenseNet
        """

        self.nodes = [3, 3, 3]
        self.output_nodes = 3
        self.activation = "relu"
        self.batch_norm = True
        super().setUp()

        self.my_dense = DenseNet(
            nodes=self.nodes,
            output_nodes=self.output_nodes,
            activation=self.activation,
            batch_norm=self.batch_norm,
            class_output_only=False,
        )

    def test_get_config(self):
        """Test returning the config values"""
        # Get configs from Dense Net
        configs = self.my_dense.get_config()

        # Test configs
        with self.subTest("Test architecture values which are returned"):
            self.assertEqual(self.output_nodes, configs["output_nodes"])
            self.assertEqual(self.activation, configs["activation"])
            self.assertEqual(self.batch_norm, configs["batch_norm"])
            self.assertEqual(self.nodes, configs["nodes"])

    def test_call(self):
        """Test the call function."""
        inputs = np.array([[0, 1, 1], [1, 1, 0]])
        expected_output = np.array(
            [
                [0.3024002, 0.2611001, 0.4364997],
                [0.3204106, 0.2945763, 0.3850131],
            ]
        )

        # Get net output
        _, out = self.my_dense(inputs=inputs)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)


class PrepareModelTestCase(unittest.TestCase):
    """Test the prepare_model function."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}/"

        self.nn_structure = ConfigObject()
        self.general = ConfigObject()
        self.general.model_name = self.tmp_test_dir + "Test_prepare_model"
        self.general.model_file = None
        self.general.continue_training = False
        self.nn_structure.load_optimiser = False

        os.makedirs(
            os.path.join(
                self.tmp_test_dir,
                self.general.model_name,
                "model_files",
            )
        )

        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/umami/test_model_file.h5",
                "--directory-prefix",
                self.tmp_test_dir,
            ],
            check=True,
        )

        run(
            [
                "cp",
                os.path.join(self.tmp_test_dir, "test_model_file.h5"),
                os.path.join(
                    self.general.model_name,
                    "model_files",
                    "model_epoch001.h5",
                ),
            ],
            check=True,
        )

    def test_init_fresh_model(self):
        """Test fresh model init."""
        model, init_epoch, load_optimiser = prepare_model(train_config=self)

        with self.subTest("Check Model"):
            self.assertIsNone(model)

        with self.subTest("Check init_epoch"):
            self.assertEqual(init_epoch, 0)

        with self.subTest("Check load_optimiser"):
            self.assertFalse(load_optimiser)

    def test_init_fresh_model_no_load_optimiser_given(self):
        """Test fresh model init with no load_optimiser given."""
        self.nn_structure.load_optimiser = None

        model, init_epoch, load_optimiser = prepare_model(train_config=self)

        with self.subTest("Check Model"):
            self.assertIsNone(model)

        with self.subTest("Check init_epoch"):
            self.assertEqual(init_epoch, 0)

        with self.subTest("Check load_optimiser"):
            self.assertFalse(load_optimiser)

    def test_load_optimiser_value_rrror(self):
        """Test load optimiser error."""
        self.nn_structure.load_optimiser = True

        with self.assertRaises(ValueError):
            prepare_model(train_config=self)

    def test_load_model_without_continue_training(self):
        """Test loading of a model without continuation."""
        self.general.model_file = os.path.join(
            self.tmp_test_dir,
            "test_model_file.h5",
        )

        model, init_epoch, load_optimiser = prepare_model(train_config=self)

        with self.subTest("Check Model"):
            self.assertTrue(isinstance(model, object))

        with self.subTest("Check init_epoch"):
            self.assertEqual(init_epoch, 0)

        with self.subTest("Check load_optimiser"):
            self.assertFalse(load_optimiser)

    def test_load_model_with_continue_training(self):
        """Test loading of a model without continuation."""
        self.general.continue_training = True
        model, init_epoch, load_optimiser = prepare_model(train_config=self)

        with self.subTest("Check Model"):
            self.assertTrue(isinstance(model, object))

        with self.subTest("Check init_epoch"):
            # The init_epoch value of keras is 0. If you start a new training
            # the new epoch will be init_epoch + 1. If you already have a training
            # the init_epoch must be the value of the last epoch saved, which is
            # in this test case the epoch 1.
            self.assertEqual(init_epoch, 1)

        with self.subTest("Check load_optimiser"):
            self.assertTrue(load_optimiser)


class ConditionalDeepSetTestCase(tf.test.TestCase):
    """Test class for the DeepSet."""

    def setUp(self):
        """
        Setting up the DeepSet network
        """

        self.nodes = [2, 2, 2]
        self.activation = "relu"
        self.mask_zero = True
        self.batch_norm = True
        super().setUp()

        self.my_deepset = ConditionalDeepSet(
            nodes=self.nodes,
            activation=self.activation,
            batch_norm=self.batch_norm,
            mask_zero=self.mask_zero,
        )

    def test_get_config(self):
        """Test the returning of the config values."""
        # Get configs from Dense Net
        configs = self.my_deepset.get_config()

        # Test configs
        with self.subTest("Test architecture values which are returned"):
            self.assertEqual(self.nodes, configs["nodes"])
            self.assertEqual(self.activation, configs["activation"])
            self.assertEqual(self.batch_norm, configs["batch_norm"])
            self.assertEqual(self.mask_zero, configs["mask_zero"])

    def test_call(self):
        """Test call function."""
        # Define an input
        inputs = (
            np.array(
                [
                    [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                    [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                ]
            ),
            np.array([[1], [0]]),
        )

        # Get the control output
        expected_output = np.array(
            [
                [
                    [0.9283064, 1.1913085],
                    [0.6515773, 0.8361782],
                    [0.7973309, 1.0232258],
                ],
                [
                    [0.56370866, 0.72341514],
                    [0.28619894, 0.36728305],
                    [0.43273303, 0.5553323],
                ],
            ]
        )

        # Get network output
        out = self.my_deepset(inputs=inputs)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_compute_mask_no_mask(self):
        """Test compute_mask function."""
        tmp = self.my_deepset.mask_zero
        self.my_deepset.mask_zero = False
        inputs = np.array(
            [
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
            ]
        )
        mask = self.my_deepset.compute_mask(
            inputs=inputs,
            mask=None,
        )
        self.my_deepset.mask_zero = tmp
        self.assertAllEqual(mask, None)
