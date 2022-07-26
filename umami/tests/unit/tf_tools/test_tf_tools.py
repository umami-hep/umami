#!/usr/bin/env python

"""
Unit test script for the functions of tf_tools.
"""

import os
import tempfile
import unittest
from subprocess import run

import numpy as np
import tensorflow as tf

from umami.configuration import logger, set_log_level
from umami.tf_tools import Attention, DeepSet, DenseNet, prepare_model

set_log_level(logger, "DEBUG")


class test_Attention(tf.test.TestCase):
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
                [0.3267786, 0.3260552, 0.3471662],
                [0.3267786, 0.3260552, 0.3471662],
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
                [0.5015888, 0.4993725, 0.5621095],
                [0.5015888, 0.4993725, 0.5621095],
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
                [0.3121045, 0.3522645, 0.335631],
                [0.3121045, 0.3522645, 0.335631],
            ]
        )

        # Get net output
        out = attention(inputs=inputs, mask=2)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_AssertionError(self):
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

    def test_compute_mask_Errors(self):
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


class test_DeepSet(tf.test.TestCase):
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
                [[0, 0.13140553], [0, 0.05430728], [0, 0.15379035]],
                [[0, 0.13140553], [0, 0.05430728], [0, 0.15379035]],
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
                [[0, 0.13153693], [0, 0.05436159], [0, 0.15394413]],
                [[0, 0.13153693], [0, 0.05436159], [0, 0.15394413]],
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
                [[0, 0.2308886], [0, 0.08622973], [0, 0.18571281]],
                [[0, 0.2308886], [0, 0.08622973], [0, 0.18571281]],
            ]
        )

        # Get net output
        out = deepset(inputs=inputs, mask=2)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)

    def test_call_AssertionError(self):
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


class test_DenseNet(tf.test.TestCase):
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
                [0.46534657, 0.23961703, 0.2950364],
                [0.35827565, 0.31311923, 0.32860515],
            ]
        )

        # Get net output
        _, out = self.my_dense(inputs=inputs)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)


class prepare_model_TestCase(unittest.TestCase):
    """Test the prepare_model function."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}/"

        self.model_name = self.tmp_test_dir + "Test_prepare_model"
        self.model_file = None
        self.NN_structure = {"load_optimiser": False}

        os.makedirs(
            os.path.join(
                self.tmp_test_dir,
                self.model_name,
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
                    self.model_name,
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
        self.NN_structure = {}

        model, init_epoch, load_optimiser = prepare_model(train_config=self)

        with self.subTest("Check Model"):
            self.assertIsNone(model)

        with self.subTest("Check init_epoch"):
            self.assertEqual(init_epoch, 0)

        with self.subTest("Check load_optimiser"):
            self.assertFalse(load_optimiser)

    def test_load_optimiser_ValueError(self):
        """Test load optimiser error."""
        self.NN_structure = {"load_optimiser": True}

        with self.assertRaises(ValueError):
            _, _, _ = prepare_model(train_config=self)

    def test_load_model_without_continue_training(self):
        """Test loading of a model without continuation."""
        self.model_file = os.path.join(
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
        model, init_epoch, load_optimiser = prepare_model(
            train_config=self,
            continue_training=True,
        )

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
