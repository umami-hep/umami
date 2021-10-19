import numpy as np
import tensorflow as tf

from umami.tf_tools import Attention, DenseNet


class test_DenseNet(tf.test.TestCase):
    def setUp(self):
        """
        Setting up the DenseNet
        """

        self.nodes = [3, 3, 3]
        self.output_nodes = 3
        self.activation = "relu"
        self.batch_norm = True
        super(test_DenseNet, self).setUp()

        self.my_dense = DenseNet(
            nodes=self.nodes,
            output_nodes=self.output_nodes,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

    def test_get_config(self):
        # Get configs from Dense Net
        configs = self.my_dense.get_config()

        # Test configs
        self.assertEqual(self.output_nodes, configs["output_nodes"])
        self.assertEqual(self.activation, configs["activation"])
        self.assertEqual(self.batch_norm, configs["batch_norm"])
        self.assertEqual(self.nodes, configs["nodes"])

    def test_call(self):
        inputs = np.array([[0, 1, 1], [1, 1, 0]])
        expected_output = np.array(
            [
                [0.46534657, 0.23961703, 0.2950364],
                [0.35827565, 0.31311923, 0.32860515],
            ]
        )

        # Get net output
        out = self.my_dense(inputs=inputs)

        # Test output
        np.testing.assert_almost_equal(expected_output, out)


class test_Attention(tf.test.TestCase):
    def setUp(self):
        """
        Setting up the Attention network
        """

        self.nodes = [3, 3, 3]
        self.activation = "relu"
        self.mask_zero = True
        self.apply_softmax = True
        super(test_Attention, self).setUp()

        self.my_attention = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=self.mask_zero,
            apply_softmax=self.apply_softmax,
        )

    def test_get_config(self):
        # Get configs from Dense Net
        configs = self.my_attention.get_config()

        # Test configs
        self.assertEqual(self.nodes, configs["nodes"])
        self.assertEqual(self.activation, configs["activation"])
        self.assertEqual(self.mask_zero, configs["mask_zero"])
        self.assertEqual(self.apply_softmax, configs["apply_softmax"])

    def test_call(self):
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
        inputs = np.array([[0, 1, 1], [1, 1, 0]])

        # Get net output
        with self.assertRaises(AssertionError):
            _ = self.my_attention(inputs=inputs)

    def test_compute_mask(self):
        inputs = np.array(
            [
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
            ]
        )
        expected_output = np.array(
            [[True, False, False], [True, False, False]]
        )

        mask = self.my_attention.compute_mask(
            inputs=inputs,
            mask=None,
        )

        self.assertAllEqual(mask, expected_output)

    def test_compute_mask_Errors(self):
        inputs = np.array(
            [
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
                [[0, 0, 0], [1, 2, 0], [1, 2, 1]],
            ]
        )

        attention_mask_zero = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=True,
            apply_softmax=self.apply_softmax,
        )

        with self.assertRaises(AssertionError):
            _ = attention_mask_zero.compute_mask(
                inputs=inputs,
                mask=2,
            )

        attention_no_mask_zero = Attention(
            nodes=self.nodes,
            activation=self.activation,
            mask_zero=False,
            apply_softmax=self.apply_softmax,
        )

        with self.assertRaises(AssertionError):
            _ = attention_no_mask_zero.compute_mask(
                inputs=inputs,
                mask=None,
            )
