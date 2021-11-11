import unittest

import numpy as np

from umami.evaluation_tools.eval_tools import discriminant_output_shape


class small_functions_TestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_discriminant_output_shape(self):
        out = discriminant_output_shape(self.array)

        np.testing.assert_array_almost_equal(out, [[1, 2, 3]])
