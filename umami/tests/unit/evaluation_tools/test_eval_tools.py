#!/usr/bin/env python

"""
Unit test script for the functions in eval_tools.py
"""

import unittest

import numpy as np


class small_functions_TestCase(unittest.TestCase):
    """Test class for small functions in eval_tools.py"""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])
