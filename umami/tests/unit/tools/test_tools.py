"""Unit test script for the tools functions."""

import unittest

from umami.configuration import logger, set_log_level
from umami.tools import flatten_list

set_log_level(logger, "DEBUG")


class FlattenListTest(unittest.TestCase):
    """Test class for flatten_list function."""

    def test_none(self):
        """Test None case."""
        self.assertIsNone(flatten_list(None))

    def test_empty(self):
        """Test empty list."""
        l_in = []
        l_flat = []

        self.assertListEqual(l_flat, flatten_list(l_in))

    def test_already_flattened(self):
        """Test already flattened list."""
        l_in = [1, 2, 3, 5, 6, 7, 8]
        l_flat = [1, 2, 3, 5, 6, 7, 8]

        self.assertListEqual(l_flat, flatten_list(l_in))

    def test_simple_neested(self):
        """Test case with simple nested list."""
        l_in = [[1, 2, 3], [5], [6, 7, 8]]
        l_flat = [1, 2, 3, 5, 6, 7, 8]

        self.assertListEqual(l_flat, flatten_list(l_in))

    def test_double_neested(self):
        """Test case with multiple nested lists."""
        l_in = [[[1, 2, 3], [5]], [6, [7], 8]]
        l_flat = [1, 2, 3, 5, 6, 7, 8]

        self.assertListEqual(l_flat, flatten_list(l_in))

    def test_neested_with_dict(self):
        """Test case when having dictionaries as list elements in nested list."""
        l_in = [[{"a": 1, "b": 2}], {"c": 3}]
        l_flat = [{"a": 1, "b": 2}, {"c": 3}]

        self.assertListEqual(l_flat, flatten_list(l_in))
