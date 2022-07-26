#!/usr/bin/env python

"""
Unit test the little helper functions from umami_tools.
"""

import os
import tempfile
import unittest
from shutil import copyfile

import pytest

from umami.configuration import logger, set_log_level
from umami.tools import (
    check_main_class_input,
    compare_leading_spaces,
    replace_line_in_file,
)
from umami.train_tools.configuration import Configuration

set_log_level(logger, "DEBUG")


class check_main_class_input_TestCase(unittest.TestCase):
    """Testing the check function for the main class."""

    def setUp(self) -> None:
        """Set up the needed variables."""
        self.main_class_str = "bjets"
        self.main_class_str_control = ["bjets"]
        self.main_class_list = ["bjets", "cjets"]
        self.main_class_set = {"bjets", "cjets"}
        self.main_class_control = ["bjets", "cjets"]
        self.main_class_int = 5

    def test_check_main_class_input_str(self):
        """Test the behaviour for strings."""
        main_class_check = check_main_class_input(self.main_class_str)
        self.assertEqual(main_class_check, self.main_class_str_control)

    def test_check_main_class_input_list(self):
        """Test the behaviour for list."""
        main_class_check = check_main_class_input(self.main_class_list)
        self.assertEqual(main_class_check, self.main_class_control)

    def test_check_main_class_input_set(self):
        """Test the behaviour for set."""
        with self.assertRaises(TypeError):
            _ = check_main_class_input(self.main_class_set)

    def test_check_main_class_input_fail(self):
        """Test the behaviour for a wrong type."""

        with self.assertRaises(TypeError):
            _ = check_main_class_input(self.main_class_int)


class replaceLineInFile_TestCase(unittest.TestCase):
    """Testing the replaceLineInFile function."""

    def setUp(self) -> None:
        """Set up the needed files."""

        # Create a temporary directory for the tests and get the path
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}"

        # Get the path for a basic config
        self.train_config_path = os.path.join(self.tmp_test_dir, "train_config.yaml")
        copyfile(
            os.path.join(os.getcwd(), "examples/Dips-PFlow-Training-config.yaml"),
            self.train_config_path,
        )

    def test_replaceLineInFile_Single_Line(self) -> None:
        """Test the standard behaviour of the function."""

        # Change the model_name
        replace_line_in_file(
            self.train_config_path,
            "model_name:",
            "model_name: Unittest_Testname",
            only_first=True,
        )

        # Load the yaml config file and check the value
        config = Configuration(self.train_config_path)
        self.assertEqual(config.model_name, "Unittest_Testname")

    def test_replaceLineInFile_Single_Line_Fail(self) -> None:
        """Test the raise error behaviour of the function."""

        # Change the model_name
        with self.assertRaises(AttributeError):
            replace_line_in_file(
                self.train_config_path,
                "Defintly_not_in_the_file:",
                "model_name: Unittest_Testname",
                only_first=True,
            )

    def test_replaceLineInFile_Multiple_Lines(self) -> None:
        """Test the standard behaviour of the function."""

        # Change the model_name
        replace_line_in_file(
            self.train_config_path,
            "model_name:",
            "model_name: Unittest_Testname",
        )

        # Load the yaml config file and check the value
        config = Configuration(self.train_config_path)
        self.assertEqual(config.model_name, "Unittest_Testname")

    def test_replaceLineInFile_Multiple_Lines_Fail(self) -> None:
        """Test the raise error behaviour of the function."""

        # Change the model_name
        with self.assertRaises(AttributeError):
            replace_line_in_file(
                self.train_config_path,
                "Defintly_not_in_the_file:",
                "model_name: Unittest_Testname",
            )


@pytest.mark.parametrize(
    "input, expected_result",
    [
        (("test134", "test789"), 0),
        (("   test134", "   test789"), 0),
        ((" test134", "   test789"), -2),
        (("   test134", " test789"), 2),
    ],
)
def test_compare_leading_spaces(  # pylint: disable=W0622
    input: tuple,
    expected_result: int,
) -> None:
    """Test different scenarios for `compare_leading_spaces` function.

    Parameters
    ----------
    input : tuple
        string tuples to test
    expected_result : int
        expected result
    """
    result = compare_leading_spaces(input[0], input[1])
    assert result == expected_result
