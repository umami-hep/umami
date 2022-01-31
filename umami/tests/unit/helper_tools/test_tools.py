#!/usr/bin/env python

"""
Unit test the little helper functions from umami_tools.
"""

import os
import tempfile
import unittest
from shutil import copyfile

from umami.tools import replaceLineInFile
from umami.train_tools.Configuration import Configuration


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
        replaceLineInFile(
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
            replaceLineInFile(
                self.train_config_path,
                "Defintly_not_in_the_file:",
                "model_name: Unittest_Testname",
                only_first=True,
            )

    def test_replaceLineInFile_Multiple_Lines(self) -> None:
        """Test the standard behaviour of the function."""

        # Change the model_name
        replaceLineInFile(
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
            replaceLineInFile(
                self.train_config_path,
                "Defintly_not_in_the_file:",
                "model_name: Unittest_Testname",
            )