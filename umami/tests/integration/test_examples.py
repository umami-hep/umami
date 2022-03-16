#!/usr/bin/env python

"""
Integration tests for the scripts located in the examples directory
"""

import os
import unittest
from subprocess import run


class TestExamples(unittest.TestCase):
    """Test class for the example scripts"""

    def setUp(self):
        """Create the folder which is used for the plots used later on in the docs"""
        os.makedirs("docs/ci_assets", exist_ok=True)
        self.umami_dir = os.getcwd()

    def test_plot_flavour_probabilities(self):
        """Test script examples/plotting/plot_flavour_probabilities.py"""

        command = (
            f"python {self.umami_dir}/examples/plotting/plot_flavour_probabilities.py"
        )
        self.assertEqual(
            run(command, shell=True, cwd="docs/ci_assets", check=True).returncode,
            0,
        )

    def test_plot_discriminant(self):
        """Test script examples/plotting/plot_discriminant_scores.py"""

        command = (
            f"python {self.umami_dir}/examples/plotting/plot_discriminant_scores.py"
        )
        self.assertEqual(
            run(command, shell=True, cwd="docs/ci_assets", check=True).returncode,
            0,
        )

    def test_plot_basic_histogram(self):
        """Test script examples/plotting/plot_basic_histogram.py"""

        command = f"python {self.umami_dir}/examples/plotting/plot_basic_histogram.py"
        self.assertEqual(
            run(command, shell=True, cwd="docs/ci_assets", check=True).returncode,
            0,
        )

    def test_plot_pt_vs_eff(self):
        """Test script examples/plotting/plot_pt_vs_eff.py"""

        command = f"python {self.umami_dir}/examples/plotting/plot_pt_vs_eff.py"
        self.assertEqual(
            run(command, shell=True, cwd="docs/ci_assets", check=True).returncode,
            0,
        )

    def test_plot_rocs(self):
        """Test script examples/plotting/plot_rocs.py"""

        command = f"python {self.umami_dir}/examples/plotting/plot_rocs.py"
        self.assertEqual(
            run(command, shell=True, cwd="docs/ci_assets", check=True).returncode,
            0,
        )
