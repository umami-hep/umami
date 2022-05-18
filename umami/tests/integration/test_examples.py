#!/usr/bin/env python

"""
Integration tests for the scripts located in the examples directory
"""

import os
from subprocess import run

import pytest

os.makedirs("docs/ci_assets", exist_ok=True)
umami_dir = os.getcwd()


@pytest.mark.parametrize(
    "command, expected_result",
    [
        ("input_correlations.py", 0),
    ],
)
def test_example_plots(command, expected_result):
    """Check the plotting of the example plots

    Parameters
    ----------
    command : str
        file containing the test to be run
    expected_result : int
        expected test result
    """
    output = run(
        f"python {umami_dir}/examples/plotting/{command}",
        shell=True,
        cwd="docs/ci_assets",
        check=True,
    ).returncode
    assert output == expected_result
