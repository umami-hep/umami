#!/usr/bin/env python

"""
This script integration tests the plotting of the training
results of the different models.
"""

import os
import unittest
from shutil import copyfile
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger  # noqa: F401
from umami.configuration import set_log_level
from umami.tools import replace_line_in_file, yaml_loader

set_log_level(logger, "DEBUG")


def get_configuration():
    """
    Load yaml file with settings for integration test of dips training.

    Returns
    -------
    object
        Loaded configuration file.

    Raises
    ------
    YAMLError
        If a needed key is not in file.
    """
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_dips"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runPlotting(config, tagger):
    """
    Call plotting_umami.py and try to plot the results of the previous tests.
    Return value `True` if training succeeded, `False` if one step did not succeed.

    Parameters
    ----------
    config : dict
        Dict with the needed configurations for the plotting.
    tagger : str
        Name of the tagger which is to be plotted.

    Raises
    ------
    AssertionError
        If the plotting step fails.

    Returns
    -------
    isSuccess : bool
        Preprocessing succeeded or not.
    """

    logger.info("Test: running plotting_umami.py for %s...", tagger)
    run_plotting_umami = run(
        [
            "python",
            "umami/plotting_umami.py",
            "-c",
            f"{config}",
            "-o",
            "plotting_umami",
            "-f",
            "pdf",
            "-p",
            "--verbose",
        ],
        check=True,
    )

    try:
        run_plotting_umami.check_returncode()

    except CalledProcessError as error:
        raise AssertionError(f"Test failed: plotting_umami.py for {tagger}.") from error

    return True


class TestPlottingUmami(unittest.TestCase):
    """Integration test class for the plotting of the training results.

    This class creates a test folder and downloads all important files.
    """

    def setUp(self):
        """Download test files for running the dips training."""
        # Get test configuration
        self.data = get_configuration()
        self.model_name_dips = self.data["test_dips"]["model_name"]
        self.model_name_umami = self.data["test_umami"]["model_name"]
        self.model_name_dl1r = self.data["test_dl1r"]["model_name"]

        # config files, will be copied to test dir
        self.config_source_dips = os.path.join(
            os.getcwd(), "examples/plotting_umami_config_dips.yaml"
        )

        self.config_source_umami = os.path.join(
            os.getcwd(), "examples/plotting_umami_config_Umami.yaml"
        )

        self.config_source_dl1r = os.path.join(
            os.getcwd(), "examples/plotting_umami_config_DL1r.yaml"
        )

        self.config_dips = os.path.join(
            os.getcwd(),
            self.model_name_dips,
            "plotting_umami_config_dips.yaml",
        )
        self.config_umami = os.path.join(
            os.getcwd(),
            self.model_name_umami,
            "plotting_umami_config_Umami.yaml",
        )
        self.config_dl1r = os.path.join(
            os.getcwd(),
            self.model_name_dl1r,
            "plotting_umami_config_DL1r.yaml",
        )

    def test_plotting_umami_dips(self):
        """
        Testing the plotting of the DIPS trainings.
        """
        # Copy the plotting yaml file
        copyfile(self.config_source_dips, self.config_dips)

        # modify copy of preprocessing config file for test
        replace_line_in_file(
            self.config_dips,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replace_line_in_file(
            self.config_dips,
            "model_name:",
            f"  model_name: {self.model_name_dips}",
        )
        replace_line_in_file(
            self.config_dips,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_dips, "dips"))

    def test_plotting_umami_umami(self):
        """
        Testing the plotting of the Umami trainings.
        """
        # Copy the plotting yaml file
        copyfile(self.config_source_umami, self.config_umami)

        replace_line_in_file(
            self.config_umami,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replace_line_in_file(
            self.config_umami,
            "model_name:",
            f"  model_name: {self.model_name_umami}",
        )
        replace_line_in_file(
            self.config_umami,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_umami, "umami"))

    def test_plotting_umami_dl1r(self):
        """
        Testing the plotting of the DL1r trainings.
        """
        # Copy the plotting yaml file
        copyfile(self.config_source_dl1r, self.config_dl1r)

        # modify copy of preprocessing config file for test
        replace_line_in_file(
            self.config_dl1r,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replace_line_in_file(
            self.config_dl1r,
            "model_name:",
            f"  model_name: {self.model_name_dl1r}",
        )
        replace_line_in_file(
            self.config_dl1r,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_dl1r, "dl1r"))
