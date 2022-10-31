"""Integration tests for variable plotting."""
import os
import tempfile
import unittest
from shutil import copyfile
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger, set_log_level
from umami.tools import yaml_loader

set_log_level(logger, "DEBUG")


def get_configuration() -> object:
    """
    Load yaml file with settings for integration test of the input vars plotting.

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
    for key in ["data_url", "test_input_vars_plot"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runPlotInputVars(config: str) -> bool:
    """Call plot_input_vars.py.

    Parameters
    ----------
    config : str
        Path to config file.

    Returns
    -------
    bool
        True if tests pass, False if tests fail.
    """

    isSuccess = True

    logger.info("Test: running plot_input_vars.py tracks...")
    run_plot_input_vars_trks = run(
        [
            "python",
            "umami/plot_input_variables.py",
            "-c",
            f"{config}",
            "--tracks",
            "--verbose",
        ],
        check=True,
    )
    try:
        run_plot_input_vars_trks.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: plot_input_variables.py.")
        isSuccess = False

    logger.info("Test: running plot_input_vars.py jets...")
    run_plot_input_vars_jets = run(
        [
            "python",
            "umami/plot_input_variables.py",
            "-c",
            f"{config}",
            "--jets",
            "--verbose",
        ],
        check=True,
    )
    try:
        run_plot_input_vars_jets.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: plot_input_variables.py.")
        isSuccess = False

    return isSuccess


class TestInput_Vars_Plotting(unittest.TestCase):
    """Integration tests for variable plotting."""

    def setUp(self):
        """Download test files for input var plots."""
        # Get test configuration
        self.data = get_configuration()

        self.test_dir_path = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = f"{self.test_dir_path.name}/"
        logger.info("Creating test directory in %s", self.test_dir)

        # input files, will be downloaded to test dir
        logger.info("Retrieving files from preprocessing...")
        self.test_file_r21 = os.path.join(self.test_dir, "plot_input_vars_r21_check.h5")
        self.test_file_r22 = os.path.join(self.test_dir, "plot_input_vars_r22_check.h5")
        self.config_path = os.path.join(self.test_dir, "config.yaml")

        copyfile("examples/plotting_input_vars.yaml", self.config_path)

        with open(self.config_path, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

        # Changing eval params
        self.config["Eval_parameters"]["n_jets"] = 3e3

        # Change datasets for all
        for plot in self.config:
            if plot != "Eval_parameters" and plot[0] != ".":
                self.config[plot]["Datasets_to_plot"]["R21"][
                    "files"
                ] = f"{self.test_dir}plot_input_vars_r21_check.h5"
                self.config[plot]["Datasets_to_plot"]["R21"]["label"] = "R21 Test"
                self.config[plot]["Datasets_to_plot"]["R22"][
                    "files"
                ] = f"{self.test_dir}plot_input_vars_r22_check.h5"
                self.config[plot]["Datasets_to_plot"]["R22"]["label"] = "R22 Test"
                self.config[plot]["Datasets_to_plot"]["R22"][
                    "tracks_name"
                ] = "tracks_loose"
                self.config[plot]["plot_settings"][
                    "SecondTag"
                ] = "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3000 Jets"

        # Change jets input vars params
        self.config["jets_input_vars"]["special_param_jets"] = {
            "SV1_NGTinSvx": {
                "lim_left": 0,
                "lim_right": 19,
            },
            "JetFitterSecondaryVertex_nTracks": {
                "lim_left": 0,
                "lim_right": 17,
            },
        }
        self.config["jets_input_vars"]["var_dict"] = {
            "SV1_NGTinSvx": 5,
            "JetFitterSecondaryVertex_nTracks": None,
            "combined_rnnip": {
                "variables": ["rnnip_pc", "rnnip_pu"],
                "binning": 50,
                "operator": "+",
            },
        }

        # Change tracks params
        self.config["tracks_input_vars"]["plot_settings"]["n_leading"] = [
            None,
            0,
        ]
        self.config["tracks_input_vars"]["var_dict"] = {
            "IP3D_signed_d0_significance": 100,
            "numberOfInnermostPixelLayerHits": [0, 4, 1],
            "dr": None,
            "numberOfSCTHits": None,
        }

        # Save changes to yaml
        with open(self.config_path, "w") as conf:
            yaml.dump(self.config, conf, default_flow_style=False)

        logger.info("Downloading test data...")
        for file in self.data["test_input_vars_plot"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_input_vars_plot"]["data_subfolder"],
                file,
            )
            logger.info("Retrieving file from path %s", path)
            run(["wget", path, "--directory-prefix", self.test_dir], check=True)

    def test_plot_input_vars(self):
        """Integration test of plot_input_vars.py script."""
        self.assertTrue(runPlotInputVars(self.config_path))
