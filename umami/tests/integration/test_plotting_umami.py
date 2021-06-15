import logging
import os
import unittest
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import global_config  # noqa: F401
from umami.tools import replaceLineInFile, yaml_loader


def getConfiguration():
    """Load yaml file with settings for integration test of dips training."""
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
    """Call plotting_umami.py.
    Return value `True` if training succeeded, `False` if one step did not succees."""
    isSuccess = True

    logging.info(f"Test: running plotting_umami.py for {tagger}...")
    run_plotting_umami = run(
        [
            "plotting_umami.py",
            "-c",
            f"{config}",
            "-o",
            "plotting_umami",
            "-f",
            "pdf",
        ]
    )
    try:
        run_plotting_umami.check_returncode()
    except CalledProcessError:
        logging.info(f"Test failed: plotting_umami.py for {tagger}.")
        isSuccess = False

    if isSuccess is True:
        run_plotting_umami

    return isSuccess


class TestPlottingUmami(unittest.TestCase):
    def setUp(self):
        """Download test files for running the dips training."""
        # Get test configuration
        self.data = getConfiguration()
        self.model_name_dips = self.data["test_dips"]["model_name"]
        self.model_name_umami = self.data["test_umami"]["model_name"]
        self.model_name_dl1 = self.data["test_dl1"]["model_name"]

        test_dir_dips = os.path.join(self.data["test_dips"]["testdir"])
        test_dir_umami = os.path.join(self.data["test_umami"]["testdir"])
        test_dir_dl1 = os.path.join(self.data["test_dl1"]["testdir"])

        # clean up, hopefully this causes no "uh oh...""
        if test_dir_dips.startswith("/tmp"):
            run(["rm", "-rf", test_dir_dips])
        run(["mkdir", "-p", test_dir_dips])

        if test_dir_umami.startswith("/tmp"):
            run(["rm", "-rf", test_dir_umami])
        run(["mkdir", "-p", test_dir_umami])

        if test_dir_dl1.startswith("/tmp"):
            run(["rm", "-rf", test_dir_dl1])
        run(["mkdir", "-p", test_dir_dl1])

        # config files, will be copied to test dir
        self.config_source_dips = os.path.join(
            os.getcwd(), "examples/plotting_umami_config_dips.yaml"
        )

        self.config_source_umami = os.path.join(
            os.getcwd(), "examples/plotting_umami_config_Umami.yaml"
        )

        self.config_source_dl1 = os.path.join(
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
        self.config_dl1 = os.path.join(
            os.getcwd(),
            self.model_name_dl1,
            "plotting_umami_config_DL1r.yaml",
        )

    def test_plotting_umami_dips(self):
        # Copy the plotting yaml file
        run(["cp", self.config_source_dips, self.config_dips])

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.config_dips,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replaceLineInFile(
            self.config_dips,
            "model_name:",
            f"  model_name: {self.model_name_dips}",
        )
        replaceLineInFile(
            self.config_dips,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_dips, "dips"))

    def test_plotting_umami_umami(self):
        # Copy the plotting yaml file
        run(["cp", self.config_source_umami, self.config_umami])

        replaceLineInFile(
            self.config_umami,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replaceLineInFile(
            self.config_umami,
            "model_name:",
            f"  model_name: {self.model_name_umami}",
        )
        replaceLineInFile(
            self.config_umami,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_umami, "umami"))

    def test_plotting_umami_dl1(self):
        # Copy the plotting yaml file
        run(["cp", self.config_source_dl1, self.config_dl1])

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.config_dl1,
            "Path_to_models_dir:",
            "  Path_to_models_dir: ./",
        )
        replaceLineInFile(
            self.config_dl1,
            "model_name:",
            f"  model_name: {self.model_name_dl1}",
        )
        replaceLineInFile(
            self.config_dl1,
            "epoch:",
            "  epoch: 1",
        )

        self.assertTrue(runPlotting(self.config_dl1, "dl1"))
