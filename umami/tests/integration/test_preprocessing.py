import logging
import os
import unittest
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import global_config  # noqa: F401
from umami.tools import replaceLineInFile, yaml_loader


def getConfiguration():
    """Load yaml file with settings for integration test of preprocessing."""
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_preprocessing"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runPreprocessing(config, var_dict, scale_dict, output, tagger):
    """Call all steps of the preprocessing for a certain configuration and variable dict input.
    Return value `True` if all steps succeeded, `False` if one step did not succees."""
    isSuccess = True

    logging.info("Test: running the undersampling...")
    if tagger == "dl1r":
        run_undersampling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--undersampling",
            ]
        )

    else:
        run_undersampling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--undersampling",
                "--tracks",
            ]
        )
    try:
        run_undersampling.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: preprocessing.py --undersampling.")
        isSuccess = False

    if isSuccess is True:
        run_undersampling

    logging.info("Test: retrieving scaling and shifting factors...")

    if tagger == "dl1r":
        run_scaling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--scaling",
            ]
        )

    else:
        run_scaling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--scaling",
                "--tracks",
            ]
        )
    try:
        run_scaling.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: preprocessing.py --scaling.")
        isSuccess = False

    if isSuccess is True:
        run_scaling

    logging.info("Test: applying shifting and scaling factors...")
    if tagger == "dl1r":
        run_apply_scales = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--apply_scales",
            ]
        )

    else:
        run_apply_scales = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--apply_scales",
                "--tracks",
            ]
        )
    try:
        run_apply_scales.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: preprocessing.py --apply_scales.")
        isSuccess = False

    if isSuccess is True:
        run_apply_scales

    logging.info(
        "Test: shuffling the samples and writing the samples to disk..."
    )
    if tagger == "dl1r":
        run_write = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--write",
            ]
        )

    else:
        run_write = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--write",
                "--tracks",
            ]
        )
    try:
        run_write.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: preprocessing.py --write.")
        isSuccess = False

    if isSuccess is True:
        run_write

        tagger_path = f"./preprocessing_{tagger}/"
        if not os.path.isdir(tagger_path):
            run(["mkdir", tagger_path])

        run(
            [
                "cp",
                "-r",
                "/tmp/umami/preprocessing/",
                tagger_path,
            ]
        )

    return isSuccess


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Download test files for running the preprocessing and dress preprocessing config file."""
        # Get test configuration
        self.data = getConfiguration()

        test_dir = os.path.join(self.data["test_preprocessing"]["testdir"])
        logging.info(f"Creating test directory in {test_dir}")
        # clean up, hopefully this causes no "uh oh...""
        if test_dir.startswith("/tmp"):
            run(["rm", "-rf", test_dir])
        run(["mkdir", "-p", test_dir])

        # inputs for test will be located in test_dir
        config_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["config"]
        )
        var_dict_umami_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["var_dict_umami"]
        )
        var_dict_dips_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["var_dict_dips"]
        )
        var_dict_dl1r_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["var_dict_dl1r"]
        )
        self.config = os.path.join(test_dir, os.path.basename(config_source))
        self.var_dict_umami = os.path.join(
            test_dir, os.path.basename(var_dict_umami_source)
        )
        self.var_dict_dips = os.path.join(
            test_dir, os.path.basename(var_dict_dips_source)
        )
        self.var_dict_dl1r = os.path.join(
            test_dir, os.path.basename(var_dict_dl1r_source)
        )
        self.scale_dict = os.path.join(test_dir, "PFlow-scale_dict.json")
        self.output = os.path.join(test_dir, "PFlow-hybrid_70-test.h5")

        logging.info(
            f"Preparing config file based on {config_source} in {self.config}..."
        )
        run(["cp", config_source, self.config])
        run(["cp", var_dict_umami_source, self.var_dict_umami])
        run(["cp", var_dict_dips_source, self.var_dict_dips])
        run(["cp", var_dict_dl1r_source, self.var_dict_dl1r])

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.config, "file_path:", f"  file_path: &file_path {test_dir}"
        )
        replaceLineInFile(
            self.config, "outfile_name:", f"outfile_name: {self.output}"
        )
        replaceLineInFile(
            self.config, "dict_file:", f"dict_file: {self.scale_dict}"
        )
        replaceLineInFile(self.config, "iterations:", "iterations: 1")

        logging.info("Downloading test data...")
        for file in self.data["test_preprocessing"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_preprocessing"]["data_subfolder"],
                file,
            )
            logging.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

    def test_preprocessing_umami(self):
        """Integration test of preprocessing.py script using Umami variables."""
        self.assertTrue(
            runPreprocessing(
                self.config,
                self.var_dict_umami,
                self.scale_dict,
                self.output,
                "umami",
            )
        )

    def test_preprocessing_dips(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        self.assertTrue(
            runPreprocessing(
                self.config,
                self.var_dict_dips,
                self.scale_dict,
                self.output,
                "dips",
            )
        )

    def test_preprocessing_dl1r(self):
        """Integration test of preprocessing.py script using DL1R variables."""
        self.assertTrue(
            runPreprocessing(
                self.config,
                self.var_dict_dl1r,
                self.scale_dict,
                self.output,
                "dl1r",
            )
        )
