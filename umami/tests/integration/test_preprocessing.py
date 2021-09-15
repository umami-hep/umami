import os
import unittest
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger
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

    logger.info("Test: running the prepare...")
    for sample in [
        "training_ttbar_bjets",
        "training_ttbar_cjets",
        "training_ttbar_ujets",
        "training_zprime_bjets",
        "training_zprime_cjets",
        "training_zprime_ujets",
    ]:
        if tagger == "dl1r":
            run_prepare = run(
                [
                    "preprocessing.py",
                    "-c",
                    f"{config}",
                    "--var_dict",
                    var_dict,
                    "--sample",
                    f"{sample}",
                    "--prepare",
                ]
            )

        else:
            run_prepare = run(
                [
                    "preprocessing.py",
                    "-c",
                    f"{config}",
                    "--var_dict",
                    var_dict,
                    "--sample",
                    f"{sample}",
                    "--tracks",
                    "--prepare",
                ]
            )
        try:
            run_prepare.check_returncode()
        except CalledProcessError:
            logger.info(f"Test failed: preprocessing.py --prepare: {sample}")
            isSuccess = False

        if isSuccess is True:
            run_prepare

    logger.info("Test: running the resampling...")
    if tagger == "dl1r":
        run_resampling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--sample",
                "count",
                "--resampling",
            ]
        )

    else:
        run_resampling = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--var_dict",
                var_dict,
                "--sample",
                "count",
                "--tracks",
                "--resampling",
            ]
        )
    try:
        run_resampling.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: preprocessing.py --resampling.")
        isSuccess = False

    if isSuccess is True:
        run_resampling

    logger.info("Test: retrieving scaling and shifting factors...")

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
        logger.info("Test failed: preprocessing.py --scaling.")
        isSuccess = False

    if isSuccess is True:
        run_scaling

    logger.info("Test: applying shifting and scaling factors...")
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
        logger.info("Test failed: preprocessing.py --apply_scales.")
        isSuccess = False

    if isSuccess is True:
        run_apply_scales

    logger.info(
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
        logger.info("Test failed: preprocessing.py --write.")
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
        logger.info(f"Creating test directory in {test_dir}")
        # clean up, hopefully this causes no "uh oh...""
        if test_dir.startswith("/tmp"):
            run(["rm", "-rf", test_dir])
        run(["mkdir", "-p", test_dir])

        # Make filepaths for basefiles
        run(["mkdir", "-p", os.path.join(test_dir, "ttbar")])
        run(["mkdir", "-p", os.path.join(test_dir, "zpext")])

        # inputs for test will be located in test_dir
        config_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["config"]
        )
        config_paths_source = os.path.join(
            os.getcwd(), self.data["test_preprocessing"]["config_paths"]
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
        self.config_paths = os.path.join(
            test_dir, os.path.basename(config_paths_source)
        )
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

        logger.info(
            f"Preparing config file based on {config_source} in {self.config}..."
        )
        run(["cp", config_source, self.config])
        run(["cp", config_paths_source, self.config_paths])
        run(["cp", var_dict_umami_source, self.var_dict_umami])
        run(["cp", var_dict_dips_source, self.var_dict_dips])
        run(["cp", var_dict_dl1r_source, self.var_dict_dl1r])

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.config_paths,
            "ntuple_path:",
            f"ntuple_path: &ntuple_path {test_dir}",
        )
        replaceLineInFile(
            self.config_paths,
            "file_path:",
            f"file_path: &file_path {test_dir}",
        )
        replaceLineInFile(
            self.config_paths,
            ".outfile_name:",
            f".outfile_name: &outfile_name {self.output}",
        )
        replaceLineInFile(
            self.config_paths,
            ".dict_file:",
            f".dict_file: &dict_file {self.scale_dict}",
        )
        replaceLineInFile(
            self.config,
            "      file_pattern: user.alfroch.410470",
            "      file_pattern: ttbar/*.h5",
        )
        replaceLineInFile(
            self.config,
            "      file_pattern: user.alfroch.427081",
            "      file_pattern: zpext/*.h5",
        )
        replaceLineInFile(self.config, "    iterations:", "    iterations: 1")

        logger.info("Downloading test data...")
        for file in self.data["test_preprocessing"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_preprocessing"]["data_subfolder"],
                file,
            )
            logger.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

        run(
            [
                "cp",
                self.config,
                "/home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/preprocessing_dips/preprocessing/",
            ]
        )
        run(
            [
                "cp",
                self.config_paths,
                "/home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/preprocessing_dips/preprocessing/",
            ]
        )
        run(
            [
                "cp",
                self.var_dict_umami,
                "/home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/preprocessing_dips/preprocessing/",
            ]
        )
        run(
            [
                "cp",
                self.var_dict_dips,
                "/home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/preprocessing_dips/preprocessing/",
            ]
        )
        run(
            [
                "cp",
                self.var_dict_dl1r,
                "/home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/preprocessing_dips/preprocessing/",
            ]
        )

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
