#!/usr/bin/env python

"""
This script integration tests the preprocessing methods.
"""

import os
import unittest
from shutil import copyfile
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


def runPreprocessing(config: dict, tagger: str, method: str) -> bool:
    """
    Call all steps of the preprocessing for a certain configuration and variable dict
    input.
    Return value `True` if all steps succeeded.

    Parameters
    ----------
    config : dict
        Dict with the needed configurations for the preprocessing.
    tagger : str
        Name of the tagger which the preprocessing should be done for.
        The difference is in saving tracks or not.
    method : str
        Define which sampling method is used.

    Raises
    ------
    AssertionError
        If the prepare step fails.
    AssertionError
        If the resampling step fails.
    AssertionError
        If the scaling step fails.
    AssertionError
        If the apply scaling step fails.
    AssertionError
        If the write step fails.
    AssertionError
        If the to records step fails.

    Returns
    -------
    isSuccess : bool
        Preprocessing succeeded or not.
    """

    logger.info(f"Starting integration test of the {method} method.")
    logger.info("Test: running the prepare...")
    for sample in [
        "training_ttbar_bjets",
        "training_ttbar_cjets",
        "training_ttbar_ujets",
        "training_zprime_bjets",
        "training_zprime_cjets",
        "training_zprime_ujets",
    ]:
        run_prepare = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--sample",
                f"{sample}",
                "--prepare",
            ],
            check=True,
        )

        try:
            run_prepare.check_returncode()

        except CalledProcessError as Error:
            raise AssertionError(
                f"Test failed: preprocessing.py --prepare: {sample}"
            ) from Error

    logger.info("Test: running the resampling...")
    run_resampling = run(
        [
            "preprocessing.py",
            "-c",
            f"{config}",
            "--resampling",
        ],
        check=True,
    )

    try:
        run_resampling.check_returncode()

    except CalledProcessError as Error:
        raise AssertionError("Test failed: preprocessing.py --resampling.") from Error

    logger.info("Test: retrieving scaling and shifting factors...")
    run_scaling = run(
        [
            "preprocessing.py",
            "-c",
            f"{config}",
            "--scaling",
            "--chunk_size",
            "1000",
        ],
        check=True,
    )

    try:
        run_scaling.check_returncode()

    except CalledProcessError as Error:
        raise AssertionError("Test failed: preprocessing.py --scaling.") from Error

    logger.info("Test: applying shifting and scaling factors...")
    run_apply_scales = run(
        [
            "preprocessing.py",
            "-c",
            f"{config}",
            "--apply_scales",
        ],
        check=True,
    )

    try:
        run_apply_scales.check_returncode()
    except CalledProcessError as Error:
        raise AssertionError("Test failed: preprocessing.py --apply_scales.") from Error

    logger.info("Test: shuffling the samples and writing the samples to disk...")
    run_write = run(
        [
            "preprocessing.py",
            "-c",
            f"{config}",
            "--write",
        ],
        check=True,
    )

    try:
        run_write.check_returncode()

    except CalledProcessError as Error:
        raise AssertionError("Test failed: preprocessing.py --write.") from Error

    logger.info(
        "Test: shuffling the samples, writing the samples to disk and convert"
        " them to tf record files..."
    )

    if tagger != "dl1r":
        run_record = run(
            [
                "preprocessing.py",
                "-c",
                f"{config}",
                "--to_records",
            ],
            check=True,
        )

        try:
            run_record.check_returncode()

        except CalledProcessError as Error:
            raise AssertionError(
                "Test failed: preprocessing.py --to_records."
            ) from Error

    tagger_path = f"./preprocessing_{tagger}/"
    if not os.path.isdir(tagger_path):
        run(["mkdir", tagger_path], check=True)

    run(
        [
            "cp",
            "-r",
            "/tmp/umami/preprocessing/",
            tagger_path,
        ],
        check=True,
    )

    # Get the path of the not needed configs
    unused_configs = os.path.join(
        tagger_path, "preprocessing/", "PFlow-Preprocessing_*.yaml"
    )

    # Rename the needed config to PFlow-Preprocessing.yaml and erase the unused
    # TODO change in python 3.10
    if method == "count":
        run(
            [f"rm -rfv {unused_configs}"],
            shell=True,
            check=True,
        )

    elif method == "pdf":
        copyfile(
            os.path.join(tagger_path, "preprocessing/", "PFlow-Preprocessing_pdf.yaml"),
            os.path.join(tagger_path, "preprocessing/", "PFlow-Preprocessing.yaml"),
        )
        run(
            [f"rm -rfv {unused_configs}"],
            shell=True,
            check=True,
        )

    elif method == "weighting":
        copyfile(
            os.path.join(
                tagger_path, "preprocessing/", "PFlow-Preprocessing_weighting.yaml"
            ),
            os.path.join(tagger_path, "preprocessing/", "PFlow-Preprocessing.yaml"),
        )
        run(
            [f"rm -rfv {unused_configs}"],
            shell=True,
            check=True,
        )

    else:
        raise KeyError(f"Method {method} is not supported by the integration test!")

    return True


class TestPreprocessing(unittest.TestCase):
    """Test class for the preprocessing.

    This class sets up the needed configs for testing the preprocessing of Umami.
    """

    def setUp(self):
        """
        Download test files for running the preprocessing and dress
        preprocessing config file.
        """
        # Get test configuration
        self.data = getConfiguration()

        test_dir = os.path.join(self.data["test_preprocessing"]["testdir"])
        logger.info(f"Creating test directory in {test_dir}")
        # clean up, hopefully this causes no "uh oh...""
        if test_dir.startswith("/tmp"):
            run(["rm", "-rf", test_dir], check=True)
        run(["mkdir", "-p", test_dir], check=True)

        # Make filepaths for basefiles
        run(["mkdir", "-p", os.path.join(test_dir, "ttbar")], check=True)
        run(["mkdir", "-p", os.path.join(test_dir, "zpext")], check=True)

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
        copyfile(config_source, self.config)
        copyfile(config_paths_source, self.config_paths)
        copyfile(var_dict_umami_source, self.var_dict_umami)
        copyfile(var_dict_dips_source, self.var_dict_dips)
        copyfile(var_dict_dl1r_source, self.var_dict_dl1r)

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.config_paths,
            "ntuple_path:",
            f"ntuple_path: &ntuple_path {test_dir}",
        )
        replaceLineInFile(
            self.config_paths,
            "sample_path:",
            f"sample_path: &sample_path {test_dir}",
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
            self.config_paths,
            ".intermediate_index_file:",
            ".intermediate_index_file: &intermediate_index_file indices.h5",
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
        replaceLineInFile(
            self.config,
            "    tracks_names:",
            "    tracks_names: ['tracks','tracks_loose']",
        )

        # copy config file and change name to pdf for pdf preprocessing config
        self.pdf_config = self.config[:].replace(".yaml", "") + "_pdf.yaml"
        copyfile(self.config, self.pdf_config)

        # Change the method to pdf and adapt options
        replaceLineInFile(self.pdf_config, "  method: count", "  method: pdf")
        replaceLineInFile(
            self.pdf_config,
            "          bins: [[0, 600000, 351], [650000, 6000000, 84]]",
            "          bins: [[0, 25e4, 100], [25e4, 6e6, 100]]",
        )
        replaceLineInFile(
            self.pdf_config,
            "          bins: [0, 2.5, 10]",
            "          bins: [[0, 2.5, 10], [0, 2.5, 10]]",
        )
        replaceLineInFile(
            self.pdf_config,
            "    njets: 25e6",
            "    njets: -1",
        )
        replaceLineInFile(
            self.pdf_config,
            "      training_ttbar_bjets: 5.5e6",
            "",
        )
        replaceLineInFile(
            self.pdf_config,
            "      training_ttbar_cjets: 11.5e6",
            "",
        )
        replaceLineInFile(
            self.pdf_config,
            "      training_ttbar_ujets: 13.5e6",
            "",
        )

        # copy config file and change name to pdf for pdf preprocessing config
        self.weight_config = self.config[:].replace(".yaml", "") + "_weighting.yaml"
        copyfile(self.config, self.weight_config)

        replaceLineInFile(self.weight_config, "  method: count", "  method: weighting")
        replaceLineInFile(
            self.weight_config,
            "    bool_attach_sample_weights: False",
            "    bool_attach_sample_weights: True",
        )

        logger.info("Downloading test data...")
        for file in self.data["test_preprocessing"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_preprocessing"]["data_subfolder"],
                file,
            )
            logger.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir], check=True)

        run(
            [
                "mv",
                os.path.join(test_dir, "ci_ttbar_basefile.h5"),
                os.path.join(test_dir, "ttbar", "ci_ttbar_basefile.h5"),
            ],
            check=True,
        )
        run(
            [
                "mv",
                os.path.join(test_dir, "ci_zpext_basefile.h5"),
                os.path.join(test_dir, "zpext", "ci_zpext_basefile.h5"),
            ],
            check=True,
        )

    def test_preprocessing_umami_count(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(runPreprocessing(self.config, tagger="umami", method="count"))

    def test_preprocessing_dips_count(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )
        self.assertTrue(runPreprocessing(self.config, tagger="dips", method="count"))

    def test_preprocessing_dl1r_count(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replaceLineInFile(
            self.config,
            "    save_tracks:",
            "    save_tracks: False",
        )

        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(runPreprocessing(self.config, tagger="dl1r", method="count"))

    def test_preprocessing_umami_pdf(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(runPreprocessing(self.pdf_config, tagger="umami", method="pdf"))

    def test_preprocessing_dips_pdf(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )

        self.assertTrue(runPreprocessing(self.pdf_config, tagger="dips", method="pdf"))

    def test_preprocessing_dl1r_pdf(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replaceLineInFile(
            self.pdf_config,
            "    save_tracks:",
            "    save_tracks: False",
        )

        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(runPreprocessing(self.pdf_config, tagger="dl1r", method="pdf"))

    def test_preprocessing_umami_weighting(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(
            runPreprocessing(self.weight_config, tagger="umami", method="weighting")
        )

    def test_preprocessing_dips_weighting(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )
        self.assertTrue(
            runPreprocessing(self.weight_config, tagger="dips", method="weighting")
        )

    def test_preprocessing_dl1r_weighting(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replaceLineInFile(
            self.weight_config,
            "    save_tracks:",
            "    save_tracks: False",
        )

        replaceLineInFile(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(
            runPreprocessing(self.weight_config, tagger="dl1r", method="weighting")
        )
