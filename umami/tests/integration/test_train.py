#!/usr/bin/env python

"""
This script integration tests the training of the different models.
"""

import os
import tempfile
import unittest
from shutil import copyfile
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger
from umami.tools import replaceLineInFile, yaml_loader


def getConfiguration():
    """Load yaml file with settings for integration test of dips training."""
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_cads"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def prepareConfig(
    tagger: str,
    test_dir: str,
    preprocess_files_from: str = None,
    useTFRecords: bool = False,
) -> dict:
    """
    Prepare the train config for the given tagger and save it.

    Parameters
    ----------
    tagger : str
        Name of the tagger for which the config is to be prepared.
    test_dir : str
        Path to the test directory where the config file is to be saved.
    preprocess_files_from : str
        Name of the preprocessing files that should be used. If not given,
        the preprocessing files from the tagger will be tried to use.
    useTFRecords : bool
        Decide, if the TFRecords files are used for training or not.

    Returns
    -------
    config : str
        Path to the created config that is to be used for the test.
    """

    # Get test configuration
    data = getConfiguration()

    if tagger != "evalute_comp_taggers":
        # For CADS, the files from the umami preprocessing are used.
        # We need to give this info so those are correctly loaded.
        if preprocess_files_from is None:
            preprocess_files = tagger

        else:
            preprocess_files = preprocess_files_from

        # config files, will be copied to test dir
        preprocessing_config_source = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            os.path.basename(data["test_preprocessing"]["config"]),
        )
        preprocessing_config_paths_source = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            os.path.basename(data["test_preprocessing"]["config_paths"]),
        )
        preprocessing_config = os.path.join(
            test_dir, os.path.basename(preprocessing_config_source)
        )
        preprocessing_config_paths = os.path.join(
            test_dir, os.path.basename(preprocessing_config_paths_source)
        )

        var_dict_source = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            os.path.basename(
                data["test_preprocessing"][f"var_dict_{preprocess_files}"]
            ),
        )
        var_dict = os.path.join(test_dir, os.path.basename(var_dict_source))

        # input files, will be downloaded to test dir
        logger.info("Retrieving files from preprocessing...")

        train_file = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            "PFlow-hybrid_70-test-resampled_scaled_shuffled.h5",
        )
        scale_dict = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            "PFlow-scale_dict.json",
        )

        # Copy preprocess configs and var dict
        copyfile(preprocessing_config_source, preprocessing_config)
        copyfile(preprocessing_config_paths_source, preprocessing_config_paths)
        copyfile(var_dict_source, var_dict)

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            preprocessing_config_paths,
            ".outfile_name:",
            f".outfile_name: &outfile_name {train_file}",
        )
        replaceLineInFile(
            preprocessing_config_paths,
            ".dict_file:",
            f".dict_file: &dict_file {scale_dict}",
        )

    # Get the train config file and copy it to test dir
    config_source = os.path.join(os.getcwd(), data[f"test_{tagger}"]["config"])
    config = os.path.join(test_dir, os.path.basename(config_source))

    # prepare config files by modifying local copies of config files
    logger.info(f"Preparing config file based on {config_source} in {config}...")
    copyfile(config_source, config)

    # Get the path to the test files
    test_file_ttbar = os.path.join(test_dir, "ci_ttbar_testing.h5")
    test_file_zprime = os.path.join(test_dir, "ci_zpext_testing.h5")

    # modify copy of training config file for test
    with open(config, "r") as con:
        config_file = yaml.load(con, Loader=yaml_loader)

    if tagger != "evalute_comp_taggers":
        config_file["preprocess_config"] = f"{preprocessing_config}"
        config_file["train_file"] = f"{train_file}"
        config_file["var_dict"] = f"{var_dict}"
        config_file["NN_structure"]["batch_size"] = 50
        config_file["NN_structure"]["epochs"] = 2
        config_file["NN_structure"]["nJets_train"] = 100

        # Add some validation files for testing
        config_file.update(
            {
                "validation_files": {
                    "ttbar_r21_val": {
                        "path": f"{test_dir}/ci_ttbar_testing.h5",
                        "label": "$t\\bar{t}$ Release 21",
                        "variable_cuts": [
                            {"pt_btagJes": {"operator": "<=", "condition": 250_000}}
                        ],
                    },
                    "zprime_r21_val": {
                        "path": f"{test_dir}/ci_zpext_testing.h5",
                        "label": "$Z'$ Release 21",
                        "variable_cuts": [
                            {"pt_btagJes": {"operator": ">", "condition": 250_000}}
                        ],
                    },
                }
            }
        )

    # Set model_name
    config_file["model_name"] = data[f"test_{tagger}"]["model_name"]

    # Erase all not used test files
    del config_file["test_files"]

    # Add only wanted test files
    config_file.update(
        {
            "test_files": {
                "ttbar_r21": {
                    "path": f"{test_file_ttbar}",
                    "variable_cuts": [
                        {"pt_btagJes": {"operator": "<=", "condition": 250_000}}
                    ],
                },
                "zpext_r21": {
                    "path": f"{test_file_zprime}",
                },
            }
        }
    )

    config_file["Eval_parameters_validation"]["n_jets"] = 4_000
    config_file["Eval_parameters_validation"]["eff_min"] = 0.60

    if useTFRecords is True:
        config_file["train_file"] = os.path.join(
            f"./test_preprocessing_{preprocess_files}/preprocessing/",
            "PFlow-hybrid_70-test-resampled_scaled_shuffled",
        )
        config_file["model_name"] = data["test_dips"]["model_name"] + "_tfrecords"

        config = config[:].replace(".yaml", "") + "_tfrecords.yaml"

    # save the copy of training config file for test
    with open(config, "w") as con:
        yaml.dump(config_file, con, default_flow_style=False)

    # Also return the config already loaded
    return config


def runTraining(config: dict, tagger: str) -> bool:
    """Call train.py for the given tagger.
    Return value `True` if training succeeded, `False` if one step did not succees.

    Parameters
    ----------
    config : dict
        Dict with the needed configurations for training.

    Raises
    ------
    AssertionError
        If train.py fails for the given tagger.
    AssertionError
        If plotting_epoch_performance.py fails for the given tagger.
    AssertionError
        If evaluate_model.py fails for the given tagger.

    Returns
    -------
    isSuccess : bool
        Training succeeded or not.
    """

    if tagger:
        logger.info(f"Test: running train.py for {tagger}...")
        run_train = run(["train.py", "-c", f"{config}"], check=True)

        try:
            run_train.check_returncode()

        except CalledProcessError as Error:
            raise AssertionError(f"Test failed: train.py for {tagger}.") from Error

        logger.info(f"Test: running plotting_epoch_performance.py for {tagger}...")
        run_plot_epoch = run(
            [
                "plotting_epoch_performance.py",
                "-c",
                f"{config}",
            ],
            check=True,
        )

        try:
            run_plot_epoch.check_returncode()

        except CalledProcessError as Error:
            raise AssertionError(
                f"Test failed: plotting_epoch_performance.py for {tagger}."
            ) from Error

    logger.info(f"Test: running evaluate_model.py for {tagger}...")
    run_evaluate_model = run(
        [
            "evaluate_model.py",
            "-c",
            f"{config}",
            "-e",
            "1",
        ],
        check=True,
    )

    try:
        run_evaluate_model.check_returncode()

    except CalledProcessError as Error:
        raise AssertionError(f"Test failed: evaluate_model.py for {tagger}.") from Error

    return True


class TestTraining(unittest.TestCase):
    """Integration test class for the training.

    This class creates a test folder and downloads all important files.
    """

    def setUp(self):
        """Download test files for running the dips training."""
        # Get test configuration
        self.data = getConfiguration()

        self.test_dir_path = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = f"{self.test_dir_path.name}"
        logger.info(f"Creating test directory in {self.test_dir}")

        logger.info("Downloading test data...")
        for file in self.data["test_umami"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_umami"]["data_subfolder"],
                file,
            )
            logger.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", self.test_dir], check=True)

    def test_train_dips_no_attention(self):
        """Integration test of train.py for DIPS script."""

        config = prepareConfig(
            tagger="dips",
            test_dir=self.test_dir,
            preprocess_files_from="dips",
        )

        self.assertTrue(runTraining(config=config, tagger="DIPS"))

    def test_train_cads(self):
        """Integration test of train.py for DIPS Conditional Attention script."""

        config = prepareConfig(
            tagger="cads",
            test_dir=self.test_dir,
            preprocess_files_from="umami",
        )

        self.assertTrue(runTraining(config=config, tagger="DIPS Conditional Attention"))

    def test_train_dl1r(self):
        """Integration test of train.py for DL1r script."""

        config = prepareConfig(
            tagger="dl1r",
            test_dir=self.test_dir,
            preprocess_files_from="dl1r",
        )

        self.assertTrue(runTraining(config=config, tagger="DL1r"))

    def test_train_umami(self):
        """Integration test of train.py for UMAMI script."""

        config = prepareConfig(
            tagger="umami",
            test_dir=self.test_dir,
            preprocess_files_from="umami",
        )

        self.assertTrue(runTraining(config=config, tagger="UMAMI"))

    def test_train_tfrecords_dips(self):
        """Integration test of train.py for DIPS script with TFRecords."""

        config = prepareConfig(
            tagger="dips",
            test_dir=self.test_dir,
            preprocess_files_from="dips",
            useTFRecords=True,
        )

        self.assertTrue(runTraining(config=config, tagger="DIPS"))

    def test_train_tfrecords_umami(self):
        """Integration test of train.py for UMAMI script with TFRecords."""

        config = prepareConfig(
            tagger="umami",
            test_dir=self.test_dir,
            preprocess_files_from="umami",
            useTFRecords=True,
        )

        self.assertTrue(runTraining(config=config, tagger="UMAMI"))

    def test_evaluate_tagger_in_files(self):
        """
        Integration test the evaluation of only the taggers available in
        the files.
        """

        config = prepareConfig(
            tagger="evalute_comp_taggers",
            test_dir=self.test_dir,
        )

        self.assertTrue(runTraining(config=config, tagger=None))
