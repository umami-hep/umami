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
    for key in ["data_url", "test_dips"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runTrainingDips(config):
    """Call train_Dips.py.
    Return value `True` if training succeeded, `False` if one step did not succees."""
    isSuccess = True

    logger.info("Test: running train_Dips.py...")
    run_train_Dips = run(["train_Dips.py", "-c", f"{config}"])
    try:
        run_train_Dips.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: run_train_Dips.py.")
        isSuccess = False

    if isSuccess is True:
        run_train_Dips

    logger.info("Test: running plotting_epoch_performance.py for DIPS...")
    run_plot_epoch_Dips = run(
        [
            "plotting_epoch_performance.py",
            "-c",
            f"{config}",
            "--tagger",
            "dips",
        ]
    )
    try:
        run_plot_epoch_Dips.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: plotting_epoch_performance.py for DIPS.")
        isSuccess = False

    if isSuccess is True:
        run_plot_epoch_Dips

    logger.info("Test: running evaluate_model.py for DIPS...")
    run_evaluate_model_Dips = run(
        [
            "evaluate_model.py",
            "-c",
            f"{config}",
            "-e",
            "1",
            "--dips",
        ]
    )
    try:
        run_evaluate_model_Dips.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: evaluate_model.py for DIPS.")
        isSuccess = False

    if isSuccess is True:
        run_evaluate_model_Dips

    return isSuccess


class TestDipsTraining(unittest.TestCase):
    def setUp(self):
        """Download test files for running the dips training."""
        # Get test configuration
        self.data = getConfiguration()

        self.test_dir_path = tempfile.TemporaryDirectory()
        test_dir = f"{self.test_dir_path.name}"
        logger.info(f"Creating test directory in {test_dir}")

        # config files, will be copied to test dir
        config_source = os.path.join(
            os.getcwd(), self.data["test_dips"]["config"]
        )
        self.config = os.path.join(test_dir, os.path.basename(config_source))

        preprocessing_config_source = os.path.join(
            "./preprocessing_dips/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["config"]),
        )
        preprocessing_config_paths_source = os.path.join(
            "./preprocessing_dips/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["config_paths"]),
        )
        self.preprocessing_config = os.path.join(
            test_dir, os.path.basename(preprocessing_config_source)
        )
        self.preprocessing_config_paths = os.path.join(
            test_dir, os.path.basename(preprocessing_config_paths_source)
        )

        var_dict_dips_source = os.path.join(
            "./preprocessing_dips/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["var_dict_dips"]),
        )
        self.var_dict_dips = os.path.join(
            test_dir, os.path.basename(var_dict_dips_source)
        )

        # input files, will be downloaded to test dir
        logger.info("Retrieving files from preprocessing...")
        self.train_file = os.path.join(
            "./preprocessing_dips/preprocessing/",
            "PFlow-hybrid_70-test-preprocessed_shuffled.h5",
        )
        self.test_file_ttbar = os.path.join(
            test_dir, "MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5"
        )
        self.test_file_zprime = os.path.join(
            test_dir, "MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5"
        )
        self.scale_dict = os.path.join(
            "./preprocessing_dips/preprocessing/", "PFlow-scale_dict.json"
        )

        # prepare config files by modifying local copies of config files
        logger.info(
            f"Preparing config file based on {config_source} in {self.config}..."
        )

        # Copy configs and var dict
        copyfile(config_source, self.config)
        copyfile(preprocessing_config_source, self.preprocessing_config)
        copyfile(
            preprocessing_config_paths_source, self.preprocessing_config_paths
        )
        copyfile(var_dict_dips_source, self.var_dict_dips)

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.preprocessing_config_paths,
            ".outfile_name:",
            f".outfile_name: &outfile_name {self.train_file}",
        )
        replaceLineInFile(
            self.preprocessing_config_paths,
            ".dict_file:",
            f".dict_file: &dict_file {self.scale_dict}",
        )

        # modify copy of training config file for test
        with open(self.config, "r") as config:
            self.config_file = yaml.load(config, Loader=yaml_loader)

        self.config_file["model_name"] = self.data["test_dips"]["model_name"]
        self.config_file["preprocess_config"] = f"{self.preprocessing_config}"
        self.config_file["train_file"] = f"{self.train_file}"
        self.config_file["validation_file"] = f"{self.test_file_ttbar}"
        self.config_file["add_validation_file"] = f"{self.test_file_zprime}"

        # Erase all not used test files
        del self.config_file["ttbar_test_files"]
        del self.config_file["zpext_test_files"]

        # Add only wanted test files
        self.config_file.update(
            {
                "ttbar_test_files": {
                    "ttbar_r21": {
                        "Path": f"{self.test_file_ttbar}",
                        "data_set_name": "ttbar",
                    }
                }
            }
        )
        self.config_file.update(
            {
                "zpext_test_files": {
                    "zpext_r21": {
                        "Path": f"{self.test_file_zprime}",
                        "data_set_name": "zpext",
                    }
                }
            }
        )
        self.config_file["var_dict"] = f"{self.var_dict_dips}"
        self.config_file["NN_structure"]["batch_size"] = 50
        self.config_file["NN_structure"]["epochs"] = 2
        self.config_file["NN_structure"]["nJets_train"] = 100
        self.config_file["Eval_parameters_validation"]["n_jets"] = 4000
        self.config_file["Eval_parameters_validation"]["low"] = 0.60

        # save the copy of training config file for test
        with open(self.config, "w") as config:
            yaml.dump(self.config_file, config, default_flow_style=False)

        logger.info("Downloading test data...")
        for file in self.data["test_dips"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_dips"]["data_subfolder"],
                file,
            )
            logger.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

    def test_train_dips(self):
        """Integration test of train_dips.py script."""
        self.assertTrue(runTrainingDips(self.config))
