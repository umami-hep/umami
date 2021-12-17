import os
import tempfile
import unittest
from shutil import copyfile
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger
from umami.tools import replaceLineInFile, yaml_loader


def getConfiguration():
    """Load yaml file with settings for integration test of DL1 training."""
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_dl1"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runTrainingDL1(config):
    """Call train.py for DL1.
    Return value `True` if training succeeded, `False` if one step did not succees."""
    isSuccess = True

    logger.info("Test: running train.py for DL1...")
    run_train_DL1 = run(["train.py", "-c", f"{config}"])
    try:
        run_train_DL1.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: train.py for DL1.")
        isSuccess = False

    if isSuccess is True:
        run_train_DL1

    logger.info("Test: running plotting_epoch_performance.py for DL1...")
    run_plot_epoch_DL1 = run(
        [
            "plotting_epoch_performance.py",
            "-c",
            f"{config}",
        ]
    )
    try:
        run_plot_epoch_DL1.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: plotting_epoch_performance.py for DL1.")
        isSuccess = False

    if isSuccess is True:
        run_plot_epoch_DL1

    logger.info("Test: running evaluate_model.py for DL1...")
    run_evaluate_model_DL1 = run(
        [
            "evaluate_model.py",
            "-c",
            f"{config}",
            "-e",
            "1",
        ]
    )
    try:
        run_evaluate_model_DL1.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: evaluate_model.py for DL1.")
        isSuccess = False

    if isSuccess is True:
        run_evaluate_model_DL1

    logger.info("Test: running evaluate_model.py for DL1 with shapley option...")
    run_evaluate_model_DL1_with_shapley = run(
        [
            "evaluate_model.py",
            "-c",
            f"{config}",
            "-e",
            "1",
            "--shapley",
        ]
    )
    try:
        run_evaluate_model_DL1_with_shapley.check_returncode()
    except CalledProcessError:
        logger.info("Test failed: evaluate_model.py for DL1 with shapley option.")
        isSuccess = False

    if isSuccess is True:
        run_evaluate_model_DL1_with_shapley

    return isSuccess


class TestDL1Training(unittest.TestCase):
    def setUp(self):
        """Download test files for running the DL1 training."""
        # Get test configuration
        self.data = getConfiguration()

        self.test_dir_path = tempfile.TemporaryDirectory()
        test_dir = f"{self.test_dir_path.name}"
        logger.info(f"Creating test directory in {test_dir}")

        # config files, will be copied to test dir
        config_source = os.path.join(os.getcwd(), self.data["test_dl1"]["config"])
        self.config = os.path.join(test_dir, os.path.basename(config_source))

        preprocessing_config_source = os.path.join(
            "./preprocessing_dl1r/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["config"]),
        )
        preprocessing_config_paths_source = os.path.join(
            "./preprocessing_dl1r/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["config_paths"]),
        )
        self.preprocessing_config = os.path.join(
            test_dir, os.path.basename(preprocessing_config_source)
        )
        self.preprocessing_config_paths = os.path.join(
            test_dir, os.path.basename(preprocessing_config_paths_source)
        )

        var_dict_dl1_source = os.path.join(
            "./preprocessing_dl1r/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["var_dict_dl1r"]),
        )
        self.var_dict_dl1 = os.path.join(
            test_dir, os.path.basename(var_dict_dl1_source)
        )

        # input files, will be downloaded to test dir
        logger.info("Retrieving files from preprocessing...")
        self.train_file = os.path.join(
            "./preprocessing_dl1r/preprocessing/",
            "PFlow-hybrid_70-test-resampled_scaled_shuffled.h5",
        )
        self.test_file_ttbar = os.path.join(
            test_dir, "MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5"
        )
        self.test_file_zprime = os.path.join(
            test_dir, "MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5"
        )
        self.scale_dict = os.path.join(
            "./preprocessing_dl1r/preprocessing/", "PFlow-scale_dict.json"
        )

        # prepare config files by modifying local copies of config files
        logger.info(
            f"Preparing config file based on {config_source} in {self.config}..."
        )

        # Copy configs and var dict
        copyfile(config_source, self.config)
        copyfile(preprocessing_config_source, self.preprocessing_config)
        copyfile(preprocessing_config_paths_source, self.preprocessing_config_paths)
        copyfile(var_dict_dl1_source, self.var_dict_dl1)

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

        self.config_file["model_name"] = self.data["test_dl1"]["model_name"]
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
        self.config_file["var_dict"] = f"{self.var_dict_dl1}"
        self.config_file["NN_structure"]["batch_size"] = 50
        self.config_file["NN_structure"]["epochs"] = 2
        self.config_file["NN_structure"]["nJets_train"] = 100
        self.config_file["Eval_parameters_validation"]["n_jets"] = 4000
        self.config_file["Eval_parameters_validation"]["eff_min"] = 0.60
        self.config_file["Eval_parameters_validation"]["variable_cuts"] = {
            "validation_file": [
                {"pt_btagJes": {"operator": "<=", "condition": 250000}}
            ],
            "add_validation_file": [
                {"pt_btagJes": {"operator": ">", "condition": 250000}}
            ],
            "ttbar": [{"pt_btagJes": {"operator": "<=", "condition": 250000}}],
            "zpext": [{"pt_btagJes": {"operator": ">", "condition": 250000}}],
        }
        self.config_file["Eval_parameters_validation"]["shapley"]["feature_sets"] = 6
        self.config_file["Eval_parameters_validation"]["shapley"]["averaged_sets"] = 2

        # save the copy of training config file for test
        with open(self.config, "w") as config:
            yaml.dump(self.config_file, config, default_flow_style=False)

        logger.info("Downloading test data...")
        for file in self.data["test_dl1"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_dl1"]["data_subfolder"],
                file,
            )
            logger.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

    def test_train_DL1(self):
        """Integration test of train.py for DL1 script."""
        self.assertTrue(runTrainingDL1(self.config))
