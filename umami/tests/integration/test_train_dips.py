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


def runTrainingDips(config):
    """Call train_Dips.py.
    Return value `True` if training succeeded, `False` if one step did not succees."""
    isSuccess = True

    logging.info("Test: running train_Dips.py...")
    run_train_Dips = run(["train_Dips.py", "-c", f"{config}"])
    try:
        run_train_Dips.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: run_train_Dips.py.")
        isSuccess = False

    if isSuccess is True:
        run_train_Dips

    logging.info("Test: running plotting_epoch_performance.py for DIPS...")
    run_plot_epoch_Dips = run(
        [
            "plotting_epoch_performance.py",
            "-c",
            f"{config}",
            "--dips",
        ]
    )
    try:
        run_plot_epoch_Dips.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: plotting_epoch_performance.py for DIPS.")
        isSuccess = False

    if isSuccess is True:
        run_plot_epoch_Dips

    return isSuccess


class TestDipsTraining(unittest.TestCase):
    def setUp(self):
        """Download test files for running the dips training."""
        # Get test configuration
        self.data = getConfiguration()

        test_dir = os.path.join(self.data["test_dips"]["testdir"])
        logging.info(f"Creating test directory in {test_dir}")

        # clean up, hopefully this causes no "uh oh...""
        if test_dir.startswith("/tmp"):
            run(["rm", "-rf", test_dir])
        run(["mkdir", "-p", test_dir])

        # config files, will be copied to test dir
        config_source = os.path.join(
            os.getcwd(), self.data["test_dips"]["config"]
        )
        self.config = os.path.join(test_dir, os.path.basename(config_source))

        preprocessing_config_source = os.path.join(
            "./preprocessing_dips/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["config"]),
        )
        self.preprocessing_config = os.path.join(
            test_dir, os.path.basename(preprocessing_config_source)
        )

        plotting_config_source = os.path.join(
            os.getcwd(), self.data["test_dips"]["plotting_config"]
        )
        self.plotting_config = os.path.join(
            test_dir, os.path.basename(plotting_config_source)
        )

        var_dict_dips_source = os.path.join(
            "./preprocessing_dips/preprocessing/",
            os.path.basename(self.data["test_preprocessing"]["var_dict_dips"]),
        )
        self.var_dict_dips = os.path.join(
            test_dir, os.path.basename(var_dict_dips_source)
        )

        # input files, will be downloaded to test dir
        logging.info("Retrieving files from preprocessing...")
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
        logging.info(
            f"Preparing config file based on {config_source} in {self.config}..."
        )

        run(["touch", self.config])
        run(["cp", preprocessing_config_source, self.preprocessing_config])
        run(["cp", plotting_config_source, self.plotting_config])
        run(["cp", var_dict_dips_source, self.var_dict_dips])

        # modify copy of preprocessing config file for test
        replaceLineInFile(
            self.preprocessing_config,
            "outfile_name:",
            f"outfile_name: {self.train_file}",
        )
        replaceLineInFile(
            self.preprocessing_config,
            "dict_file:",
            f"dict_file: {self.scale_dict}",
        )

        # modify copy of training config file for test
        with open(self.config, "w") as config:
            config.write(
                "model_name: {}\n".format(self.data["test_dips"]["model_name"])
            )
            config.write(f"preprocess_config: {self.preprocessing_config}\n")
            config.write(f"train_file: {self.train_file}\n")
            config.write(f"validation_file: {self.test_file_ttbar}\n")
            config.write(f"add_validation_file: {self.test_file_zprime}\n")
            config.write("ttbar_test_files:\n")
            config.write("zpext_test_files:\n")
            config.write(f"var_dict: {self.var_dict_dips}\n")
            config.write("bool_use_taus: False\n")
            config.write("exclude: []\n")
            config.write("NN_structure:\n")
            config.write("    lr: 0.001\n")
            config.write("    batch_size: 50\n")
            config.write("    epochs: 2\n")
            config.write("    nJets_train: 100\n")
            config.write("    dropout: 0\n")
            config.write("    nClasses: 3\n")
            config.write("    Batch_Normalisation: True\n")
            config.write("    ppm_sizes: [100, 100, 128]\n")
            config.write("    dense_sizes: [100, 100, 100, 30]\n")
            config.write("Eval_parameters_validation:\n")
            config.write("    n_jets: 100\n")
            config.write("    fc_value: 0.018\n")
            config.write("    WP_b: 0.77\n")
            config.write("    UseAtlasTag: True\n")
            config.write('    AtlasTag: "Internal Simulation"\n')
            config.write(r'    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow jets"')
            config.write("\n")
            config.write('    plot_datatype: "pdf"\n')

        logging.info("Downloading test data...")
        for file in self.data["test_dips"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_dips"]["data_subfolder"],
                file,
            )
            logging.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

    def test_train_dips(self):
        """Integration test of train_dips.py script."""
        self.assertTrue(runTrainingDips(self.config))
