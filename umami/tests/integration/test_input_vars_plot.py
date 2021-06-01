import logging
import os
import unittest
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import global_config  # noqa: F401
from umami.tools import yaml_loader


def getConfiguration():
    """Load yaml file with settings for integration test of the input vars plotting."""
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_input_vars_plot"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def runPlotInputVars(config):
    """Call plot_input_vars.py.
    Return value `True` if training succeeded, `False` if one step did not succees."""
    isSuccess = True

    logging.info("Test: running plot_input_vars.py tracks...")
    run_plot_input_vars_trks = run(
        [
            "python",
            "umami/plot_input_variables.py",
            "-c",
            f"{config}",
            "--tracks",
        ]
    )
    try:
        run_plot_input_vars_trks.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: plot_input_variables.py.")
        isSuccess = False

    if isSuccess is True:
        run_plot_input_vars_trks

    logging.info("Test: running plot_input_vars.py jets...")
    run_plot_input_vars_jets = run(
        [
            "python",
            "umami/plot_input_variables.py",
            "-c",
            f"{config}",
            "--jets",
        ]
    )
    try:
        run_plot_input_vars_jets.check_returncode()
    except CalledProcessError:
        logging.info("Test failed: plot_input_variables.py.")
        isSuccess = False

    if isSuccess is True:
        run_plot_input_vars_jets

    return isSuccess


class TestInput_Vars_Plotting(unittest.TestCase):
    def setUp(self):
        """Download test files for input var plots."""
        # Get test configuration
        self.data = getConfiguration()

        test_dir = os.path.join(self.data["test_input_vars_plot"]["testdir"])
        logging.info(f"Creating test directory in {test_dir}")

        # clean up, hopefully this causes no "uh oh...""
        if test_dir.startswith("/tmp"):
            run(["rm", "-rf", test_dir])
        run(["mkdir", "-p", test_dir])

        # input files, will be downloaded to test dir
        logging.info("Retrieving files from preprocessing...")
        self.test_file_r21 = os.path.join(
            test_dir, "plot_input_vars_r21_check.h5"
        )
        self.test_file_r22 = os.path.join(
            test_dir, "plot_input_vars_r22_check.h5"
        )
        self.config = os.path.join(test_dir, "config.yaml")

        run(["touch", self.config])

        # modify copy of training config file for test
        with open(self.config, "w") as config:
            config.write("Eval_parameters:\n")
            config.write("  nJets: 3e3\n")
            config.write("  var_dict: umami/configs/Dips_Variables.yaml\n")
            config.write("jets_input_vars:\n")
            config.write('  variables: "jets"\n')
            config.write("  folder_to_save: jets_input_vars\n")
            config.write("  Datasets_to_plot:\n")
            config.write("    R21:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r21_check.h5\n"
            )
            config.write('      label: "R21 Test"\n')
            config.write("    R22:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r22_check.h5\n"
            )
            config.write('      label: "R22 Test"\n')
            config.write("  plot_settings:\n")
            config.write("    Log: True\n")
            config.write("    UseAtlasTag: True\n")
            config.write('    AtlasTag: "Internal Simulation"\n')
            config.write(
                r'    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3000 Jets"'
            )
            config.write("\n")
            config.write("    yAxisAtlasTag: 0.925\n")
            config.write("    yAxisIncrease: 2\n")
            config.write("    figsize: [7, 5]\n")
            config.write("    bool_use_taus: True\n")
            config.write("  special_param_jets:\n")
            config.write("    IP2D_cu:\n")
            config.write("      lim_left: -30\n")
            config.write("      lim_right: 30\n")
            config.write("    IP2D_bu:\n")
            config.write("      lim_left: -30\n")
            config.write("      lim_right: 30\n")
            config.write("  binning:\n")
            config.write("    IP2D_cu: 5\n")
            config.write("    IP2D_bu:\n")
            config.write("  flavors:\n")
            config.write("    b: 5\n")
            config.write("    c: 4\n")
            config.write("    u: 0\n")
            config.write("    tau: 15\n")
            config.write("\n")
            config.write("nTracks_Test:\n")
            config.write('  variables: "tracks"\n')
            config.write("  folder_to_save: nTracks_Test\n")
            config.write("  nTracks: True\n")
            config.write("  Datasets_to_plot:\n")
            config.write("    R21:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r21_check.h5\n"
            )
            config.write('      label: "R21 Test"\n')
            config.write("    R22:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r22_check.h5\n"
            )
            config.write('      label: "R22 Test"\n')
            config.write("  plot_settings:\n")
            config.write("    Log: True\n")
            config.write("    UseAtlasTag: True\n")
            config.write('    AtlasTag: "Internal Simulation"\n')
            config.write(
                r'    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3000 Jets"'
            )
            config.write("\n")
            config.write("    yAxisAtlasTag: 0.925\n")
            config.write("    yAxisIncrease: 2\n")
            config.write("    figsize: [7, 5]\n")
            config.write("    Ratio_Cut: [0.5, 2]\n")
            config.write("  flavors:\n")
            config.write("    b: 5\n")
            config.write("    c: 4\n")
            config.write("    u: 0\n")
            config.write("\n")
            config.write("Tracks_Test:\n")
            config.write('  variables: "tracks"\n')
            config.write("  folder_to_save: Tracks_Test\n")
            config.write("  Datasets_to_plot:\n")
            config.write("    R21:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r21_check.h5\n"
            )
            config.write('      label: "R21 Test"\n')
            config.write("    R22:\n")
            config.write(
                f"      files: {test_dir}plot_input_vars_r22_check.h5\n"
            )
            config.write('      label: "R22 Test"\n')
            config.write("  plot_settings:\n")
            config.write("    Log: True\n")
            config.write("    UseAtlasTag: True\n")
            config.write('    AtlasTag: "Internal Simulation"\n')
            config.write(
                r'    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3000 Jets"'
            )
            config.write("\n")
            config.write("    yAxisAtlasTag: 0.925\n")
            config.write("    yAxisIncrease: 2\n")
            config.write("    figsize: [7, 5]\n")
            config.write("    Ratio_Cut: [0.5, 2]\n")
            config.write("    bool_use_taus: False\n")
            config.write("  flavors:\n")
            config.write("    b: 5\n")
            config.write("    c: 4\n")
            config.write("    u: 0\n")
            config.write("\n")
            config.write("  plot_settings:\n")
            config.write('    sorting_variable: "ptfrac"\n')
            config.write(
                "    n_Leading: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n"
            )
            config.write("    Log: True\n")
            config.write("    UseAtlasTag: True\n")
            config.write('    AtlasTag: "Internal Simulation"\n')
            config.write(
                r'    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3M Jets"'
            )
            config.write("\n")
            config.write("    yAxisAtlasTag: 0.925\n")
            config.write("    yAxisIncrease: 2\n")
            config.write("    figsize: [7, 5]\n")
            config.write("    Ratio_Cut: [0.5, 1.5]\n")
            config.write("    bool_use_taus: False\n")
            config.write("  binning:\n")
            config.write("    IP3D_signed_d0_significance: 100\n")
            config.write("    numberOfInnermostPixelLayerHits: [0, 4, 1]\n")
            config.write("    dr:\n")
            config.write("  flavors:\n")
            config.write("    b: 5\n")
            config.write("    c: 4\n")
            config.write("    u: 0\n")

        logging.info("Downloading test data...")
        for file in self.data["test_input_vars_plot"]["files"]:
            path = os.path.join(
                self.data["data_url"],
                self.data["test_input_vars_plot"]["data_subfolder"],
                file,
            )
            logging.info(f"Retrieving file from path {path}")
            run(["wget", path, "--directory-prefix", test_dir])

    def test_plot_input_vars(self):
        """Integration test of plot_input_vars.py script."""
        self.assertTrue(runPlotInputVars(self.config))
