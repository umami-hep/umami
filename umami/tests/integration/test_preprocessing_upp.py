"""This script integration tests the preprocessing methods."""

import os
import unittest
from pathlib import Path
from shutil import copyfile
from subprocess import CalledProcessError, run

import yaml

from umami.configuration import logger, set_log_level
from umami.tools import replace_line_in_file, yaml_loader

set_log_level(logger, "DEBUG")


def get_configuration():
    """
    Load yaml file with settings for integration test of preprocessing.

    Returns
    -------
    object
        Loaded configuration file.
    """
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    # for key in ["data_url", "test_preprocessing_upp"]:
    #     if key not in conf_setup.keys():
    #         raise yaml.YAMLError(
    #             f"Missing key in yaml file ({path_configuration}): {key}"
    #         )
    return conf_setup


def run_preprocessing(
    config: dict,
    tagger: str,
    method: str,
    string_id: str,
    test_dir: str,
) -> bool:
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
    string_id : str
        Unique identifier to further specify which preprocessing was done.
    test_dir : str
        Path to the directory where all the files are downloaded to etc.

    Returns
    -------
    isSuccess : bool
        Preprocessing succeeded or not.
    """

    logger.info("Starting integration test of the %s method.", method)

    logger.info("Test: running the resampling...")
    run_resampling = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--resampling",
            "--verbose",
        ],
        check=True,
    )

    run_resampling.check_returncode()
    # Commented out due to the fact that tests are also reflected in test coverage
    # and in case of succesful test coverage, the test coverage is not 100%
    # due to lines in except
    # try:
    #     run_resampling.check_returncode()

    # except CalledProcessError as error:
    #     raise AssertionError("Test failed: preprocessing.py --resampling.") from error

    logger.info("Test: retrieving scaling and shifting factors...")
    run_scaling = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--scaling",
            "--verbose",
        ],
        check=True,
    )
    run_scaling.check_returncode()
    # try:
    #     run_scaling.check_returncode()

    # except CalledProcessError as error:
    #     raise AssertionError("Test failed: preprocessing.py --scaling.") from error

    logger.info("Test: shuffling the samples and writing the samples to disk...")
    run_write = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--write",
            "--verbose",
        ],
        check=True,
    )
    run_write.check_returncode()
    # try:
    #     run_write.check_returncode()
    # except CalledProcessError as error:
    #     raise AssertionError("Test failed: preprocessing.py --write.") from error

    logger.info(
        "Test: shuffling the samples, writing the samples to disk and convert"
        " them to tf record files..."
    )

    run_record = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--to_records",
            "--verbose",
        ],
        check=True,
    )
    run_record.check_returncode()
    # try:
    #     run_record.check_returncode()
    # except CalledProcessError as error:
    #     raise AssertionError("Test failed: preprocessing.py --to_records.") from error

    tagger_path = Path(f"./test_preprocessing_{tagger}_{string_id}_{method}/")
    tagger_path_full = Path(f"./test_preprocessing_{tagger}_{string_id}_{method}_full/")

    # Copy the artifacts from tmp to a local folder and the folder which will be
    # uploaded
    for copy_path in [tagger_path, tagger_path_full]:
        run(
            [
                "cp",
                "-rf",
                f"{test_dir}/",
                copy_path,
            ],
            check=True,
        )

    return True


class TestPreprocessingUPP(unittest.TestCase):
    """Test class for the preprocessing with UPP.

    This class sets up the needed configs for testing the preprocessing of UPP.
    """

    def setUp(self):
        """
        Download test files for running the preprocessing and dress
        preprocessing config file.
        """
        # Get test configuration
        data = get_configuration()
        data_url = data["data_url"]
        data = data["test_preprocessing_upp"]

        self.test_dir = Path("user_test_upp")
        logger.info("Creating test directory in %s", self.test_dir)

        # Make filepaths for basefiles
        run(["mkdir", "-p", Path(self.test_dir) / "ttbar"], check=True)
        run(["mkdir", "-p", Path(self.test_dir) / "zpext"], check=True)

        # inputs for test will be located in test_dir
        config_source = Path(data["config"])
        config_paths_source = Path(data["config_paths"])
        var_dict_umami_source = Path(data["var_dict_umami"])
        var_dict_dips_source = Path(data["var_dict_dips"])
        var_dict_gn1_source = Path(data["var_dict_gn1"])
        var_dict_dl1r_source = Path(data["var_dict_dl1r"])
        var_dict_dips_hits_source = Path(data["var_dict_dips_hits"])
        self.pdf_config = self.test_dir / os.path.basename(config_source)
        self.config_paths = self.test_dir / config_paths_source.name
        self.var_dict_umami = self.test_dir / var_dict_umami_source.name
        self.var_dict_dips = self.test_dir / var_dict_dips_source.name
        self.var_dict_gn1 = self.test_dir / var_dict_gn1_source.name
        self.var_dict_dl1r = self.test_dir / var_dict_dl1r_source.name
        self.var_dict_dips_hits = self.test_dir / var_dict_dips_hits_source.name
        self.scale_dict = self.test_dir / "scale_dict.json"
        self.output = self.test_dir / "PFlow-hybrid_70-test.h5"
        self.output_validation = self.test_dir / "PFlow-hybrid-validation-test.h5"
        self.indices = self.test_dir / "indices.h5"
        self.indices_validation = self.test_dir / "indices_validation.h5"

        logger.info(
            "Preparing config file based on %s in %s ...",
            config_source,
            self.pdf_config,
        )
        copyfile(config_source, self.pdf_config)
        copyfile(config_paths_source, self.config_paths)
        copyfile(var_dict_umami_source, self.var_dict_umami)
        copyfile(var_dict_dips_source, self.var_dict_dips)
        copyfile(var_dict_gn1_source, self.var_dict_gn1)
        copyfile(var_dict_dl1r_source, self.var_dict_dl1r)
        copyfile(var_dict_dips_hits_source, self.var_dict_dips_hits)

        replace_line_in_file(
            self.pdf_config,
            "  ntuple_dir:",
            f"  ntuple_dir: {self.test_dir.resolve()}",
        )

        replace_line_in_file(
            self.pdf_config,
            "  base_dir:",
            f"  base_dir: {self.test_dir.resolve()}/output",
        )

        replace_line_in_file(
            self.pdf_config,
            "    dict_file:",
            f"    dict_file: {self.test_dir.resolve()}/output/PFlow-scale_dict.json",
        )

        replace_line_in_file(
            self.pdf_config,
            "  pattern: user.alfroch.410470",
            "  pattern: ttbar/ci_ttbar_basefile.h5",
        )
        replace_line_in_file(
            self.pdf_config,
            "  pattern: user.alfroch.427081",
            "  pattern: zpext/ci_zpext_basefile.h5",
        )
        replace_line_in_file(
            self.pdf_config,
            "      tracks_names:",
            "      tracks_names: ['tracks']",
        )
        replace_line_in_file(
            self.pdf_config,
            "      n_jets_to_plot:",
            "      n_jets_to_plot: null",
        )

        replace_line_in_file(
            self.pdf_config,
            "  sampling_fraction:",
            "  sampling_fraction: 2",
        )

        # copy config file and change name to pdf for pdf preprocessing config
        self.cup_config = str(self.pdf_config)[:].replace(".yaml", "") + "_cup.yaml"
        copyfile(self.pdf_config, self.cup_config)

        replace_line_in_file(
            self.cup_config,
            "  upscale_pdf: 2",
            " ",
        )

        # Change the method to pdf and adapt options
        replace_line_in_file(self.cup_config, "  method: pdf", "  method: countup")

        logger.info("Downloading test data...")
        for file in data["files"]:
            path = os.path.join(
                data_url,
                data["data_subfolder"],
                file,
            )
            logger.info("Retrieving file from path %s", path)
            run(["wget", path, "--directory-prefix", self.test_dir], check=True)

        run(
            [
                "mv",
                os.path.join(self.test_dir, "ci_ttbar_basefile.h5"),
                os.path.join(self.test_dir, "ttbar", "ci_ttbar_basefile.h5"),
            ],
            check=True,
        )
        run(
            [
                "mv",
                os.path.join(self.test_dir, "ci_zpext_basefile.h5"),
                os.path.join(self.test_dir, "zpext", "ci_zpext_basefile.h5"),
            ],
            check=True,
        )
        copyfile(
            os.path.join(self.test_dir, "ci_hits_basefile.h5"),
            os.path.join(self.test_dir, "ttbar", "ci_hits_basefile.h5"),
        )
        run(
            [
                "mv",
                os.path.join(self.test_dir, "ci_hits_basefile.h5"),
                os.path.join(self.test_dir, "zpext", "ci_hits_basefile.h5"),
            ],
            check=True,
        )

    def test_preprocessing_dips_upp_countup(self):
        """Integration test of preprocessing.py with UPP preprocessing."""

        self.assertTrue(
            run_preprocessing(
                self.cup_config,
                tagger="dips",
                method="countup",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_upp_pdf(self):
        """Integration test of preprocessing.py with UPP preprocessing."""

        self.assertTrue(
            run_preprocessing(
                self.pdf_config,
                tagger="dips",
                method="pdf",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_upp_flags(self):
        """Integration test of preprocessing.py with UPP flags."""
        run_resampling = run(
            [
                "python",
                "umami/preprocessing.py",
                "-c",
                f"{self.pdf_config}",
                "--resampling",
                "--verbose",
                "--upp=1",
            ],
            check=True,
        )

        run_resampling.check_returncode()

        with self.assertRaises(CalledProcessError):
            run_resampling = run(
                [
                    "python",
                    "umami/preprocessing.py",
                    "-c",
                    f"{self.pdf_config}",
                    "--resampling",
                    "--verbose",
                    "--upp=2",
                ],
                check=True,
            )

        with self.assertRaises(CalledProcessError):
            run_resampling = run(
                [
                    "python",
                    "umami/preprocessing.py",
                    "-c",
                    f"{self.pdf_config}",
                    "--resampling",
                    "--verbose",
                    "--upp=0",
                ],
                check=True,
            )

        with self.assertRaises(CalledProcessError):
            run_resampling = run(
                [
                    "python",
                    "umami/preprocessing.py",
                    "-c",
                    f"{self.pdf_config}",
                    "--prepare",
                    "--verbose",
                    "--upp=0",
                ],
                check=True,
            )
