"""This script integration tests the preprocessing methods."""

import os
import tempfile
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

    Raises
    ------
    YAMLError
        If a needed key is not in file.
    """
    path_configuration = "umami/tests/integration/fixtures/testSetup.yaml"
    with open(path_configuration, "r") as conf:
        conf_setup = yaml.load(conf, Loader=yaml_loader)
    for key in ["data_url", "test_preprocessing"]:
        if key not in conf_setup.keys():
            raise yaml.YAMLError(
                f"Missing key in yaml file ({path_configuration}): {key}"
            )
    return conf_setup


def run_preprocessing(
    config: dict,
    tagger: str,
    method: str,
    string_id: str,
    test_dir: str,
    flavours_to_process: list = None,
    sample_type_list: list = None,
    sample_usecase_list: list = None,
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
    flavours_to_process : list, optional
        List with the flavours that are to be processed. By default
        None
    sample_type_list: list, optional
        List with the sample types to prepare. By default this will be
        ['ttbar', 'zpext']. By default None
    sample_usecase_list: list, optional
        List with the sample usecases to prepare. By default this will be
        ['training', 'validation'] for DL1r and ["training"] for all other
        taggers. By default None

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
    KeyError
        If the resampling method is not supported by the test.

    Returns
    -------
    isSuccess : bool
        Preprocessing succeeded or not.
    """

    logger.info("Starting integration test of the %s method.", method)
    logger.info("Test: running the prepare...")

    # Check if default needs to be set
    if flavours_to_process is None:
        flavours_to_process = ["ujets", "cjets", "bjets"]

    if sample_type_list is None:
        sample_type_list = ["ttbar", "zprime"]

    if sample_usecase_list is None:
        sample_usecase_list = ["training", "validation"]

    # Generate list of samples
    sample_list = [
        f"{sample_usecase}_{sample_type}_{flavour}"
        for sample_type in sample_type_list
        for flavour in flavours_to_process
        for sample_usecase in sample_usecase_list
    ]

    # Prepare all samples in sample_list
    for sample in sample_list:
        run_prepare = run(
            [
                "python",
                "umami/preprocessing.py",
                "-c",
                f"{config}",
                "--sample",
                f"{sample}",
                "--prepare",
                "--verbose",
            ],
            check=True,
        )

        try:
            run_prepare.check_returncode()

        except CalledProcessError as error:
            raise AssertionError(
                f"Test failed: preprocessing.py --prepare: {sample}"
            ) from error

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

    try:
        run_resampling.check_returncode()

    except CalledProcessError as error:
        raise AssertionError("Test failed: preprocessing.py --resampling.") from error

    # Running the hybrid validation resampling
    logger.info("Test: running the hybrid validation resampling...")
    run_hybrid_validation_resampling = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--resampling",
            "--verbose",
            "--hybrid_validation",
        ],
        check=True,
    )

    try:
        run_hybrid_validation_resampling.check_returncode()

    except CalledProcessError as error:
        raise AssertionError(
            "Test failed: preprocessing.py --resampling --hybrid_validation."
        ) from error

    logger.info("Test: retrieving scaling and shifting factors...")
    run_scaling = run(
        [
            "python",
            "umami/preprocessing.py",
            "-c",
            f"{config}",
            "--scaling",
            "--chunk_size",
            "1000",
            "--verbose",
        ],
        check=True,
    )

    try:
        run_scaling.check_returncode()

    except CalledProcessError as error:
        raise AssertionError("Test failed: preprocessing.py --scaling.") from error

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

    try:
        run_write.check_returncode()

    except CalledProcessError as error:
        raise AssertionError("Test failed: preprocessing.py --write.") from error

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

    try:
        run_record.check_returncode()

    except CalledProcessError as error:
        raise AssertionError("Test failed: preprocessing.py --to_records.") from error

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

        # Get the path of the not needed configs
        unused_configs = copy_path / "PFlow-Preprocessing_*.yaml"

        # Rename the needed config to PFlow-Preprocessing.yaml and erase the unused
        # TODO change in python 3.10
        if method == "count":
            if string_id == "hits":
                copyfile(
                    copy_path / "PFlow-Preprocessing_hits.yaml",
                    copy_path / "PFlow-Preprocessing.yaml",
                )
            run(
                [f"rm -rfv {unused_configs}"],
                shell=True,
                check=True,
            )

        elif method in ["pdf", "weighting", "importance_no_replace"]:
            copyfile(
                copy_path / f"PFlow-Preprocessing_{method}.yaml",
                copy_path / "PFlow-Preprocessing.yaml",
            )
            run(
                [f"rm -rfv {unused_configs}"],
                shell=True,
                check=True,
            )
        else:
            raise KeyError(f"Method {method} is not supported by the integration test!")

        # replace all options which pointed to the test directory `test_dir` and replace
        # with copy_path - here the `Preprocessing-parameters.yaml` file is concerned
        param_content = ""
        with open(copy_path / "Preprocessing-parameters.yaml", "r") as par_conf:
            for line in par_conf:
                if str(test_dir) in line:
                    line = line.replace(str(test_dir), str(copy_path))
                param_content += line
        with open(copy_path / "Preprocessing-parameters.yaml", "w") as par_conf:
            par_conf.write(param_content)

    # Remove all intermediate files which are not needed for the training
    run(
        [f"rm -rfv {tagger_path}/ttbar/"],
        shell=True,
        check=True,
    )
    run(
        [f"rm -rfv {tagger_path}/zpext/"],
        shell=True,
        check=True,
    )
    run(
        [f"rm -rfv {tagger_path}/PDF_sampling/"],
        shell=True,
        check=True,
    )
    run(
        [f"rm -rfv {tagger_path}/*_training_*.h5"],
        shell=True,
        check=True,
    )
    run(
        [f"rm -rfv {tagger_path}/*_validation_*.h5"],
        shell=True,
        check=True,
    )
    run(
        [f"rm -rfv {tagger_path}/PFlow-hybrid_70-test-resampled.h5"],
        shell=True,
        check=True,
    )

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
        data = get_configuration()
        data_url = data["data_url"]
        data = data["test_preprocessing"]

        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = Path(self.tmp_dir.name)
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
        self.config = self.test_dir / os.path.basename(config_source)
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
            "Preparing config file based on %s in %s ...", config_source, self.config
        )
        copyfile(config_source, self.config)
        copyfile(config_paths_source, self.config_paths)
        copyfile(var_dict_umami_source, self.var_dict_umami)
        copyfile(var_dict_dips_source, self.var_dict_dips)
        copyfile(var_dict_gn1_source, self.var_dict_gn1)
        copyfile(var_dict_dl1r_source, self.var_dict_dl1r)
        copyfile(var_dict_dips_hits_source, self.var_dict_dips_hits)

        for elem in data["add_configs"]:
            path_elem = Path(elem)
            copyfile(path_elem, self.test_dir / path_elem.name)

        # modify copy of preprocessing config file for test
        replace_line_in_file(
            self.config_paths,
            "ntuple_path:",
            f"ntuple_path: &ntuple_path {self.test_dir}",
        )
        replace_line_in_file(
            self.config_paths,
            "sample_path:",
            f"sample_path: &sample_path {self.test_dir}",
        )
        replace_line_in_file(
            self.config_paths,
            "file_path:",
            f"file_path: &file_path {self.test_dir}",
        )
        replace_line_in_file(
            self.config_paths,
            ".outfile_name:",
            f".outfile_name: &outfile_name {self.output}",
        )
        replace_line_in_file(
            self.config_paths,
            ".outfile_name_validation:",
            ".outfile_name_validation: &outfile_name_validation"
            f" {self.output_validation}",
        )
        replace_line_in_file(
            self.config_paths,
            ".dict_file:",
            f".dict_file: &dict_file {self.scale_dict}",
        )
        replace_line_in_file(
            self.config_paths,
            ".intermediate_index_file:",
            f".intermediate_index_file: &intermediate_index_file {self.indices}",
        )
        replace_line_in_file(
            self.config_paths,
            ".intermediate_index_file_validation:",
            ".intermediate_index_file_validation: &intermediate_index_file_validation"
            f" {self.indices_validation}",
        )
        replace_line_in_file(
            self.config,
            "      file_pattern: user.alfroch.410470",
            "      file_pattern: ttbar/ci_ttbar_basefile.h5",
        )
        replace_line_in_file(
            self.config,
            "      file_pattern: user.alfroch.427081",
            "      file_pattern: zpext/ci_zpext_basefile.h5",
        )
        replace_line_in_file(
            self.config,
            "    tracks_names:",
            "    tracks_names: ['tracks','tracks_loose']",
        )
        replace_line_in_file(
            self.config,
            "    n_jets_to_plot:",
            "    n_jets_to_plot: null",
        )
        replace_line_in_file(
            self.config,
            "  N_add_vars:",
            "  N_add_vars: 2",
        )

        # copy config file and change name to pdf for pdf preprocessing config
        self.pdf_config = str(self.config)[:].replace(".yaml", "") + "_pdf.yaml"
        copyfile(self.config, self.pdf_config)

        # Change the method to pdf and adapt options
        replace_line_in_file(self.pdf_config, "  method: count", "  method: pdf")
        replace_line_in_file(
            self.pdf_config,
            "          bins: [[0, 600000, 351], [650000, 6000000, 84]]",
            "          bins: [[0, 25e4, 100], [25e4, 6e6, 100]]",
        )
        replace_line_in_file(
            self.pdf_config,
            "          bins: [0, 2.5, 10]",
            "          bins: [[0, 2.5, 10], [0, 2.5, 10]]",
        )
        replace_line_in_file(
            self.pdf_config,
            "    n_jets: 25e6",
            "    n_jets: 1e4",
        )
        replace_line_in_file(
            self.pdf_config,
            "    n_jets_validation: 4e6",
            "    n_jets_validation: 1e4",
        )
        replace_line_in_file(
            self.pdf_config,
            "      training_ttbar_bjets: 5.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "      training_ttbar_cjets: 11.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "      training_ttbar_ujets: 13.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "      validation_ttbar_bjets: 5.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "      validation_ttbar_cjets: 11.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "      validation_ttbar_ujets: 13.5e6",
            "",
        )
        replace_line_in_file(
            self.pdf_config,
            "    n_jets_to_plot:",
            "    n_jets_to_plot: 3e4",
        )

        # copy config file and change name to pdf for pdf preprocessing config
        self.weight_config = (
            str(self.config)[:].replace(".yaml", "") + "_weighting.yaml"
        )
        copyfile(self.config, self.weight_config)

        replace_line_in_file(
            self.weight_config, "  method: count", "  method: weighting"
        )
        replace_line_in_file(
            self.weight_config,
            "    bool_attach_sample_weights: False",
            "    bool_attach_sample_weights: True",
        )
        replace_line_in_file(
            self.weight_config,
            "    n_jets_to_plot:",
            "    n_jets_to_plot: null",
        )

        self.importance_no_replace_config = (
            str(self.config)[:].replace(".yaml", "") + "_importance_no_replace.yaml"
        )
        copyfile(self.config, self.importance_no_replace_config)

        replace_line_in_file(
            self.importance_no_replace_config,
            "  method: count",
            "  method: importance_no_replace",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "    weighting_target_flavour: 'bjets'",
            "    target_distribution: 'bjets'",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "          bins: [[0, 600000, 351], [650000, 6000000, 84]]",
            "          bins: [[0, 600000, 100], [600500, 6000000, 20]]",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "          bins: [0, 2.5, 10]",
            "          bins: [0, 2.5, 2]",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "    custom_n_jets_initial:",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      training_ttbar_bjets: 5.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      training_ttbar_cjets: 11.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      training_ttbar_ujets: 13.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      validation_ttbar_bjets: 5.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      validation_ttbar_cjets: 11.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      validation_ttbar_ujets: 13.5e6",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "    fractions:",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      ttbar: 0.7",
            "",
        )
        replace_line_in_file(
            self.importance_no_replace_config,
            "      zprime: 0.3",
            "",
        )

        # copy config file and change name to hits for hit preprocessing test
        self.hits_config = str(self.config)[:].replace(".yaml", "") + "_hits.yaml"
        copyfile(self.config, self.hits_config)

        replace_line_in_file(
            self.hits_config,
            "      file_pattern: ttbar/ci_ttbar_basefile.h5",
            "      file_pattern: ttbar/ci_hits_basefile.h5",
        )
        replace_line_in_file(
            self.hits_config,
            "      file_pattern: zpext/ci_zpext_basefile.h5",
            "      file_pattern: zpext/ci_hits_basefile.h5",
        )
        replace_line_in_file(
            self.hits_config,
            "    tracks_names:",
            "    tracks_names: ['hits']",
        )
        replace_line_in_file(
            self.hits_config,
            "      ttbar: 0.7",
            "      ttbar: 0.0",
        )
        replace_line_in_file(
            self.hits_config,
            "      zprime: 0.3",
            "      zprime: 1.0",
        )
        replace_line_in_file(
            self.hits_config,
            "    save_track_labels: True",
            "    save_track_labels: False",
        )
        replace_line_in_file(
            self.hits_config,
            "      ttbar:",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - training_ttbar_bjets",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - training_ttbar_cjets",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - training_ttbar_ujets",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - validation_ttbar_bjets",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - validation_ttbar_cjets",
            "",
        )
        replace_line_in_file(
            self.hits_config,
            "        - validation_ttbar_ujets",
            "",
        )

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

    def test_preprocessing_additional_jet_labels(self):
        """Integration test of preprocessing.py script using GN1 variables, and
        allowing for additional jet labels"""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_gn1}",
        )
        self.assertTrue(
            run_preprocessing(
                self.config,
                tagger="gn1",
                method="count",
                string_id="additional_jet_labels",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_umami_count(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(
            run_preprocessing(
                self.config,
                tagger="umami",
                method="count",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_count(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )
        self.assertTrue(
            run_preprocessing(
                self.config,
                tagger="dips",
                method="count",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_hits_count(self):
        """Integration test of preprocessing.py script using DIPS with hits
        variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips_hits}",
        )
        self.assertTrue(
            run_preprocessing(
                self.hits_config,
                tagger="dips",
                method="count",
                string_id="hits",
                test_dir=self.test_dir,
                sample_type_list=["zprime"],
            )
        )

    def test_preprocessing_dl1r_count(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replace_line_in_file(
            self.config,
            "    save_tracks:",
            "    save_tracks: False",
        )
        replace_line_in_file(
            self.config,
            "    tracks_names:",
            "    tracks_names: null",
        )
        replace_line_in_file(
            self.config,
            "    save_track_labels:",
            "    save_track_labels: False",
        )
        replace_line_in_file(
            self.config,
            "    track_truth_variables:",
            "    track_truth_variables: null",
        )
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(
            run_preprocessing(
                self.config,
                tagger="dl1r",
                method="count",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_umami_pdf(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(
            run_preprocessing(
                self.pdf_config,
                tagger="umami",
                method="pdf",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_pdf(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )

        self.assertTrue(
            run_preprocessing(
                self.pdf_config,
                tagger="dips",
                method="pdf",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_four_classes_pdf(self):
        """Integration test of preprocessing.py script using DIPS variables and four
        classes."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )

        replace_line_in_file(
            self.pdf_config,
            "    n_jets_to_plot:",
            "    n_jets_to_plot: null",
        )

        replace_line_in_file(
            self.pdf_config,
            "  class_labels:",
            "  class_labels: [ujets, cjets, bjets, taujets]",
        )

        replace_line_in_file(
            self.pdf_config,
            "        - training_ttbar_ujets",
            "        - training_ttbar_ujets\n        - training_ttbar_taujets",
        )

        replace_line_in_file(
            self.pdf_config,
            "        - training_zprime_ujets",
            "        - training_zprime_ujets\n        - training_zprime_taujets",
        )

        self.assertTrue(
            run_preprocessing(
                self.pdf_config,
                tagger="dips",
                method="pdf",
                string_id="four_classes",
                test_dir=self.test_dir,
                flavours_to_process=["ujets", "cjets", "bjets", "taujets"],
            )
        )

    def test_preprocessing_dl1r_pdf(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replace_line_in_file(
            self.pdf_config,
            "    save_tracks:",
            "    save_tracks: False",
        )
        replace_line_in_file(
            self.config,
            "    tracks_names:",
            "    tracks_names: null",
        )
        replace_line_in_file(
            self.config,
            "    save_track_labels:",
            "    save_track_labels: False",
        )
        replace_line_in_file(
            self.config,
            "    track_truth_variables:",
            "    track_truth_variables: null",
        )
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(
            run_preprocessing(
                self.pdf_config,
                tagger="dl1r",
                method="pdf",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_umami_weighting(self):
        """Integration test of preprocessing.py script using Umami variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(
            run_preprocessing(
                self.weight_config,
                tagger="umami",
                method="weighting",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dips_weighting(self):
        """Integration test of preprocessing.py script using DIPS variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dips}",
        )
        self.assertTrue(
            run_preprocessing(
                self.weight_config,
                tagger="dips",
                method="weighting",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_dl1r_weighting(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replace_line_in_file(
            self.weight_config,
            "    save_tracks:",
            "    save_tracks: False",
        )
        replace_line_in_file(
            self.config,
            "    tracks_names:",
            "    tracks_names: null",
        )
        replace_line_in_file(
            self.config,
            "    save_track_labels:",
            "    save_track_labels: False",
        )
        replace_line_in_file(
            self.config,
            "    track_truth_variables:",
            "    track_truth_variables: null",
        )
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_dl1r}",
        )

        self.assertTrue(
            run_preprocessing(
                self.weight_config,
                tagger="dl1r",
                method="weighting",
                string_id="base",
                test_dir=self.test_dir,
            )
        )

    def test_preprocessing_umami_importance_no_replace(self):
        """Integration test of preprocessing.py script using DL1r variables."""
        replace_line_in_file(
            self.config_paths,
            ".var_file:",
            f".var_file: &var_file {self.var_dict_umami}",
        )

        self.assertTrue(
            run_preprocessing(
                self.importance_no_replace_config,
                tagger="umami",
                method="importance_no_replace",
                string_id="base",
                test_dir=self.test_dir,
            )
        )
