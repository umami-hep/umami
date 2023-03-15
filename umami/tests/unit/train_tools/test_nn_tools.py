"""Unit test script for the NN_tools functions."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile
from subprocess import run

import numpy as np

from umami.configuration import logger, set_log_level
from umami.tools import replace_line_in_file
from umami.train_tools.configuration import TrainConfiguration
from umami.train_tools.nn_tools import (
    CallbackBase,
    MyCallback,
    MyCallbackUmami,
    create_metadata_folder,
    get_dropout_rates,
    get_epoch_from_string,
    get_jet_feature_indices,
    get_jet_feature_position,
    get_metrics_file_name,
    get_model_path,
    get_parameters_from_validation_dict_name,
    get_test_file,
    get_test_sample,
    get_test_sample_trks,
    get_unique_identifiers,
    load_validation_data,
    setup_output_directory,
)

set_log_level(logger, "DEBUG")


class ConfigObject:
    """Object class with attributes."""


class MinimalPreprocessingConfig:
    """
    Init a object with the scale dict as attribute
    """

    def __init__(
        self,
        scale_dict: str,
    ):
        """
        Initalise the minimal preprocessing config so the loading
        of the files is done correctly

        Parameters
        ----------
        scale_dict : str
            Path to the used scale dict.
        """
        self.general = ConfigObject()
        setattr(self.general, "dict_file", scale_dict)


class GetUniqueIdentifiersTestCase(unittest.TestCase):
    """Test class for the unique identifiers."""

    def setUp(self) -> None:
        self.testdict = {
            "X_valid_ttbar": np.random.normal(10),
            "Y_valid_ttbar": np.random.normal(10),
            "X_valid_trks_ttbar": np.random.normal(10),
            "X_valid_zprime": np.random.normal(10),
            "Y_valid_zprime": np.random.normal(10),
            "X_valid_trks_zprime": np.random.normal(10),
        }

    def test_hardcoded_dict(self):
        """Test the hardcoded dicts."""
        self.assertEqual(
            get_unique_identifiers(
                self.testdict.keys(),
                prefix="Y_valid",
            ),
            sorted(["ttbar", "zprime"]),
        )


class GetModelPathTestCase(unittest.TestCase):
    """Test class for the get_model_path function."""

    def setUp(self):
        self.model_name = "dips_test"
        self.epoch = 50
        self.control_model_path = "dips_test/model_files/model_epoch050.h5"

    def test_get_model_path(self):
        """Test nominal behaviour."""
        test_model_path = get_model_path(model_name=self.model_name, epoch=self.epoch)

        self.assertEqual(self.control_model_path, test_model_path)


class GetEpochFromStringTestCase(unittest.TestCase):
    """Test class for the get_epoch_from_string function."""

    def setUp(self):
        self.test_string = "model_epoch11.h5"
        self.int = 11

    def test_get_epoch_from_string(self):
        """Test nominal behaviour."""
        test_int = get_epoch_from_string(self.test_string)

        self.assertEqual(int(test_int), self.int)


class SetupOutputDirectoryTestCase(unittest.TestCase):
    """Test class for the setup_output_directory function."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}"

        # Create file inside the test dir
        os.makedirs(os.path.join(self.tmp_test_dir, "model_files"), exist_ok=True)
        Path(f"{self.tmp_test_dir}/model_files/model_epoch.h5").touch()
        Path(f"{self.tmp_test_dir}/test_val_dict.json").touch()

    def test_setup_output_directory(self):
        """Test nominal behaviour."""
        # Run test function
        setup_output_directory(
            dir_name=f"{self.tmp_test_dir}",
            continue_training=False,
        )

        self.assertFalse(
            os.path.isfile(f"{self.tmp_test_dir}/" + "model_files/model_epoch.h5")
        )

    def test_setup_output_directory_clean(self):
        """Test nominal behaviour with a clean fresh start."""
        # Remove the complete dir
        run(
            ["rm", "-rfv", f"{self.tmp_test_dir}"],
            check=True,
        )

        # Run test function
        setup_output_directory(f"{self.tmp_test_dir}")

        # Check that the function created the folder
        self.assertTrue(os.path.isdir(f"{self.tmp_test_dir}"))

    def test_setup_output_directory_continue(self):
        """Test nominal behaviour with a continuation of the training."""
        # Run test function
        setup_output_directory(
            dir_name=f"{self.tmp_test_dir}",
            continue_training=True,
        )

        # Assert file is still there
        self.assertTrue(
            os.path.isfile(f"{self.tmp_test_dir}/" + "model_files/model_epoch.h5")
        )

    def test_setup_output_directory_error(self):
        """Test error behaviour if file already exists."""
        # Run function and test that error is thrown
        with self.assertRaises(FileExistsError):
            setup_output_directory(
                dir_name=f"{self.tmp_test_dir}/model_files/model_epoch.h5",
                continue_training=False,
            )


class GetMetricsFileNameTestCase(unittest.TestCase):
    """Test class for get_metrics_file_name function."""

    def setUp(self):
        self.dir_name = "test"
        self.dict_name = "validation_WP0p77_300000jets_Dict.json"
        self.working_point = 0.77
        self.n_jets = 300000

    def test_get_dict_name(self):
        """Test nominal behaviour."""
        train_metrics_file_name, val_metrics_file_name = get_metrics_file_name(
            working_point=self.working_point,
            n_jets=self.n_jets,
            dir_name=self.dir_name,
        )

        with self.subTest("Test train dict"):
            self.assertEqual(
                train_metrics_file_name,
                self.dir_name + "/" + "train_metrics_dict.json",
            )

        with self.subTest("Test the validation dict."):
            self.assertEqual(
                val_metrics_file_name,
                self.dir_name + "/" + self.dict_name,
            )

    def test_get_parameters(self):
        """Test the nominal behaviour of the
        get_parameters_from_validation_dict_name function."""
        parameters = get_parameters_from_validation_dict_name(
            self.dir_name + "/" + self.dict_name
        )

        with self.subTest("Test working point parameter"):
            self.assertEqual(parameters["working_point"], self.working_point)

        with self.subTest("Test working n_jets parameter"):
            self.assertEqual(parameters["n_jets"], self.n_jets)

        with self.subTest("Test working dir_name parameter"):
            self.assertEqual(parameters["dir_name"], self.dir_name)

    def test_get_parameters_exception(self):
        """Test execption if wrong parameter is extracted."""
        with self.assertRaises(Exception):
            _ = get_parameters_from_validation_dict_name(
                self.dir_name + "/" + "validation_WP0p77_0jets_Dict.json"
            )

    def test_get_dict_name_without_val_dict_path(self):
        """Test name retrieval without validation dict."""
        train_metrics_file_name, val_metrics_file_name = get_metrics_file_name(
            working_point=self.working_point,
            n_jets=None,
            dir_name=self.dir_name,
        )

        self.assertEqual(
            train_metrics_file_name,
            self.dir_name + "/" + "train_metrics_dict.json",
        )
        self.assertIsNone(val_metrics_file_name)


class CreateMetadataFolderTestCase(unittest.TestCase):
    """Test class for the create_metadata_folder function."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = Path(self.tmp_dir.name)
        self.model_name = Path(self.tmp_test_dir) / "test_model"
        self.train_config_path = Path(self.tmp_test_dir) / "train_config.yaml"
        self.preprocess_config = Path(self.tmp_test_dir) / "preprocess_config.yaml"
        self.preprocess_config_include = (
            Path(self.tmp_test_dir) / "Preprocessing-parameters.yaml"
        )
        self.var_dict_path = Path(self.tmp_test_dir) / "Var_Dict.yaml"
        self.scale_dict_path = Path(self.tmp_test_dir) / "scale_dict.json"
        self.model_file_path = Path(self.tmp_test_dir) / "test_model_file.h5"

        self.var_dict_path.touch()
        self.scale_dict_path.touch()
        self.model_file_path.touch()

        copyfile(
            Path("examples/training/Dips-PFlow-Training-config.yaml"),
            self.train_config_path,
        )
        copyfile(
            Path("examples/preprocessing/PFlow-Preprocessing.yaml"),
            self.preprocess_config,
        )
        copyfile(
            Path("examples/preprocessing/Preprocessing-parameters.yaml"),
            self.preprocess_config_include,
        )
        copyfile(
            Path("examples/preprocessing/Preprocessing-cut_parameters.yaml"),
            Path(self.tmp_test_dir) / "Preprocessing-cut_parameters.yaml",
        )
        copyfile(
            Path("examples/preprocessing/Preprocessing-samples.yaml"),
            Path(self.tmp_test_dir) / "Preprocessing-samples.yaml",
        )

        replace_line_in_file(
            self.preprocess_config,
            "var_file:",
            f"var_file: {self.var_dict_path}",
        )

        replace_line_in_file(
            self.preprocess_config,
            "dict_file:",
            f"dict_file: {self.scale_dict_path}",
        )

    def test_create_metadata_folder(self):
        """Test the nominal behaviour."""
        create_metadata_folder(
            train_config_path=self.train_config_path,
            var_dict_path=self.var_dict_path,
            model_name=self.model_name,
            preprocess_config_path=self.preprocess_config,
            overwrite_config=False,
        )

        with self.subTest("Test train config"):
            self.assertTrue(
                os.path.isfile(
                    os.path.join(self.model_name, "metadata/train_config.yaml")
                )
            )

        with self.subTest("Test preprocess config"):
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        self.model_name,
                        "metadata/preprocess_config.yaml",
                    )
                )
            )

    def test_create_metadata_folder_overwrite(self):
        """Test the overwrite behaviour."""
        create_metadata_folder(
            train_config_path=self.train_config_path,
            var_dict_path=self.var_dict_path,
            model_name=self.model_name,
            preprocess_config_path=self.preprocess_config,
            model_file_path=self.model_file_path,
            overwrite_config=True,
        )

        with self.subTest("Test train config"):
            self.assertTrue(
                os.path.isfile(
                    os.path.join(self.model_name, "metadata", "train_config.yaml")
                )
            )

        with self.subTest("Test preprocess config"):
            self.assertTrue(
                os.path.isfile(
                    os.path.join(self.model_name, "metadata", "preprocess_config.yaml")
                )
            )


class ConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file_dl1r = (
            Path(os.path.dirname(__file__))
            / "../../../../examples/training/DL1r-PFlow-Training-config.yaml"
        )
        self.config_file_dips = (
            Path(os.path.dirname(__file__))
            / "../../../../examples/training/Dips-PFlow-Training-config.yaml"
        )
        self.config_file_cads = (
            Path(os.path.dirname(__file__))
            / "../../../../examples/training/CADS-PFlow-Training-config.yaml"
        )
        self.config_file_umami = (
            Path(os.path.dirname(__file__))
            / "../../../../examples/training/umami-PFlow-Training-config.yaml"
        )

    def test_dips_config(self):
        """Test call of DIPS configuration file."""
        _ = TrainConfiguration(self.config_file_dips)

    def test_cads_config(self):
        """Test call of CADS configuration file."""
        _ = TrainConfiguration(self.config_file_cads)

    def test_umami_config(self):
        """Test call of Umami configuration file."""
        _ = TrainConfiguration(self.config_file_umami)

    def test_no_val_no_eval_batch_size(self):
        """Test the no validation and no evaluation batch size given case."""
        config = TrainConfiguration(self.config_file_dl1r)

        with self.subTest("Test validation batch size"):
            self.assertEqual(
                config.nn_structure.batch_size,
                config.validation_settings.val_batch_size,
            )

        with self.subTest("Test evaluation batch size"):
            self.assertEqual(
                config.validation_settings.val_batch_size,
                config.evaluation_settings.eval_batch_size,
            )

    def test_no_val_batch_size(self):
        """Test the no validation batch size given case."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["evaluation_settings"]["eval_batch_size"] = 50
        config.get_configuration()

        with self.subTest("Test validation batch size"):
            self.assertEqual(
                config.nn_structure.batch_size,
                config.validation_settings.val_batch_size,
            )

        with self.subTest("Test evaluation batch size"):
            self.assertNotEqual(
                config.validation_settings.val_batch_size,
                config.evaluation_settings.eval_batch_size,
            )

    def test_no_eval_batch_size(self):
        """Test the no evaluation batch size given case."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["validation_settings"]["val_batch_size"] = 50
        config.get_configuration()

        with self.subTest("Test evaluation batch size"):
            self.assertEqual(
                config.nn_structure.batch_size,
                config.evaluation_settings.eval_batch_size,
            )

        with self.subTest("Test different val and eval batch sizes"):
            self.assertNotEqual(
                config.validation_settings.val_batch_size,
                config.evaluation_settings.eval_batch_size,
            )

    def test_no_batch_size(self):
        """Test no batch size given error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["batch_size"] = None
        config.config["validation_settings"]["val_batch_size"] = None
        config.config["evaluation_settings"]["eval_batch_size"] = None
        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_missing_key_error(self):
        """Test missing key error."""
        config = TrainConfiguration(self.config_file_dl1r)
        del config.config["model_name"]
        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_double_label_value(self):
        """Test double label error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["class_labels"] = [
            "bjets",
            "singlebjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_double_defined_b_jets(self):
        """Test double defined bjets error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["class_labels"] = [
            "bjets",
            "bbjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_double_defined_c_jets(self):
        """Test double defined cjets error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["class_labels"] = [
            "bjets",
            "ccjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_cads_without_cond_info(self):
        """Test cads without conditions error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["n_conditions"] = 0
        config.config["nn_structure"]["tagger"] = "cads"

        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_dips_att_with_cond_info(self):
        """Test dips attention with conditions error."""
        config = TrainConfiguration(self.config_file_dl1r)
        config.config["nn_structure"]["n_conditions"] = 2
        config.config["nn_structure"]["tagger"] = "dips_attention"

        with self.assertRaises(ValueError):
            config.get_configuration()


class CallbackBaseTestCase(unittest.TestCase):
    """
    Unit tests for the callback base class
    """

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.model_path = f"{self.test_dir.name}/test_model"
        self.n_classes = len(self.class_labels)
        self.target_beff = 0.77
        self.n_jets = 300
        self.val_data_dict = {}
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

    def test_init_no_continue(self):
        """Test init without continue training."""
        CallbackBase(
            model_name=self.model_path,
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=self.n_jets,
            use_lrr=False,
            continue_training=False,
        )

    def test_init_with_continue_new_file_init(self):
        """Test init with continue training."""
        CallbackBase(
            model_name=self.model_path,
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=self.n_jets,
            use_lrr=False,
            continue_training=True,
        )

    def test_init_with_continue_no_file_init(self):
        """Test init with continue training but no file given."""
        # Create dirs and files
        os.makedirs(self.model_path, exist_ok=True)
        test_list = [{"train1": 1}, {"train2": 2}, {"train": 3}]

        with open(
            os.path.join(self.model_path, "train_metrics_dict.json"), "w"
        ) as train_outfile:
            json.dump(test_list, train_outfile, indent=4)
        with open(
            os.path.join(
                self.model_path,
                f"validation_WP0p{int(self.target_beff * 100)}_"
                f"{self.n_jets}jets_Dict.json",
            ),
            "w",
        ) as train_outfile:
            json.dump(test_list, train_outfile, indent=4)

        CallbackBase(
            model_name=self.model_path,
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=self.n_jets,
            use_lrr=False,
            continue_training=True,
        )

    def test_init_no_val_data(self):
        """Test init without valdidation data."""
        CallbackBase(
            model_name=self.model_path,
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=None,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=self.n_jets,
            use_lrr=False,
            continue_training=True,
        )


class MyCallbackTestCase(unittest.TestCase):
    """
    Test the Callback implementation for DIPS
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.n_classes = len(self.class_labels)
        self.target_beff = 0.77
        self.val_data_dict = {}
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

    def test_my_callback(self):
        """Test the MyCallback class"""
        MyCallback(
            model_name=f"{self.test_dir.name}",
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=300,
            use_lrr=False,
        )


class MyCallbackUmamiTestCase(unittest.TestCase):
    """
    Test the Callback implementation for UMAMI
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.n_features_jets = 41
        self.n_trks = 40
        self.n_features_trks = 15
        self.n_classes = len(self.class_labels)
        self.target_beff = 0.77
        self.val_data_dict = {}
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

    def test_my_callback_umami(self):
        """Test the MyCallbackUmami class"""
        MyCallbackUmami(
            model_name=f"{self.test_dir.name}",
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            n_jets=300,
            use_lrr=False,
        )


class GetJetFeatureIndicesTestCase(unittest.TestCase):
    """
    Test the jet features indices slicing.
    """

    def setUp(self):
        self.variable_config = {
            "JetKinematics": ["absEta_btagJes", "pt_btagJes"],
            "JetFitter": ["JetFitter_isDefaults", "JetFitter_mass"],
        }
        self.cutted_variables = [
            "pt_btagJes",
            "JetFitter_isDefaults",
            "JetFitter_mass",
        ]
        self.exclude = ["absEta_btagJes"]
        self.position = 0

    def test_get_jet_feature_indices(self):
        """Test nominal behaviour."""
        variables, excluded_variables, position = get_jet_feature_indices(
            variable_header=self.variable_config,
            exclude=self.exclude + ["Not_existing_var"],
        )

        with self.subTest("Test cutted variables"):
            self.assertEqual(variables, self.cutted_variables)

        with self.subTest("Test excluded variables"):
            self.assertEqual(excluded_variables, self.exclude)

        with self.subTest("Test position of the variables"):
            self.assertEqual(position[0], self.position)

    def test_exclude_in_header(self):
        """Test exclude in header option."""
        variables, excluded_variables, position = get_jet_feature_indices(
            variable_header=self.variable_config,
            exclude=self.exclude + ["JetFitter"],
        )
        self.assertEqual(variables, ["pt_btagJes"])
        self.assertEqual(
            excluded_variables,
            ["JetFitter_isDefaults", "JetFitter_mass", "absEta_btagJes"],
        )
        self.assertEqual(position[0], self.position)


class GetJetFeaturePositionTestCase(unittest.TestCase):
    """
    Test the jet features indices finding.
    """

    def setUp(self):
        self.variable_config = [
            "absEta_btagJes",
            "pt_btagJes",
            "JetFitter_isDefaults",
            "JetFitter_mass",
        ]

        self.repeat_variables = ["pt_btagJes"]
        self.faulty_repeat_variable = ["non_existing_var"]
        self.position = [1]

    def test_get_jet_feature_position(self):
        """Test nominal behaviour."""
        feature_connect_indices = get_jet_feature_position(
            self.repeat_variables, self.variable_config
        )
        self.assertEqual(feature_connect_indices, self.position)

    def test_get_jet_feature_position_value_error(self):
        """Test raise of ValueError."""
        with self.assertRaises(ValueError):
            get_jet_feature_position(self.faulty_repeat_variable, self.variable_config)


class GetSamplesTestCase(unittest.TestCase):
    """
    Test all functions that uses the GetSamples functions
    """

    def setUp(self):
        self.evaluation_settings = {}
        self.tracks_name = "tracks"
        self.nn_structure = {
            "class_labels": ["bjets", "cjets", "ujets"],
            "tagger": "dips",
        }
        self.sampling = {"class_labels": ["bjets", "cjets", "ujets"]}
        self.preparation = {"class_labels": ["bjets", "cjets", "ujets"]}
        self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.validation_files = {
            "ttbar_r21_val": {
                "path": f"{self.test_dir.name}/ci_ttbar_testing.h5",
                "label": "$t\\bar{t}$ Release 21",
                "variable_cuts": [
                    {"pt_btagJes": {"operator": "<=", "condition": 250_000}}
                ],
            },
            "zprime_r21_val": {
                "path": f"{self.test_dir.name}/ci_zpext_testing.h5",
                "label": "$Z'$ Release 21",
                "variable_cuts": None,
            },
        }
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.class_labels_extended = [
            "singlebjets",
            "cjets",
            "ujets",
            "bbjets",
        ]
        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/preprocessing/"
                "ci_ttbar_testing.h5",
                "--directory-prefix",
                self.test_dir.name,
            ],
            check=True,
        )
        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/preprocessing/"
                "ci_zpext_testing.h5",
                "--directory-prefix",
                self.test_dir.name,
            ],
            check=True,
        )

        self.var_dict = os.path.join(
            os.path.dirname(__file__), "fixtures/var_dict_test.yaml"
        )

        self.dict_file = os.path.join(
            os.path.dirname(__file__), "fixtures/scale_dict_test.json"
        )

        self.exclude = ["pt_btagJes"]
        self.n_jets = 1000
        self.length_track_variables = 5
        self.n_trks = 40
        self.config = {"exclude": self.exclude}
        self.preprocess_config = MinimalPreprocessingConfig(
            scale_dict=self.dict_file,
        )

        # object wise train config
        self.test_object = ConfigObject()
        self.test_object.general = ConfigObject()
        self.test_object.nn_structure = ConfigObject()
        self.test_object.nn_structure.class_labels = self.nn_structure["class_labels"]
        self.test_object.nn_structure.tagger = self.nn_structure["tagger"]
        self.test_object.general.exclude = self.exclude
        self.test_object.general.tracks_name = self.tracks_name
        self.test_object.general.validation_files = self.validation_files
        self.test_object.general.var_dict = self.var_dict
        self.test_object.general.preprocess_config = self.preprocess_config

    def test_get_test_sample_trks(self):
        """Test nominal behaviour."""
        x_trk, y_trk = get_test_sample_trks(
            input_file=self.validation_files["ttbar_r21_val"]["path"],
            var_dict=self.var_dict,
            scale_dict=self.dict_file,
            class_labels=self.class_labels,
            tracks_name=self.tracks_name,
            n_jets=self.n_jets,
        )

        with self.subTest("Test track to label length"):
            self.assertEqual(len(x_trk), len(y_trk))

        with self.subTest("Test track shape"):
            self.assertEqual(
                x_trk.shape,
                (len(x_trk), self.n_trks, self.length_track_variables),
            )

        with self.subTest("Test label shape"):
            self.assertEqual(y_trk.shape, (len(y_trk), 3))

    def test_get_test_sample_trks_extended_labeling(self):
        """Test get test sample trks for extended labelling."""
        self.sampling = {"class_labels": ["singlebjets", "cjets", "ujets", "bbjets"]}

        x_trk, y_trk = get_test_sample_trks(
            input_file=self.validation_files["ttbar_r21_val"]["path"],
            var_dict=self.var_dict,
            scale_dict=self.dict_file,
            class_labels=self.class_labels_extended,
            tracks_name=self.tracks_name,
            n_jets=self.n_jets,
        )

        with self.subTest("Test track to label length"):
            self.assertEqual(len(x_trk), len(y_trk))

        with self.subTest("Test track shape"):
            self.assertEqual(
                x_trk.shape,
                (len(x_trk), self.n_trks, self.length_track_variables),
            )

        with self.subTest("Test label shape"):
            self.assertEqual(y_trk.shape, (len(y_trk), 4))

    def test_get_test_sample(self):
        """Test nominal behaviour of get_test_sample"""
        x_test, y_test = get_test_sample(
            input_file=self.validation_files["ttbar_r21_val"]["path"],
            var_dict=self.var_dict,
            scale_dict=self.dict_file,
            class_labels=self.class_labels,
            n_jets=self.n_jets,
            exclude=self.exclude,
        )

        with self.subTest("Test jet to label length"):
            self.assertEqual(len(x_test), len(y_test))

        with self.subTest("Test jet shape"):
            self.assertEqual(x_test.shape, (len(x_test), 3))

        with self.subTest("Test label shape"):
            self.assertEqual(y_test.shape, (len(y_test), 3))

        with self.subTest("Test jet variables"):
            self.assertEqual(
                list(x_test.keys()),
                ["absEta_btagJes", "JetFitter_isDefaults", "JetFitter_mass"],
            )

    def test_get_test_sample_extended_labeling(self):
        """Test get test sample for the extended labelling."""
        self.sampling = {"class_labels": ["singlebjets", "cjets", "ujets", "bbjets"]}

        x_test, y_test = get_test_sample(
            input_file=self.validation_files["ttbar_r21_val"]["path"],
            var_dict=self.var_dict,
            scale_dict=self.dict_file,
            class_labels=self.class_labels_extended,
            n_jets=self.n_jets,
            jet_variables=["pt_btagJes"],
        )
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(x_test.shape, (len(x_test), 1))
        self.assertEqual(y_test.shape, (len(y_test), 4))
        self.assertEqual(
            list(x_test.keys()),
            ["pt_btagJes"],
        )

    def test_get_test_sample_errors(self):
        """Test errors risen by get_test_sample."""
        # Check ValueError if jet_variables and exclude are given
        with self.subTest("Test Value Error"):
            with self.assertRaises(ValueError):
                get_test_sample(
                    input_file=self.validation_files["ttbar_r21_val"]["path"],
                    var_dict=self.var_dict,
                    scale_dict=self.dict_file,
                    class_labels=self.class_labels,
                    n_jets=self.n_jets,
                    exclude=self.exclude,
                    jet_variables=["pt_btagJes"],
                )

        # Check empty filepaths
        with self.subTest("Test Runtime Error"):
            with self.assertRaises(RuntimeError):
                get_test_sample(
                    input_file=f"{self.test_dir.name}/not_existing_file.h5",
                    var_dict=self.var_dict,
                    scale_dict=self.dict_file,
                    class_labels=self.class_labels,
                    n_jets=self.n_jets,
                    exclude=self.exclude,
                )

        # Request variable which is not in scale dict
        with self.subTest("Test Key Error"):
            with self.assertRaises(KeyError):
                get_test_sample(
                    input_file=self.validation_files["ttbar_r21_val"]["path"],
                    var_dict=self.var_dict,
                    scale_dict=self.dict_file,
                    class_labels=self.class_labels,
                    n_jets=self.n_jets,
                    jet_variables=["pt_btagJes", "JetFitter_energyFraction"],
                )

        with self.subTest("Test Runtime Error"):
            with self.assertRaises(RuntimeError):
                get_test_sample_trks(
                    input_file=f"{self.test_dir.name}/not_existing_file.h5",
                    var_dict=self.var_dict,
                    scale_dict=self.dict_file,
                    class_labels=self.class_labels,
                    tracks_name=self.tracks_name,
                    n_jets=self.n_jets,
                )

    def test_get_test_file(self):
        """Test nominal behaviour."""
        (x_valid, x_valid_trk, y_valid,) = get_test_file(
            input_file=self.validation_files["ttbar_r21_val"]["path"],
            var_dict=self.var_dict,
            scale_dict=self.dict_file,
            class_labels=self.class_labels,
            tracks_name=self.tracks_name,
            n_jets=self.n_jets,
            exclude=self.exclude,
        )

        with self.subTest("Test X_valid shape"):
            self.assertEqual(x_valid.shape, (len(x_valid), 3))

        with self.subTest("Test X_valid_trk shape"):
            self.assertEqual(
                x_valid_trk.shape,
                (len(x_valid_trk), self.n_trks, self.length_track_variables),
            )

        with self.subTest("Test Y_valid shape"):
            self.assertEqual(y_valid.shape, (len(y_valid), 3))

    def test_load_validation_data(self):
        """Test the loading of the validation data for umami."""

        for convert_to_tensor in [True, False]:
            for iter_tagger in ["umami", "umami_cond_att", "cads"]:
                with self.subTest(f"{iter_tagger}_tensor_{convert_to_tensor}"):
                    self.test_object.nn_structure.tagger = iter_tagger

                    val_data_dict = load_validation_data(
                        train_config=self.test_object,
                        n_jets=self.n_jets,
                        convert_to_tensor=convert_to_tensor,
                    )

                    self.assertEqual(
                        list(val_data_dict.keys()),
                        [
                            "X_valid_ttbar_r21_val",
                            "X_valid_trk_ttbar_r21_val",
                            "Y_valid_ttbar_r21_val",
                            "X_valid_zprime_r21_val",
                            "X_valid_trk_zprime_r21_val",
                            "Y_valid_zprime_r21_val",
                        ],
                    )

            for iter_tagger in ["dips", "dips_attention", "dl1"]:
                with self.subTest(f"{iter_tagger}_tensor_{convert_to_tensor}"):
                    self.test_object.nn_structure.tagger = iter_tagger

                    val_data_dict = load_validation_data(
                        train_config=self.test_object,
                        n_jets=self.n_jets,
                        convert_to_tensor=convert_to_tensor,
                    )

                    self.assertEqual(
                        list(val_data_dict.keys()),
                        [
                            "X_valid_ttbar_r21_val",
                            "Y_valid_ttbar_r21_val",
                            "X_valid_zprime_r21_val",
                            "Y_valid_zprime_r21_val",
                        ],
                    )

    def test_load_validation_data_unsupported_tagger(self):
        """Test behaviour when not supported tagger is provided."""
        self.test_object.nn_structure.tagger = "not_supported_tagger"

        with self.assertRaises(ValueError):
            load_validation_data(
                train_config=self.test_object,
                n_jets=self.n_jets,
            )

    def test_load_validation_data_no_var_cuts(self):
        """Test the loading of the validation data for umami with no variable cuts."""
        self.test_object.nn_structure.tagger = "umami"

        val_data_dict = load_validation_data(
            train_config=self.test_object,
            n_jets=self.n_jets,
        )

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid_ttbar_r21_val",
                "X_valid_trk_ttbar_r21_val",
                "Y_valid_ttbar_r21_val",
                "X_valid_zprime_r21_val",
                "X_valid_trk_zprime_r21_val",
                "Y_valid_zprime_r21_val",
            ],
        )


class GetDropoutRatesTestCase(unittest.TestCase):
    """Test class for the helper function get_dropout_rates."""

    def setUp(self) -> None:
        self.ppm_sizes = [100, 100, 128]
        self.dropout_rates_phi = [0, 0.1, 0]
        self.dense_sizes = [100, 100, 100, 30]
        self.dropout_rate = [0.1, 0.1, 0.1, 0.1]

    def test_correct_config_dips(self):
        """Test the case for DIPS"""
        with self.subTest():
            self.assertListEqual(
                [0, 0.1, 0],
                get_dropout_rates("dropout_rates_phi", "ppm_sizes", self),
            )
        with self.subTest():
            self.assertListEqual(
                [0.1, 0.1, 0.1, 0.1],
                get_dropout_rates("dropout_rate", "dense_sizes", self),
            )

    def test_wrong_length(self):
        """Test if error is raised if dropout rate is not defined for each layer."""
        self.dropout_rate = [0.1, 0]
        with self.assertRaises(ValueError):
            get_dropout_rates("dropout_rate", "dense_sizes", self)
