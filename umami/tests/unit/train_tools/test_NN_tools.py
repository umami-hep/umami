import os
import tempfile
import unittest
from shutil import copyfile
from subprocess import run

import numpy as np

from umami.configuration import global_config
from umami.tools import replaceLineInFile
from umami.train_tools.Configuration import Configuration
from umami.train_tools.NN_tools import (
    GetModelPath,
    GetTestFile,
    GetTestSample,
    GetTestSampleTrks,
    MyCallback,
    MyCallbackUmami,
    create_metadata_folder,
    get_epoch_from_string,
    get_jet_feature_indices,
    get_jet_feature_position,
    get_parameters_from_validation_dict_name,
    get_unique_identifiers,
    get_validation_dict_name,
    get_variable_cuts,
    load_validation_data_dips,
    load_validation_data_umami,
    setup_output_directory,
)


class get_unique_identifiers_TestCase(unittest.TestCase):
    def test_hardcoded_dict(self):
        self.testdict = {
            "X_valid_ttbar": np.random.normal(10),
            "Y_valid_ttbar": np.random.normal(10),
            "X_valid_trks_ttbar": np.random.normal(10),
            "X_valid_zprime": np.random.normal(10),
            "Y_valid_zprime": np.random.normal(10),
            "X_valid_trks_zprime": np.random.normal(10),
        }
        self.assertEqual(
            get_unique_identifiers(
                self.testdict.keys(),
                prefix="Y_valid",
            ),
            sorted(["ttbar", "zprime"]),
        )


class GetModelPath_TestCase(unittest.TestCase):
    def setUp(self):
        self.model_name = "dips_test"
        self.epoch = 50
        self.control_model_path = "dips_test/model_files/model_epoch050.h5"

    def test_GetModelPath(self):
        test_model_path = GetModelPath(model_name=self.model_name, epoch=self.epoch)

        self.assertEqual(self.control_model_path, test_model_path)


class get_variable_cuts_TestCase(unittest.TestCase):
    def setUp(self):
        self.Eval_parameters = {
            "variable_cuts": {
                "validation_file": {
                    "pt_btagJes": {
                        "operator": "<=",
                        "condition": 250000,
                    }
                }
            }
        }

        self.file = "validation_file"
        self.control_dict = {
            "pt_btagJes": {
                "operator": "<=",
                "condition": 250000,
            }
        }

    def test_get_variable_cuts(self):
        # Get dict
        cut_dict = get_variable_cuts(self.Eval_parameters, self.file)

        # Check dict
        self.assertEqual(cut_dict, self.control_dict)

    def testtest_get_variable_cuts_None(self):
        # Get dict
        cut_dict = get_variable_cuts(self.Eval_parameters, "error")

        # Check dict
        self.assertEqual(cut_dict, None)


class get_epoch_from_string_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_string = "model_epoch11.h5"
        self.int = 11

    def test_get_epoch_from_string(self):
        test_int = get_epoch_from_string(self.test_string)

        self.assertEqual(int(test_int), self.int)


class setup_output_directory_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_test_dir = f"{self.tmp_dir.name}"

    def test_setup_output_directory(self):
        # Create file inside the test dir
        run(["touch", f"{self.tmp_test_dir}/" + "model.h5"])

        # Run test function
        setup_output_directory(f"{self.tmp_test_dir}")

        self.assertFalse(os.path.isfile(f"{self.tmp_test_dir}/" + "model.h5"))

    def test_setup_output_directory_clean(self):
        run(["rm", "-rfv", f"{self.tmp_test_dir}"])
        setup_output_directory(f"{self.tmp_test_dir}")

        self.assertTrue(os.path.isdir(f"{self.tmp_test_dir}"))


class dict_name_TestCase(unittest.TestCase):
    def setUp(self):
        self.dir_name = "test"
        self.dict_name = "validation_WP0p77_300000jets_Dict.json"
        self.WP = 0.77
        self.n_jets = 300000

    def test_get_dict_name(self):
        self.assertEqual(
            get_validation_dict_name(
                WP=self.WP,
                n_jets=self.n_jets,
                dir_name=self.dir_name,
            ),
            self.dir_name + "/" + self.dict_name,
        )

    def test_get_parameters(self):
        parameters = get_parameters_from_validation_dict_name(
            self.dir_name + "/" + self.dict_name
        )

        self.assertEqual(parameters["WP"], self.WP)
        self.assertEqual(parameters["n_jets"], self.n_jets)
        self.assertEqual(parameters["dir_name"], self.dir_name)


class create_metadata_folder_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.model_name = os.path.join(self.tmp_test_dir, "test_model")
        self.train_config_path = os.path.join(self.tmp_test_dir, "train_config.yaml")
        self.preprocess_config = os.path.join(
            self.tmp_test_dir, "preprocess_config.yaml"
        )
        self.preprocess_config_include = os.path.join(
            self.tmp_test_dir, "Preprocessing-parameters.yaml"
        )
        self.var_dict_path = os.path.join(self.tmp_test_dir, "Var_Dict.yaml")
        self.scale_dict_path = os.path.join(self.tmp_test_dir, "scale_dict.json")

        run(["touch", f"{self.var_dict_path}"])
        run(["touch", f"{self.scale_dict_path}"])

        copyfile(
            os.path.join(os.getcwd(), "examples/Dips-PFlow-Training-config.yaml"),
            self.train_config_path,
        )
        copyfile(
            os.path.join(os.getcwd(), "examples/PFlow-Preprocessing.yaml"),
            self.preprocess_config,
        )
        copyfile(
            os.path.join(os.getcwd(), "examples/Preprocessing-parameters.yaml"),
            self.preprocess_config_include,
        )

        replaceLineInFile(
            self.train_config_path,
            "var_dict:",
            f"var_dict: {self.var_dict_path}",
        )

        replaceLineInFile(
            self.preprocess_config,
            "dict_file:",
            f"dict_file: {self.scale_dict_path}",
        )

    def test_create_metadata_folder(self):
        create_metadata_folder(
            train_config_path=self.train_config_path,
            var_dict_path=self.var_dict_path,
            model_name=self.model_name,
            preprocess_config_path=self.preprocess_config,
            overwrite_config=False,
        )

        self.assertTrue(
            os.path.isfile(os.path.join(self.model_name, "metadata/train_config.yaml"))
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.model_name,
                    "metadata/preprocess_config.yaml",
                )
            )
        )

        create_metadata_folder(
            train_config_path=self.train_config_path,
            var_dict_path=self.var_dict_path,
            model_name=self.model_name,
            preprocess_config_path=self.preprocess_config,
            overwrite_config=True,
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.model_name, "metadata", "train_config.yaml")
            )
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.model_name, "metadata", "preprocess_config.yaml")
            )
        )


class Configuration_TestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures/test_train_config.yaml"
        )

    def test_missing_key_error(self):
        config = Configuration(self.config_file)
        del config.config["model_name"]
        with self.assertRaises(KeyError):
            config.GetConfiguration()

    def test_double_label_value(self):
        config = Configuration(self.config_file)
        config.NN_structure["class_labels"] = [
            "bjets",
            "singlebjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.GetConfiguration()

    def test_double_defined_b_jets(self):
        config = Configuration(self.config_file)
        config.NN_structure["class_labels"] = [
            "bjets",
            "bbjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.GetConfiguration()

    def test_double_defined_c_jets(self):
        config = Configuration(self.config_file)
        config.NN_structure["class_labels"] = [
            "bjets",
            "ccjets",
            "cjets",
            "ujets",
        ]

        with self.assertRaises(ValueError):
            config.GetConfiguration()


class MyCallback_TestCase(unittest.TestCase):
    """
    Test the Callback implementation for DIPS
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.nTrks = 40
        self.nFeatures = 15
        self.nClasses = len(self.class_labels)
        self.target_beff = 0.77
        self.val_data_dict = {
            "X_valid": np.random.random((10000, self.nTrks, self.nFeatures)),
            "Y_valid": np.random.random((10000, self.nClasses)),
            "X_valid_add": np.random.random((10000, self.nTrks, self.nFeatures)),
            "Y_valid_add": np.random.random((10000, self.nClasses)),
        }
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

    def test_MyCallback(self):
        MyCallback(
            model_name=f"{self.test_dir.name}",
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            dict_file_name=get_validation_dict_name(
                WP=self.target_beff,
                n_jets=300,
                dir_name=f"{self.test_dir.name}",
            ),
        )


class MyCallbackUmami_TestCase(unittest.TestCase):
    """
    Test the Callback implementation for UMAMI
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.main_class = "bjets"
        self.nFeatures_Jets = 41
        self.nTrks = 40
        self.nFeatures_Trks = 15
        self.nClasses = len(self.class_labels)
        self.target_beff = 0.77
        self.val_data_dict = {
            "X_valid": np.random.random((10000, self.nFeatures_Jets)),
            "X_valid_add": np.random.random((10000, self.nFeatures_Jets)),
            "X_valid_trk": np.random.random((10000, self.nTrks, self.nFeatures_Trks)),
            "X_valid_trk_add": np.random.random(
                (10000, self.nTrks, self.nFeatures_Trks)
            ),
            "Y_valid": np.random.random((10000, self.nClasses)),
            "Y_valid_add": np.random.random((10000, self.nClasses)),
        }
        self.frac_dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        }

    def test_MyCallbackUmami(self):
        MyCallbackUmami(
            model_name=f"{self.test_dir.name}",
            class_labels=self.class_labels,
            main_class=self.main_class,
            val_data_dict=self.val_data_dict,
            target_beff=self.target_beff,
            frac_dict=self.frac_dict,
            dict_file_name=get_validation_dict_name(
                WP=self.target_beff,
                n_jets=300,
                dir_name=f"{self.test_dir.name}",
            ),
        )


class get_jet_feature_indices_TestCase(unittest.TestCase):
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
        variables, excluded_variables, position = get_jet_feature_indices(
            self.variable_config, self.exclude
        )
        self.assertEqual(variables, self.cutted_variables)
        self.assertEqual(excluded_variables, self.exclude)
        self.assertEqual(position[0], self.position)


class get_jet_feature_position_TestCase(unittest.TestCase):
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
        self.position = [1]

    def test_get_jet_feature_position(self):
        feature_connect_indices = get_jet_feature_position(
            self.repeat_variables, self.variable_config
        )
        self.assertEqual(feature_connect_indices, self.position)


class GetSamples_TestCase(unittest.TestCase):
    """
    Test all functions that uses the GetSamples functions
    """

    def setUp(self):
        self.Eval_parameters_validation = {}
        self.NN_structure = {"class_labels": ["bjets", "cjets", "ujets"]}
        self.preparation = {"class_labels": ["bjets", "cjets", "ujets"]}
        self.test_dir = tempfile.TemporaryDirectory()
        self.validation_file = (
            f"{self.test_dir.name}/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5"
        )
        self.add_validation_file = (
            f"{self.test_dir.name}/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5"
        )
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
                "https://umami-ci-provider.web.cern.ch/umami/MC16d_hybrid"
                "_odd_100_PFlow-no_pTcuts-file_0.h5",
                "--directory-prefix",
                self.test_dir.name,
            ]
        )
        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/umami/MC16d_hybrid-"
                "ext_odd_0_PFlow-no_pTcuts-file_0.h5",
                "--directory-prefix",
                self.test_dir.name,
            ]
        )

        self.var_dict = os.path.join(
            os.path.dirname(__file__), "fixtures/var_dict_test.yaml"
        )

        self.dict_file = os.path.join(
            os.path.dirname(__file__), "fixtures/scale_dict_test.json"
        )

        self.exclude = ["pt_btagJes"]
        self.nJets = 1000
        self.length_track_variables = 5
        self.nTracks = 40
        self.config = {"exclude": self.exclude}

    def test_GetTestSampleTrks(self):
        X_trk, Y_trk = GetTestSampleTrks(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            class_labels=self.class_labels,
            nJets=self.nJets,
        )
        self.assertEqual(len(X_trk), len(Y_trk))
        self.assertEqual(
            X_trk.shape,
            (len(X_trk), self.nTracks, self.length_track_variables),
        )
        self.assertEqual(Y_trk.shape, (len(Y_trk), 3))

    def test_GetTestSampleTrks_Different_class_labels(self):
        self.class_labels_given = ["ujets", "cjets", "bjets"]

        with self.assertRaises(AssertionError):
            X_trk, Y_trk = GetTestSampleTrks(
                input_file=self.validation_file,
                var_dict=self.var_dict,
                preprocess_config=self,
                class_labels=self.class_labels_given,
                nJets=self.nJets,
            )

    def test_GetTestSampleTrks_Extended_Labeling(self):
        self.preparation = {"class_labels": ["singlebjets", "cjets", "ujets", "bbjets"]}

        X_trk, Y_trk = GetTestSampleTrks(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            class_labels=self.class_labels_extended,
            nJets=self.nJets,
        )
        self.assertEqual(len(X_trk), len(Y_trk))
        self.assertEqual(
            X_trk.shape,
            (len(X_trk), self.nTracks, self.length_track_variables),
        )
        self.assertEqual(Y_trk.shape, (len(Y_trk), 4))

    def test_GetTestSample(self):
        X, Y = GetTestSample(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            class_labels=self.class_labels,
            nJets=self.nJets,
            exclude=self.exclude,
        )
        self.assertEqual(len(X), len(Y))
        self.assertEqual(X.shape, (len(X), 3))
        self.assertEqual(Y.shape, (len(Y), 3))
        self.assertEqual(
            list(X.keys()),
            ["absEta_btagJes", "JetFitter_isDefaults", "JetFitter_mass"],
        )

    def test_GetTestSample_Different_class_labels(self):
        self.class_labels_given = ["ujets", "cjets", "bjets"]

        with self.assertRaises(AssertionError):
            X, Y = GetTestSample(
                input_file=self.validation_file,
                var_dict=self.var_dict,
                preprocess_config=self,
                class_labels=self.class_labels_given,
                nJets=self.nJets,
                exclude=self.exclude,
            )

    def test_GetTestSample_Extended_Labeling(self):
        self.preparation = {"class_labels": ["singlebjets", "cjets", "ujets", "bbjets"]}

        X, Y = GetTestSample(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            class_labels=self.class_labels_extended,
            nJets=self.nJets,
            exclude=self.exclude,
        )
        self.assertEqual(len(X), len(Y))
        self.assertEqual(X.shape, (len(X), 3))
        self.assertEqual(Y.shape, (len(Y), 4))
        self.assertEqual(
            list(X.keys()),
            ["absEta_btagJes", "JetFitter_isDefaults", "JetFitter_mass"],
        )

    def test_GetTestFile(self):
        (X_valid, X_valid_trk, Y_valid,) = GetTestFile(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            class_labels=self.class_labels,
            nJets=self.nJets,
            exclude=self.exclude,
        )
        self.assertEqual(X_valid.shape, (len(X_valid), 3))
        self.assertEqual(
            X_valid_trk.shape,
            (len(X_valid_trk), self.nTracks, self.length_track_variables),
        )
        self.assertEqual(Y_valid.shape, (len(Y_valid), 3))

    def test_load_validation_data_umami(self):
        self.Eval_parameters_validation = {
            "variable_cuts": {
                "validation_file": [
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "<=",
                            "condition": 50_000,
                        }
                    }
                ],
                "add_validation_file": [
                    {
                        f"{global_config.pTvariable}": {
                            "operator": ">",
                            "condition": 50_000,
                        }
                    }
                ],
            }
        }
        val_data_dict = load_validation_data_umami(self, self, self.nJets)

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid",
                "X_valid_trk",
                "Y_valid",
                "X_valid_add",
                "X_valid_trk_add",
                "Y_valid_add",
            ],
        )

    def test_load_validation_data_dips(self):
        self.Eval_parameters_validation = {
            "variable_cuts": {
                "validation_file": [
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "<=",
                            "condition": 50_000,
                        }
                    }
                ],
                "add_validation_file": [
                    {
                        f"{global_config.pTvariable}": {
                            "operator": ">",
                            "condition": 50_000,
                        }
                    }
                ],
            }
        }
        val_data_dict = load_validation_data_dips(self, self, self.nJets)

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid",
                "Y_valid",
                "X_valid_add",
                "Y_valid_add",
            ],
        )

    def test_load_validation_data_umami_no_var_cuts(self):
        val_data_dict = load_validation_data_umami(self, self, self.nJets)

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid",
                "X_valid_trk",
                "Y_valid",
                "X_valid_add",
                "X_valid_trk_add",
                "Y_valid_add",
            ],
        )

    def test_load_validation_data_dips_no_var_cuts(self):
        val_data_dict = load_validation_data_dips(self, self, self.nJets)

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid",
                "Y_valid",
                "X_valid_add",
                "Y_valid_add",
            ],
        )
