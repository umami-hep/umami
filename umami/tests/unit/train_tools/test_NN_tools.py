import os
import tempfile
import unittest
from shutil import copyfile
from subprocess import run

import numpy as np

from umami.tools import replaceLineInFile
from umami.train_tools.Configuration import Configuration
from umami.train_tools.NN_tools import (
    GetRejection,
    GetTestFile,
    GetTestSample,
    GetTestSampleTrks,
    MyCallback,
    MyCallbackDips,
    MyCallbackUmami,
    create_metadata_folder,
    filter_taus,
    get_jet_feature_indices,
    get_parameters_from_validation_dict_name,
    get_validation_dict_name,
    load_validation_data,
)


class GetRejection_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Create a temporary directory
        self.y_pred = np.random.randint(low=0, high=3, size=(100, 3))
        self.y_true = np.random.randint(low=0, high=3, size=(100, 3))
        self.y_pred_tau = np.random.randint(low=0, high=4, size=(100, 4))
        self.y_true_tau = np.random.randint(low=0, high=4, size=(100, 4))

    def test_rejection_noTaus_dtype_b_case(self):
        c_rej, light_rej, _ = GetRejection(
            y_pred=self.y_pred,
            y_true=self.y_true,
            d_type="b",
            taufrac=None,
            use_taus=False,
        )

    def test_rejection_noTaus_dtype_c_case(self):
        b_rej, light_rej, _ = GetRejection(
            y_pred=self.y_pred,
            y_true=self.y_true,
            d_type="c",
            taufrac=None,
            use_taus=False,
        )

    def test_rejection_Taus_dtype_b_case(self):
        c_rej, light_rej, tau_rej, _ = GetRejection(
            y_pred=self.y_pred_tau,
            y_true=self.y_true_tau,
            d_type="b",
            taufrac=0.3,
            use_taus=True,
        )

    def test_rejection_Taus_dtype_c_case(self):
        b_rej, light_rej, tau_rej, _ = GetRejection(
            y_pred=self.y_pred_tau,
            y_true=self.y_true_tau,
            d_type="c",
            taufrac=0.3,
            use_taus=True,
        )


class dict_name_TestCase(unittest.TestCase):
    def setUp(self):
        self.dir_name = "test"
        self.dict_name = "validation_WP0p77_fc0p018_300000jets_Dict.json"
        self.WP_b = 0.77
        self.fc_value = 0.018
        self.n_jets = 300000

    def test_get_dict_name(self):
        self.assertEqual(
            get_validation_dict_name(
                WP_b=self.WP_b,
                fc_value=self.fc_value,
                n_jets=self.n_jets,
                dir_name=self.dir_name,
            ),
            self.dir_name + "/" + self.dict_name,
        )

    def test_get_parameters(self):
        parameters = get_parameters_from_validation_dict_name(
            self.dir_name + "/" + self.dict_name
        )

        self.assertEqual(parameters["WP_b"], self.WP_b)
        self.assertEqual(parameters["fc_value"], self.fc_value)
        self.assertEqual(parameters["n_jets"], self.n_jets)
        self.assertEqual(parameters["dir_name"], self.dir_name)


class create_metadata_folder_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.model_name = os.path.join(self.tmp_test_dir, "test_model")
        self.train_config_path = os.path.join(
            self.tmp_test_dir, "train_config.yaml"
        )
        self.preprocess_config = os.path.join(
            self.tmp_test_dir, "preprocess_config.yaml"
        )
        self.var_dict_path = os.path.join(self.tmp_test_dir, "Var_Dict.yaml")
        self.scale_dict_path = os.path.join(
            self.tmp_test_dir, "scale_dict.json"
        )

        run(["touch", f"{self.var_dict_path}"])
        run(["touch", f"{self.scale_dict_path}"])

        copyfile(
            os.path.join(
                os.getcwd(), "examples/Dips-PFlow-Training-config.yaml"
            ),
            self.train_config_path,
        )
        copyfile(
            os.path.join(os.getcwd(), "examples/PFlow-Preprocessing.yaml"),
            self.preprocess_config,
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
            model_name=self.model_name,
            preprocess_config_path=self.preprocess_config,
            overwrite_config=False,
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.model_name, "metadata/train_config.yaml")
            )
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
                os.path.join(
                    self.model_name, "metadata", "preprocess_config.yaml"
                )
            )
        )


class filter_taus_TestCase(unittest.TestCase):
    def setUp(self):
        self.test = np.random.randint(low=0, high=2, size=(100, 3))
        self.train = np.random.randint(low=0, high=10, size=(100, 3))
        self.test_tau = np.random.randint(low=0, high=2, size=(100, 4))
        self.train_tau = np.random.randint(low=0, high=10, size=(100, 4))

    def test_filter_taus_no_taus(self):
        train_set, test_set = filter_taus(
            train_set=self.train,
            test_set=self.test,
        )

        self.assertEqual(train_set.shape, self.train.shape)
        self.assertEqual(test_set.shape, self.test.shape)

    def test_filter_taus_with_taus(self):
        train_set, test_set = filter_taus(
            train_set=self.train_tau,
            test_set=self.test_tau,
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


class MyCallback_TestCase(unittest.TestCase):
    """
    Test the Callback implementation for DL1
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.val_data_dict = {
            "X_valid": np.random.random((10000, 41)),
            "X_valid_add": np.random.random((10000, 41)),
            "Y_valid": np.random.random((10000, 3)),
            "Y_valid_add": np.random.random((10000, 3)),
        }

        self.Eval_parameters = {
            "n_jets": 3e5,
            "fc_value": 0.018,
            "fb_value": 0.2,
            "ftauforc_value": None,
            "ftauforb_value": None,
            "WP_b": 0.77,
            "WP_c": 0.4,
            "acc_ymin": 0.59,
            "acc_ymax": 1.0,
            "add_variables_eval": ["actualInteractionsPerCrossing"],
            "UseAtlasTag": True,
            "AtlasTag": "Internal Simulation",
            "SecondTag": "\n$\\sqrt{s}=13$ TeV, PFlow jets",
            "plot_datatype": "pdf",
        }

    def test_MyCallback_no_taus(self):
        MyCallback(
            model_name=f"{self.test_dir.name}",
            X_valid=self.val_data_dict["X_valid"],
            Y_valid=self.val_data_dict["Y_valid"],
            X_valid_add=self.val_data_dict["X_valid_add"],
            Y_valid_add=self.val_data_dict["Y_valid_add"],
            include_taus=False,
            eval_config=self.Eval_parameters,
        )

    def test_MyCallback_with_taus(self):
        MyCallback(
            model_name=f"{self.test_dir.name}",
            X_valid=self.val_data_dict["X_valid"],
            Y_valid=self.val_data_dict["Y_valid"],
            X_valid_add=self.val_data_dict["X_valid_add"],
            Y_valid_add=self.val_data_dict["Y_valid_add"],
            include_taus=True,
            eval_config=self.Eval_parameters,
        )


class MyCallbackDips_TestCase(unittest.TestCase):
    """
    Test the Callback implementation for DIPS
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.nTrks = 40
        self.nFeatures = 15
        self.nClasses = 3
        self.val_data_dict = {
            "X_valid": np.random.random((10000, self.nTrks, self.nFeatures)),
            "Y_valid": np.random.random((10000, self.nClasses)),
            "X_valid_add": np.random.random(
                (10000, self.nTrks, self.nFeatures)
            ),
            "Y_valid_add": np.random.random((10000, self.nClasses)),
        }

    def test_MyCallbackDips(self):
        MyCallbackDips(
            model_name=f"{self.test_dir.name}",
            val_data_dict=self.val_data_dict,
            target_beff=0.77,
            charm_fraction=0.018,
            dict_file_name=get_validation_dict_name(
                WP_b=0.77,
                fc_value=0.018,
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
        self.nFeatures_Jets = 41
        self.nTrks = 40
        self.nFeatures_Trks = 15
        self.nClasses = 3
        self.val_data_dict = {
            "X_valid": np.random.random((10000, self.nFeatures_Jets)),
            "X_valid_add": np.random.random((10000, self.nFeatures_Jets)),
            "X_valid_trk": np.random.random(
                (10000, self.nTrks, self.nFeatures_Trks)
            ),
            "X_valid_trk_add": np.random.random(
                (10000, self.nTrks, self.nFeatures_Trks)
            ),
            "Y_valid": np.random.random((10000, self.nClasses)),
            "Y_valid_add": np.random.random((10000, self.nClasses)),
        }

    def test_MyCallbackUmami(self):
        MyCallbackUmami(
            model_name=f"{self.test_dir.name}",
            val_data_dict=self.val_data_dict,
            target_beff=0.77,
            charm_fraction=0.018,
            dict_file_name=get_validation_dict_name(
                WP_b=0.77,
                fc_value=0.018,
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


class GetSamples_TestCase(unittest.TestCase):
    """
    Test all functions that uses the GetSamples functions
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.validation_file = f"{self.test_dir.name}/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5"
        self.add_validation_file = f"{self.test_dir.name}/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5"
        run(
            [
                "wget",
                "https://umami-docs.web.cern.ch/umami-docs/ci/umami/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5",
                "--directory-prefix",
                self.test_dir.name,
            ]
        )
        run(
            [
                "wget",
                "https://umami-docs.web.cern.ch/umami-docs/ci/umami/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5",
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
        self.nJets = 10
        self.length_track_variables = 5
        self.nTracks = 40
        self.config = {"exclude": self.exclude}

    def test_GetTestSampleTrks(self):
        X_trk, Y_trk = GetTestSampleTrks(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            nJets=self.nJets,
        )
        self.assertEqual(
            X_trk.shape,
            (self.nJets, self.nTracks, self.length_track_variables),
        )
        self.assertEqual(Y_trk.shape, (self.nJets, 1))

    def test_GetTestSample(self):
        X, Y = GetTestSample(
            input_file=self.validation_file,
            var_dict=self.var_dict,
            preprocess_config=self,
            nJets=self.nJets,
            exclude=self.exclude,
        )
        self.assertEqual(X.shape, (self.nJets, 3))
        self.assertEqual(Y.shape, (self.nJets, 1))
        self.assertEqual(
            list(X.keys()),
            ["absEta_btagJes", "JetFitter_isDefaults", "JetFitter_mass"],
        )

    def test_GetTestFile(self):
        (X_valid, X_valid_trk, Y_valid,) = GetTestFile(
            self.validation_file,
            self.var_dict,
            self,
            nJets=self.nJets,
            exclude=self.exclude,
        )
        self.assertEqual(X_valid.shape, (self.nJets, 3))
        self.assertEqual(
            X_valid_trk.shape,
            (self.nJets, self.nTracks, self.length_track_variables),
        )
        self.assertEqual(Y_valid.shape, (self.nJets, 1))

    def test_load_validation_data(self):
        val_data_dict = load_validation_data(self, self, self.nJets)

        self.assertEqual(
            list(val_data_dict.keys()),
            [
                "X_valid",
                "X_valid_trk",
                "Y_valid",
                "X_valid_add",
                "Y_valid_add",
                "X_valid_trk_add",
            ],
        )
