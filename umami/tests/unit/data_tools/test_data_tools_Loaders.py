import os
import tempfile
import unittest
from subprocess import run

from umami.configuration import global_config, logger, set_log_level
from umami.data_tools.Loaders import LoadJetsFromFile, LoadTrksFromFile

set_log_level(logger, "DEBUG")


class Load_Files_TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.nJets = 3000

        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/preprocessing/"
                "ci_ttbar_testing.h5",
                "--directory-prefix",
                self.tmp_test_dir,
            ]
        )

    def test_LoadJetsFromFile(self):
        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            nJets=self.nJets,
        )

        self.assertEqual(len(jets), len(labels))

        with self.assertRaises(RuntimeError):
            jets, labels = LoadJetsFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
            )

        for operator in ["<=", "==", ">=", "<", ">"]:
            jets, labels = LoadJetsFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": operator,
                            "condition": 250000,
                        }
                    }
                ],
            )

            self.assertEqual(len(jets), len(labels))

        with self.assertRaises(KeyError):
            jets, labels = LoadJetsFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "=",
                            "condition": 250000,
                        }
                    }
                ],
            )

        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            nJets=self.nJets,
            chunk_size=1000,
        )

        self.assertEqual(len(jets), len(labels))

        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            nJets=self.nJets,
            variables=[
                "dipsLoose20210729_pu",
                "dipsLoose20210729_pb",
                "not_existing_variable",
            ],
            print_logger=True,
            cut_vars_dict=[
                {
                    f"{global_config.pTvariable}": {
                        "operator": "<=",
                        "condition": 250000,
                    }
                }
            ],
        )

        self.assertEqual(len(jets), len(labels))

        with self.assertRaises(KeyError):
            jets, labels = LoadJetsFromFile(
                filepath=666,
                class_labels=self.class_labels,
                nJets=self.nJets,
            )

        with self.assertRaises(RuntimeError):
            jets, labels = LoadJetsFromFile(
                filepath="",
                class_labels=self.class_labels,
                nJets=self.nJets,
            )

    def test_LoadTrksFromFile(self):
        trks, labels = LoadTrksFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            nJets=self.nJets,
        )

        self.assertEqual(len(trks), len(labels))

        with self.assertRaises(RuntimeError):
            trks, labels = LoadTrksFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
            )

        for operator in ["<=", "==", ">=", "<", ">"]:
            trks, labels = LoadTrksFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": operator,
                            "condition": 250000,
                        }
                    }
                ],
            )

            self.assertEqual(len(trks), len(labels))

        with self.assertRaises(KeyError):
            trks, labels = LoadTrksFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                nJets=self.nJets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "=",
                            "condition": 250000,
                        }
                    }
                ],
            )

        trks, labels = LoadTrksFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            nJets=self.nJets,
            chunk_size=1000,
        )

        self.assertEqual(len(trks), len(labels))

        with self.assertRaises(KeyError):
            trks, labels = LoadTrksFromFile(
                filepath=666,
                class_labels=self.class_labels,
                nJets=self.nJets,
            )

        with self.assertRaises(RuntimeError):
            trks, labels = LoadTrksFromFile(
                filepath="",
                class_labels=self.class_labels,
                nJets=self.nJets,
            )
