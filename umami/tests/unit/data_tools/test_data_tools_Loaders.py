"""Test scripts for the data tools loaders."""
import os
import tempfile
import unittest
from subprocess import run

from umami.configuration import global_config, logger, set_log_level
from umami.data_tools.Loaders import LoadJetsFromFile, LoadTrksFromFile

set_log_level(logger, "DEBUG")


class Load_Files_TestCase(unittest.TestCase):
    """Test class for the different loading functions."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.n_jets = 3000

        run(
            [
                "wget",
                "https://umami-ci-provider.web.cern.ch/preprocessing/"
                "ci_ttbar_testing.h5",
                "--directory-prefix",
                self.tmp_test_dir,
            ],
            check=True,
        )

    def test_LoadJetsFromFile(self):
        """Test the loading of the jet variables from file."""
        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
        )

        self.assertEqual(len(jets), len(labels))

    def test_LoadJetsFromFile_h5_errors(self):
        """Test the raise of errors for wrong h5 file."""
        with self.assertRaises(RuntimeError):
            LoadJetsFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(KeyError):
            LoadJetsFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "=",
                            "condition": 250000,
                        }
                    }
                ],
            )

    def test_LoadJetsFromFile_cut_operators(self):
        """Test different operators for the cuts."""
        for operator in ["<=", "==", ">=", "<", ">"]:
            with self.subTest(f"Testing operator {operator}"):
                jets, labels = LoadJetsFromFile(
                    filepath=os.path.join(
                        self.tmp_test_dir,
                        "ci_ttbar_testing.h5",
                    ),
                    class_labels=self.class_labels,
                    n_jets=self.n_jets,
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

    def test_LoadJetsFromFile_different_chunk_size(self):
        """Test different chunk size."""
        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
            chunk_size=1000,
        )

        self.assertEqual(len(jets), len(labels))

    def test_LoadJetsFromFile_cut_vars(self):
        """Test cut vars."""
        jets, labels = LoadJetsFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
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

    def test_LoadJetsFromFile_wrong_filepath(self):
        """Test raise of error for wrong filepath."""
        with self.assertRaises(KeyError):
            LoadJetsFromFile(
                filepath=666,
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(RuntimeError):
            LoadJetsFromFile(
                filepath="",
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

    def test_LoadTrksFromFile(self):
        """Test the correct loading of the track variables."""
        trks, labels = LoadTrksFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
        )

        self.assertEqual(len(trks), len(labels))

    def test_LoadTrksFromFile_different_operators(self):
        """Test different operators."""
        for operator in ["<=", "==", ">=", "<", ">"]:
            with self.subTest(f"Testing operator {operator}"):
                trks, labels = LoadTrksFromFile(
                    filepath=os.path.join(
                        self.tmp_test_dir,
                        "ci_ttbar_testing.h5",
                    ),
                    class_labels=self.class_labels,
                    n_jets=self.n_jets,
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

    def test_LoadTrksFromFile_different_chunk_size(self):
        """Test different chunk size."""
        trks, labels = LoadTrksFromFile(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
            chunk_size=1000,
        )

        self.assertEqual(len(trks), len(labels))

    def test_LoadTrksFromFile_raise_errors(self):
        """Test the raise of the different errors."""
        with self.assertRaises(RuntimeError):
            LoadTrksFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(KeyError):
            LoadTrksFromFile(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "ci_ttbar_testing.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
                cut_vars_dict=[
                    {
                        f"{global_config.pTvariable}": {
                            "operator": "=",
                            "condition": 250000,
                        }
                    }
                ],
            )

        with self.assertRaises(KeyError):
            LoadTrksFromFile(
                filepath=666,
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(RuntimeError):
            LoadTrksFromFile(
                filepath="",
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )
