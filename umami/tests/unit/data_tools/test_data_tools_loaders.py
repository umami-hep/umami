"""Test scripts for the data tools loaders."""
import os
import tempfile
import unittest
from subprocess import run

from umami.configuration import global_config, logger, set_log_level
from umami.data_tools.loaders import load_jets_from_file, load_trks_from_file

set_log_level(logger, "DEBUG")


class LoadFilesTestCase(unittest.TestCase):
    """Test class for the different loading functions."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.class_labels = ["bjets", "cjets", "ujets"]
        self.n_jets = 3000

        run(
            [
                "wget",
                (
                    "https://umami-ci-provider.web.cern.ch/preprocessing/"
                    "ci_ttbar_testing.h5"
                ),
                "--directory-prefix",
                self.tmp_test_dir,
            ],
            check=True,
        )

    def test_load_jets_from_file(self):
        """Test the loading of the jet variables from file."""
        jets, labels = load_jets_from_file(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
        )

        self.assertEqual(len(jets), len(labels))

    def test_h5_errors(self):
        """Test the raise of errors for wrong h5 file."""
        with self.assertRaises(RuntimeError):
            load_jets_from_file(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(KeyError):
            load_jets_from_file(
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

    def test_cut_operators(self):
        """Test different operators for the cuts."""
        for operator in ["<=", "==", ">=", "<", ">"]:
            with self.subTest(f"Testing operator {operator}"):
                jets, labels = load_jets_from_file(
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

    def test_different_chunk_size(self):
        """Test different chunk size."""
        jets, labels = load_jets_from_file(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
            chunk_size=1000,
        )

        self.assertEqual(len(jets), len(labels))

    def test_cut_vars(self):
        """Test cut vars."""
        jets, labels = load_jets_from_file(
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

    def test_wrong_filepath(self):
        """Test raise of error for wrong filepath."""
        with self.assertRaises(KeyError):
            load_jets_from_file(
                filepath=666,
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(RuntimeError):
            load_jets_from_file(
                filepath="",
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

    def test_load_trks_from_file(self):
        """Test the correct loading of the track variables."""
        trks, labels = load_trks_from_file(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
        )

        self.assertEqual(len(trks), len(labels))

    def test_trks_different_operators(self):
        """Test different operators."""
        for operator in ["<=", "==", ">=", "<", ">"]:
            with self.subTest(f"Testing operator {operator}"):
                trks, labels = load_trks_from_file(
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

    def test_trks_different_chunk_size(self):
        """Test different chunk size."""
        trks, labels = load_trks_from_file(
            filepath=os.path.join(
                self.tmp_test_dir,
                "ci_ttbar_testing.h5",
            ),
            class_labels=self.class_labels,
            n_jets=self.n_jets,
            chunk_size=1000,
        )

        self.assertEqual(len(trks), len(labels))

    def test_trks_raise_errors(self):
        """Test the raise of the different errors."""
        with self.assertRaises(RuntimeError):
            load_trks_from_file(
                filepath=os.path.join(
                    self.tmp_test_dir,
                    "test_to_fail*.h5",
                ),
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(KeyError):
            load_trks_from_file(
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
            load_trks_from_file(
                filepath=666,
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )

        with self.assertRaises(RuntimeError):
            load_trks_from_file(
                filepath="",
                class_labels=self.class_labels,
                n_jets=self.n_jets,
            )
