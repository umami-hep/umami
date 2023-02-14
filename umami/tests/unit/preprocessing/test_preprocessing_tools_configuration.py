"""Unit tests for configuration of preprocessing_tools."""
import os
import tempfile
import unittest
from pathlib import Path

from umami.preprocessing_tools.configuration import Preparation, PreprocessConfiguration


class PreparationTestCase(unittest.TestCase):
    """
    Test the implementation of the Preparation class.
    """

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())
        (self.tmp_path / "ttbar_dummy_files").mkdir(parents=True, exist_ok=True)
        self.example_input_h5 = {
            "ttbar": {"path": self.tmp_path, "file_pattern": "ttbar_dummy_files/*.h5"}
        }
        self.example_samples = {
            "training_ttbar_bjets": {
                "type": "ttbar",
                "category": "bjets",
                "n_jets": 100,
                "output_name": str(self.tmp_path) + "/bjets_training_ttbar.h5",
            }
        }
        self.prep_example = {
            "input_h5": self.example_input_h5,
            "samples": self.example_samples,
        }

    def test_default_batch_size(self):
        """Testing if batch size is correctly initialised if none is given."""
        prep = Preparation(self.prep_example)
        self.assertEqual(prep.batch_size, 500_000)

    def test_failed_get_sample(self):
        """Testing if wrong sample name is given to raise KeyError."""
        prep = Preparation(self.prep_example)
        with self.assertRaises(KeyError):
            prep.get_sample("dummy_not_exist_sample")

    def test_double_output_name(self):
        """Testing if wrong sample name is given to raise KeyError."""
        self.prep_example["samples"]["training_ttbar_bjets"]["f_output"] = None
        with self.assertRaises(KeyError):
            Preparation(self.prep_example)


class PreprocessConfigurationTestCase(unittest.TestCase):
    """
    Test the implementation of the Configuration class.
    """

    def setUp(self):
        """
        Set a example config file.
        """
        self.config_file = (
            Path(os.path.dirname(__file__)) / "fixtures" / "test_preprocess_config.yaml"
        )

    def test_missing_key_error(self):
        """Test missing key error."""
        config = PreprocessConfiguration(self.config_file)
        del config.config["outfile_name"]
        with self.assertRaises(ValueError):
            config.get_configuration()

    def test_get_file_name_no_input(self):
        """Test filename without input."""
        config = PreprocessConfiguration(self.config_file)
        out_file = config.get_file_name()
        self.assertEqual(out_file, config.general.outfile_name)

    def test_get_file_name_no_iterations(self):
        """Test no iterations"""
        config = PreprocessConfiguration(self.config_file)
        with self.subTest():
            self.assertNotIn("test", config.general.outfile_name)
        out_file = config.get_file_name(option="test")
        with self.subTest():
            self.assertIn("test", out_file)

    def test_get_file_name_no_iterations_no_input(self):
        """Test no iterations and no input."""
        config = PreprocessConfiguration(self.config_file)
        out_file = config.get_file_name()
        self.assertEqual(config.general.outfile_name, out_file)
