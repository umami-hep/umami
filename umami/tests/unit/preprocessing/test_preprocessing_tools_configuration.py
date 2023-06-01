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

    def test_sampling_class_as_dict(self):
        """Test return of all class attributes as dict."""
        config = PreprocessConfiguration(self.config_file)
        out_dict = config.sampling.as_dict()

        # Define reference dict
        ref_dict = {
            "class_labels": ["bjets", "cjets", "ujets"],
            "method": "count",
            "use_validation_samples": False,
        }

        # Check the ref with output
        self.assertEqual(out_dict, ref_dict)

    def test_sampling_options_class_as_dict(self):
        """Test return of all class attributes as dict."""
        config = PreprocessConfiguration(self.config_file)
        out_dict = config.sampling.options.as_dict()

        # Define reference dict
        ref_dict = {
            "sampling_variables": [
                {"pt_btagJes": {"bins": [[0, 600000, 351], [650000, 6000000, 84]]}},
                {"absEta_btagJes": {"bins": [0, 2.5, 10]}},
            ],
            "samples": {
                "ttbar": [
                    "validation_ttbar_bjets",
                    "validation_ttbar_cjets",
                    "validation_ttbar_ujets",
                ],
                "zprime": [
                    "validation_zprime_bjets",
                    "validation_zprime_cjets",
                    "validation_zprime_ujets",
                ],
            },
            "custom_n_jets_initial": None,
            "fractions": {"ttbar": 0.65, "zprime": 0.35},
            "max_upsampling_ratio": None,
            "n_jets": 5500000.0,
            "n_jets_validation": None,
            "n_jets_scaling": None,
            "save_tracks": True,
            "tracks_names": ["tracks"],
            "save_track_labels": False,
            "intermediate_index_file": "indices.h5",
            "intermediate_index_file_validation": None,
            "weighting_target_flavour": None,
            "bool_attach_sample_weights": None,
            "n_jets_to_plot": None,
            "target_distribution": None,
        }

        # Check the ref with output
        self.assertEqual(out_dict, ref_dict)

    def test_general_settings_class_as_dict(self):
        """Test return of all class attributes as dict."""
        config = PreprocessConfiguration(self.config_file)
        out_dict = config.general.as_dict()

        # Define reference dict
        ref_dict = {
            "outfile_name": "dummy_out.h5",
            "outfile_name_validation": None,
            "plot_name": "dummy_plot",
            "plot_type": "pdf",
            "apply_atlas_style": True,
            "use_atlas_tag": True,
            "atlas_first_tag": "Simulation Internal",
            "atlas_second_tag": None,
            "legend_sample_category": True,
            "var_file": "fixtures/dummy_var_file.yaml",
            "dict_file": "test.json",
            "compression": None,
            "precision": "float32",
            "concat_jet_tracks": False,
            "convert_to_tfrecord": {"chunk_size": 5000},
        }

        # Check the ref with output
        self.assertEqual(out_dict, ref_dict)

    def test_genera_settings_class_plot_options_as_dict(self):
        """Test return of all class attributes as dict."""
        config = PreprocessConfiguration(self.config_file)
        out_dict = config.general.plot_options_as_dict()

        # Define reference dict
        ref_dict = {
            "apply_atlas_style": True,
            "use_atlas_tag": True,
            "atlas_first_tag": "Simulation Internal",
            "atlas_second_tag": None,
        }

        # Check the ref with output
        self.assertEqual(out_dict, ref_dict)
