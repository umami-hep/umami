"""Unit test for preprocessing tools Scaling."""
# pylint: disable=attribute-defined-outside-init
import os
import unittest

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from umami.configuration import logger, set_log_level
from umami.preprocessing_tools import (
    CalculateScaling,
    PreprocessConfiguration,
    apply_scaling_jets,
    apply_scaling_trks,
)

set_log_level(logger, "DEBUG")


class ApplyScalingTrksTestCase(unittest.TestCase):
    """Test class for apply_scaling_trks function."""

    def setUp(self):
        self.var_config = {
            "track_train_variables": {
                "tracks": {
                    "noNormVars": ["IP3D_signed_d0_significance"],
                    "logNormVars": ["ptfrac"],
                    "jointNormVars": ["numberOfPixelHits"],
                }
            }
        }
        self.scale_dict = {
            "ptfrac": {"shift": 5, "scale": 2},
            "numberOfPixelHits": {"shift": 4, "scale": 2},
        }
        self.trks = np.array(
            [(50, 2, 30, True), (100, 2, 40, True)],
            dtype=[
                ("ptfrac", "f4"),
                ("numberOfPixelHits", "i4"),
                ("IP3D_signed_d0_significance", "f4"),
                ("valid", "?"),
            ],
        )
        self.control_trks = np.array([[30, -0.54398847, -1], [40, -0.19741488, -1]])

    def test_apply_scaling_trks(self):
        """Test apply scaling fot tracks."""
        scaled_trks, _, _ = apply_scaling_trks(
            trks=self.trks,
            variable_config=self.var_config,
            scale_dict=self.scale_dict,
            tracks_name="tracks",
        )

        np.testing.assert_array_almost_equal(scaled_trks, self.control_trks)


class ApplyScalingJetsTestCase(unittest.TestCase):
    """Test class for apply_scaling_jets function."""

    def setUp(self) -> None:
        self.variable_list = [
            "absEta_btagJes",
            "pt_btagJes",
            "JetFitter_isDefaults",
        ]
        self.jets = pd.DataFrame(
            {
                var: np.ones(5) * counter
                for counter, var in enumerate(self.variable_list)
            }
        )
        self.structured_array_jets = np.array(
            [
                (0, 1, 2),
                (0, 1, 2),
                (0, 1, 2),
                (0, 1, 2),
                (0, 1, 2),
            ],
            dtype={
                "names": self.variable_list,
                "formats": ["<f4", "<f4", "<f4"],
            },
        )
        self.control_jets = pd.DataFrame(
            {
                "absEta_btagJes": np.ones(5) * -0.5,
                "pt_btagJes": np.zeros(5),
                "JetFitter_isDefaults": np.ones(5) * 2,
            }
        )
        self.scale_dict = {
            var: {
                "shift": 1,
                "scale": 2,
                "default": 0,
            }
            for var in self.variable_list
        }

    def test_scaling_jets(self):
        """Testing the default behaviour."""
        jets = apply_scaling_jets(
            jets=self.jets,
            variables_list=self.variable_list,
            scale_dict=self.scale_dict,
        )

        for var in self.variable_list:
            with self.subTest(f"Testing variable {var}"):
                np.testing.assert_array_almost_equal(jets[var], self.control_jets[var])

    def test_scaling_structured_array_jets(self):
        """Testing default behaviour for structured numpy array."""
        jets = apply_scaling_jets(
            jets=self.structured_array_jets,
            variables_list=self.variable_list,
            scale_dict=self.scale_dict,
        )

        for var in self.variable_list:
            with self.subTest(f"Testing variable {var}"):
                np.testing.assert_array_almost_equal(jets[var], self.control_jets[var])

    def test_key_error_scale_dict(self):
        """Test error raise if variable not in scale dict."""
        with self.assertRaises(KeyError):
            apply_scaling_jets(
                jets=self.jets,
                variables_list=self.variable_list + ["JetFitter_mass"],
                scale_dict=self.scale_dict,
            )

    def test_value_error_scale_value(self):
        """Test error raise if variable not in scale dict."""
        faulty_scale_dict = {
            "absEta_btagJes": {
                "shift": 1,
                "scale": 0,
                "default": 0,
            },
            "pt_btagJes": {
                "shift": 1,
                "scale": np.inf,
                "default": 0,
            },
            "JetFitter_mass": {
                "shift": 1,
                "scale": 0,
                "default": 0,
            },
        }

        with self.assertRaises(ValueError):
            apply_scaling_jets(
                jets=self.jets,
                variables_list=self.variable_list,
                scale_dict=faulty_scale_dict,
            )


class ScalingTestCase(unittest.TestCase):
    """
    Unit test the functions inside the Scaling class.
    """

    def setUp(self):
        self.config_file = os.path.join(
            os.path.dirname(__file__), "fixtures", "test_preprocess_config.yaml"
        )
        self.config = PreprocessConfiguration(self.config_file)
        self.config.var_file = os.path.join(
            os.path.dirname(__file__), "fixtures/dummy_var_file.yaml"
        )

    def test_join_scale_dicts_trks(self):
        """Test joining of track scale dicts."""
        self.first_scale_dict = {
            "variable": {"shift": 5, "scale": 2},
            "variable_2": {"shift": 4, "scale": 2},
        }
        self.second_scale_dict = {
            "variable": {"shift": 3, "scale": 1},
            "variable_2": {"shift": 1, "scale": 0.5},
        }
        self.control_dict = {
            "variable": {"shift": 3.8, "scale": 1.7776388834631178},
            "variable_2": {"shift": 2.2, "scale": 1.977371993328519},
        }

        scaling_class = CalculateScaling(self.config)

        combined_scale_dict, combined_n_trks = scaling_class.join_scale_dicts(
            first_scale_dict=self.first_scale_dict,
            second_scale_dict=self.second_scale_dict,
            first_n=10,
            second_n=15,
        )

        with self.subTest():
            self.assertEqual(combined_n_trks, 25)

        for key, var in combined_scale_dict.items():
            with self.subTest():
                self.assertEqual(
                    var["shift"],
                    self.control_dict[key]["shift"],
                )
            with self.subTest():
                self.assertEqual(
                    var["scale"],
                    self.control_dict[key]["scale"],
                )

    def test_join_scale_dicts(self):
        """Test joining scale dicts of jets."""
        self.first_scale_dict = {
            "variable": {"shift": 5, "scale": 2, "default": None},
            "variable_2": {"shift": 4, "scale": 2, "default": None},
        }
        self.second_scale_dict = {
            "variable": {"shift": 3, "scale": 1, "default": None},
            "variable_2": {"shift": 1, "scale": 0.5, "default": None},
        }
        self.control_dict = {
            "variable": {
                "shift": 3.8,
                "scale": 1.7776388834631178,
                "default": None,
            },
            "variable_2": {
                "shift": 2.2,
                "scale": 1.977371993328519,
                "default": None,
            },
        }

        scaling_class = CalculateScaling(self.config)

        combined_scale_dict, combined_n_jets = scaling_class.join_scale_dicts(
            first_scale_dict=self.first_scale_dict,
            second_scale_dict=self.second_scale_dict,
            first_n=10,
            second_n=15,
        )

        with self.subTest():
            self.assertEqual(combined_n_jets, 25)

        for varname, scale_dict in combined_scale_dict.items():
            with self.subTest():
                self.assertEqual(
                    scale_dict["shift"],
                    self.control_dict[varname]["shift"],
                )
            with self.subTest():
                self.assertEqual(
                    scale_dict["scale"],
                    self.control_dict[varname]["scale"],
                )

    def test_get_scaling_tracks(self):
        """Test scaling of tracks."""
        njet = 3
        ntrack = 40
        nvar = 2
        self.var_names = ["variable", "variable_2"]
        dtype = np.dtype([(n, "f4") for n in self.var_names])
        self.data = u2s(np.arange(0, 240, 1).reshape((njet, ntrack, nvar)), dtype)
        self.track_mask = np.ones((njet, ntrack)).astype(bool)

        self.n_trks_control = 120
        self.scale_dict_control = {
            "variable": {"shift": 119, "scale": 69.27962493896484},
            "variable_2": {"shift": 120, "scale": 69.27962493896484},
        }

        scaling_class = CalculateScaling(self.config)

        n_trk, scale_dict, _ = scaling_class.get_scaling_tracks(
            data=self.data,
            var_names=self.var_names,
            track_mask=self.track_mask,
        )

        with self.subTest():
            self.assertEqual(n_trk, self.n_trks_control)

        for key, var in scale_dict.items():
            with self.subTest():
                self.assertEqual(
                    var["shift"],
                    self.scale_dict_control[key]["shift"],
                )
            with self.subTest():
                self.assertEqual(
                    var["scale"],
                    self.scale_dict_control[key]["scale"],
                )

    def test_get_scaling(self):
        """Test retrieving scaling."""
        self.vec = np.arange(0, 100, 1)
        self.weight = np.ones_like(self.vec)
        self.varname = "variable"
        self.custom_defaults_vars = {"variable_2": 1}

        self.average_control = 49.5
        self.std_control = 28.86607004772212

        scaling_class = CalculateScaling(self.config)

        varname, scale_dict = scaling_class.get_scaling(
            vec=self.vec,
            varname=self.varname,
            custom_defaults_vars=self.custom_defaults_vars,
        )

        with self.subTest():
            self.assertEqual(varname, self.varname)
        with self.subTest():
            self.assertEqual(scale_dict["shift"], self.average_control)
        with self.subTest():
            self.assertEqual(scale_dict["scale"], self.std_control)
        with self.subTest():
            self.assertEqual(scale_dict["default"], self.average_control)

        varname, scale_dict = scaling_class.get_scaling(
            vec=self.vec,
            varname="variable_2",
            custom_defaults_vars=self.custom_defaults_vars,
        )

        with self.subTest():
            self.assertEqual(varname, "variable_2")
        with self.subTest():
            self.assertEqual(scale_dict["shift"], self.average_control)
        with self.subTest():
            self.assertEqual(scale_dict["scale"], self.std_control)
        with self.subTest():
            self.assertEqual(scale_dict["default"], 1)
