"""Unit test script for the generator functions of tf_tools."""

import os
import unittest

import numpy as np

import umami.preprocessing_tools as upt
from umami.configuration import logger, set_log_level
from umami.tf_tools.generators import (
    CadsGenerator,
    DipsGenerator,
    Dl1Generator,
    ModelGenerator,
    UmamiConditionGenerator,
    UmamiGenerator,
)
from umami.tf_tools.tddgenerators import (
    TDDCadsGenerator,
    TDDDipsGenerator,
    TDDDl1Generator,
    TDDGenerator,
    TDDUmamiConditionGenerator,
    TDDUmamiGenerator,
    get_generator,
)

set_log_level(logger, "DEBUG")


class TestTDDGenerator(unittest.TestCase):
    """Test the TDDGenerator class."""

    def setUp(self) -> None:
        # Create a temporary file with data from TDD saved via writer from preprocessing
        config = upt.PreprocessConfiguration(
            os.path.join(os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml")
        )
        config.sampling.options.n_jets_to_plot = 0
        config.general.precision = "float32"
        writer = upt.TrainSampleWriter(config, compression=None, shuffling=False)
        writer.write_train_sample(
            input_file="umami/tests/unit/tf_tools/fixtures/TDD-10jets",
            out_file="umami/tests/unit/tf_tools/fixtures/training-10jets-written.h5",
            chunk_size=10,
        )
        writer.write_train_sample(
            input_file="umami/tests/unit/tf_tools/fixtures/TDD-10jets-weights",
            out_file=(
                "umami/tests/unit/tf_tools/fixtures/training-10jets-written-weight.h5"
            ),
            chunk_size=10,
        )

    def tearDown(self) -> None:
        # Delete a temporary file with data from TDD saved via writer from preprocessing
        os.remove("umami/tests/unit/tf_tools/fixtures/training-10jets-written.h5")
        os.remove(
            "umami/tests/unit/tf_tools/fixtures/training-10jets-written-weight.h5"
        )

    def test_circular_modelgen_eq_tddgen(self):
        """Test that the TrainSampleWriter+ModelGenerator
        and TDDGenerator retrieve the same data."""

        # Read the data from the temporary file
        generator = ModelGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/training-10jets-written.h5"
            ),
            x_name="jets/inputs",
            y_name="jets/labels_one_hot",
            x_trk_name="tracks/inputs",
            ############
            n_jets=None,
            batch_size=10,
            n_conds=2,
            excluded_var=None,
            sample_weights=False,
        )

        # Read the data from the original TDD
        tddgenerator = TDDGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/TDD-10jets"
            ),
            ############
            n_jets=None,
            excluded_var=None,
            batch_size=10,
            n_conds=2,
            sample_weights=False,
            ############
            config_file=os.path.join(
                os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml"
            ),
            tracks_name="tracks",
        )

        generator.load_in_memory(part=0, load_jets=True, load_tracks=True)
        tddgenerator.load_in_memory(part=0, load_jets=True, load_tracks=True)

        np.testing.assert_array_almost_equal(
            generator.x_trk_in_mem, tddgenerator.x_trk_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.x_in_mem, tddgenerator.x_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.y_in_mem, tddgenerator.y_in_mem, decimal=8
        )
        self.assertEqual(generator.weight_in_mem, tddgenerator.weight_in_mem)
        self.assertEqual(generator.n_conds, tddgenerator.n_conds)
        self.assertEqual(generator.get_n_jets(), tddgenerator.get_n_jets())
        self.assertEqual(generator.get_n_dim(), tddgenerator.get_n_dim())
        self.assertEqual(
            generator.get_n_jet_features(), tddgenerator.get_n_jet_features()
        )
        self.assertEqual(
            generator.get_n_trk_features(), tddgenerator.get_n_trk_features()
        )
        self.assertEqual(generator.get_n_trks(), tddgenerator.get_n_trks())

    def test_circular_modelgen_eq_tddgen_old_scaling(self):
        """Test that the TrainSampleWriter+ModelGenerator
        and TDDGenerator retrieve the same data."""

        # Read the data from the temporary file
        generator = ModelGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/training-10jets-written.h5"
            ),
            x_name="jets/inputs",
            y_name="jets/labels_one_hot",
            x_trk_name="tracks/inputs",
            ############
            n_jets=None,
            batch_size=10,
            n_conds=2,
            excluded_var=None,
            sample_weights=False,
        )

        # Read the data from the original TDD
        tddgenerator = TDDGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/TDD-10jets"
            ),
            ############
            n_jets=None,
            excluded_var=None,
            batch_size=10,
            n_conds=2,
            sample_weights=False,
            ############
            config_file=os.path.join(
                os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml"
            ),
            tracks_name="tracks",
            old_scaling_functions=True,
        )

        generator.load_in_memory(part=0, load_jets=True, load_tracks=True)
        tddgenerator.load_in_memory(part=0, load_jets=True, load_tracks=True)

        np.testing.assert_array_almost_equal(
            generator.x_trk_in_mem, tddgenerator.x_trk_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.x_in_mem, tddgenerator.x_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.y_in_mem, tddgenerator.y_in_mem, decimal=8
        )
        self.assertEqual(generator.weight_in_mem, tddgenerator.weight_in_mem)
        self.assertEqual(generator.n_conds, tddgenerator.n_conds)
        self.assertEqual(generator.get_n_jets(), tddgenerator.get_n_jets())
        self.assertEqual(generator.get_n_dim(), tddgenerator.get_n_dim())
        self.assertEqual(
            generator.get_n_jet_features(), tddgenerator.get_n_jet_features()
        )
        self.assertEqual(
            generator.get_n_trk_features(), tddgenerator.get_n_trk_features()
        )
        self.assertEqual(generator.get_n_trks(), tddgenerator.get_n_trks())

    def test_circular_with_weights_in_tdd(self):
        """Test that the TrainSampleWriter+ModelGenerator
        and TDDGenerator retrieve the same data and weights"""

        # Read the data from the temporary file
        generator = ModelGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/training-10jets-written-weight.h5"
            ),
            x_name="jets/inputs",
            y_name="jets/labels_one_hot",
            x_trk_name="tracks/inputs",
            ############
            n_jets=None,
            batch_size=10,
            excluded_var=None,
            sample_weights=True,
        )

        # Read the data from the original TDD
        tddgenerator = TDDGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/TDD-10jets-weights"
            ),
            ############
            n_jets=None,
            excluded_var=None,
            batch_size=10,
            sample_weights=True,
            ############
            config_file=os.path.join(
                os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml"
            ),
            tracks_name="tracks",
        )

        generator.load_in_memory(part=0, load_jets=True, load_tracks=True)
        tddgenerator.load_in_memory(part=0, load_jets=True, load_tracks=True)

        np.testing.assert_array_almost_equal(
            generator.x_trk_in_mem, tddgenerator.x_trk_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.x_in_mem, tddgenerator.x_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.y_in_mem, tddgenerator.y_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.weight_in_mem, tddgenerator.weight_in_mem, decimal=8
        )
        self.assertEqual(generator.n_conds, tddgenerator.n_conds)

    def test_circular_with_excluded_var(self):
        """Test that the TrainSampleWriter+ModelGenerator
        and TDDGenerator retrieve the same data and manage excluded variables"""

        # Read the data from the temporary file
        generator = ModelGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/training-10jets-written.h5"
            ),
            x_name="jets/inputs",
            y_name="jets/labels_one_hot",
            x_trk_name="tracks/inputs",
            ############
            n_jets=None,
            batch_size=10,
            excluded_var=[2],
            sample_weights=False,
        )

        # Read the data from the original TDD
        tddgenerator = TDDGenerator(
            train_file_path=os.path.join(
                os.path.dirname(__file__), "fixtures/TDD-10jets"
            ),
            ############
            n_jets=None,
            excluded_var=[2],
            batch_size=10,
            sample_weights=False,
            ############
            config_file=os.path.join(
                os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml"
            ),
            tracks_name="tracks",
        )

        generator.load_in_memory(part=0, load_jets=True, load_tracks=True)
        tddgenerator.load_in_memory(part=0, load_jets=True, load_tracks=True)

        np.testing.assert_array_almost_equal(
            generator.x_trk_in_mem, tddgenerator.x_trk_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.x_in_mem, tddgenerator.x_in_mem, decimal=8
        )
        np.testing.assert_array_almost_equal(
            generator.y_in_mem, tddgenerator.y_in_mem, decimal=8
        )
        self.assertEqual(generator.weight_in_mem, tddgenerator.weight_in_mem)

    def test_get_generator(self):
        """Test that the get_generator function returns the correct generator."""
        args_dict = {
            "train_file_path": os.path.join(
                os.path.dirname(__file__), "fixtures/training-10jets-written.h5"
            ),
            "x_name": "jets/inputs",
            "y_name": "jets/labels_one_hot",
            "x_trk_name": "tracks/inputs",
            "n_jets": None,
            "batch_size": 5,
            "excluded_var": None,
            "sample_weights": True,
            "config_file": os.path.join(
                os.path.dirname(__file__), "fixtures/PFlow-Preprocessing.yaml"
            ),
            "tracks_name": "tracks",
        }

        generator = get_generator("Dl1", args_dict, small=True)
        self.assertIsInstance(generator, Dl1Generator)
        generator = get_generator("DIPS", args_dict, small=True)
        self.assertIsInstance(generator, DipsGenerator)
        generator = get_generator("CADS", args_dict, small=True)
        self.assertIsInstance(generator, CadsGenerator)
        generator = get_generator("Umami", args_dict, small=True)
        self.assertIsInstance(generator, UmamiGenerator)
        generator = get_generator("UmamiCondition", args_dict, small=True)
        self.assertIsInstance(generator, UmamiConditionGenerator)

        args_dict["train_file_path"] = os.path.join(
            os.path.dirname(__file__), "fixtures/TDD-10jets-weights"
        )
        generator = get_generator("Dl1", args_dict, small=True)
        self.assertIsInstance(generator, TDDDl1Generator)
        generator = get_generator("DIPS", args_dict, small=True)
        self.assertIsInstance(generator, TDDDipsGenerator)
        generator = get_generator("CADS", args_dict, small=True)
        self.assertIsInstance(generator, TDDCadsGenerator)
        generator = get_generator("Umami", args_dict, small=True)
        self.assertIsInstance(generator, TDDUmamiGenerator)
        generator = get_generator("UmamiCondition", args_dict, small=True)
        self.assertIsInstance(generator, TDDUmamiConditionGenerator)

    # @unittest.skip("Not implemented yet")
    # def test_circular_with_weights_in_TDD_calc_sample_weights(self):
    #    pass
