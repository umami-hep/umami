"""Unit test script for the generator functions of tf_tools."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from umami.configuration import logger, set_log_level
from umami.tf_tools.generators import (
    CadsGenerator,
    DipsGenerator,
    Dl1Generator,
    ModelGenerator,
    UmamiConditionGenerator,
    UmamiGenerator,
)

set_log_level(logger, "DEBUG")


class BaseGeneratorTest(unittest.TestCase):
    """Base generator test class of the tf_tools."""

    def setUp(self) -> None:
        """Download a test training file"""

        self.test_dir_path = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = f"{self.test_dir_path.name}"
        logger.info("Creating test directory in %s", self.test_dir)

        self.x_name = "jets/inputs"
        self.x_trk_name = "tracks_loose/inputs"
        self.y_name = "jets/labels"
        self.batch_size = 100
        self.n_jets = 2500
        self.chunk_size = 500
        self.sample_weights = True
        self.jet_features = 2
        self.n_tracks_per_jet = 40
        self.track_features = 21
        self.n_conds = 2
        self.print_logger = True

        self.train_file_path = Path(self.test_dir) / "dummy_train_file.h5"

        # implementaiton for new group setup
        # jets
        key_words_jets = ["inputs", "labels", "labels_one_hot", "weight"]
        shapes_jets = [[self.jet_features], [], [3], []]

        # tracks
        key_words_tracks = ["inputs", "labels", "valid"]
        shapes_tracks = [
            [self.n_tracks_per_jet, self.track_features],
            [self.n_tracks_per_jet, self.jet_features],
            [self.n_tracks_per_jet],
        ]

        rng = np.random.default_rng(seed=65)
        with h5py.File(self.train_file_path, "w") as f_h5:
            g_jets = f_h5.create_group("jets")
            for key, shape in zip(key_words_jets, shapes_jets):
                arr = rng.random((self.n_jets, *shape))
                g_jets.create_dataset(key, data=arr)

            g_tracks = f_h5.create_group("tracks_loose")
            for key, shape in zip(key_words_tracks, shapes_tracks):
                arr = rng.random((self.n_jets, *shape))
                g_tracks.create_dataset(key, data=arr)

        self.shared_settings = {
            "train_file_path": self.train_file_path,
            "y_name": self.y_name,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "print_logger": self.print_logger,
        }


class TestModelGenerator(BaseGeneratorTest):
    """Test Generator base class of the tf_tools."""

    def test_init(self):
        """Test the init from the base class"""
        base_generator = ModelGenerator(
            x_name=self.x_name,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        with self.subTest("x_in_mem check"):
            self.assertEqual(base_generator.x_in_mem, None)

        with self.subTest("n_jets check"):
            self.assertEqual(base_generator.n_jets, int(self.n_jets))

    def test_init_no_n_jets_x_name_given(self):
        """Test case for init without n_jets given with X_Name."""
        base_generator = ModelGenerator(
            x_name=self.x_name,
            x_trk_name=None,
            n_jets=None,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        self.assertEqual(base_generator.n_jets, self.n_jets)

    def test_init_no_n_jets_x_trk_name_given(self):
        """Test case for init without n_jets given with X_trk_Name."""
        base_generator = ModelGenerator(
            x_name=None,
            x_trk_name=self.x_trk_name,
            n_jets=None,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        self.assertEqual(base_generator.n_jets, self.n_jets)

    def test_init_no_n_jets_no_x_given(self):
        """Test case for init without n_jets given without X."""

        with self.assertRaises(ValueError):
            _ = ModelGenerator(
                x_name=None,
                x_trk_name=None,
                n_jets=None,
                sample_weights=self.sample_weights,
                **self.shared_settings,
            )

    def test_load_in_memory(self):
        """Test the basic load_in_memory function."""
        base_generator = ModelGenerator(
            x_name=self.x_name,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Load the jets and tracks
        base_generator.load_in_memory(
            load_jets=True,
            load_tracks=True,
        )

        with self.subTest("Check length tracks in memory"):
            self.assertEqual(len(base_generator.x_trk_in_mem), self.chunk_size)

        with self.subTest("Check length jets in memory"):
            self.assertEqual(len(base_generator.x_in_mem), self.chunk_size)

        with self.subTest("Check length labels in memory"):
            self.assertEqual(len(base_generator.y_in_mem), self.chunk_size)

    def test_load_in_memory_x_name_error(self):
        """Test the error when trying to load jets without X_Name given."""
        base_generator = ModelGenerator(
            x_name=None,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        with self.assertRaises(ValueError):
            base_generator.load_in_memory(
                load_jets=True,
                load_tracks=True,
            )

    def test_load_in_memory_x_trk_name_error(self):
        """Test the error when trying to load tracks without X_trk_Name given."""
        base_generator = ModelGenerator(
            x_name=self.x_name,
            x_trk_name=None,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        with self.assertRaises(ValueError):
            base_generator.load_in_memory(
                load_jets=True,
                load_tracks=True,
            )


class TestCadsGenerator(BaseGeneratorTest):
    """Test CADS Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        cads_generator = CadsGenerator(
            x_name=self.x_name,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            n_conds=self.n_conds,
            **self.shared_settings,
        )

        # Get the generator
        generator = cads_generator()

        for _ in range(cads_generator.length):
            x_input, y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    x_input["input_1"].shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest(
                "Check shape of conditional info yielded by the generator"
            ):
                self.assertEqual(
                    x_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.n_conds,
                    ),
                )

            with self.subTest("Check shape of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)


class TestDipsGenerator(BaseGeneratorTest):
    """Test DIPS Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        dips_generator = DipsGenerator(
            x_name=None,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = dips_generator()

        for _ in range(dips_generator.length):
            x_input, y_input, weights = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    x_input.shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)

            with self.subTest("Check length of weights yielded by the generator"):
                self.assertEqual(len(weights), self.batch_size)

    def test_call_without_weights(self):
        """Test the call of the generator."""
        dips_generator = DipsGenerator(
            x_name=None,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=False,
            **self.shared_settings,
        )

        # Get the generator
        generator = dips_generator()

        for _ in range(dips_generator.length):
            x_input, y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    x_input.shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest("Check shape of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)


class TestDL1Generator(BaseGeneratorTest):
    """Test DL1 Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        dl1_generator = Dl1Generator(
            x_name=self.x_name,
            x_trk_name=None,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = dl1_generator()

        for _ in range(dl1_generator.length):
            x_input, y_input, weights = next(generator)

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    x_input.shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)

            with self.subTest("Check length of weights yielded by the generator"):
                self.assertEqual(len(weights), self.batch_size)

    def test_call_without_weights(self):
        """Test the call of the generator."""
        dl1_generator = Dl1Generator(
            x_name=self.x_name,
            x_trk_name=None,
            n_jets=self.n_jets,
            sample_weights=False,
            **self.shared_settings,
        )

        # Get the generator
        generator = dl1_generator()

        for _ in range(dl1_generator.length):
            x_input, y_input = next(generator)

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    x_input.shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)


class TestUmamiCondAttGenerator(BaseGeneratorTest):
    """Test Umami Cond Att Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        umami_cond_att_generator = UmamiConditionGenerator(
            x_name=self.x_name,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            n_conds=self.n_conds,
            **self.shared_settings,
        )

        # Get the generator
        generator = umami_cond_att_generator()

        for _ in range(umami_cond_att_generator.length):
            x_input, y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    x_input["input_1"].shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest(
                "Check shape of conditional info yielded by the generator"
            ):
                self.assertEqual(
                    x_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.n_conds,
                    ),
                )

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    x_input["input_3"].shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)


class TestUmamiGenerator(BaseGeneratorTest):
    """Test Umami Cond Att Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        umami_generator = UmamiGenerator(
            x_name=self.x_name,
            x_trk_name=self.x_trk_name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = umami_generator()

        for _ in range(umami_generator.length):
            x_input, y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    x_input["input_1"].shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest(
                "Check shape of conditional info yielded by the generator"
            ):
                self.assertEqual(
                    x_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(y_input), self.batch_size)
