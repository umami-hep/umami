import os
import tempfile
import unittest
from subprocess import run

from umami.configuration import logger, set_log_level
from umami.tf_tools.generators import (
    Model_Generator,
    cads_generator,
    dips_generator,
    dl1_generator,
    umami_condition_generator,
    umami_generator,
)

set_log_level(logger, "DEBUG")


class BaseGeneratorTest(unittest.TestCase):
    """Base generator test class of the tf_tools."""

    def setUp(self) -> None:
        """Download a test training file"""

        self.test_dir_path = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.test_dir = f"{self.test_dir_path.name}"
        logger.info(f"Creating test directory in {self.test_dir}")

        logger.info("Downloading test data...")
        path = os.path.join(
            "https://umami-ci-provider.web.cern.ch/",
            "umami",
            "PFlow-hybrid_70-test-preprocessed_shuffled.h5",
        )

        logger.info(f"Retrieving file from path {path}")
        run(
            [
                "wget",
                path,
                "--directory-prefix",
                self.test_dir,
            ],
            check=True,
        )

        self.train_file_path = os.path.join(
            self.test_dir,
            "PFlow-hybrid_70-test-preprocessed_shuffled.h5",
        )

        self.X_Name = "X_train"
        self.X_trk_Name = "X_trk_train"
        self.Y_Name = "Y_train"
        self.batch_size = 100
        self.n_jets = 2500
        self.chunk_size = 500
        self.sample_weights = True
        self.jet_features = 41
        self.n_tracks_per_jet = 40
        self.track_features = 15
        self.nConds = 2
        self.print_logger = True

        self.shared_settings = {
            "train_file_path": self.train_file_path,
            "Y_Name": self.Y_Name,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "print_logger": self.print_logger,
        }


class TestModelGenerator(BaseGeneratorTest):
    """Test Generator base class of the tf_tools."""

    def test_init(self):
        """Test the init from the base class"""
        base_generator = Model_Generator(
            X_Name=self.X_Name,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        with self.subTest("x_in_mem check"):
            self.assertEqual(base_generator.x_in_mem, None)

        with self.subTest("n_jets check"):
            self.assertEqual(base_generator.n_jets, int(self.n_jets))

    def test_init_no_n_jets_X_Name_given(self):
        """Test case for init without n_jets given with X_Name."""
        base_generator = Model_Generator(
            X_Name=self.X_Name,
            X_trk_Name=None,
            n_jets=None,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        self.assertEqual(base_generator.n_jets, 2517)

    def test_init_no_n_jets_X_trk_Name_given(self):
        """Test case for init without n_jets given with X_trk_Name."""
        base_generator = Model_Generator(
            X_Name=None,
            X_trk_Name=self.X_trk_Name,
            n_jets=None,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        self.assertEqual(base_generator.n_jets, 2517)

    def test_init_no_n_jets_no_X_given(self):
        """Test case for init without n_jets given without X."""

        with self.assertRaises(ValueError):
            _ = Model_Generator(
                X_Name=None,
                X_trk_Name=None,
                n_jets=None,
                sample_weights=self.sample_weights,
                **self.shared_settings,
            )

    def test_load_in_memory(self):
        """Test the basic load_in_memory function."""
        base_generator = Model_Generator(
            X_Name=self.X_Name,
            X_trk_Name=self.X_trk_Name,
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

    def test_load_in_memory_X_Name_error(self):
        """Test the error when trying to load jets without X_Name given."""
        base_generator = Model_Generator(
            X_Name=None,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        with self.assertRaises(ValueError):
            base_generator.load_in_memory(
                load_jets=True,
                load_tracks=True,
            )

    def test_load_in_memory_X_trk_Name_error(self):
        """Test the error when trying to load tracks without X_trk_Name given."""
        base_generator = Model_Generator(
            X_Name=self.X_Name,
            X_trk_Name=None,
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
        CADS_generator = cads_generator(
            X_Name=self.X_Name,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            nConds=self.nConds,
            **self.shared_settings,
        )

        # Get the generator
        generator = CADS_generator()

        for _ in range(CADS_generator.length):
            X_input, Y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    X_input["input_1"].shape,
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
                    X_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.nConds,
                    ),
                )

            with self.subTest("Check shape of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)


class TestDipsGenerator(BaseGeneratorTest):
    """Test DIPS Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        DIPS_generator = dips_generator(
            X_Name=None,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = DIPS_generator()

        for _ in range(DIPS_generator.length):
            X_input, Y_input, weights = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    X_input.shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)

            with self.subTest("Check length of weights yielded by the generator"):
                self.assertEqual(len(weights), self.batch_size)

    def test_call_without_weights(self):
        """Test the call of the generator."""
        DIPS_generator = dips_generator(
            X_Name=None,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=False,
            **self.shared_settings,
        )

        # Get the generator
        generator = DIPS_generator()

        for _ in range(DIPS_generator.length):
            X_input, Y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    X_input.shape,
                    (
                        self.batch_size,
                        self.n_tracks_per_jet,
                        self.track_features,
                    ),
                )

            with self.subTest("Check shape of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)


class TestDL1Generator(BaseGeneratorTest):
    """Test DL1 Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        DL1_generator = dl1_generator(
            X_Name=self.X_Name,
            X_trk_Name=None,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = DL1_generator()

        for _ in range(DL1_generator.length):
            X_input, Y_input, weights = next(generator)

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    X_input.shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)

            with self.subTest("Check length of weights yielded by the generator"):
                self.assertEqual(len(weights), self.batch_size)

    def test_call_without_weights(self):
        """Test the call of the generator."""
        DL1_generator = dl1_generator(
            X_Name=self.X_Name,
            X_trk_Name=None,
            n_jets=self.n_jets,
            sample_weights=False,
            **self.shared_settings,
        )

        # Get the generator
        generator = DL1_generator()

        for _ in range(DL1_generator.length):
            X_input, Y_input = next(generator)

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    X_input.shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)


class TestUmamiCondAttGenerator(BaseGeneratorTest):
    """Test Umami Cond Att Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        UmamiCondAtt_generator = umami_condition_generator(
            X_Name=self.X_Name,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            nConds=self.nConds,
            **self.shared_settings,
        )

        # Get the generator
        generator = UmamiCondAtt_generator()

        for _ in range(UmamiCondAtt_generator.length):
            X_input, Y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    X_input["input_1"].shape,
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
                    X_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.nConds,
                    ),
                )

            with self.subTest("Check shape of jets yielded by the generator"):
                self.assertEqual(
                    X_input["input_3"].shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)


class TestUmamiGenerator(BaseGeneratorTest):
    """Test Umami Cond Att Generator class of the tf_tools."""

    def test_call(self):
        """Test the call of the generator."""
        Umami_generator = umami_generator(
            X_Name=self.X_Name,
            X_trk_Name=self.X_trk_Name,
            n_jets=self.n_jets,
            sample_weights=self.sample_weights,
            **self.shared_settings,
        )

        # Get the generator
        generator = Umami_generator()

        for _ in range(Umami_generator.length):
            X_input, Y_input = next(generator)

            with self.subTest("Check shape of tracks yielded by the generator"):
                self.assertEqual(
                    X_input["input_1"].shape,
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
                    X_input["input_2"].shape,
                    (
                        self.batch_size,
                        self.jet_features,
                    ),
                )

            with self.subTest("Check length of labels yielded by the generator"):
                self.assertEqual(len(Y_input), self.batch_size)
