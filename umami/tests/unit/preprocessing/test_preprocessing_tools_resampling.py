"""Unit tests for Resampling in the preprocessing tools."""
import os
import shutil
import tempfile
import unittest
from subprocess import run

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config, logger, set_log_level
from umami.preprocessing_tools import (
    PDFSampling,
    PreprocessConfiguration,
    UnderSampling,
    UnderSamplingNoReplace,
    calculate_binning,
    correct_fractions,
    sampling_generator,
)

set_log_level(logger, "DEBUG")


class CorrectFractionsTestCase(unittest.TestCase):
    """
    Test the implementation of the CorrectFractions function.
    """

    def test_zero_length(self):
        """Tests zero length."""
        with self.assertRaises(ValueError):
            correct_fractions([], [])

    def test_different_input_lengths(self):
        """Test different input lengths."""
        with self.assertRaises(AssertionError):
            correct_fractions([1, 2, 3, 4], [0.2, 0.8, 0.0])

    def test_not_fraction_sum_one(self):
        """Test not fraction sum one."""
        with self.assertRaises(ValueError):
            correct_fractions([1, 2, 3, 4], [0.2, 0.5, 0.2, 0.2])

    def test_different_input_length_class_names(self):
        """Test different input length class names."""
        with self.assertRaises(AssertionError):
            correct_fractions([5000, 6000, 3000], [0.2, 0.6, 0.2], ["Zjets", "ttbar"])

    def test_zero_n_jets(self):
        """Test zero n_jets."""
        with self.assertRaises(ValueError):
            correct_fractions([0, 6000, 3000], [0.2, 0.6, 0.2])

    def test_twice_same_fractions(self):
        """Test twice same fractions."""
        self.assertListEqual(
            list(correct_fractions([1000, 6000, 3000], [0.2, 0.6, 0.2])),
            [1000, 3000, 1000],
        )

    def test_input_correct_fractions(self):
        """Test input correct fractions."""
        n_jets = [2000, 6000, 2000]
        self.assertListEqual(list(correct_fractions(n_jets, [0.2, 0.6, 0.2])), n_jets)

    def test_scaling_down_largest(self):
        """Test scaling down largest."""
        self.assertListEqual(
            list(correct_fractions([3000, 6000, 3000], [0.3, 0.5, 0.2])),
            [3000, 5000, 2000],
        )

    def test_scaling_down_small(self):
        """Test scaling down small."""
        self.assertListEqual(
            list(correct_fractions([10000, 6000, 7000], [0.4, 0.5, 0.1])),
            [4800, 6000, 1200],
        )


class CalculateBinningTestCase(unittest.TestCase):
    """
    Test the implementation of the CalculateBinning function.
    """

    def test_non_list_case(self):
        """Test no list case."""
        with self.assertRaises(TypeError):
            calculate_binning(1)

    @staticmethod
    def test_single_list_case():
        """Test single list case."""
        np.testing.assert_array_equal(
            calculate_binning([1, 2, 3]), np.linspace(1, 2, 3)
        )

    @staticmethod
    def test_nested_list_case():
        """Test nested lists."""
        bins = [[1, 2, 3], [3, 4, 5]]
        expected_outcome = np.concatenate([np.linspace(*elem) for elem in bins])
        np.testing.assert_array_equal(calculate_binning(bins), expected_outcome)


class SamplingGeneratorTestCase(unittest.TestCase):
    """
    Test the implementation of the SamplingGenerator function.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.tmp_test_dir = f"{self.tmp_dir.name}"
        self.label_classes = ["bjets", "cjets", "ujets"]
        np.random.seed(42)
        self.indices = sorted(np.random.randint(0, 3000, 3000))
        self.label = 2
        self.seed = 42
        self.chunk_size = 100
        self.duplicate = True
        self.save_tracks = True
        self.tracks_names = ["tracks", "tracks_loose"]
        self.n_tracks = 40
        self.test_file = os.path.join(self.tmp_test_dir, "ci_ttbar_testing.h5")

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

    def test_no_tracks(self):
        """Testing no tracks yield"""
        generator = sampling_generator(
            file=self.test_file,
            indices=self.indices,
            label=self.label,
            label_classes=self.label_classes,
            save_tracks=False,
            tracks_names=None,
            chunk_size=self.chunk_size,
            seed=self.seed,
            duplicate=self.duplicate,
        )

        jets, labels = next(generator)

        with self.subTest("Test jets shape"):
            self.assertEqual(jets.shape, (self.chunk_size,))

        with self.subTest("Test labels shape"):
            self.assertEqual(labels.shape, (self.chunk_size,))

    def test_tracks(self):
        """Testing with tracks yield"""
        generator = sampling_generator(
            file=self.test_file,
            indices=self.indices,
            label=self.label,
            label_classes=self.label_classes,
            save_tracks=self.save_tracks,
            tracks_names=self.tracks_names,
            chunk_size=self.chunk_size,
            seed=self.seed,
            duplicate=self.duplicate,
        )

        jets, tracks, labels = next(generator)
        with self.subTest("Test jets shape"):
            self.assertEqual(jets.shape, (self.chunk_size,))

        with self.subTest("Test tracks shape"):
            self.assertEqual(tracks[0].shape, (self.chunk_size, self.n_tracks))
            self.assertEqual(tracks[1].shape, (self.chunk_size, self.n_tracks))

        with self.subTest("Test labels shape"):
            self.assertEqual(labels.shape, (self.chunk_size,))

    def test_no_duplicate(self):
        """Testing with tracks yield with no duplicates."""
        generator = sampling_generator(
            file=self.test_file,
            indices=np.arange(0, 3000, 1),
            label=self.label,
            label_classes=self.label_classes,
            save_tracks=self.save_tracks,
            tracks_names=self.tracks_names,
            chunk_size=self.chunk_size,
            seed=self.seed,
            duplicate=False,
        )

        jets, tracks, labels = next(generator)
        with self.subTest("Test jets shape"):
            self.assertEqual(jets.shape, (self.chunk_size,))

        with self.subTest("Test tracks shape"):
            self.assertEqual(tracks[0].shape, (self.chunk_size, self.n_tracks))
            self.assertEqual(tracks[1].shape, (self.chunk_size, self.n_tracks))

        with self.subTest("Test labels shape"):
            self.assertEqual(labels.shape, (self.chunk_size,))

    def test_no_duplicate_no_tracks(self):
        """Testing no tracks yield with no duplicates."""
        generator = sampling_generator(
            file=self.test_file,
            indices=np.arange(0, 3000, 1),
            label=self.label,
            label_classes=self.label_classes,
            save_tracks=False,
            tracks_names=None,
            chunk_size=self.chunk_size,
            seed=self.seed,
            duplicate=False,
        )

        jets, labels = next(generator)
        with self.subTest("Test jets shape"):
            self.assertEqual(jets.shape, (self.chunk_size,))

        with self.subTest("Test labels shape"):
            self.assertEqual(labels.shape, (self.chunk_size,))

    def test_duplicate_error(self):
        """Test duplicate TypeError."""
        with self.assertRaises(TypeError):
            generator = sampling_generator(
                file=self.test_file,
                indices=self.indices,
                label=self.label,
                label_classes=self.label_classes,
                save_tracks=self.save_tracks,
                tracks_names=self.tracks_names,
                chunk_size=self.chunk_size,
                seed=self.seed,
                duplicate=False,
            )

            next(generator)


class ResamplingTestCase(unittest.TestCase):
    """
    Test the implementation of the Resampling base class.
    """

    # TODO: write tests


class ResamplingTestCaseComplexDistr(unittest.TestCase):
    """
    Class that handle creation of a somewhat coplex dataset of test data
    useful to be inherited by other test classes
    """

    def generate_test_data(self):
        """
        Generates a relatively complex dataset of test data
        """
        np.random.seed(42)
        self.data = {
            "training_ttbar_bjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(250000, 30000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.25, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.ones(10000) * 5,
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
            "training_ttbar_cjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(220000, 28000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.4, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.ones(10000) * 4,
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
            "training_ttbar_ujets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(230000, 25000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.0, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.zeros(10000),
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
            "training_zprime_bjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(260000, 30000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.5, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.ones(10000) * 5,
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
            "training_zprime_cjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(260000, 28000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.6, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.ones(10000) * 4,
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
            "training_zprime_ujets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.normal(350000, 25000, 10000)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.2, 1, 10000)),
                    "HadronConeExclTruthLabelID": np.zeros(10000),
                    "eventNumber": np.arange(0, 10000, 1),
                    "jetPtRank": np.random.random_integers(1, 4, 10000),
                }
            ),
        }
        training_ttbar_samples = [
            "training_ttbar_bjets",
            "training_ttbar_cjets",
            "training_ttbar_ujets",
            "training_zprime_bjets",
            "training_zprime_cjets",
            "training_zprime_ujets",
        ]
        for sample in training_ttbar_samples:
            test_h5_file_name = self.config.preparation.get_sample(sample).output_name
            dir_path = os.path.dirname(test_h5_file_name)
            if dir_path != "" and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with h5py.File(test_h5_file_name, "w") as f_h5:
                jets = f_h5.create_dataset(
                    "jets",
                    (10000),
                    dtype=np.dtype(
                        [
                            (global_config.pTvariable, "f"),
                            (global_config.etavariable, "f"),
                            ("HadronConeExclTruthLabelID", "i"),
                            ("eventNumber", "i"),
                            ("jetPtRank", "i"),
                        ]
                    ),
                )
                jets[global_config.pTvariable] = self.data[sample][
                    global_config.pTvariable
                ]
                jets[global_config.etavariable] = self.data[sample][
                    global_config.etavariable
                ]
                jets["HadronConeExclTruthLabelID"] = self.data[sample][
                    "HadronConeExclTruthLabelID"
                ]
                jets["eventNumber"] = self.data[sample]["eventNumber"]
                jets["jetPtRank"] = self.data[sample]["jetPtRank"]

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Read in the configuration file
        self.config_file = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "test_preprocess_config_complex_sampling.yaml",
        )
        self.config = PreprocessConfiguration(self.config_file)
        self.sampling_config = self.config.sampling
        self.generate_test_data()

    def tearDown(self):
        """
        Delete the workspace folder after the test.
        """
        shutil.rmtree("test_preprocessing_workspace")
        # hardcoding here is intentional so that in case
        # of modifications to the test config file
        # we don't delete the wrong folder by accident (e.g. the root folder)
        # plese modify this line if you change the test_parameters.yaml file


class ResamplingTestCaseEasyDistr(unittest.TestCase):
    """
    Class that handle creation of a somewhat simple dataset of test data
    useful to be inherited by other test classes
    """

    def generate_test_data(self):
        """
        Generates a relatively simple dataset of test data
        """
        np.random.seed(42)
        n_events = 100000  # number of events to generate
        self.data = {
            "training_ttbar_bjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.25, 1, n_events)),
                }
            ),
            "training_ttbar_cjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.4, 1, n_events)),
                }
            ),
            "training_ttbar_ujets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.0, 1, n_events)),
                }
            ),
            "training_zprime_bjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.5, 1, n_events)),
                }
            ),
            "training_zprime_cjets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.6, 1, n_events)),
                }
            ),
            "training_zprime_ujets": pd.DataFrame(
                {
                    global_config.pTvariable: abs(
                        np.random.uniform(0, 6000000, n_events)
                    ),
                    global_config.etavariable: abs(np.random.normal(1.2, 1, n_events)),
                }
            ),
        }
        training_ttbar_samples = [
            "training_ttbar_bjets",
            "training_ttbar_cjets",
            "training_ttbar_ujets",
            "training_zprime_bjets",
            "training_zprime_cjets",
            "training_zprime_ujets",
        ]
        for sample in training_ttbar_samples:
            test_h5_file_name = self.config.preparation.get_sample(sample).output_name
            dir_path = os.path.dirname(test_h5_file_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with h5py.File(test_h5_file_name, "w") as f_h5:
                jets = f_h5.create_dataset(
                    "jets",
                    (n_events),
                    dtype=np.dtype(
                        [
                            (global_config.pTvariable, "f"),
                            (global_config.etavariable, "f"),
                        ]
                    ),
                )
                jets[global_config.pTvariable] = self.data[sample][
                    global_config.pTvariable
                ]
                jets[global_config.etavariable] = self.data[sample][
                    global_config.etavariable
                ]

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        # Read in the configuration file
        self.config_file = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "test_preprocess_config_easy_samples.yaml",
        )
        self.config = PreprocessConfiguration(self.config_file)
        self.sampling_config = self.config.sampling
        self.generate_test_data()

    def tearDown(self):
        """
        Delete the workspace folder after the test.
        """
        shutil.rmtree("test_preprocessing_workspace")
        # TODO: uncomment this line before merge
        # hardcoding here is intentional so that in case of
        # modifications to the test config file
        # we don't delete the wrong folder by accident (e.g. the root folder)
        # plese modify this line if you change the test_parameters.yaml file


class UnderSamplingTestCase(ResamplingTestCaseComplexDistr):
    """
    Test the implementation of the UnderSampling class.
    """

    def test_count_no_samples_defined(self):
        """Test no samples defined."""
        del self.sampling_config.options.samples_training
        us_norepl = UnderSampling(self.config)
        with self.assertRaises(KeyError):
            us_norepl.initialise_samples()

    def test_different_samples_per_category(self):
        """Test different samples per category."""
        del self.sampling_config.options.samples_training["zprime"][1]
        us_norepl = UnderSampling(self.config)
        with self.assertRaises(RuntimeError):
            us_norepl.initialise_samples()

    def test_run(self):
        """
        Test the run method. and ensure the deterministic sampling for a seed.
        Most of the major code changes would brake this test.
        """
        countsampler = UnderSampling(self.config)
        countsampler.Run()
        indices = countsampler.get_indices()
        # ensure the deterministic sampling for a seed
        np.testing.assert_array_equal(
            indices["training_ttbar_bjets"][:10],
            np.array([8, 12, 13, 16, 18, 27, 28, 29, 33, 35]),
        )
        np.testing.assert_array_equal(
            indices["training_zprime_ujets"][:10],
            np.array([7, 16, 49, 51, 59, 62, 79, 86, 102, 154]),
        )

    # def testDESIRED_equal_length(self):
    #     """Test equal fractions."""
    #     # ensure the total number of jets is divisible by the number of categories (3)
    #     n_jets_initial = self.config.sampling["options"]["n_jets"]
    #     n_jets_to_plot_initial = self.config.sampling["options"]["n_jets_to_plot"]
    #     self.config.sampling["options"]["n_jets"] = 3e5
    #     self.config.sampling["options"]["n_jets_to_plot"] = 3e5
    #     countsampler = UnderSampling(self.config)
    #     countsampler.Run()
    #     indices = countsampler.get_indices()
    #     n_b = len(indices["training_ttbar_bjets"]) + len(
    #         indices["training_zprime_bjets"]
    #     )
    #     n_c = len(indices["training_ttbar_cjets"]) + len(
    #         indices["training_zprime_cjets"]
    #     )
    #     n_u = len(indices["training_ttbar_ujets"]) + len(
    #         indices["training_zprime_ujets"]
    #     )
    #     self.assertEqual(n_b, n_c)
    #     self.assertEqual(n_b, n_u)
    #     self.config.sampling["options"]["n_jets"] = n_jets_initial
    #     self.config.sampling["options"]["n_jets_to_plot"] = n_jets_to_plot_initial

    def test_equal_length_smalln(self):
        """Test equal fractions."""
        # ensure the total number of jets is divisible by
        # the number of categories (3) and small
        n_jets_initial = self.config.sampling.options.n_jets
        n_jets_to_plot_initial = self.config.sampling.options.n_jets_to_plot
        self.config.sampling.options.n_jets = 900
        self.config.sampling.options.n_jets_to_plot = 900
        countsampler = UnderSampling(self.config)
        countsampler.Run()
        indices = countsampler.get_indices()
        n_b = len(indices["training_ttbar_bjets"]) + len(
            indices["training_zprime_bjets"]
        )
        n_c = len(indices["training_ttbar_cjets"]) + len(
            indices["training_zprime_cjets"]
        )
        n_u = len(indices["training_ttbar_ujets"]) + len(
            indices["training_zprime_ujets"]
        )
        self.assertEqual(n_b, n_c)
        self.assertEqual(n_b, n_u)
        self.config.sampling.options.n_jets = n_jets_initial
        self.config.sampling.options.n_jets_to_plot = n_jets_to_plot_initial


class UnderSamplingTestCaseEasyDistr(ResamplingTestCaseEasyDistr):
    """
    Test the implementation of the UnderSampling class, using a simple example.
    """

    def test_equal_length(self):
        """Test equal fractions."""
        countsampler = UnderSampling(self.config)
        countsampler.Run()
        indices = countsampler.get_indices()

        n_b = len(indices["training_ttbar_cjets"])
        n_c = len(indices["training_ttbar_cjets"])
        n_u = len(indices["training_ttbar_ujets"])
        self.assertEqual(n_b, n_c)
        self.assertEqual(n_b, n_u)


class PDFResamplingTestCase(ResamplingTestCaseComplexDistr):
    """
    Test the run method. and ensure the deterministic sampling for a seed.
    Most of the major code changes would brake this test.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        super().setUp()

        # tweek configs so that they wrk for PDF sampling
        self.sampling_config.options.target_distribution = "bjets"
        self.sampling_config.options.sampling_variables[0][global_config.pTvariable][
            "bins"
        ] = [
            [0, 2.5e5, 11],
            [
                2.6e5,
                5e5,
                11,
            ],
        ]
        self.sampling_config.options.sampling_variables[1][global_config.etavariable][
            "bins"
        ] = [
            [0, 1.25, 5],
            [0, 1.25, 5],
        ]

    def test_no_samples_defined(self):
        """Test no samples defined."""
        del self.sampling_config.options.samples_training
        self.assertRaises(AttributeError, lambda: PDFSampling(self.config))

    def test_run(self):
        """Test equal fractions."""
        pdfsampler = PDFSampling(self.config)
        pdfsampler.Run()
        output_name = pdfsampler.config.get_file_name(
            option="resampled",
            use_val=pdfsampler.use_validation_samples,
        )
        with h5py.File(output_name, "r") as file:
            jets = np.array(file["jets"]["pt_btagJes"][:10])
        jets_must = np.array(
            [
                264299.53,
                272481.4,
                271023.8,
                282889.47,
                253087.52,
                305471.3,
                223912.48,
                256265.61,
                279714.03,
                260276.78,
            ]
        )
        np.testing.assert_allclose(jets, jets_must)


class UnderSamplingNoReplaceTestCase(ResamplingTestCaseComplexDistr):
    """
    Test the implementation of the UnderSamplingNoReplace class.
    """

    def setUp(self):
        """
        Create a default dataset for testing.
        """
        super().setUp()
        self.sampling_config.options.target_distribution = "bjets"

    def test_no_samples_defined(self):
        """Test no samples defined."""
        del self.sampling_config.options.samples_training
        us_norepl = UnderSamplingNoReplace(self.config)
        with self.assertRaises(KeyError):
            us_norepl.initialise_samples()

    def test_different_samples_per_category(self):
        """Test different samples per category."""
        del self.sampling_config.options.samples_training["zprime"][1]
        us_norepl = UnderSamplingNoReplace(self.config)
        with self.assertRaises(RuntimeError):
            us_norepl.initialise_samples()

    def test_equal_fractions(self):
        """Test equal fractions."""
        us_norepl = UnderSamplingNoReplace(self.config)
        us_norepl.initialise_samples()
        indices = us_norepl.get_indices()
        with self.subTest():
            self.assertEqual(
                len(indices["training_ttbar_bjets"])
                + len(indices["training_zprime_bjets"]),
                len(indices["training_ttbar_cjets"])
                + len(indices["training_zprime_cjets"]),
            )
        with self.subTest():
            self.assertEqual(
                len(indices["training_ttbar_bjets"])
                + len(indices["training_zprime_bjets"]),
                len(indices["training_ttbar_ujets"])
                + len(indices["training_zprime_ujets"]),
            )
