import os
import tempfile
import unittest

import numpy as np

from umami.preprocessing_tools import PDFSampling


class PDFSampling_TestCase(unittest.TestCase):
    """
    Unit test the PDFSampling class
    """

    def setUp(self):
        """
        Set-Up a few testarrays for PDF Sampling
        """

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_func_dir = f"{self.tmp_dir.name}/"
        self.func_dir = os.path.join(os.path.dirname(__file__), "fixtures/")
        self.x_y_target = (
            np.random.default_rng().uniform(-1, 1, 10000),
            np.random.default_rng().uniform(-1, 1, 10000),
        )

        self.x_y_original = (
            np.random.default_rng().normal(0, 1, size=10000),
            np.random.default_rng().normal(0, 1, size=10000),
        )

        self.ratio_max = 1
        self.bins = [50, 50]

    def test_Init_Properties(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Check if Interpolation function is None
        self.assertTrue(Sampler.Inter_Func is None)
        self.assertTrue(Sampler.ratio is None)

    def test_CalculatePDF(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Calculate Interpolation function
        Sampler.CalculatePDF(
            x_y_target=self.x_y_target,
            x_y_original=self.x_y_original,
            bins=self.bins,
        )

        self.assertTrue((self.bins[0], self.bins[1]) == Sampler.ratio.shape)
        self.assertTrue(Sampler.Inter_Func is not None)

    def test_CalculatePDFRatio(self):
        # Init new Sampler
        Sampler = PDFSampling()

        h_target, x_bin_edges, y_bin_edges = np.histogram2d(
            x=self.x_y_target[0],
            y=self.x_y_target[1],
            bins=self.bins,
        )

        h_original, _, _ = np.histogram2d(
            x=self.x_y_original[0],
            y=self.x_y_original[1],
            bins=[x_bin_edges, y_bin_edges],
        )

        Sampler.CalculatePDFRatio(
            h_target=h_target,
            h_original=h_original,
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            ratio_max=self.ratio_max,
        )

        self.assertTrue((self.bins[0], self.bins[1]) == Sampler.ratio.shape)
        self.assertTrue(Sampler.Inter_Func is not None)

    def test_inMemoryResample(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Calculate Interpolation function
        Sampler.CalculatePDF(
            x_y_target=self.x_y_target,
            x_y_original=self.x_y_original,
            bins=self.bins,
        )

        x_values, y_values = Sampler.inMemoryResample(
            x_values=self.x_y_original[0],
            y_values=self.x_y_original[1],
            size=1000,
        )

        self.assertEqual(
            len(x_values),
            1000,
        )

        self.assertEqual(
            len(y_values),
            1000,
        )

    def test_Resample_Array(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Calculate Interpolation function
        Sampler.CalculatePDF(
            x_y_target=self.x_y_target,
            x_y_original=self.x_y_original,
            bins=self.bins,
        )

        x_values, y_values = Sampler.Resample(
            x_values=self.x_y_original[0],
            y_values=self.x_y_original[1],
        )

        self.assertEqual(
            len(x_values),
            len(y_values),
        )

    def test_Resample_Float(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Calculate Interpolation function
        Sampler.CalculatePDF(
            x_y_target=self.x_y_target,
            x_y_original=self.x_y_original,
            bins=self.bins,
        )

        x_values, y_values = Sampler.Resample(
            y_values=2,
            x_values=2,
        )

        self.assertEqual(
            len(x_values),
            len(y_values),
        )

    def test_save(self):
        # Init new Sampler
        Sampler = PDFSampling()

        # Calculate Interpolation function
        Sampler.CalculatePDF(
            x_y_target=self.x_y_target,
            x_y_original=self.x_y_original,
            bins=self.bins,
        )

        # Save function to pickle file
        Sampler.save(
            os.path.join(self.tmp_func_dir, "PDF_interpolation_function.pkl")
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.tmp_func_dir, "PDF_interpolation_function.pkl"
                )
            )
        )

    def test_load(self):
        # Init new Sampler
        Sampler = PDFSampling()

        Sampler.load(
            os.path.join(self.func_dir, "PDF_interpolation_function.pkl")
        )

        x_values, y_values = Sampler.inMemoryResample(
            x_values=self.x_y_original[0],
            y_values=self.x_y_original[1],
            size=1000,
        )

        self.assertEqual(
            len(x_values),
            1000,
        )

        self.assertEqual(
            len(y_values),
            1000,
        )
