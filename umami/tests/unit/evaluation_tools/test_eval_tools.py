import unittest

import numpy as np

from umami.evaluation_tools.PlottingFunctions import (
    GetCutDiscriminant,
    GetScore,
    GetScoreC,
    discriminant_output_shape,
    eff_err,
    getDiscriminant,
)


class Small_funcs_TestCase(unittest.TestCase):
    """
    Test eff_err, discriminant_output_shape, getDiscriminant
    """

    def setUp(self):
        self.x = 0.5
        self.N = 1
        self.result = 0.5
        self.x_keras = np.array([[0.20, 0.20, 0.60], [0.20, 0.40, 0.20]])
        self.fc = 0.018

    def test_eff_err(self):
        score = eff_err(x=self.x, N=self.N)

        self.assertEqual(score, self.result)

    def test_discriminant_output_shape(self):
        shape = discriminant_output_shape((2, 4))

        self.assertEqual(shape, (2,))

    def test_getDiscriminant(self):
        discriminant = getDiscriminant(x=self.x_keras, fc=self.fc)

        self.assertEqual(discriminant.shape, (len(self.x_keras),))

        results = np.array([1.0986122886681096, -0.01783991812833112])
        for i in range(len(results)):
            self.assertEqual(discriminant.numpy()[i], results[i])


class GetScore_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create numbers for testing
        """
        # Create a temporary directory
        self.pb = np.array([0.33, 0.33, 0.33])
        self.pc = np.array([0.33, 0.33, 0.33])
        self.pu = np.array([0.33, 0.33, 0.33])
        self.ptau = np.array([0.01, 0.01, 0.01])
        self.fc = 0.018
        self.ftau = 0.018
        self.WP = 0.77
        self.add_small = 1e-10

    def test_GetScore_no_taus(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=None,
            fc=self.fc,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.0)

        self.assertEqual(scores.shape, (3,))

    def test_GetScore_taus_no_ftau(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == -0.029323411188329818)

        self.assertEqual(scores.shape, (3,))

    def test_GetScore_taus_with_ftau(self):
        scores = GetScore(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=self.ftau,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.017608672135089463)

        self.assertEqual(scores.shape, (3,))

    def test_GetCutDiscriminant(self):
        cut_value = GetCutDiscriminant(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fc=self.fc,
            ftau=self.ftau,
            wp=self.WP,
        )

        self.assertEqual(cut_value, 0.017608672135089463)
        self.assertEqual(cut_value.shape, ())


class GetScoreC_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Create numbers for testing
        """
        # Create a temporary directory
        self.pb = np.array([0.33, 0.33, 0.33])
        self.pc = np.array([0.33, 0.33, 0.33])
        self.pu = np.array([0.33, 0.33, 0.33])
        self.ptau = np.array([0.01, 0.01, 0.01])
        self.fb = 0.018
        self.ftau = 0.018
        self.add_small = 1e-10

    def test_GetScoreC_no_taus(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=None,
            fb=self.fb,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.0)

        self.assertEqual(scores.shape, (3,))

    def test_GetScoreC_taus_no_ftau(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fb=self.fb,
            ftau=None,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == -0.029323411188329818)

        self.assertEqual(scores.shape, (3,))

    def test_GetScoreC_taus_with_ftau(self):
        scores = GetScoreC(
            pb=self.pb,
            pc=self.pc,
            pu=self.pu,
            ptau=self.ptau,
            fb=self.fb,
            ftau=self.ftau,
        )

        for i, score in enumerate(scores):
            self.assertTrue(score == 0.017608672135089463)

        self.assertEqual(scores.shape, (3,))
