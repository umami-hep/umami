import os
import pickle

import numpy as np
from scipy.interpolate import RectBivariateSpline

from umami.configuration import logger


class PDFSampling(object):
    """
    Sampling method using ratios between distributions to sample training
    file.
    An importance sampling approach
    """

    def __init__(self):
        self.inter_func = None
        self._ratio = None

    def CalculatePDF(
        self,
        x_y_target,
        x_y_original,
        bins=[100, 9],
        ratio_max: float = 1,
    ):
        """
        Calculates the histograms of the input data and uses them to
        calculate the PDF Ratio.
        CalculatePDFRatio is invoked here.

        Inputs:
        x_y_target: A 2D tuple of the target datapoints of x and y.
        x_y_original: A 2D tuple of the to resample datapoints of x and y.
        bins: This can be all possible binning inputs as for numpy
              histogram2d.
        ratio_max: Maximum Ratio difference which is used for upsampling
                   the inputs.

        Output:
        Provides the PDF interpolation function which is used for sampling.
        This can be returned with Inter_Func. It is a property of the class.
        """

        # Calculate the corresponding histograms
        h_target, self._x_bin_edges, self._y_bin_edges = np.histogram2d(
            x_y_target[0], x_y_target[1], bins
        )

        h_original, _, _ = np.histogram2d(
            x_y_original[0],
            x_y_original[1],
            [self._x_bin_edges, self._y_bin_edges],
        )

        # Calculate the PDF Ratio
        self.CalculatePDFRatio(
            h_target,
            h_original,
            self._x_bin_edges,
            self._y_bin_edges,
            ratio_max=ratio_max,
        )

    def CalculatePDFRatio(
        self,
        h_target,
        h_original,
        x_bin_edges,
        y_bin_edges,
        ratio_max: float = 1,
    ):
        """
        Receives the histograms of the target and original data, the bins
        and a max ratio value. Latter is optional.

        Inputs:
        h_target: Output of numpy histogram2D for the target datapoints
        h_original: Output of numpy histogram2D for the original datapoints
        bins: The bin edges of the binning used for the numpy histogram2D.
              This is also returned from numpy histgram2D
        ratio_max: Maximum Ratio difference which is used for upsampling
                   the inputs.

        Output:
        Provides the PDF interpolation function which is used for sampling.
        This can be returned with Inter_Func. It is a property of the class.
        """

        # Normalise the histograms to unity
        h_target = h_target / np.sum(h_target)
        h_original = h_original / np.sum(h_original)

        # Transform bin edges to bin centres
        self.x_bins = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        self.y_bins = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

        # Calculating the ratio of the reference distribution w.r.t. the target distribution
        ratio = np.divide(
            h_target,
            h_original,
            out=np.zeros(
                h_original.shape,
                dtype=float,
            ),
            where=(h_original != 0),
        )

        # Setting max ratio value
        self._ratio = ratio / ratio.max() * ratio_max

        # Calculate interpolation function
        logger.info("Retrieve interpolation function")
        self.inter_func = RectBivariateSpline(
            self.x_bins, self.y_bins, self._ratio
        )

    def save(self, file_name: str, overwrite: bool = False):
        """
        Save the interpolation function to file

        Input:
        file_name: Path where the pickle file is saved.

        Output:
        Pickle file of the PDF interpolation function.
        """

        if self.inter_func is not None:
            if os.path.isfile(file_name) is True:
                if overwrite is True:
                    logger.warning(
                        "File already exists at given path! Overwrite existing file!"
                    )

                    # Dump function into pickle file
                    with open(file_name, "wb") as file:
                        pickle.dump(self.inter_func, file)

                else:
                    logger.warning(
                        "File already exists at given path! PDF interpolation function not saved!"
                    )

            else:
                # Dump function into pickle file
                with open(file_name, "wb") as file:
                    pickle.dump(self.inter_func, file)

        else:
            raise ValueError("Interpolation function not calculated/given!")

    def load(self, file_name: str):
        """
        Load the interpolation function from file.

        Input:
        file_name: Path where the pickle file is saved.

        Output:
        PDF interpolation function of the pickle file is added as property
        to the class.
        """

        with open(file_name, "rb") as file:
            self.inter_func = pickle.load(file)

    # the resampling is so far only working for a batch which is being normalised
    # TODO: rename
    def inMemoryResample(self, x_values, y_values, size):
        """
        Resample all of the datapoints at once. Requirement for that
        is that all datapoints fit in the RAM.

        Input:
        x_values: x values of the datapoints which are to be resampled from (i.e pT)
        y_values: y values of the datapoints which are to be resampled from (i.e eta)
        size: Number of jets which are resampled.

        Output:
        Resampled jets
        """

        if type(x_values) == float or type(x_values) == int:
            x_values = np.asarray([x_values])

        if type(y_values) == float or type(y_values) == int:
            y_values = np.asarray([y_values])

        # Check for sizes of x_values and y_values
        if len(y_values) != len(x_values):
            raise ValueError("x_values and y_values need to have same size!")

        # Evaluate the datapoints with the PDF function
        r_resamp = self.inter_func.ev(x_values, y_values)

        # Discard all datapoints where the ratio is 0 or less
        indices = np.where(r_resamp >= 0)[0]
        r_resamp = r_resamp[indices]

        # Normalise the datapoints for sampling
        r_resamp = r_resamp / np.sum(r_resamp)

        # Resample the datapoints based on their PDF Ratio value
        sampled_indices = np.random.default_rng().choice(
            indices, p=r_resamp, size=size
        )

        # Return the resampled datapoints
        return x_values[sampled_indices], y_values[sampled_indices]

    # TODO: rename
    def Resample(self, x_values, y_values):
        """
        Resample a batch of datapoints at once. This function is used
        if multiple files need to resampled and also if the datapoints
        does not fit in the RAM.

        Input:
        x_values: x values of the datapoints which are to be resampled from (i.e pT)
        y_values: y values of the datapoints which are to be resampled from (i.e eta)

        Output:
        Resampled jets
        """

        if type(x_values) == float or type(x_values) == int:
            x_values = np.asarray([x_values])

        if type(y_values) == float or type(y_values) == int:
            y_values = np.asarray([y_values])

        # Check for sizes of x_values and y_values
        if len(y_values) != len(x_values):
            raise ValueError("x_values and y_values need to have same size!")

        # Evaluate the datapoints with the PDF function
        r_resamp = self.inter_func.ev(x_values, y_values)

        # Get random numbers from generator
        rnd_numbers = np.random.default_rng().uniform(0, 1, len(r_resamp))

        # Decide, based on the PDF values for the datapoints and the random
        # numbers which datapoints are sampled
        sampled_indices = np.where(rnd_numbers < r_resamp)

        # Return sampled datapoints
        return x_values[sampled_indices], y_values[sampled_indices]

    @property
    def ratio(self):
        return self._ratio

    @property
    def Inter_Func(self):
        return self.inter_func
