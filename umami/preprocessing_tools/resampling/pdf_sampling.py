"""PDF sampling module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import itertools
import json
import os
import pickle

import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from umami.configuration import logger
from umami.plotting_tools import plot_resampling_variables, preprocessing_plots
from umami.preprocessing_tools.resampling.resampling_base import (
    JsonNumpyEncoder,
    ResamplingTools,
    correct_fractions,
    read_dataframe_repetition,
    sampling_generator,
)
from umami.preprocessing_tools.utils import get_variable_dict


class PDFSampling(ResamplingTools):  # pylint: disable=too-many-public-methods
    """
    An importance sampling approach using ratios between distributions to sample
    and a target as importance weights.
    """

    def __init__(
        self,
        config: object,
        flavour: int = None,
    ) -> None:
        """
        Initialise class.  to 'target' or an int corresponding to the index
        of the flavour to process (in the list of samples from config samples).

        Parameters
        ----------
        config : object
            Loaded preprocessing config.
        flavour : int, optional
            Set flavour to a flavour corresponding to the given int (index entry). If
            None is given, the 'target' will be used, by default None.

        Raises
        ------
        ValueError
            if given flavour index is out of range.
        """

        super().__init__(config)

        # Set up the dicts and set random seed
        self.inter_func_dict = {}
        self._ratio_dict = {}
        self._bin_edges_dict = {}
        self.rnd_seed = 42

        # Check for usage of multiprocessing
        self.use_multiprocessing = True
        self.n_processes = 4

        # Get the samples which will be used for resampling
        self.samples_to_resample = (
            self.options.samples_validation
            if self.use_validation_samples
            else self.options.samples_training
        )

        # Setting some limits: important for good spline approximation
        sampling_var = self.options.sampling_variables

        # Init a list for the bin info and ranges info
        bin_info = []
        ranges_info = []
        extreme_ranges = []

        # Iterate over the chosen variables which are used for resampling
        for samp_var in sampling_var:

            # Iterate over the variables
            for _, var in enumerate(list(samp_var.keys())):

                # Add the bin and range info to the lists
                while len(bin_info) < len(samp_var[var]["bins"]):
                    bin_info.append([])
                    ranges_info.append([])

                # Init a default min and max
                themin, themax = 1e8, 0

                # Add bin infos of the categories
                for i, bin_cat in enumerate(samp_var[var]["bins"]):
                    bin_info[i].append(bin_cat[2])
                    ranges_info[i].append((bin_cat[0], bin_cat[1]))
                    if bin_cat[0] < themin:
                        themin = bin_cat[0]
                    if bin_cat[1] > themax:
                        themax = bin_cat[1]

                # Add them to the extreme ranges list
                extreme_ranges.append([themin, themax])

        # Init a dict with the bins, ranges and extreme ranges
        self.limit = {
            "bins": bin_info,
            "ranges": ranges_info,
            "extreme_ranges": extreme_ranges,
        }

        # Init a dict for the number to sample
        self.number_to_sample = {}

        # Get the max flavour index
        flavour_index = len(
            self.samples_to_resample[list(self.samples_to_resample.keys())[0]]
        )

        # Check if flavour is given
        if flavour is not None:
            self.do_target = False
            self.do_plotting = False
            self.do_combination = False

            new_flavour_list = []
            for item in flavour:
                if "target" == item:
                    self.do_target = True
                elif "plotting" == item:
                    self.do_plotting = True
                elif "combining" == item:
                    self.do_combination = True
                elif int(item) < flavour_index:
                    new_flavour_list.append(int(item))
                else:
                    raise ValueError(
                        f"Flavour key {item} unrecognised or out of range."
                    )
            self.do_flavours = new_flavour_list

        else:
            self.do_target = True
            self.do_plotting = True
            self.do_combination = True
            self.do_flavours = np.arange(flavour_index)

    @property
    def ratio(self):
        """
        Return the dict with the ratios inside.

        Returns
        -------
        Ratio_dict
            Dict with the ratios inside.
        """
        return self._ratio_dict

    @property
    def inter_func(self):
        """
        Return the dict with the interpolation functions inside.

        Returns
        -------
        inter_func_dict
            Dict with the interpolation functions inside.
        """
        return self.inter_func_dict

    def load_samples_generator(
        self,
        sample_category: str,
        sample_id: int,
        chunk_size: int,
    ):
        """
        Generator for the loading of the samples.

        Parameters
        ----------
        sample_category : str
            Sample category that is loaded.
        sample_id : int
            Index of the sample.
        chunk_size : int
            Chunk size of the jets that are loaded and yielded.

        Yields
        -------
        sample : str
            Name of the sample. "training_ttbar_bjets" for example.
        samples : dict
            Dict with the loaded jet info needed for resampling.
        Next_chunk : bool
            True if more chunks can be loaded, False if this was
            the last chunk.
        n_jets_initial : int
            Number of jets available.
        start_index : int
            Start index of the chunk.
        """

        # Get the sample name and the preparation configs (cuts etc.)
        sample, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Get path of the input file
        in_file = self.config.preparation.get_sample(
            preparation_sample.name
        ).output_name

        # Open input file
        with h5py.File(in_file, "r") as f_in:

            # Get the number of jets inside the file
            n_jets_initial = len(f_in["jets"])

            # Check if custom inital n_jets are given for this sample
            if self.options.custom_n_jets_initial is not None and sample in list(
                self.options.custom_n_jets_initial
            ):
                logger.warning(
                    "You selected the PDF resampling method but provided "
                    "custom_n_jets_initial! This option can't be used with "
                    "the PDF sampling! Ignoring custom_n_jets_initial!"
                )

            # Set start and end index
            start_ind = 0
            end_ind = int(start_ind + chunk_size)

            # Get a list for the tupled index paris
            tupled_indices = []

            # Iterate until the number of asked jets is reached
            while end_ind <= n_jets_initial or start_ind == 0:

                # Check that not an end index is chosen, which is not available
                # 100 jets available, chunk size: 30, end index: 80
                # Adding this up would cause an error
                if end_ind + chunk_size > n_jets_initial:
                    # Missing less then a chunk, joining to last chunk
                    end_ind = n_jets_initial

                # Append the start and end index pairs to list
                tupled_indices.append((start_ind, end_ind))

                # Set the old end index as new start index
                start_ind = end_ind

                # Get new end index
                end_ind = int(start_ind + chunk_size)

            # Loop over the index paris
            for index_tuple in tupled_indices:

                # Get the chunk of jets that is to be loaded
                to_load = f_in["jets"][index_tuple[0] : index_tuple[1]]

                # Load the two resampling variables from the jets
                jets_x = np.asarray(to_load[self.var_x])
                jets_y = np.asarray(to_load[self.var_y])

                # Stack the jet variables
                sample_vector = np.column_stack((jets_x, jets_y))

                # Get a dict with the file, the stacked jets and the category
                samples = {
                    "file": in_file,
                    "sample_vector": sample_vector,
                    "category": preparation_sample.category,
                }

                # Yield the sample name, the dict with the info,
                # a bool value if this was the last chunk (False) if not (True)
                # and the available jets and the start index of the chunk
                yield sample, samples, index_tuple[
                    1
                ] != n_jets_initial, n_jets_initial, index_tuple[0]

    def load_index_generator(self, in_file: str, chunk_size: int):
        """
        Generator that yields the indicies of the jets that are
        to be loaded.

        Parameters
        ----------
        in_file : str
            Filepath of the input file.
        chunk_size : int
            Chunk size of the jets that are loaded and yielded.

        Yields
        -------
        indices : np.ndarray
            Indicies of the jets which are to be loaded.
        index_tuple : int
            End index of the chunk.
        """

        # Open input file
        with h5py.File(in_file, "r") as f_in:

            # Get the number of available jets
            n_indices = len(f_in["jets"])

            # Set start and end index
            start_ind = 0
            end_ind = int(start_ind + chunk_size)

            # Init list for the index pairs (start, end)
            tupled_indices = []

            # Get the start and end index pairs for the given
            # chunk size
            while end_ind <= n_indices or start_ind == 0:
                if end_ind + chunk_size > n_indices:
                    # Missing less then a chunk, joining to last chunk
                    end_ind = n_indices

                # Create the tuple and append it
                tupled_indices.append((start_ind, end_ind))

                # Set new start and end index values
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            # Iterate over the index pairs
            for index_tuple in tupled_indices:

                # Get the indicies which are to be loaded
                loading_indices = np.arange(index_tuple[0], index_tuple[1])

                # Get the jets based on their indicies
                indices = np.asarray(f_in["jets"][loading_indices])

                # Yield the indicies
                yield indices, index_tuple[1] != n_indices

    def load_samples(self, sample_category: str, sample_id: int):
        """
        Load the input file of the specified category and id.

        Parameters
        ----------
        sample_category : str
            Sample category that is loaded.
        sample_id : int
            Index of the sample.

        Returns
        -------
        sample : str
            Name of the sample which is loaded.
        samples : dict
            Dict with the info retrieved for resampling.
        """

        # Get the sample name and the preparation configs (cuts etc.)
        sample, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Get path of the input file
        in_file = self.config.preparation.get_sample(preparation_sample).output_name

        # Open input file
        with h5py.File(in_file, "r") as f_in:
            # Get the number of jets inside the file
            n_jets_initial = len(f_in["jets"])

            # Check if custom inital n_jets are given for this sample
            if (
                "custom_n_jets_initial" in self.options
                and self.options["custom_n_jets_initial"] is not None
                and sample in list(self.options["custom_n_jets_initial"])
                and self.options["custom_n_jets_initial"][sample] is not None
            ):
                logger.warning(
                    "You selected the PDF resampling method but provided "
                    "custom_n_jets_initial! This option can't be used with "
                    "the PDF sampling! Ignoring custom_n_jets_initial!"
                )

            # Get the jets which are to be loaded
            to_load = f_in["jets"][:n_jets_initial]

            # Retrieve the resampling variables from the jets
            jets_x = np.asarray(to_load[self.var_x])
            jets_y = np.asarray(to_load[self.var_y])
            logger.info(
                "Loaded %i %s jets from %s.",
                len(jets_x),
                preparation_sample.category,
                sample,
            )

        # Stack the jets
        sample_vector = np.column_stack((jets_x, jets_y))

        # Create dict with the info for resampling
        samples = {
            "file": in_file,
            "sample_vector": sample_vector,
            "category": preparation_sample.category,
        }

        # Return sample name and the dict
        return sample, samples

    def file_to_histogram(
        self,
        sample_category: str,
        category_ind: int,
        sample_id: int,
        iterator: bool = True,
        chunk_size: int = 1e4,
        bins: list = None,
        hist_range: list = None,
    ) -> dict:
        """
        Convert the provided sample into a 2d histogram
        which is used to calculate the PDF functions.

        Parameters
        ----------
        sample_category : str
            Sample category that is loaded.
        category_ind : int
            Index of the category which is used.
        sample_id : int
            Index of the sample that is used.
        iterator : bool, optional
            Decide, if the iterative approach is used (True) or
            the in memory approach (False), by default True.
        chunk_size : int, optional
            Chunk size for loading the jets in the iterative approach,
            by default 1e4.
        bins : list, optional
            List with the bins to use for the 2d histogram,
            by default None.
        hist_range : list, optional
            List with histogram ranges for the 2d histogram function,
            by default None.

        Returns
        -------
        Results_dict : dict
            Dict with the 2d histogram info.
        """

        if bins is None:
            bins = self.limit["bins"][category_ind]

        if hist_range is None:
            hist_range = self.limit["ranges"][category_ind]

        # Init the available numbers
        available_numbers = 0

        # Check if the iterative method is used
        if iterator:
            # Get the generator which loads the sample
            generator = self.load_samples_generator(
                sample_category=sample_category,
                sample_id=sample_id,
                chunk_size=chunk_size,
            )

            # Set load in chunks and chunk counter
            load_chunk = True
            chunk_counter = 0

            # Iterate over until all chunks are loaded
            while load_chunk:

                # Get the dict with the resampling info, the info if more
                # chunks can be loaded and the number of total jets available
                try:
                    _, target_dist, load_more, total, _ = next(generator)

                # If the no more chunks can be loaded, break
                except StopIteration:
                    break

                # Get the info if more chunks can be loaded or not
                load_chunk = load_more

                # Check if this is the first chunk
                if chunk_counter == 0:
                    # If this is the first chunk, init the progress bar
                    pbar = tqdm(total=total)

                    # Get the 2d histogram of the two resampling variables
                    h_target, x_bin_edges, y_bin_edges = np.histogram2d(
                        target_dist["sample_vector"][:, 0],
                        target_dist["sample_vector"][:, 1],
                        bins=bins,
                        range=hist_range,
                    )

                else:

                    # Get the 2d histogram of the two resampling variables
                    new_hist, _, _ = np.histogram2d(
                        target_dist["sample_vector"][:, 0],
                        target_dist["sample_vector"][:, 1],
                        bins=bins,
                        range=hist_range,
                    )

                    # Add the new loaded one to the old
                    h_target += new_hist

                # Get the number of jets that were loaded in this step
                n_jets_added = len(target_dist["sample_vector"])

                # Update the progres bar
                pbar.update(n_jets_added)

                # Add the number of loaded jets to the availabe number
                available_numbers += n_jets_added

                # Set the chunk counter up
                chunk_counter += 1

            # Close progress bar after finishing the loop
            pbar.close()

        # Or the in memory approach
        else:

            # Get the target dist from the sample
            _, target_dist = self.load_samples(sample_category, sample_id)

            # Get the number of available jets
            available_numbers = len(target_dist["sample_vector"])

            # Create the 2d histogram
            h_target, x_bin_edges, y_bin_edges = np.histogram2d(
                target_dist["sample_vector"][:, 0],
                target_dist["sample_vector"][:, 1],
                bins=bins,
                range=hist_range,
            )

        # Get the preparation info (cuts, etc.) for the chosen sample
        _, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Add the results to a dict
        return_dict = {
            "hist": h_target,
            "xbins": x_bin_edges,
            "ybins": y_bin_edges,
            "available_numbers": available_numbers,
            "category": preparation_sample.category,
        }

        # If the in memory approach is chosen, set the target distribution in dict
        if not iterator:
            return_dict["target_dist"] = target_dist

        # Return the dict with the 2d histogram info
        return return_dict

    def initialise_flavour_samples(self) -> None:
        """
        Initialising input files: this one just creates the map.
        (based on UnderSampling one). At this point the arrays of
        the 2 variables are loaded which are used
        for the sampling and saved into class variables.
        """

        # saving a list of sample categories with associated IDs
        self.sample_categories = {
            elem: i for i, elem in enumerate(list(self.samples_to_resample.keys()))
        }
        self.check_sample_consistency(self.samples_to_resample)

        self.upsampling_max_rate = {}
        self.sample_map = {elem: {} for elem in list(self.samples_to_resample.keys())}
        self.sample_file_map = {}
        self.max_upsampling = {}
        self.sampling_fraction = {}

        sample_id = 0
        for sample_category in self.sample_categories:
            self.sample_file_map[sample_category] = {}
            for sample_id, sample in enumerate(
                self.samples_to_resample[sample_category]
            ):
                preparation_sample = self.preparation_config.get_sample(sample)
                self.sample_file_map[sample_category][sample_id] = (
                    sample,
                    preparation_sample,
                )
                # If max upsampling ratio set, need to save the number of jets available
                if self.options.max_upsampling_ratio is not None and sample in list(
                    self.options.max_upsampling_ratio
                ):
                    max_upsampling = float(self.options.max_upsampling_ratio[sample])
                    in_file = self.config.preparation.get_sample(
                        preparation_sample.name
                    ).output_name
                    with h5py.File(in_file, "r") as f_h5:
                        num_available = len(f_h5["jets"])
                    self.max_upsampling[sample] = (
                        max_upsampling,
                        num_available,
                    )
                if self.options.sampling_fraction is not None and sample in list(
                    self.options.sampling_fraction
                ):
                    self.sampling_fraction[sample] = self.options.sampling_fraction[
                        sample
                    ]

    def check_sample_consistency(self, samples: dict) -> None:
        """
        Helper function to check if each sample category has the same amount
        of samples with same category (e.g. Z' and ttbar both have b, c & light)

        Parameters
        ----------
        samples : dict
            Dict with the categories (ttbar, zpext) and their corresponding
            sample names.

        Raises
        ------
        KeyError
            If the sample which is requested is not in the preparation stage.
        RuntimeError
            Your specified samples in the sampling/samples block need to have
            the same category in each sample category.
        """

        # Init a dict with the sample categories
        check_consistency = {elem: [] for elem in self.sample_categories}

        # Iterate over the categories
        for category in self.sample_categories:

            # Iterate over the samples in the category
            for sample in samples[category]:

                # Get the preparation configs (cuts etc.) for the sample
                preparation_sample = self.preparation_config.get_sample(sample)

                # Check that the samples are also defined in the preparation stage
                if preparation_sample is None:
                    raise KeyError(
                        f"'{sample}' was requested in sampling/samples block,"
                        "however, it is not defined in preparation/samples in"
                        "the preprocessing config file"
                    )

                # Add the sample to the check dict
                check_consistency[category].append(preparation_sample.category)

        # Get the combinations of the dict keys
        combs = list(itertools.combinations(check_consistency.keys(), 2))

        # Create a list with the checks of the combinations
        combs_check = [
            sorted(check_consistency[elem[0]]) == sorted(check_consistency[elem[1]])
            for elem in combs
        ]

        # Check that all combinations are valied
        if not all(combs_check):
            raise RuntimeError(
                "Your specified samples in the sampling/samples "
                "block need to have the same category in each sample category."
            )

        # Get the class categories
        self.class_categories = check_consistency[next(iter(self.sample_categories))]

    def calculate_pdf(
        self,
        store_key: str,
        x_y_original: tuple = None,
        x_y_target: tuple = None,
        target_hist: np.ndarray = None,
        original_hist: np.ndarray = None,
        target_bins: tuple = None,
        bins: list = None,
        limits: list = None,
    ) -> None:
        """
        Calculates the histograms of the input data and uses them to
        calculate the PDF Ratio. Works either on dataframe or pre-made histograms
        CalculatePDFRatio is invoked here.
        Provides the PDF interpolation function which is used for sampling
        (entry in a dict). It is a property of the class.

        Parameters
        ----------
        store_key : str
            Key of the interpolation function to be added
            to self.inter_func_dict (and self._ratio_dict)
        x_y_original : tuple, optional
            A 2D tuple of the to resample datapoints of x and y, by default None.
        x_y_target : tuple, optional
            A 2D tuple of the target datapoints of x and y, by default None.
        target_hist : np.ndarray, optional
            Histogram for the target, by default None
        original_hist : np.ndarray, optional
            Histogram for the original flavour, by default None
        target_bins : tuple, optional
            If using target_hist, need to define target_bins, a tuple with
            (binx, biny), by default None.
        bins : list, optional
            This can be all possible binning inputs as for numpy
            histogram2d. Not used if hist are passed instead of arrays,
            by default None.
        limits : list, optional
            Limits for the binning. Not used if hist are passed instead of arrays,
            by default None.

        Raises
        ------
        ValueError
            If feeding a histogram but not the bins in PDF calculation.
        ValueError
            If improper target input for PDF calculation of the store_key.
        ValueError
            If improper original flavour input for PDF calculation of store_key
        """

        # Check if bins are defined and set value
        if bins is None:
            bins = [100, 9]

        # Calculate the corresponding histograms if target histogram is defined
        if target_hist is not None:
            if target_bins is not None:
                h_target, self._x_bin_edges, self._y_bin_edges = (
                    target_hist,
                    target_bins[0],
                    target_bins[1],
                )
            else:
                raise ValueError(
                    "Feeding a histogram but not the bins in PDF calculation."
                )

        # Calculate the corresponding histograms if the xy target is defined
        elif x_y_target is not None:
            h_target, self._x_bin_edges, self._y_bin_edges = np.histogram2d(
                x_y_target[:, 0], x_y_target[:, 1], bins, range=limits
            )

        else:
            raise ValueError(
                f"Improper target input for PDF calculation of {store_key}."
            )

        # Check if the original histogram is provided
        if original_hist is not None:
            h_original = original_hist

        # Check if the original x y is provided
        elif x_y_original is not None:
            h_original, _, _ = np.histogram2d(
                x_y_original[:, 0],
                x_y_original[:, 1],
                bins=[self._x_bin_edges, self._y_bin_edges],
            )

        else:
            raise ValueError(
                f"Improper original flavour input for PDF calculation of {store_key}."
            )

        # Calculate the PDF Ratio
        self.calculate_pdf_ratio(
            store_key=store_key,
            h_target=h_target,
            h_original=h_original,
            x_bin_edges=self._x_bin_edges,
            y_bin_edges=self._y_bin_edges,
        )

    def calculate_pdf_ratio(
        self,
        store_key: str,
        h_target: np.ndarray,
        h_original: np.ndarray,
        x_bin_edges: np.ndarray,
        y_bin_edges: np.ndarray,
    ) -> None:
        """
        Receives the histograms of the target and original data, the bins
        and a max ratio value. Latter is optional. Provides the PDF
        interpolation function which is used for sampling.
        This can be returned with inter_func. It is a property of the class.

        Parameters
        ----------
        store_key : str
            Key of the interpolation function to be added
            to self.inter_func_dict (and self._ratio_dict)
        h_target : np.ndarray
            Output of numpy histogram2D for the target datapoints.
        h_original : np.ndarray
            Output of numpy histogram2D for the original datapoints.
        x_bin_edges : np.ndarray
            Array with the x axis bin edges.
        y_bin_edges : np.ndarray
            Array with the y axis bin edges.
        """

        # Transform bin edges to bin centres
        self.x_bins = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        self.y_bins = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

        # Calculating the ratio of the reference distribution w.r.t. the target
        # distribution
        ratio = np.divide(
            h_target,
            h_original,
            out=np.zeros(
                h_original.shape,
                dtype=float,
            ),
            where=(h_original != 0),
        )

        # Store the ratio unter the store key as a class attribute
        self._ratio_dict[store_key] = ratio
        self._bin_edges_dict[store_key] = (
            x_bin_edges[1:-1],
            y_bin_edges[1:-1],
        )

        # Get the interpolation function and store it in the inter func dict
        # which is an attribute of the class
        self.inter_func_dict[store_key] = RectBivariateSpline(
            self.x_bins,
            self.y_bins,
            self._ratio_dict[store_key],
            kx=3,
            ky=3,
            bbox=[
                x_bin_edges[0],
                x_bin_edges[-1],
                y_bin_edges[0],
                y_bin_edges[-1],
            ],
        )

        # Get the path where the interpolation function will be saved
        save_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            f"inter_func_{store_key}_training"
            if not self.use_validation_samples
            else f"inter_func_{store_key}_validation",
        )

        # Save the interpolation function to file
        self.save(self.inter_func_dict[store_key], save_name)

    def save(
        self,
        inter_func: RectBivariateSpline,
        file_name: str,
        overwrite: bool = True,
    ) -> None:
        """
        Save the interpolation function to file.

        Parameters
        ----------
        inter_func : RectBivariateSpline
            Interpolation function.
        file_name : str
            Path where the pickle file is saved.
        overwrite : bool, optional
            Decide if the file is overwritten if it exists already,
            by default True

        Raises
        ------
        ValueError
            If no interpolation function is given.
        """

        if inter_func is not None:
            if os.path.isfile(file_name) is True:
                if overwrite is True:
                    logger.warning(
                        "File already exists at given path! Overwrite existing"
                        " file to save interpolation function!"
                    )

                    # Dump function into pickle file
                    with open(file_name, "wb") as file:
                        pickle.dump(inter_func, file)

                else:
                    logger.warning(
                        "File already exists at given path! PDF interpolation"
                        " function not saved!"
                    )

            else:
                # Dump function into pickle file
                with open(file_name, "wb") as file:
                    pickle.dump(inter_func, file)

        else:
            raise ValueError("Interpolation function not calculated/given!")

    def load(self, file_name: str) -> RectBivariateSpline:
        """
        Load the interpolation function from file.

        Parameters
        ----------
        file_name : str
            Path where the pickle file is saved.

        Returns
        -------
        RectBivariateSpline
            The loaded interpolation function.
        """

        with open(file_name, "rb") as file:
            inter_func = pickle.load(file)

        return inter_func

    def in_memory_resample(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        size: int,
        store_key: str,
        replacement: bool = True,
    ) -> np.ndarray:
        """
        Resample all of the datapoints at once. Requirement for that
        is that all datapoints fit in the RAM.

        Parameters
        ----------
        x_values : np.ndarray
            x values of the datapoints which are to be resampled from (i.e pT)
        y_values : np.ndarray
            y values of the datapoints which are to be resampled from (i.e eta)
        size : int
            Number of jets which are resampled.
        store_key : str
            Key of the interpolation function to be added to
            self.inter_func_dict (and self._ratio_dict)
        replacement : bool, optional
            Decide, if replacement is used in the resampling, by default True.

        Returns
        -------
        sampled_indices : np.ndarray
            The indicies of the sampled jets.

        Raises
        ------
        ValueError
            If x_values and y_values have different shapes.
        """

        # Check for replacement and log an info
        if replacement is False:
            logger.info("PDF sampling without replacement for given set!")

        # Check that the given values are np.ndarrays
        if isinstance(x_values, (float, int)):
            x_values = np.asarray([x_values])

        if isinstance(y_values, (float, int)):
            y_values = np.asarray([y_values])

        # Check for sizes of x_values and y_values
        if len(y_values) != len(x_values):
            raise ValueError("x_values and y_values need to have same size!")

        # Evaluate the datapoints with the PDF function
        r_resamp = self.return_unnormalised_pdf_weights(x_values, y_values, store_key)

        # Normalise the datapoints for sampling
        r_resamp = r_resamp / np.sum(r_resamp)

        if logger.level <= 10:
            # When debugging, this snippet plots the
            # weights and interpolated values vs pT.

            # Import plt for plotting
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

            # Histogram the x values with and without weights
            h_n, _ = np.histogram(x_values, bins=self._x_bin_edges)
            h_sy, _ = np.histogram(x_values, bins=self._x_bin_edges, weights=r_resamp)

            # Get mean
            mean = np.divide(
                h_sy,
                h_n,
                out=np.zeros(
                    h_sy.shape,
                    dtype=float,
                ),
                where=(h_n != 0),
            )

            # Plot the Distributions with and without weights
            plt.figure()
            plt.plot(self.x_bins, mean, label=f"effective weights {store_key}")
            plt.legend(loc="best", ncol=1, fontsize=7)
            plt.xlabel("pT (MeV)")
            plt.ylabel("Weights")
            plt.tight_layout()
            plt.savefig(f"{store_key}_weights.pdf")

            plt.figure()
            plt.plot(self.x_bins, h_n, label=f"Distribution {store_key}")
            plt.legend(loc="best", ncol=1, fontsize=7)
            plt.xlabel("pT (MeV)")
            plt.ylabel("Weights")
            plt.tight_layout()
            plt.savefig(f"{store_key}_distribution.pdf")

        # Resample the datapoints based on their PDF Ratio value
        sampled_indices = self.resample_chunk(r_resamp, size, replacement)

        # Return sampled jet indicies
        return sampled_indices

    def return_unnormalised_pdf_weights(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        store_key: str,
    ) -> np.ndarray:
        """
        Calculate the unnormalised PDF weights and return them.

        Parameters
        ----------
        x_values : np.ndarray
            x values of the datapoints which are to be resampled from (i.e pT)
        y_values : np.ndarray
            y values of the datapoints which are to be resampled from (i.e eta)
        store_key : str
            Key of the interpolation function to be added to
            self.inter_func_dict (and self._ratio_dict)

        Returns
        -------
        r_resamp : np.ndarray
            Array with the PDF weights.
        """

        # Get the filepath of the interpolation function
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            f"inter_func_{store_key}_training"
            if not self.use_validation_samples
            else f"inter_func_{store_key}_validation",
        )

        # Load the interpolation function
        inter_func = self.load(load_name)

        # Evaluate the given values with the interpolation function
        r_resamp = inter_func.ev(x_values, y_values)

        # Neutralise all datapoints where the ratio is less than 0
        indices = np.where(r_resamp < 0)[0]
        r_resamp[indices] = 0

        # Return the evaluated result
        return r_resamp

    def resample_chunk(
        self,
        r_resamp: np.ndarray,
        size: int,
        replacement: bool = True,
    ) -> np.ndarray:
        """
        Get sampled indicies from the PDF weights.

        Parameters
        ----------
        r_resamp : np.ndarray
            PDF weights.
        size : int
            Number of jets to sample
        replacement : bool, optional
            Decide, if replacement is used, by default True.

        Returns
        -------
        sampled_indices : np.ndarray
            Indicies of the resampled jets which are to use.
        """

        sampled_indices = np.random.default_rng().choice(
            len(r_resamp), p=r_resamp, size=size, replace=replacement
        )

        return sampled_indices

    def resample_iterator(
        self,
        sample_category: str,
        sample_id: int,
        save_name: str,
        sample_name: str,
        chunk_size: int = 1e6,
    ) -> None:
        """
        Resample with the data not completely stored in memory.
        Will load the jets in chunks, computing first the sum of PDF
        weights and then sampling with replacement based on the normalised
        weights.

        Parameters
        ----------
        sample_category : str
            Sample category to resample.
        sample_id : int
            Index of the sample which to be resampled.
        save_name : str
            Filepath + Filename and ending of the file where to save the
            resampled jets to.
        sample_name : str
            Name of the sample to use.
        chunk_size : int, optional
            Chunk size which is loaded per step, by default 1e6.
        """

        # Get the preparation options (cuts etc.) for the given sample
        _, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Create a store_key for the sample
        store_key = sample_category + "_" + preparation_sample.category

        # Get filepath where the info of the target data are saved
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Load the target info from file
        with open(load_name, "r") as load_file:
            target_data = json.load(load_file)

        # Load number to sample
        number_to_sample = int(target_data["number_to_sample"][sample_name])

        # First pass over data to get sum of weights
        # Assuming chunk size is large enough to avoid the loop
        # (all chunk have the same weight).
        # load_chunk = True
        # generator = self.Load_Samples_Generator(
        #     sample_category=sample_category,
        #     sample_id=sample_id,
        #     chunk_size=chunk_size,
        # )
        # sum_of_weights = 0
        # while load_chunk:
        #     try:
        #         _, target_dist, load_more, _, _ = next(generator)
        #     except StopIteration:
        #         break
        #     load_chunk = load_more
        #     weights = self.Return_unnormalised_PDF_weights(
        #         target_dist["sample_vector"][:, 0],
        #         target_dist["sample_vector"][:, 1],
        #         store_key=store_key,
        #     )
        #     sum_of_weights += np.sum(weights)

        # second pass over data, normalising weights and sampling

        # Init chunk load bool
        load_chunk = True

        # Get the generator which loads the samples
        generator = self.load_samples_generator(
            sample_category=sample_category,
            sample_id=sample_id,
            chunk_size=chunk_size,
        )

        # Set bool for creation of file and init number of sampled jets
        create_file = True
        sampled_jets = 0

        # Init a progress bar
        pbar = tqdm(total=number_to_sample)

        # Loop over the chunks until no chunks can be loaded anymore
        while load_chunk:

            # Try to get chunk from generator
            try:
                _, target_dist, load_more, total_number, start_ind = next(generator)

            # Break if no chunk can be loaded anymore
            except StopIteration:
                break

            # Check if another chunk can be loaded or if this was the last
            load_chunk = load_more

            # Get the unnormalised PDF weights of the chunk
            weights = self.return_unnormalised_pdf_weights(
                target_dist["sample_vector"][:, 0],
                target_dist["sample_vector"][:, 1],
                store_key=store_key,
            )

            # Statistical weight of the chunk
            chunk_weights = len(weights) / total_number

            # Sample a fraction of jets proportional to the chunk weight
            to_sample = number_to_sample * chunk_weights

            # If this is the last chunk, get the number of jets that still needs to be
            # sampled
            if not load_chunk:
                # last chunk
                to_sample = number_to_sample - sampled_jets

            # Calculate normalised weights
            weights = weights / np.sum(weights)

            # Get the selected indicies based on the normalised weights
            selected_ind = self.resample_chunk(weights, size=round(to_sample))

            # Sort the selected indicies
            selected_indices = np.sort(selected_ind).astype(int)

            # Need to turn chunk indices to full list indices
            selected_indices += start_ind
            sampled_jets += len(selected_indices)

            # Update progress bar
            pbar.update(selected_indices.size)

            # If this is the first chunk, create a new file
            if create_file:

                # Set the creation to false
                create_file = False
                with h5py.File(save_name, "w") as f_h5:

                    # Create new dataset with the indicies inside
                    f_h5.create_dataset(
                        "jets",
                        data=selected_indices,
                        compression=self.config.general.compression,
                        chunks=True,
                        maxshape=(None,),
                    )

            # If this is not the first chunk, extend existing file
            else:

                with h5py.File(save_name, "a") as f_h5:
                    # Append indicies to existing ones
                    f_h5["jets"].resize(
                        (f_h5["jets"].shape[0] + selected_indices.shape[0]),
                        axis=0,
                    )
                    f_h5["jets"][-selected_indices.shape[0] :] = selected_indices

        # Close progress bar
        pbar.close()

    def save_partial_iterator(
        self,
        sample_category: str,
        sample_id: int,
        selected_indices: np.ndarray,
        chunk_size: int = 1e6,
    ) -> None:
        """
        Save the selected data to an output file with an iterative approach
        (generator) for writing only, writing in chunk of size chunk_size.
        The file is read in one go.

        Parameters
        ----------
        sample_category : str
            Sample category to save
        sample_id : int
            Sample index which is to be saved
        selected_indices : np.ndarray
            Array with the selected indicies
        chunk_size : int, optional
            Chunk size which is loaded per step, by default 1e6.
        """

        # Get the preparation options of the given sample
        _, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Get the filepath of the input file
        in_file = self.config.preparation.get_sample(preparation_sample).output_name

        # Get the number of indicies
        sample_lengths = len(selected_indices)

        # Return max sample length
        max_sample = np.amax(sample_lengths)

        # Calculate number of chunks based on max_sample and chunksize
        n_chunks = round(max_sample / chunk_size + 0.5)

        # Get rounded chunk sizes
        chunk_sizes = sample_lengths / n_chunks

        # Get the sampling generator
        generators = sampling_generator(
            file=in_file,
            indices=selected_indices,
            chunk_size=chunk_sizes,
            label=self.class_labels_map[preparation_sample.category],
            label_classes=list(range(len(self.class_labels_map))),
            save_tracks=self.save_tracks,
            tracks_names=self.tracks_names,
            seed=42,
            duplicate=True,
        )

        # Init a chunk counter
        chunk_counter = 0

        # Get the path where to save the jets
        save_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            self.samples_to_resample[sample_category][sample_id] + "_selected.h5",
        )

        logger.info("Writing to file %s.", save_name)

        # Init a new progress bar
        pbar = tqdm(total=np.sum(sample_lengths))

        # Iterate over the chunks
        while chunk_counter < n_chunks + 1:

            # Get jets, tracks and labels of the chunk
            try:
                if self.save_tracks:
                    jets, tracks, labels = next(generators)

                else:
                    jets, labels = next(generators)

            except StopIteration:
                break

            # Update progress bar
            pbar.update(jets.size)

            # Init a index list
            rng_index = np.arange(len(jets))

            # Shuffle the index list
            rng = np.random.default_rng(seed=self.rnd_seed)
            rng.shuffle(rng_index)

            # Shuffle the jets, labels (and tracks)
            jets = jets[rng_index]
            labels = labels[rng_index]
            if self.save_tracks:
                tracks = [trk[rng_index] for trk in tracks]

            # Create a new file if this is the first chunk
            if chunk_counter == 0:

                # Write to file by creating dataset
                with h5py.File(save_name, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression=self.config.general.compression,
                        chunks=True,
                        maxshape=(None,),
                    )
                    out_file.create_dataset(
                        "labels",
                        data=labels,
                        compression=self.config.general.compression,
                        chunks=True,
                        maxshape=(None, labels.shape[1]),
                    )
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file.create_dataset(
                                tracks_name,
                                data=tracks[i],
                                compression=self.config.general.compression,
                                chunks=True,
                                maxshape=(None, tracks[i].shape[1]),
                            )

            # If not the first chunk, extend datasets
            else:
                # appending to existing dataset
                with h5py.File(save_name, "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    out_file["labels"].resize(
                        (out_file["labels"].shape[0] + labels.shape[0]),
                        axis=0,
                    )
                    out_file["labels"][-labels.shape[0] :] = labels
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file[tracks_name].resize(
                                (out_file[tracks_name].shape[0] + tracks[i].shape[0]),
                                axis=0,
                            )
                            out_file[tracks_name][-tracks[i].shape[0] :] = tracks[i]

            # Add up chunk counter
            chunk_counter += 1

        # Close progress bar
        pbar.close()

    def save_complete_iterator(
        self,
        sample_category: str,
        sample_id: int,
        chunk_size: int = 1e5,
    ) -> None:
        """
        Save the selected data to an output file with an iterative approach
        (generator) for both writing and reading, in chunk of size chunk_size.

        Parameters
        ----------
        sample_category : str
            Sample category to save
        sample_id : int
            Sample index which is to be saved
        chunk_size : int, optional
            Chunk size which is loaded per step, by default 1e5.
        """

        # Get the sample name of the config
        sample_name = self.samples_to_resample[sample_category][sample_id]

        # Get the preparation options of the sample
        _, preparation_sample = self.sample_file_map[sample_category][sample_id]

        # Get the filepath of the input sample
        in_file = self.config.preparation.get_sample(
            preparation_sample.name
        ).output_name

        # Get filepath where the info of the target data are saved
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Load the target info from file
        with open(load_name, "r") as load_file:
            target_data = json.load(load_file)

        # Load number to sample
        number_to_sample = target_data["number_to_sample"][sample_name]

        # Get the path of the index file
        index_file = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            self.samples_to_resample[sample_category][sample_id] + "_indices.h5",
        )

        # Get the generator which loads the indicies from file
        index_generator = self.load_index_generator(
            in_file=index_file,
            chunk_size=chunk_size,
        )

        # Get the labels
        label = self.class_labels_map[preparation_sample.category]

        # Set duplicate to True for resampling
        duplicate = True

        # Get path where to save the resampled jets
        save_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            self.samples_to_resample[sample_category][sample_id] + "_selected.h5",
        )

        logger.info("Writing to file %s.", save_name)

        # Create bool for loop and file creation
        load_chunk = True
        create_file = True

        # Init new progress bar
        pbar = tqdm(total=number_to_sample)

        # Iterate over chunks
        while load_chunk:

            # Load the chunk with the generator
            try:
                indices, load_more = next(index_generator)
            except StopIteration:
                break

            # Check if another chunk can be loaded
            load_chunk = load_more

            # save labels as int labels 0, 1, ..., nclasses-1
            labels = (np.ones(len(indices)) * label).astype(int)

            # Open the input file and read the jets and tracks
            # in a fancy way which allows double index loading
            with h5py.File(in_file, "r") as file_df:
                jets, tracks = read_dataframe_repetition(
                    file_df=file_df,
                    loading_indices=indices,
                    duplicate=duplicate,
                    save_tracks=self.save_tracks,
                    tracks_names=self.tracks_names,
                )

            # Update the progress bar
            pbar.update(jets.size)

            # Init a index list
            rng_index = np.arange(len(jets))

            # Shuffle the index list
            rng = np.random.default_rng(seed=self.rnd_seed)
            rng.shuffle(rng_index)

            # Shuffle the jets, labels (and tracks)
            jets = jets[rng_index]
            labels = labels[rng_index]
            if self.save_tracks:
                tracks = [trk[rng_index] for trk in tracks]

            # Check if this is the first chunk
            if create_file:

                # Set file creation to false
                create_file = False

                # write to file by creating dataset
                with h5py.File(save_name, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression=self.config.general.compression,
                        chunks=True,
                        maxshape=(None,),
                    )
                    out_file.create_dataset(
                        "labels",
                        data=labels,
                        compression=self.config.general.compression,
                        chunks=True,
                        maxshape=(None,),
                    )
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file.create_dataset(
                                tracks_name,
                                data=tracks[i],
                                compression=self.config.general.compression,
                                chunks=True,
                                maxshape=(None, tracks[i].shape[1]),
                            )

            # If this is not the first chunk, open file and extend the datasets
            else:
                # appending to existing dataset
                with h5py.File(save_name, "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    out_file["labels"].resize(
                        (out_file["labels"].shape[0] + labels.shape[0]),
                        axis=0,
                    )
                    out_file["labels"][-labels.shape[0] :] = labels
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file[tracks_name].resize(
                                (out_file[tracks_name].shape[0] + tracks[i].shape[0]),
                                axis=0,
                            )
                            out_file[tracks_name][-tracks[i].shape[0] :] = tracks[i]

        # Close the progress bar
        pbar.close()

    def generate_target_pdf(self, iterator: bool = True) -> None:
        """
        This method creates the target distribution (seperated) and store the associated
        histogram in memory (use for sampling) as well as the target numbers.
        Save to memory the target histogram, binning info, and target numbers.

        Parameters
        ----------
        iterator : bool, optional
            Whether to use the iterator approach or load the whole sample in memory,
            by default True.
        """

        logger.info("Generating target PDF.")
        self.target_fractions = []
        available_numbers = []
        self.target_histo = {}

        # Iterate over the samples
        for cat_ind, sample_category in enumerate(self.samples_to_resample):
            logger.info("Loading target in category %s.", sample_category)

            # Get the histogram from selected file
            reading_dict = self.file_to_histogram(
                sample_category=sample_category,
                category_ind=cat_ind,
                sample_id=0,
                iterator=iterator,
            )

            # Create target histogram dict for the category
            self.target_histo[sample_category] = {
                "hist": reading_dict["hist"],
                "xbins": reading_dict["xbins"],
                "ybins": reading_dict["ybins"],
            }

            # Add the available number and the fraction of jets
            available_numbers.append(reading_dict["available_numbers"])
            self.target_fractions.append(self.options.fractions[sample_category])

        # Correct target numbers
        n_jets_asked = (
            self.options.n_jets
            if not self.use_validation_samples
            else self.options.n_jets_validation
        )
        target_numbers_corr = correct_fractions(
            n_jets=available_numbers,
            target_fractions=self.target_fractions,
            verbose=False,
        )

        # Check for how many jets are requested
        if n_jets_asked == -1:
            logger.info("Maximising number of jets to target distribution.")

        else:
            logger.info("Requesting %s in total from target.", n_jets_asked)

            # Correct the fractions
            total_corr = sum(target_numbers_corr)
            if total_corr < n_jets_asked:
                logger.info("Requesting more jets from target than available.")
                logger.info("PDF sampling will thus upsample target too.")
                ratio = float(n_jets_asked / total_corr)
                target_numbers_corr = [int(num * ratio) for num in target_numbers_corr]
            elif total_corr > n_jets_asked:
                ratio = float(n_jets_asked / total_corr)
                target_numbers_corr = [int(num * ratio) for num in target_numbers_corr]

        # Add the target number
        self.target_number = {
            list(self.samples_to_resample.keys())[ind]: target
            for ind, target in enumerate(target_numbers_corr)
        }

        # Iterate over the samples and log the info
        for cat_ind, sample_category in enumerate(self.samples_to_resample):
            logger.info(
                "target - category %s: selected %i/%i jets, giving the requested "
                "fraction of %s",
                sample_category,
                self.target_number[sample_category],
                available_numbers[cat_ind],
                self.target_fractions[cat_ind],
            )

        # Save the info to a json file
        logger.info("Saving target histogram and numbers to sample")
        save_data = {
            "target_histo": self.target_histo,
            "target_number": self.target_number,
            "target_fraction": self.target_fractions,
        }

        # Get path of file where to save the target info
        save_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Ensure the output path exists
        os.makedirs(os.path.join(self.resampled_path, "PDF_sampling"), exist_ok=True)

        # Dump the target info to json
        with open(save_name, "w") as write_file:
            json.dump(save_data, write_file, cls=JsonNumpyEncoder)

    def generate_number_sample(self, sample_id: int) -> None:
        """
        For a given sample, sets the target numbers, respecting flavour ratio and
        upsampling max ratio (if given).

        Parameters
        ----------
        sample_id : int
            Position of the flavour in the sample list.
        """

        # Get the path of the target data file
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Load target data
        with open(load_name, "r") as load_file:
            target_data = json.load(load_file)

        # Init list for the flavour names
        flavour_names = []

        # Iterate over the samples
        for cat_ind, sample_category in enumerate(self.samples_to_resample):

            # Get the flavour name and append it to list
            flavour_name = self.sample_file_map[sample_category][sample_id][0]
            flavour_names.append(flavour_name)

        # If the flavour name is in the max upsampling section
        if any(flavour_name in self.max_upsampling for flavour_name in flavour_names):

            # Init list for asked number of jets
            asked_num = []

            # Iterate over samples
            for cat_ind, flavour_name in enumerate(flavour_names):

                # Get the number of asked jets
                num_asked = target_data["target_number"][
                    list(self.samples_to_resample)[cat_ind]
                ]

                # Check if the flavour is in max upsampling
                if flavour_name in self.max_upsampling:
                    upsampling, num_av = self.max_upsampling[flavour_name]
                    upsampling_asked = float(num_asked) / num_av

                    # Check if the max upsampling ratio is reached
                    if upsampling < upsampling_asked:
                        num_corrected = int(num_av * upsampling)
                        asked_num.append(num_corrected)
                        logger.warning(
                            "Upsampling ratio demanded to %s is %s, over limit of %s.",
                            flavour_name,
                            upsampling_asked,
                            upsampling,
                        )
                        logger.warning(
                            "Number of %s demanded will therefore be limited.",
                            flavour_name,
                        )
                    else:
                        logger.info(
                            "Upsampling ratio demanded to %s is %s, below limit of %s.",
                            flavour_name,
                            upsampling_asked,
                            upsampling,
                        )
                        asked_num.append(num_asked)
                else:
                    logger.info(
                        "No maximum upsampling ratio demanded to %s.", flavour_name
                    )
                    asked_num.append(num_asked)

            # Correct the number of asked jets
            asked_num_corr = correct_fractions(
                n_jets=asked_num,
                target_fractions=target_data["target_fraction"],
                verbose=False,
            )
            for flavour_name, num in zip(flavour_names, asked_num_corr):
                self.number_to_sample[flavour_name] = int(num)
                logger.info("For %s, demanding %i jets.", flavour_name, num)
        else:
            for sample_category, flavour_name in zip(
                self.samples_to_resample, flavour_names
            ):
                self.number_to_sample[flavour_name] = target_data["target_number"][
                    sample_category
                ]
                logger.info(
                    "For %s, demanding %s jets.",
                    flavour_name,
                    target_data["target_number"][sample_category],
                )
        if any(
            flavour_name in self.sampling_fraction for flavour_name in flavour_names
        ):
            self.number_to_sample[flavour_name] = int(
                self.number_to_sample[flavour_name]
                * self.sampling_fraction[flavour_name]
            )

        # Save the number to sample to json
        target_data["number_to_sample"] = self.number_to_sample
        with open(load_name, "w") as load_file:
            json.dump(target_data, load_file, cls=JsonNumpyEncoder)

    def generate_flavour_pdf(
        self,
        sample_category: str,
        category_id: int,
        sample_id: int,
        iterator: bool = True,
    ) -> dict:
        """
        This method:
            - create the flavour distribution (also seperated),
            - produce the PDF between the flavour and the target.

        Parameters
        ----------
        sample_category : str
            The name of the category study.
        category_id : int
            The location of the category in the list.
        sample_id : int
            The location of the sample flavour in the category dict.
        iterator : bool, optional
            Whether to use the iterator approach or load the whole
            sample in memory, by default True.

        Returns
        -------
        reading_dict : dict
            Add a dictionary object to the class pointing to the
            interpolation functions (also saves them). Returns
            None or dataframe of flavour (if iterator or not)
            and the histogram of the flavour.
        """

        # Get filepath of the target data
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Load the target data
        with open(load_name, "r") as load_file:
            target_data = json.load(load_file)

        # Create the histogram dict with the target info
        reading_dict = self.file_to_histogram(
            sample_category=sample_category,
            category_ind=category_id,
            sample_id=sample_id,
            iterator=iterator,
        )
        logger.info(
            "Computing PDF in %s for the %s.", sample_category, reading_dict["category"]
        )

        # Calculate the PDF
        self.calculate_pdf(
            store_key=f"{sample_category}_{reading_dict['category']}",
            target_hist=np.asarray(
                target_data["target_histo"][sample_category]["hist"]
            ),
            original_hist=reading_dict["hist"],
            target_bins=(
                np.asarray(target_data["target_histo"][sample_category]["xbins"]),
                np.asarray(target_data["target_histo"][sample_category]["ybins"]),
            ),
            bins=self.limit["bins"][category_id],
            limits=self.limit["ranges"][category_id],
        )

        if iterator:
            return None

        return reading_dict["target_dist"]

    def sample_flavour(
        self,
        sample_category: str,
        sample_id: int,
        iterator: bool = True,
        flavour_distribution: np.ndarray = None,
    ) -> np.ndarray:
        """
        This method:
            - samples the required amount based on PDF and fractions
            - storing the indices selected to memory.

        Parameters
        ----------
        sample_category : str
            The name of the category study.
        sample_id : int
            The location of the sample flavour in the category dict.
        iterator : bool, optional
            Whether to use the iterator approach or load the whole sample
            in memory, by default True.
        flavour_distribution : np.ndarray, optional
            None or numpy array, the loaded data (for the flavour).
            If it is None, an iterator method is used, by default None.

        Returns
        -------
        selected_indices : np.ndarray
            Returns (and stores to memory, if iterator is false) the selected indices
            for the flavour studied. If iterator is True, a None will be returned.
        """

        # Get the sample name from config
        sample_name = self.samples_to_resample[sample_category][sample_id]

        # Get filepath where to save the indicies
        save_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            self.samples_to_resample[sample_category][sample_id] + "_indices.h5",
        )

        # Get filepath of the target data
        load_name = os.path.join(
            self.resampled_path,
            "PDF_sampling",
            "target_data_training.json"
            if not self.use_validation_samples
            else "target_data_validation.json",
        )

        # Load number to sample
        with open(load_name, "r") as load_file:
            target_data = json.load(load_file)
        number_to_sample = target_data["number_to_sample"][sample_name]

        logger.info("Selecting indices for %s.", sample_name)
        logger.info("Saving indices to: %s", save_name)

        # Use the resample iterator if iterator is True
        if flavour_distribution is None or iterator:
            logger.info("Using iterating approach.")
            self.resample_iterator(
                sample_category=sample_category,
                sample_id=sample_id,
                save_name=save_name,
                sample_name=sample_name,
            )
            return None

        logger.info("Using in-memory approach.")

        # Resample in Memory
        selected_ind = self.in_memory_resample(
            x_values=flavour_distribution["sample_vector"][:, 0],
            y_values=flavour_distribution["sample_vector"][:, 1],
            size=number_to_sample,
            store_key=sample_category + "_" + flavour_distribution["category"],
        )

        # Open file where the indicies will be saved
        with h5py.File(save_name, "w") as f_h5:

            # Sort the indicies
            selected_indices = np.sort(selected_ind).astype(int)

            # Create new dataset with the indicies
            f_h5.create_dataset(
                "jets",
                data=selected_indices,
                compression=self.config.general.compression,
            )

        # Return the selected indicies
        return selected_indices

    def save_flavour(
        self,
        sample_category: str,
        sample_id: int,
        selected_indices: dict = None,
        chunk_size: int = 1e5,
        iterator: bool = True,
    ):
        """
        This method stores the selected date to memory (based on given indices).

        Parameters
        ----------
        sample_category : str
            The name of the category study.
        sample_id : int
            The location of the category in the list.
        selected_indices : dict, optional
            The location of the sample flavour in the category dict, by default None
        chunk_size : int, optional
            The size of the chunks (the last chunk may be at most 2 * chunk_size),
            by default 1e5
        iterator : bool, optional
            Whether to use the iterator approach or load the whole sample
            in memory, by default True
        """

        logger.info(
            "Sampling %s for %s.",
            self.samples_to_resample[sample_category][sample_id],
            sample_category,
        )

        if iterator or selected_indices is None:
            logger.info("Using complete iterating approach for saving.")
            self.save_complete_iterator(
                sample_category,
                sample_id,
                chunk_size,
            )

        else:
            logger.info(
                "Using partial iterating approach for saving (indices in memory)."
            )
            self.save_partial_iterator(
                sample_category,
                sample_id,
                selected_indices,
                chunk_size,
            )

    def combine_flavours(self, chunk_size: int = 1e6):
        """
        This method loads the stored resampled flavour samples and combines
        them iteratively into a single file.

        Parameters
        ----------
        chunk_size : int, optional
            Number of jets that are loaded in one chunk, by default 1e6
        """

        # Get the output name of the final single file
        output_name = self.config.get_file_name(
            option="resampled",
            use_val=self.use_validation_samples,
        )

        # Log infos
        logger.info("Combining all the flavours into a single file.")
        logger.info("Storing to %s", self.outfile_path)
        logger.info("Storing to %s", output_name)

        # Set the create_file to True for first loop
        create_file = True

        # TODO add contextmanager and dataclass
        # Create dicts to for the files from which the jets are loaded
        sample_dict = {}
        sample_length = {}
        sample_sum = {}
        sample_start_ind = {}
        sample_end_ind = {}

        # Init a list for the sample paths
        sample_paths = []

        # Loop over the different flavour types (like bjets, cjets etc.)
        for sample_id, _ in enumerate(
            self.samples_to_resample[list(self.sample_categories.keys())[0]]
        ):

            # Iterate over the different sample types (ttbar, zpext)
            for sample_category in self.samples_to_resample:

                # Get the name of the file where the selected jets are stored in
                # for the given combination of flavour and sample type
                load_name = os.path.join(
                    self.resampled_path,
                    "PDF_sampling",
                    self.samples_to_resample[sample_category][sample_id]
                    + "_selected.h5",
                )

                # Add the path of the file to the sample_paths list
                sample_paths.append(load_name)

                # Add the h5 file to dict to loop over it
                sample_dict[f"{sample_category}_{sample_id}"] = h5py.File(
                    load_name, "r"
                )

                # Get the total number of jets in the file
                sample_length[f"{sample_category}_{sample_id}"] = len(
                    sample_dict[f"{sample_category}_{sample_id}"]["/jets"]
                )

                # Set the start and end indicies for the file
                sample_start_ind[f"{sample_category}_{sample_id}"] = 0
                sample_end_ind[f"{sample_category}_{sample_id}"] = 0

            # Get the total number of jets for this flavour
            sample_sum[f"{sample_id}"] = np.sum(
                [
                    sample_length[f"{sample_category}_{sample_id}"]
                    for sample_category in self.samples_to_resample
                ]
            )

        # Check for variable differences in the files and get the common variables
        # available for all input files
        common_vars = self.retrieve_common_variables(sample_paths=sample_paths)

        # Set the counter for the chunks
        chunk_number = 0

        # Get the number of chunks
        sample_sum = np.sum([item for _, item in sample_length.items()])
        number_of_chunks = round(sample_sum / chunk_size + 0.5)

        # Define a progress bar for the writing
        pbar = tqdm(total=np.sum(number_of_chunks))

        # Get a random seed for shuffling at the end of the chunk
        seed_generator = np.random.default_rng(seed=self.rnd_seed)

        # Loop over all chunks until limit is reached
        while chunk_number < number_of_chunks:

            # Loop over the files
            for file_counter, (dict_key, df_in) in enumerate(sample_dict.items()):

                # Get the chunk size for this file
                chunk_size_file = int(
                    np.round((sample_length[dict_key] / sample_sum) * chunk_size)
                )

                # Get the end index of the chunk
                sample_end_ind[dict_key] = int(
                    sample_start_ind[dict_key] + chunk_size_file
                )

                # If the last chunk is processed, set the
                # end index to the total number of jets in this file
                if chunk_number == number_of_chunks - 1:
                    sample_end_ind[dict_key] = int(sample_length[dict_key])

                # Get a chunk of labels, tracks and jets
                if file_counter == 0:
                    jets = df_in["jets"].fields(common_vars["jets"])[
                        sample_start_ind[dict_key] : sample_end_ind[dict_key]
                    ]
                    labels = df_in["labels"][
                        sample_start_ind[dict_key] : sample_end_ind[dict_key]
                    ]
                    if self.save_tracks:
                        tracks = [
                            df_in[tracks_name].fields(common_vars[tracks_name])[
                                sample_start_ind[dict_key] : sample_end_ind[dict_key]
                            ]
                            for tracks_name in self.tracks_names
                        ]

                else:
                    jets = np.lib.recfunctions.stack_arrays(
                        [
                            jets,
                            df_in["jets"].fields(common_vars["jets"])[
                                sample_start_ind[dict_key] : sample_end_ind[dict_key]
                            ],
                        ]
                    )
                    labels = np.concatenate(
                        (
                            labels,
                            df_in["labels"][
                                sample_start_ind[dict_key] : sample_end_ind[dict_key]
                            ],
                        )
                    )
                    if self.save_tracks:
                        for track_counter, tracks_name in enumerate(self.tracks_names):
                            tracks[track_counter] = np.vstack(
                                (
                                    tracks[track_counter],
                                    df_in[tracks_name].fields(common_vars[tracks_name])[
                                        sample_start_ind[dict_key] : sample_end_ind[
                                            dict_key
                                        ]
                                    ],
                                )
                            )

                # Set the old end index as new start index for the next chunk
                sample_start_ind[dict_key] = sample_end_ind[dict_key]

            # Get a shuffled index array
            chunk_shuffle_seed = seed_generator.integers(low=0, high=1000, size=1)
            rng = np.random.default_rng(seed=chunk_shuffle_seed)
            idx_array = np.arange(start=0, stop=len(jets))
            rng.shuffle(idx_array)

            # Shuffle jets, labels and tracks
            jets = jets[idx_array]
            labels = labels[idx_array]

            if self.save_tracks:
                for track_counter, _ in enumerate(tracks):
                    tracks[track_counter] = tracks[track_counter][idx_array]

            # Check if this is the first chunk that is written to the final
            # output file.
            if create_file:

                # Create the output directory if not exist
                os.makedirs(output_name.rsplit("/", 1)[0], exist_ok=True)

                # Open the final output file and create the datasets
                # needed.
                with h5py.File(output_name, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
                    out_file.create_dataset(
                        "labels",
                        data=labels,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )

                    # If tracks are used, save them also
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file.create_dataset(
                                tracks_name,
                                data=tracks[i],
                                compression="gzip",
                                chunks=True,
                                maxshape=(
                                    None,
                                    tracks[i].shape[1],
                                ),
                            )

                # Set create file to False because it is created now
                create_file = False

            else:

                # Open the already existing output file and datasets
                # and append the chunk to it
                with h5py.File(output_name, "a") as out_file:
                    # Save jets
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets

                    # Save labels
                    out_file["labels"].resize(
                        (out_file["labels"].shape[0] + labels.shape[0]),
                        axis=0,
                    )
                    out_file["labels"][-labels.shape[0] :] = labels

                    # If tracks are used, save them also
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file[tracks_name].resize(
                                (out_file[tracks_name].shape[0] + tracks[i].shape[0]),
                                axis=0,
                            )
                            out_file[tracks_name][-tracks[i].shape[0] :] = tracks[i]

            # Update the progress bar
            pbar.update(1)

            # Set the chunk counter up by one
            chunk_number += 1

        # Close the progress bar
        pbar.close()

        # Close all h5 files
        for _, item in sample_dict.items():
            item.close()

    def Run(self):
        """Run function for PDF sampling class."""
        # Get the samples before resampling for plotting

        if self.do_plotting:
            logger.info("Start plotting resampling variables before sampling.")
            # Get the samples from the files and concatenate them for plotting
            self.initialise_samples(n_jets=int(1e6))
            self.concatenate_samples()

            # Make the resampling plots for the resampling variables before resampling
            plot_resampling_variables(
                concat_samples=self.concat_samples,
                var_positions=[0, 1],
                variable_names=[self.var_x, self.var_y],
                sample_categories=self.config.preparation.sample_categories,
                output_dir=os.path.join(
                    self.resampled_path,
                    "plots/resampling/",
                ),
                bins_dict={
                    self.var_x: 200,
                    self.var_y: 20,
                },
                atlas_second_tag=self.config.general.plot_sample_label,
                logy=True,
                ylabel="Normalised number of jets",
            )

        logger.info("Starting PDFsampling...")

        # Get the samples for pdf sampling
        self.initialise_flavour_samples()

        # Whether to use iterator approach or in-memory (one file at a time).
        iterator = True

        # Retrieve the PDF between target and all distribution
        if self.do_target:
            self.generate_target_pdf(iterator=iterator)

        else:
            logger.warning("Skipping target computation (not in list to execute).")

        for sample_id, sample in enumerate(
            self.samples_to_resample[list(self.sample_categories.keys())[0]]
        ):
            # Before starting, get the number to sample
            self.generate_number_sample(sample_id)
            for cat_ind, sample_category in enumerate(self.samples_to_resample):
                if sample_id not in self.do_flavours:
                    logger.warning(
                        "Skipping %s - %s (not in list to execute).",
                        sample_category,
                        sample,
                    )
                    continue

                # First step: generate the PDF of the Flavour to Target.
                flavour_dist = self.generate_flavour_pdf(
                    sample_category=sample_category,
                    category_id=cat_ind,
                    sample_id=sample_id,
                    iterator=iterator,
                )

                # Second step: use the flavour pdf to select indices (target included)
                selected_indices = self.sample_flavour(
                    sample_category=sample_category,
                    sample_id=sample_id,
                    flavour_distribution=flavour_dist,
                    iterator=iterator,
                )

                # Third step: use the selected indices to save the thus sampled file
                self.save_flavour(
                    sample_category=sample_category,
                    sample_id=sample_id,
                    selected_indices=selected_indices,
                    iterator=iterator,
                    chunk_size=1e4,
                )

        # Now that everything is saved, load each file and concatenate them into a
        # single large file
        if self.do_combination:
            self.combine_flavours()

            # Plot the variables from the output file of the resampling process
            if self.options.n_jets_to_plot:
                logger.info("Plotting resampled distributions...")
                preprocessing_plots(
                    sample=self.config.get_file_name(option="resampled"),
                    var_dict=get_variable_dict(self.config.general.var_file),
                    class_labels=self.config.sampling.class_labels,
                    plots_dir=os.path.join(
                        self.resampled_path,
                        "plots/resampling/",
                    ),
                    track_collection_list=self.options.tracks_names
                    if self.options.save_tracks is True
                    else None,
                    n_jets=self.options.n_jets_to_plot,
                    atlas_second_tag=self.config.general.plot_sample_label,
                    logy=True,
                    ylabel="Normalised number of jets",
                )

        else:
            logger.warning("Skipping combining step (not in list to execute).")

        logger.info("PDFsampling finished.")
