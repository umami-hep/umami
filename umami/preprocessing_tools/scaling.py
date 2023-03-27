"""Scaling module to perform variable scaling and shifting."""
# pylint: disable=no-self-use
import json
import os

import h5py
import numpy as np
import pandas as pd

from umami.configuration import logger
from umami.preprocessing_tools.utils import get_variable_dict


def as_full(data_type: np.dtype):
    """
    Convert float type to full precision

    Parameters
    ----------
    data_type: np.dtype
        type to check for float

    Returns
    -------
    np.dtype
        Return an element of a dtype as a full precision float if we
        stored half.

    """
    data_type = np.dtype(data_type)
    if data_type.kind == "f" and data_type.itemsize == 2:
        return np.dtype("f4")
    return data_type


def get_track_mask(tracks: np.ndarray) -> np.ndarray:
    """Return the mask for the tracks

    Parameters
    ----------
    tracks : np.ndarray
        Loaded tracks with shape (n_jets, nTrks, nTrkFeatures). Note, the
        input tracks should not already be converted with np.nan_to_num, as this
        function relies on a np.isnan check in the case where the valid flag is
        not present.

    Returns
    -------
    np.ndarray
        A bool array (n_jets, nTrks), True for tracks that are present.

    Raises
    ------
    ValueError
        If no 'valid' flag or at least one float variable in your input tracks.
    """

    # try to use the valid flag, present in newer samples
    if "valid" in tracks.dtype.names:
        return tracks["valid"]

    # instead look for a float variable to use, which will be NaN
    # for the padded tracks
    for var, dtype in tracks.dtype.fields.items():
        if "f" in dtype[0].str:
            return ~np.isnan(tracks[var])

    raise ValueError(
        "Need 'valid' flag or at least one float variable in your input tracks."
    )


def apply_scaling_jets(
    jets: pd.DataFrame,
    variables_list: dict,
    scale_dict: dict,
) -> pd.DataFrame:
    """
    Apply the jet scaling and shifting for the given jets.

    Parameters
    ----------
    jets : pd.DataFrame
        Loaded jets which are to be scaled/shifted.
    variables_list : dict
        Train variables which will be scaled/shifted. For all
        variables, the scaling/shifting values must be in the
        scaling dict.
    scale_dict : dict
        Loaded scaling dict with the scaling/shifting values
        for the variables defined in variables_list.

    Returns
    -------
    pd.DataFrame
        Scaled/Shifted jets with the variables defined in variables_list

    Raises
    ------
    ValueError
        When jets is neither a pandas DataFrame nor a structured numpy ndarray
    KeyError
        When for the variable which is to be scaled no shift/scale values
        are available in the scale dict.
    ValueError
        If the scale parameter for the variable is either 0 or inf.
    ValueError
        If the scaled/shifted variable has infs or NaNs.
    """

    # Check type of input
    if isinstance(jets, pd.DataFrame):
        is_dataframe = True

    elif isinstance(jets, np.ndarray):
        is_dataframe = False

    else:
        raise ValueError(
            "Only pandas DataFrame and structured numpy ndarrays are"
            " valid input datatypes for jets!"
        )

    # Generate the defaults dict out of the scale dict
    default_dict = {
        k: v["default"] for k, v in scale_dict.items() if "isDefaults" not in k
    }

    if is_dataframe:
        # Get rid of all not-wanted variables
        jets = jets[variables_list]

        # Replace infinity values with nan values
        jets = jets.replace([np.inf, -np.inf], np.nan)

        # Fill the nan values with the default values from the default dict
        jets = jets.fillna(default_dict)

    # Loop over the variables requested
    for var in variables_list:

        # Skipping default variables
        if "isDefaults" in var or "weight" in var:
            continue

        # Check if scaling/shifting values are availabe for this variable
        if var not in scale_dict.keys():
            raise KeyError(
                f"Requested {var} to be used but no values for this "
                "variable is available in the scale dict!"
            )

        if not is_dataframe:
            # Replace inf and nans
            jets[var] = np.nan_to_num(
                x=jets[var],
                nan=default_dict[var],
                posinf=default_dict[var],
                neginf=default_dict[var],
            )

        # Ensuring correct scaling factors
        if scale_dict[var]["scale"] == 0 or np.isinf(scale_dict[var]["scale"]):
            raise ValueError(
                f"Scale parameter for jet var {var} is {scale_dict[var]['scale']}."
            )

        # Shift and scale the variables
        jets[var] = (jets[var] - scale_dict[var]["shift"]) / scale_dict[var]["scale"]

        # Check for NaNs or Infs and raise an error if so
        if np.isnan(jets[var]).any() or np.isinf(jets[var]).any():
            raise ValueError(
                "Inf or NaN value(s) encountered when applying the scaling/shifting "
                f"for the jets! The probablematic variable is {var}. "
                f'Scale value: {scale_dict[var]["scale"]}, '
                f'Shift value: {scale_dict[var]["shift"]}'
            )

    return jets


def apply_scaling_trks(
    trks: np.ndarray,
    variable_config: dict,
    scale_dict: dict,
    tracks_name: str,
    save_track_labels: bool = False,
    track_label_variables: list = None,
):
    """
    Apply the scaling/shifting to the tracks.

    Parameters
    ----------
    trks : np.ndarray
        Loaded tracks as numpy array.
    variable_config : dict
        Loaded variable config.
    scale_dict : dict
        Loaded scale dict.
    tracks_name : str
        Name of the tracks.
    save_track_labels : bool
        Save the track labels
    track_label_variables : list
        List of the track label variables which are
        to be saved.

    Returns
    -------
    scaled_trks : np.ndarray
        The tracks scaled and shifted.
    valid : np.ndarray
        Bool array specifying which tracks are valid vs padding.
    trk_labels : np.ndarray
        The track labels, if defined in the variable config.

    Raises
    ------
    ValueError
        If a value of a variable which is to be used in log form
        is zero/negative.
    ValueError
        If scale is found to be 0 or inf for any track variable.
    ValueError
        If the scaled/shifted variable has infs or NaNs.
    """

    # Init a list for the variables
    var_arr_list = []

    # Get track mask
    track_mask = get_track_mask(trks)

    # Load the variables which are scaled/shifted
    trk_vars = []
    trk_vars_lists_dict = {}

    for var_type in ["noNormVars", "logNormVars", "jointNormVars"]:
        if (
            var_type in variable_config["track_train_variables"][tracks_name]
            and variable_config["track_train_variables"][tracks_name][var_type]
            is not None
        ):
            trk_vars_lists_dict[var_type] = variable_config["track_train_variables"][
                tracks_name
            ][var_type]

        else:
            logger.warning("No %s in variable dict for %s!", var_type, tracks_name)
            trk_vars_lists_dict[var_type] = []

        # Combine all variables into one list
    for _, item in trk_vars_lists_dict.items():
        trk_vars += item

    # Iterate over variables and scale/shift it
    for var in trk_vars:
        trk_array = trks[var]

        # Get the scaling/shifting values for this variable in full precision
        if var in scale_dict:
            shift = np.float32(scale_dict[var].get("shift"))
            scale = np.float32(scale_dict[var].get("scale"))

        else:
            shift, scale = None, None

        # Check if the variable needs to be in log
        if var in trk_vars_lists_dict["logNormVars"]:

            # Check for negative values in the log vars
            if (trk_array < 0).any():
                raise ValueError(
                    f"Negative values encountered in {var}! "
                    "This variable is supposed to "
                    "be used in log and is not allowed to be zero! Please check!"
                )

            # Check for 0 values in the variables that are to
            # be used in log
            if (trk_array == 0).any():
                raise ValueError(
                    f"Zeros encountered in {var}! "
                    "This variable is supposed to "
                    "be used in log and is not allowed to be zero! Please check "
                    "the precision that was used for this variable!"
                )

            trk_array = np.log(trk_array)

        # Check if the variable is to be scaled/shifted
        if (
            var in trk_vars_lists_dict["jointNormVars"]
            or var in trk_vars_lists_dict["logNormVars"]
        ):

            # Check against 0 or inf scale value
            if scale == 0 or np.isinf(scale):
                raise ValueError(f"Scale parameter for track var {var} is {scale}.")

            # Apply scaling and shifting
            trk_array = np.where(
                track_mask,
                (trk_array - shift) / scale,
                trk_array,
            )

        # Convert NaNs from padded tracks to zeros
        trk_array = np.where(~track_mask, np.nan_to_num(trk_array), trk_array)

        # Check for NaNs or Infs and raise an error if so
        if np.isnan(trk_array).any() or np.isinf(trk_array).any():
            raise ValueError(
                "Inf value(s) encountered when applying the scaling/shifting "
                f"for the tracks! The probablematic variable is {var}. "
                f"Scale value: {scale}, Shift value: {shift}"
            )

        # Append track variable values to list for correct stacking later
        var_arr_list.append(trk_array)

    # Stack the results for new dataset
    scaled_trks = np.stack(var_arr_list, axis=-1)

    # track vertex and origin labels
    trk_labels = None
    if save_track_labels and track_label_variables:
        trk_labels = np.stack(
            [np.nan_to_num(trks[v]) for v in track_label_variables],
            axis=-1,
        )

    # Return the scaled and tracks, the valid flag and, if defined, the track labels
    return scaled_trks, track_mask, trk_labels


class CalculateScaling:
    """
    Scaling class. Can calculate the scaling and shifting for training dataset
    and can apply it.
    """

    def __init__(self, config: object) -> None:
        """
        Init the needed configs and variables

        Parameters
        ----------
        config : object
            Loaded config file for the preprocessing.

        Raises
        ------
        ValueError
            If the given track label variables are not a list
            nor a string
        ValueError
            If the defined track weight variables are not a dict
            or a single string
        ValueError
            If for a track collection, the defined track weight variable
            is not defined in the track label variables
        """

        self.scale_dict_path = config.dict_file
        self.sampling_options = config.sampling["options"]
        self.save_tracks = (
            config.sampling["options"]["save_tracks"]
            if "save_tracks" in config.sampling["options"].keys()
            else False
        )
        self.save_track_labels = (
            config.sampling["options"]["save_track_labels"]
            if "save_track_labels" in config.sampling["options"].keys()
            else False
        )

        self.tracks_names = config.sampling["options"]["tracks_names"]
        self.compression = config.compression

        logger.info("Using variable dict at %s", config.var_file)
        self.variable_config = get_variable_dict(config.var_file)

        self.track_label_variables = self.variable_config.get("track_truth_variables")
        self.track_weight_variables = self.variable_config.get("track_weight_variables")

        # Adding the full config to retrieve the correct paths
        self.config = config

        if self.save_track_labels:
            if isinstance(self.track_label_variables, str):
                self.track_label_variables = {
                    self.tracks_names[0]: [self.track_label_variables]
                }
            elif isinstance(self.track_label_variables, list):
                self.track_label_variables = {
                    self.tracks_names[0]: self.track_label_variables
                }
            elif not isinstance(self.track_label_variables, dict):
                raise ValueError(
                    """
                    Given track truth label variables are not a dict nor a list nor a
                    single string!
                    """
                )

        # Check that the track weight variable can be used
        if self.track_weight_variables:
            if isinstance(self.track_weight_variables, str):
                self.track_weight_variables = {
                    self.tracks_names[0]: [self.track_weight_variables]
                }

            elif isinstance(self.track_weight_variables, list):
                self.track_weight_variables = {
                    self.tracks_names[0]: self.track_weight_variables
                }

            elif isinstance(self.track_weight_variables, dict):
                for (
                    trk_collection,
                    trk_weight_var,
                ) in self.track_weight_variables.items():
                    if isinstance(trk_weight_var, str):
                        self.track_weight_variables[trk_collection] = [trk_weight_var]

                    elif not isinstance(trk_weight_var, list):
                        raise ValueError(
                            f"""
                            The track weight variables given for {trk_collection} are
                            not a string nor a list of strings!
                            """
                        )

            elif not isinstance(self.track_weight_variables, dict):
                raise ValueError(
                    """
                    Given track weight variables are not a dict! Please
                    define them as a dict with the track collection name
                    as the key and the track weight variable for this
                    track collection as the item!
                    """
                )

            for track_collection in self.tracks_names:
                if self.track_weight_variables[track_collection] is not None:
                    for trk_wght_var in self.track_weight_variables[track_collection]:
                        if (
                            trk_wght_var
                            not in self.track_label_variables[track_collection]
                        ):
                            raise ValueError(
                                f"""
                                Given track weight variable {trk_wght_var} for
                                track group
                                {track_collection} is not given in the track
                                label variables! Please define it also there!
                                """
                            )

    def join_scale_dicts(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        first_n: int,
        second_n: int,
    ):
        """
        Combining the scale dicts of two track chunks.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict to join.
        second_scale_dict : dict
            Second scale dict to join.
        first_n : int
            Number for the first scale dict.
        second_n : int
            Number for the second scale dict.

        Returns
        -------
        combined_scale_dict : dict
            The combined scale dict.
        combined_n : int
            The combined number of objects.
        """

        # Init a new combined scale dict
        combined_scale_dict = {}

        for var in first_scale_dict:
            # Add var to combined dict
            combined_scale_dict[var] = {}

            # Combine
            mean = first_scale_dict[var]["shift"]
            tmp_mean = second_scale_dict[var]["shift"]
            std = first_scale_dict[var]["scale"]
            tmp_std = second_scale_dict[var]["scale"]

            combined_mean = (mean * first_n + tmp_mean * second_n) / (
                first_n + second_n
            )
            combined_scale_dict[var]["shift"] = combined_mean

            combined_scale_dict[var]["scale"] = np.sqrt(
                (
                    (
                        (((mean - combined_mean) ** 2 + std**2) * first_n)
                        + (((tmp_mean - combined_mean) ** 2 + tmp_std**2) * second_n)
                    )
                )
                / (first_n + second_n)
            )

            if "default" in first_scale_dict[var]:
                combined_scale_dict[var]["default"] = first_scale_dict[var]["default"]

        # Sum of nTrks corresponding to combined scale dict
        combined_n = first_n + second_n

        return combined_scale_dict, combined_n

    def get_scaling_tracks(
        self,
        data: np.ndarray,
        var_names: list,
        track_mask: np.ndarray,
        tracks_name: str,
    ):
        """
        Calculate the scale dict for the tracks and return the dict.

        Parameters
        ----------
        data : np.ndarray
            Structured tracks array with shape (n_jets, n_trks, n_trk_features)
        var_names : list
            List of variables which are to be scaled
        track_mask : np.ndarray
            Boolen array where False denotes padded tracks,
            with shape (n_jets, n_trks)
        tracks_name : str
            Name of used tracks collection
        Returns
        -------
        scale_dict : dict
            Scale dict with scaling/shifting values for each variable
        n_trks : int
            Number of tracks used to calculate the scaling/shifting
        """

        # Get origin weights
        origin_weights = {}

        # Initalise scale dict
        scale_dict = {}

        # For each track variable
        for var in var_names:
            array = data[var][track_mask]
            n_trks = len(array)

            # Calculate normalisation parameters
            mean, scale = array.mean(), array.std()
            scale_dict[var] = {"shift": float(mean), "scale": float(scale)}

        if self.save_track_labels:

            # Get the track label variables and the track weight variable
            trk_label_variables = self.track_label_variables.get(tracks_name)
            trk_weight_variables = self.track_weight_variables.get(tracks_name)

            # If everything is defined, calculate the track weights by their number
            # of appearance
            if trk_weight_variables is not None and trk_label_variables is not None:
                for trk_weight_var in trk_weight_variables:
                    counts = np.unique(data[trk_weight_var], return_counts=True)
                    counts = dict(zip(*counts))
                    counts.pop(-1, None)
                    origin_weights[trk_weight_var] = {
                        str(lab): n for lab, n in counts.items()
                    }

        return n_trks, scale_dict, origin_weights

    def get_scaling(
        self,
        vec: np.ndarray,
        varname: str,
        custom_defaults_vars: dict,
    ):
        """
        Calculates the weighted average and std for vector vec.

        Parameters
        ----------
        vec : np.ndarray
            Array with variable values for the jets
        varname : str
            Name of the variable which is to be scaled
        custom_defaults_vars : dict
            Dict with custom default variable values

        Returns
        -------
        varname : str
            Name of the variable
        average : float
            Average of the variable
        std : float
            Std of the variable
        default : float
            Default value of the variable
        """

        # find NaN values
        nans = np.isnan(vec)

        # check if variable has predefined default value
        if custom_defaults_vars is not None and varname in custom_defaults_vars:
            default = custom_defaults_vars[varname]

        # NaN values are not considered in calculation for average
        else:
            vec_without_nan = vec[~nans]
            default = np.float64(np.ma.average(vec_without_nan))

        # replace NaN values with default values
        vec[nans] = default
        average = np.float64(np.ma.average(vec))
        std = np.float64(np.sqrt(np.average((vec - average) ** 2)))

        return varname, {"shift": average, "scale": std, "default": default}

    def get_scale_dict(
        self,
        input_file: str = None,
        chunk_size: int = 100_000,
    ):
        """
        Calculates the scaling, shifting and default values and saves them to json.

        Parameters
        ----------
        input_file : str, optional
            File which is used to calculate scaling/shifting, by default None
        chunk_size : int, optional
            Scale dict calculated using the given file, by default 1e5

        Raises
        ------
        ValueError
            If one of the scaling/shifting values is inf
        """
        # Get input filename to calculate scaling and shifting
        if input_file is None:
            input_file = self.config.get_file_name(option="resampled")

        logger.info("Calculating scaling and shifting values for the jet variables")
        logger.info("Using %s for calculation of scaling/shifting", input_file)

        # Get the file_length
        file_length = len(h5py.File(input_file, "r")["/jets"])
        logger.info("Found %i jets in file.", file_length)
        n_jets_scaling = int(
            self.config.sampling["options"].get("n_jets_scaling", file_length)
        )
        n_jets_scaling = file_length if n_jets_scaling is None else int(n_jets_scaling)
        logger.info("Using %i jets to calculate scaling and shifting.", n_jets_scaling)

        # Get the number of chunks we need to load
        n_chunks = int(np.ceil(n_jets_scaling / chunk_size))

        # Get the jets scaling generator
        jets_scaling_generator = self.get_scaling_dict_generator(
            input_file=input_file,
            n_jets=n_jets_scaling,
            chunk_size=chunk_size,
        )

        # Loop over chunks
        for chunk_counter in range(n_chunks):
            logger.info(
                "Calculating jet scales for chunk %i of %i", chunk_counter + 1, n_chunks
            )
            # Check if this is the first time loading from the generator
            if chunk_counter == 0:
                # Get the first chunk of scales from the generator
                scale_dict, n_jets_loaded = next(jets_scaling_generator)
            else:
                # Get the next chunk of scales from the generator
                tmp_scale_dict, tmp_n_jets_loaded = next(jets_scaling_generator)

                # Combine the scale dicts coming from the generator
                scale_dict, n_jets_loaded = self.join_scale_dicts(
                    first_scale_dict=scale_dict,
                    second_scale_dict=tmp_scale_dict,
                    first_n=n_jets_loaded,
                    second_n=tmp_n_jets_loaded,
                )

        logger.info("Calculating scaling and shifting values for the track variables")

        # Init a empty scale dict for the tracks
        scale_dict_trk_all = {}

        # Check if tracks are used or not
        if self.save_tracks is True:

            # Loop over all tracks selections
            for tracks_name in self.tracks_names:
                scale_dict_trk = {}
                # Load generator
                trks_scaling_generator = self.get_scaling_tracks_generator(
                    input_file=input_file,
                    n_jets=n_jets_scaling,
                    tracks_name=tracks_name,
                    chunk_size=chunk_size,
                )

                # Loop over chunks
                for chunk_counter in range(n_chunks):
                    logger.info(
                        "Calculating track scales for %s for chunk %i of %i",
                        tracks_name,
                        chunk_counter + 1,
                        n_chunks,
                    )
                    # Check if this is the first time loading from the generator
                    if chunk_counter == 0:
                        # Get the first chunk of scales from the generator
                        n_trks, scale_dict_trk, origin_weights = next(
                            trks_scaling_generator
                        )
                    else:
                        # Get the next chunk of scales from the generator
                        tmp_n_trks, tmp_scale_dict_trk, tmp_ow = next(
                            trks_scaling_generator
                        )

                        # Combine the scale dicts coming from the generator
                        (scale_dict_trk, n_trks) = self.join_scale_dicts(
                            first_scale_dict=scale_dict_trk,
                            second_scale_dict=tmp_scale_dict_trk,
                            first_n=n_trks,
                            second_n=tmp_n_trks,
                        )

                        # combine origin counts
                        for trk_weight_var in origin_weights:
                            for k in origin_weights[trk_weight_var]:
                                origin_weights[trk_weight_var][k] += tmp_ow[
                                    trk_weight_var
                                ].get(k, 0)

                # Checking for inf values in scaling/shifting values
                inf_error = False
                for iter_var, iter_dict in scale_dict_trk.items():
                    for shift_scale, shift_scale_value in iter_dict.items():
                        if np.isinf(shift_scale_value) or np.isnan(shift_scale_value):
                            logger.error(
                                "Track variable %s in track collection %s has an"
                                " inf/nan value for %s value!",
                                iter_var,
                                tracks_name,
                                shift_scale,
                            )
                            inf_error = True

                if inf_error:
                    raise ValueError(
                        "One or more track variables have scale/shift"
                        " values equal to infinity!"
                    )

                # Add a dict for the origin weights for this track collection
                scale_dict_trk_all[f"{tracks_name}_origin_weights"] = {}

                # turn origin counts into weights
                for trk_wgh_var in origin_weights:
                    origin_weights[trk_wgh_var] = {
                        k: sum(origin_weights[trk_wgh_var].values()) / weight
                        for k, weight in origin_weights[trk_wgh_var].items()
                    }

                    # Add the origin weights to the scale dict
                    scale_dict_trk_all[f"{tracks_name}_origin_weights"][
                        f"{trk_wgh_var}"
                    ] = origin_weights[trk_wgh_var]

                # Add scale dict for given tracks selection to the more general one
                scale_dict_trk_all[tracks_name] = scale_dict_trk

        # TODO: change in python 3.9
        # save scale/shift dictionary to json file
        scale_dict = {"jets": scale_dict}
        scale_dict.update(scale_dict_trk_all)
        os.makedirs(os.path.dirname(self.scale_dict_path), exist_ok=True)
        with open(self.scale_dict_path, "w") as outfile:
            json.dump(scale_dict, outfile, indent=4)
        logger.info("Saved scale dictionary as %s", self.scale_dict_path)

    def get_scaling_dict_generator(
        self,
        input_file: str,
        n_jets: int,
        chunk_size: int = 100_000,
    ):
        """
        Set up a generator that loads the jets in chunks and calculates the mean/std.

        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        n_jets : int
            Number of jets which are to be scaled.
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default 1_000_000

        Yields
        ------
        scale_dict_trk : dict
            Dict with the scale/shift values for each variable.
        n_jets : int
            Number of jets used for scaling/shifting.
        """

        # Extract the correct variables
        variables_header = self.variable_config["train_variables"]
        var_list = [i for j in variables_header for i in variables_header[j]]

        # Open the h5 file
        with h5py.File(input_file, "r") as infile_all:
            # Get the indices
            start_ind = 0
            tupled_indices = []

            # Loop over indicies
            while start_ind < n_jets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunk_size)

                # Check if end index is bigger than n_jets
                end_ind = min(end_ind, n_jets)

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            # Loop over the chunks
            for index_tuple in tupled_indices:
                # Init a list for the scale values for the variables
                scale_dict = {}

                # Load jets
                jets = pd.DataFrame(
                    infile_all["/jets"].fields(var_list)[
                        index_tuple[0] : index_tuple[1]
                    ]
                )

                # Loop over the columns and change all floats to full precision
                for iter_var in var_list:
                    if jets[iter_var].dtype.kind == "f":
                        jets[iter_var] = jets[iter_var].astype(np.float32)

                # Replace inf values
                jets.replace([np.inf, -np.inf], np.nan, inplace=True)

                if "weight" not in jets:
                    length = n_jets if n_jets < chunk_size else len(jets)
                    jets["weight"] = np.ones(int(length))

                # Iterate over the vars of the jets
                for var in jets.columns.values:
                    # Skip all label/weight variables
                    if var in [self.variable_config["label"], "weight"]:
                        continue

                    # Set Default values for isDefaults variables
                    if "isDefaults" in var:
                        logger.debug(
                            "Default scaling/shifting values (0, 1) are used for %s",
                            var,
                        )
                        scale_dict[var] = {"shift": 0.0, "scale": 1.0, "default": None}

                    # Calculate scaling/shifting value for given variable
                    else:

                        # Get the dict entry
                        var, vals = self.get_scaling(
                            vec=jets[var].values,
                            varname=var,
                            custom_defaults_vars=self.variable_config[
                                "custom_defaults_vars"
                            ]
                            if "custom_defaults_vars" in self.variable_config
                            else None,
                        )
                        scale_dict[var] = vals

                # Yield the scale dict and the number of jets
                yield scale_dict, len(jets)

    def get_scaling_tracks_generator(
        self,
        input_file: str,
        n_jets: int,
        tracks_name: str,
        chunk_size: int = 100_000,
    ):
        """
        Set up a generator that loads the tracks in chunks and calculates the mean/std.


        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        n_jets : int
            Number of jets which are to be scaled.
        tracks_name : str
            Name of the tracks
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(1_000_000)

        Yields
        ------
        scale_dict_trk : dict
            Dict with the scale/shift values for each variable.
        nTrks : int
            Number of tracks used for scaling/shifting.
        """

        # Load the variables which are scaled/shifted
        trk_vars = []
        var_lists_dict = {}

        for var_type in ["logNormVars", "jointNormVars"]:
            if (
                var_type in self.variable_config["track_train_variables"][tracks_name]
                and self.variable_config["track_train_variables"][tracks_name][var_type]
                is not None
            ):
                var_lists_dict[var_type] = self.variable_config[
                    "track_train_variables"
                ][tracks_name][var_type]

            else:
                logger.warning("No %s in variable dict for %s!", var_type, tracks_name)
                var_lists_dict[var_type] = []

        # Combine all variables into one list
        for _, item in var_lists_dict.items():
            trk_vars += item

        # Open h5 file
        with h5py.File(input_file, "r") as infile_all:

            # Get the indices
            start_ind = 0
            tupled_indices = []

            # Retrieving the dtypes of the variables to load
            to_load_dtype = [
                (n, as_full(x)) for n, x in infile_all[f"/{tracks_name}"].dtype.descr
            ]

            # Loop over indicies
            while start_ind < n_jets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunk_size)

                # Check if end index is bigger than n_jets
                end_ind = min(end_ind, n_jets)

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            # Loop over the chunks
            for idx in tupled_indices:

                # Load tracks
                trks = np.asarray(
                    infile_all[f"/{tracks_name}"][idx[0] : idx[1]],
                    dtype=to_load_dtype,
                )

                # Get the masking
                track_mask = get_track_mask(trks)

                # Take the log of the desired variables
                eps = 1e-8  # Add small value, so log(0) does not happen
                for var in var_lists_dict["logNormVars"]:
                    trks[var][track_mask] = np.log(trks[var][track_mask] + eps)

                # compute scalings
                n_trks, scale_dict, origin_weights = self.get_scaling_tracks(
                    data=trks,
                    var_names=trk_vars,
                    track_mask=track_mask,
                    tracks_name=tracks_name,
                )

                yield n_trks, scale_dict, origin_weights
