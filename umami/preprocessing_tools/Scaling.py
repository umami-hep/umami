"""Scaling module to perform variable scaling and shifting."""
# pylint: disable=no-self-use
import json
import os

import h5py
import numpy as np
import pandas as pd

from umami.configuration import logger

from .utils import GetVariableDict, preprocessing_plots


def Gen_default_dict(scale_dict: dict) -> dict:
    """
    Generates default value dictionary from scale/shift dictionary.

    Parameters
    ----------
    scale_dict : dict
        Scale dict loaded from json.

    Returns
    -------
    dict
        Returns a dict with all given variables but with
        default values. With these, NaNs can be filled.
    """

    default_dict = {}
    for elem in scale_dict:
        if "isDefaults" in elem["name"]:
            continue
        default_dict[elem["name"]] = elem["default"]
    return default_dict


def get_track_mask(tracks: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    tracks : np.ndarray
        Loaded tracks with shape (nJets, nTrks, nTrkFeatures). Note, the
        input tracks should not already be converted with np.nan_to_num, as this
        function relies on a np.isnan check in the case where the valid flag is
        not present.

    Returns
    -------
    np.ndarray
        A bool array (nJets, nTrks), True for tracks that are present.

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


def apply_scaling_trks(
    trks: np.ndarray,
    variable_config: dict,
    scale_dict: dict,
    tracks_name: str,
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

    Returns
    -------
    scaled_trks : np.ndarray
        The tracks scaled and shifted.
    trk_labels : np.ndarray
        The track labels, if defined in the variable config.

    Raises
    ------
    ValueError
        If scale is found to be 0 or inf for any track variable.
    """

    # Init a list for the variables
    var_arr_list = []

    # Get track mask
    track_mask = get_track_mask(trks)

    # Get the track variables
    tracks_noNormVars = variable_config["track_train_variables"][tracks_name][
        "noNormVars"
    ]
    tracks_logNormVars = variable_config["track_train_variables"][tracks_name][
        "logNormVars"
    ]
    tracks_jointNormVars = variable_config["track_train_variables"][tracks_name][
        "jointNormVars"
    ]
    tracks_variables = tracks_noNormVars + tracks_logNormVars + tracks_jointNormVars

    # Iterate over variables and scale/shift it
    for var in tracks_variables:
        x = trks[var]

        if var in tracks_logNormVars:
            x = np.log(x)
        if var in tracks_jointNormVars or var in tracks_logNormVars:
            shift = np.float32(scale_dict[var]["shift"])
            scale = np.float32(scale_dict[var]["scale"])
            if scale == 0 or np.isinf(scale):
                raise ValueError(f"Scale parameter for track var {var} is {scale}.")
            x = np.where(
                track_mask,
                x - shift,
                x,
            )
            x = np.where(
                track_mask,
                x / scale,
                x,
            )
        var_arr_list.append(np.nan_to_num(x))

        # track vertex and origin labels
        if f"{tracks_name}_labels" in variable_config:
            trkLabels = variable_config[f"{tracks_name}_labels"]
            trk_labels = np.stack(
                [np.nan_to_num(trks[v]) for v in trkLabels],
                axis=-1,
            )
        else:
            trk_labels = None

    # Stack the results for new dataset
    scaled_trks = np.stack(var_arr_list, axis=-1)

    # Return the scaled and tracks and, if defined, the track labels
    return scaled_trks, trk_labels


class Scaling:
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
        """

        self.config = config
        self.scale_dict_path = config.dict_file
        self.bool_use_tracks = config.sampling["options"]["save_tracks"]
        self.tracks_names = self.config.sampling["options"]["tracks_names"]
        self.compression = self.config.compression

        logger.info(f"Using variable dict at {config.var_file}")
        self.variable_config = GetVariableDict(config.var_file)

    def join_mean_scale(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        variable: str,
        first_N: int,
        second_N: int,
    ):
        """
        Combine the mean and scale of the two input scale dict.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict with the variable and
            their respective mean/std inside.
        second_scale_dict : dict
            Second scale dict with the variable and
            their respective mean/std inside.
        variable : str
            Variable which is to be combined.
        first_N : int
            Number of tracks/jets used to calculate mean/std for first dict.
        second_N : int
            Number of tracks/jets used to calculate mean/std for second dict.

        Returns
        -------
        combined_mean : float
            Combined mean/shift.
        combined_std : float
            Combined std/scale.
        """

        # Get the values in variables
        mean = first_scale_dict[variable]["shift"]
        tmp_mean = second_scale_dict[variable]["shift"]
        std = first_scale_dict[variable]["scale"]
        tmp_std = second_scale_dict[variable]["scale"]

        # Combine the means
        combined_mean = (mean * first_N + tmp_mean * second_N) / (first_N + second_N)

        # Combine the std
        combined_std = np.sqrt(
            (
                (
                    (((mean - combined_mean) ** 2 + std**2) * first_N)
                    + (((tmp_mean - combined_mean) ** 2 + tmp_std**2) * second_N)
                )
            )
            / (first_N + second_N)
        )

        return combined_mean, combined_std

    def join_scale_dicts_trks(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        first_nTrks: int,
        second_nTrks: int,
    ):
        """
        Combining the scale dicts of two track chunks.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict to join.
        second_scale_dict : dict
            Second scale dict to join.
        first_nTrks : int
            Number of tracks used for the first scale dict.
        second_nTrks : int
            Number of tracks used for the second scale dict.

        Returns
        -------
        combined_scale_dict : dict
            The combined scale dict.
        combined_nTrks : int
            The combined number of tracks.
        """

        # Init a new combined scale dict
        combined_scale_dict = {}

        for var in first_scale_dict:
            # Add var to combined dict
            combined_scale_dict[var] = {}

            # Combine the means
            (
                combined_scale_dict[var]["shift"],
                combined_scale_dict[var]["scale"],
            ) = self.join_mean_scale(
                first_scale_dict=first_scale_dict,
                second_scale_dict=second_scale_dict,
                variable=var,
                first_N=first_nTrks,
                second_N=second_nTrks,
            )

        # Sum of nTrks corresponding to combined scale dict
        combined_nTrks = first_nTrks + second_nTrks

        return combined_scale_dict, combined_nTrks

    def join_scale_dicts_jets(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        first_nJets: int,
        second_nJets: int,
    ):
        """
        Combine two scale dicts for jet variables.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict to join.
        second_scale_dict : dict
            Second scale dict to join.
        first_nJets : int
            Number of jets used for the first scale dict.
        second_nJets : int
            Number of jets used for the second scale dict.

        Returns
        -------
        combined_scaled_dict : list
            The combined scale dict list.
        combined_nJets : int
            The combined number of jets (corresponding to the combined dicts).
        """

        # Init a combined list for the dicts
        combined_dict_list = []

        # Loop over the list with the dicts from the variables
        for counter, dict_i in enumerate(first_scale_dict):
            # Ensure the same variables are merged
            if dict_i["name"] == second_scale_dict[counter]["name"]:
                # Combine the means
                combined_average, combined_std = self.join_mean_scale(
                    first_scale_dict=first_scale_dict,
                    second_scale_dict=second_scale_dict,
                    variable=counter,
                    first_N=first_nJets,
                    second_N=second_nJets,
                )

                # Combine the mean/shift in a dict and append it
                combined_dict_list.append(
                    self.dict_in(
                        varname=dict_i["name"],
                        average=combined_average,
                        std=combined_std,
                        default=dict_i["default"],
                    )
                )

        # Sum of nJets corresponding to combined scale dict
        combined_nJets = first_nJets + second_nJets

        return combined_dict_list, combined_nJets

    def get_scaling_tracks(
        self,
        data: np.ndarray,
        var_names: list,
        track_mask: np.ndarray,
    ):
        """
        Calculate the scale dict for the tracks and return the dict.

        Parameters
        ----------
        data : np.ndarray
            Loaded tracks with shape (nJets, nTrks, nTrkFeatures)
        var_names : list
            List of variables which are to be scaled
        track_mask : np.ndarray
            Boolen array where False denotes padded tracks,
            with shape (nJets, nTrks)

        Returns
        -------
        scale_dict : dict
            Scale dict with scaling/shifting values for each variable
        nTrks : int
            Number of tracks used to calculate the scaling/shifting
        """
        # TODO add weight support for tracks

        # Initalise scale dict
        scale_dict = {}

        # For each track variable
        for v, name in enumerate(var_names):
            f = data[:, :, v]

            # Get tracks
            slc = f[track_mask]
            nTrks = len(slc)

            # Caculate normalisation parameters
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {"shift": float(m), "scale": float(s)}

        return scale_dict, nTrks

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
        return varname, average, std, default

    def dict_in(
        self,
        varname: str,
        average: float,
        std: float,
        default: float,
    ) -> dict:
        """
        Creates dictionary entry containing scale and shift parameters.

        Parameters
        ----------
        varname : str
            Name of the variable
        average : float
            Average of the variable
        std : float
            Standard deviation of the variable
        default : float
            Default value of the variable

        Returns
        -------
        Scale_dict : dict
            Dict with the name, shift, scale and default
        """

        return {
            "name": varname,
            "shift": average,
            "scale": std,
            "default": default,
        }

    def GetScaleDict(
        self,
        input_file: str = None,
        chunk_size: int = 1e5,
    ):
        """
        Calculates the scaling, shifting and default values and saves them to json.

        Parameters
        ----------
        input_file : str, optional
            File which is used to calculate scaling/shifting, by default None
        chunk_size : int, optional
            Scale dict calculated using the given file, by default 1e5
        """

        # Get input filename to calculate scaling and shifting
        if input_file is None:
            input_file = self.config.GetFileName(option="resampled")

        logger.info("Calculating scaling and shifting values for the jet variables")
        logger.info(f"Using {input_file} for calculation of scaling/shifting")

        # Extract the correct variables
        variables_header = self.variable_config["train_variables"]
        var_list = [i for j in variables_header for i in variables_header[j]]

        # Get the file_length
        file_length = len(h5py.File(input_file, "r")["/jets"].fields(var_list[0])[:])

        # Get the number of chunks we need to load
        n_chunks = int(np.ceil(file_length / chunk_size))

        # Get the jets scaling generator
        jets_scaling_generator = self.get_scaling_generator(
            input_file=input_file,
            nJets=file_length,
            chunk_size=chunk_size,
        )

        # Loop over chunks
        for chunk_counter in range(n_chunks):
            logger.info(
                f"Calculating jet scales for chunk {chunk_counter+1} of {n_chunks}"
            )
            # Check if this is the first time loading from the generator
            if chunk_counter == 0:
                # Get the first chunk of scales from the generator
                scale_dict, nJets_loaded = next(jets_scaling_generator)
            else:
                # Get the next chunk of scales from the generator
                tmp_scale_dict, tmp_nJets_loaded = next(jets_scaling_generator)

                # Combine the scale dicts coming from the generator
                scale_dict, nJets_loaded = self.join_scale_dicts_jets(
                    first_scale_dict=scale_dict,
                    second_scale_dict=tmp_scale_dict,
                    first_nJets=nJets_loaded,
                    second_nJets=tmp_nJets_loaded,
                )

        logger.info("Calculating scaling and shifting values for the track variables")

        # Init a empty scale dict for the tracks
        scale_dict_trk = {}

        # Check if tracks are used or not
        if self.bool_use_tracks is True:

            # Loop over all tracks selections
            for tracks_name in self.tracks_names:
                scale_dict_trk_selection = {}
                # Load generator
                trks_scaling_generator = self.get_scaling_tracks_generator(
                    input_file=input_file,
                    nJets=file_length,
                    tracks_name=tracks_name,
                    chunk_size=chunk_size,
                )

                # Loop over chunks
                for chunk_counter in range(n_chunks):
                    logger.info(
                        f"Calculating track scales for {tracks_name} for chunk"
                        f" {chunk_counter+1} of {n_chunks}"
                    )
                    # Check if this is the first time loading from the generator
                    if chunk_counter == 0:
                        # Get the first chunk of scales from the generator
                        scale_dict_trk_selection, nTrks_loaded = next(
                            trks_scaling_generator
                        )
                    else:
                        # Get the next chunk of scales from the generator
                        tmp_dict_trk, tmp_nTrks_loaded = next(trks_scaling_generator)

                        # Combine the scale dicts coming from the generator
                        (
                            scale_dict_trk_selection,
                            nTrks_loaded,
                        ) = self.join_scale_dicts_trks(
                            first_scale_dict=scale_dict_trk_selection,
                            second_scale_dict=tmp_dict_trk,
                            first_nTrks=nTrks_loaded,
                            second_nTrks=tmp_nTrks_loaded,
                        )

                # Add scale dict for given tracks selection to the more general one
                scale_dict_trk.update({tracks_name: scale_dict_trk_selection})

        # TODO: change in python 3.9
        # save scale/shift dictionary to json file
        scale_dict = {"jets": scale_dict}
        scale_dict.update(scale_dict_trk)
        os.makedirs(os.path.dirname(self.scale_dict_path), exist_ok=True)
        with open(self.scale_dict_path, "w") as outfile:
            json.dump(scale_dict, outfile, indent=4)
        logger.info(f"Saved scale dictionary as {self.scale_dict_path}")

    def get_scaling_generator(
        self,
        input_file: str,
        nJets: int,
        chunk_size: int = int(10000),
    ):
        """
        Set up a generator that loads the jets in chunks and calculates the mean/std.

        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        nJets : int
            Number of jets which are to be scaled.
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(10000)

        Yields
        ------
        scale_dict_trk : dict
            Dict with the scale/shift values for each variable.
        nJets : int
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
            while start_ind < nJets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunk_size)

                # Check if end index is bigger than Njets
                end_ind = min(end_ind, nJets)

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            # Loop over the chunks
            for index_tuple in tupled_indices:
                # Init a list for the scale values for the variables
                scale_dict = []

                # Load jets
                jets = pd.DataFrame(
                    infile_all["/jets"].fields(var_list)[
                        index_tuple[0] : index_tuple[1]
                    ]
                )

                # Replace inf values
                jets.replace([np.inf, -np.inf], np.nan, inplace=True)

                if "weight" not in jets:
                    length = nJets if nJets < chunk_size else len(jets)
                    jets["weight"] = np.ones(int(length))

                # Iterate over the vars of the jets
                for var in jets.columns.values:
                    # Skip all label/weight variables
                    if var in [self.variable_config["label"], "weight"]:
                        continue

                    # Set Default values for isDefaults variables
                    if "isDefaults" in var:
                        logger.debug(
                            f"Default scaling/shifting values (0, 1) are used for {var}"
                        )
                        scale_dict.append(self.dict_in(var, 0.0, 1.0, None))

                    # Calculate scaling/shifting value for given variable
                    else:

                        # Get the dict entry
                        dict_entry = self.get_scaling(
                            vec=jets[var].values,
                            varname=var,
                            custom_defaults_vars=self.variable_config[
                                "custom_defaults_vars"
                            ],
                        )
                        scale_dict.append(self.dict_in(*dict_entry))

                # Yield the scale dict and the number of jets
                yield scale_dict, len(jets)

    def get_scaling_tracks_generator(
        self,
        input_file: str,
        nJets: int,
        tracks_name: str,
        chunk_size: int = int(10000),
    ):
        """
        Set up a generator that loads the tracks in chunks and calculates the mean/std.


        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        nJets : int
            Number of jets which are to be scaled.
        tracks_name : str
            Name of the tracks
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(10000)

        Yields
        ------
        scale_dict_trk : dict
            Dict with the scale/shift values for each variable.
        nTrks : int
            Number of tracks used for scaling/shifting.
        """

        # Load the variables which are scaled/shifted
        logNormVars = self.variable_config["track_train_variables"][tracks_name][
            "logNormVars"
        ]
        jointNormVars = self.variable_config["track_train_variables"][tracks_name][
            "jointNormVars"
        ]
        trkVars = logNormVars + jointNormVars

        # Open h5 file
        with h5py.File(input_file, "r") as infile_all:

            # Get the indices
            start_ind = 0
            tupled_indices = []

            # Loop over indicies
            while start_ind < nJets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunk_size)

                # Check if end index is bigger than Njets
                end_ind = min(end_ind, nJets)

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            # Loop over the chunks
            for index_tuple in tupled_indices:

                # Load tracks
                trks = np.asarray(
                    infile_all[f"/{tracks_name}"][index_tuple[0] : index_tuple[1]]
                )

                # Get the masking
                track_mask = get_track_mask(trks)

                # Stack the arrays by their variable
                X_trk_train = np.stack(
                    [np.nan_to_num(trks[v]) for v in trkVars], axis=-1
                )

                # Add small value, so log(0) does not happen
                eps = 1e-8

                # Take the log of the desired variables
                for i, _ in enumerate(logNormVars):
                    X_trk_train[:, :, i][track_mask] = np.log(
                        X_trk_train[:, :, i][track_mask] + eps
                    )

                # Scale the variables
                scale_dict_trk, nTrks = self.get_scaling_tracks(
                    data=X_trk_train[:, :, :],
                    var_names=logNormVars + jointNormVars,
                    track_mask=track_mask,
                )

                # Yield the scale dict and the number jets
                yield scale_dict_trk, nTrks

    def scale_generator(
        self,
        input_file: str,
        jets_variables: list,
        jets_scale_dict: dict,
        jets_default_dict: dict,
        nJets: int,
        tracks_scale_dict: dict = None,
        chunk_size: int = int(10000),
    ):
        """
        Set up a generator who applies the scaling/shifting for the given
        jet variables.

        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        jets_variables : list
            Variables of the jets which are to be scaled.
        jets_scale_dict : dict
            Scale dict of the jet variables with the values inside.
        jets_default_dict : dict
            Default scale dict of the jets.
        nJets : int
            Number of jets which are to be scaled.
        tracks_scale_dict : dict, optional
            Scale dict of the track variables., by default None
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(10000)

        Yields
        ------
        jets : np.ndarray
            Yielded jets
        tracks : np.ndarray
            Yielded tracks
        labels : np.ndarray
            Yielded labels
        tracks_labels : np.ndarray
            Yielded track labels
        flavour : np.ndarray
            Yielded flavours

        Raises
        ------
        ValueError
            If scale is found to be 0 or inf for any jet variable.
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0
            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunk_size)
                end_ind = min(end_ind, nJets)
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:

                # Load jets
                jets = pd.DataFrame(f["/jets"][index_tuple[0] : index_tuple[1]])
                labels = pd.DataFrame(f["/labels"][index_tuple[0] : index_tuple[1]])
                if "weight" not in jets:
                    length = nJets if nJets < chunk_size else len(jets)
                    jets["weight"] = np.ones(int(length))

                if "weight" not in jets_variables:
                    jets_variables += ["weight"]

                # keep the jet flavour
                flavour = jets[self.variable_config["label"]]

                # Remove inf values
                jets = jets[jets_variables]
                jets = jets.replace([np.inf, -np.inf], np.nan)

                # Fill the nans with default values
                jets = jets.fillna(jets_default_dict)

                for elem in jets_scale_dict:
                    if "isDefaults" in elem["name"] or "weight" in elem["name"]:
                        continue

                    if elem["scale"] == 0 or np.isinf(elem["scale"]):
                        raise ValueError(
                            f"Scale parameter for jet var {elem['name']} is "
                            f"{elem['scale']}."
                        )
                    jets[elem["name"]] -= elem["shift"]
                    jets[elem["name"]] /= elem["scale"]

                if self.bool_use_tracks is False:
                    yield jets, labels, flavour

                elif self.bool_use_tracks is True:
                    tracks, tracks_labels = [], []
                    # Loop on each track selection
                    for tracks_name in self.tracks_names:
                        trk_scale_dict = tracks_scale_dict[tracks_name]
                        # Load tracks
                        trks = np.asarray(
                            h5py.File(input_file, "r")[f"/{tracks_name}"][
                                index_tuple[0] : index_tuple[1]
                            ]
                        )

                        # Apply scaling to the tracks
                        trks, trk_labels = apply_scaling_trks(
                            trks=trks,
                            variable_config=self.variable_config,
                            scale_dict=trk_scale_dict,
                            tracks_name=tracks_name,
                        )
                        tracks.append(trks)
                        tracks_labels.append(trk_labels)

                    # Yield jets, labels and tracks
                    yield jets, tracks, labels, tracks_labels, flavour

            # TODO: Add plotting

    def ApplyScales(
        self,
        input_file: str = None,
        chunk_size: int = 1e6,
    ):
        """
        Apply the scaling and shifting.

        Parameters
        ----------
        input_file : str, optional
            File which is to be scaled., by default None
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default 1e6
        """

        # Get input filename to calculate scaling and shifting
        if input_file is None:
            input_file = self.config.GetFileName(option="resampled")

        logger.info(f"Scale/Shift jets from {input_file}")
        logger.info(f"Using scales in {self.scale_dict_path}")

        # Extract the correct variables
        variables_header_jets = self.variable_config["train_variables"]
        jets_variables = [
            i for j in variables_header_jets for i in variables_header_jets[j]
        ]

        file_length = len(h5py.File(input_file, "r")["/jets"][jets_variables[0]][:])

        n_chunks = int(np.ceil(file_length / chunk_size))

        # Get scale dict
        with open(self.scale_dict_path, "r") as infile:
            jets_scale_dict = json.load(infile)["jets"]

        # Define Scale dict with default values
        jets_default_dict = Gen_default_dict(jets_scale_dict)

        # Check if tracks are used
        if self.bool_use_tracks:
            tracks_scale_dict = {}
            # Get the scale dict for tracks
            with open(self.scale_dict_path, "r") as infile:
                full_scale_dict = json.load(infile)
                for tracks_name in self.tracks_names:
                    tracks_scale_dict[tracks_name] = full_scale_dict[f"{tracks_name}"]

        else:
            tracks_scale_dict = None

        # Load jets
        scale_generator = self.scale_generator(
            input_file=input_file,
            jets_variables=jets_variables,
            jets_scale_dict=jets_scale_dict,
            jets_default_dict=jets_default_dict,
            nJets=file_length,
            tracks_scale_dict=tracks_scale_dict,
            chunk_size=chunk_size,
        )

        logger.info("Applying scaling and shifting.")
        out_file = self.config.GetFileName(option="resampled_scaled")
        with h5py.File(out_file, "w") as h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            while chunk_counter <= n_chunks:
                logger.info(
                    f"Applying scales for chunk {chunk_counter+1} of {n_chunks+1}."
                )
                try:
                    # Load jets from file
                    if self.bool_use_tracks is False:
                        jets, labels, flavour = next(scale_generator)

                    else:
                        (
                            jets,
                            tracks,
                            labels,
                            track_labels,
                            flavour,
                        ) = next(scale_generator)

                    if chunk_counter == 0:
                        h5file.create_dataset(
                            "jets",
                            data=jets.to_records(index=False),
                            compression="gzip",
                            maxshape=(None,),
                        )
                        h5file.create_dataset(
                            "labels",
                            data=labels,
                            compression="gzip",
                            maxshape=(None, labels.shape[1]),
                        )
                        h5file.create_dataset(
                            "flavour",
                            data=flavour,
                            compression="gzip",
                            maxshape=(None,),
                        )

                    if chunk_counter == 0 and self.bool_use_tracks is True:
                        for i, tracks_name in enumerate(self.tracks_names):
                            h5file.create_dataset(
                                tracks_name,
                                data=tracks[i],
                                compression="lzf",
                                chunks=((100,) + tracks[i].shape[1:]),
                                maxshape=(
                                    None,
                                    tracks[i].shape[1],
                                    tracks[i].shape[2],
                                ),
                            )
                            if track_labels[i] is not None:
                                h5file.create_dataset(
                                    f"{tracks_name}_labels",
                                    data=track_labels[i],
                                    compression="lzf",
                                    chunks=((100,) + track_labels[i].shape[1:]),
                                    maxshape=(
                                        None,
                                        track_labels[i].shape[1],
                                        track_labels[i].shape[2],
                                    ),
                                )

                    else:
                        # appending to existing dataset
                        h5file["jets"].resize(
                            (h5file["jets"].shape[0] + jets.shape[0]),
                            axis=0,
                        )
                        h5file["jets"][-jets.shape[0] :] = jets.to_records(
                            index=False
                        )  # jets

                        h5file["labels"].resize(
                            (h5file["labels"].shape[0] + labels.shape[0]),
                            axis=0,
                        )
                        h5file["labels"][-labels.shape[0] :] = labels

                        h5file["flavour"].resize(
                            (h5file["flavour"].shape[0] + flavour.shape[0]),
                            axis=0,
                        )
                        h5file["flavour"][-flavour.shape[0] :] = flavour

                        if self.bool_use_tracks is True:
                            for i, tracks_name in enumerate(self.tracks_names):
                                h5file[tracks_name].resize(
                                    (h5file[tracks_name].shape[0] + tracks[i].shape[0]),
                                    axis=0,
                                )
                                h5file[tracks_name][-tracks[i].shape[0] :] = tracks[i]

                                if track_labels[i] is not None:
                                    h5file[f"{tracks_name}_labels"].resize(
                                        (
                                            h5file[f"{tracks_name}_labels"].shape[0]
                                            + track_labels[i].shape[0]
                                        ),
                                        axis=0,
                                    )
                                    h5file[f"{tracks_name}_labels"][
                                        -track_labels[i].shape[0] :
                                    ] = track_labels[i]

                except StopIteration:
                    break

                chunk_counter += 1

        # Plot the variables from the output file of the resampling process
        if (
            "njets_to_plot" in self.config.sampling["options"]
            and self.config.sampling["options"]["njets_to_plot"]
        ):
            preprocessing_plots(
                sample=self.config.GetFileName(option="resampled_scaled"),
                var_dict=self.variable_config,
                class_labels=self.config.sampling["class_labels"],
                plots_dir=os.path.join(
                    self.config.config["parameters"]["file_path"],
                    "plots/scaling/",
                ),
                track_collection_list=self.config.sampling["options"]["tracks_names"]
                if "tracks_names" in self.config.sampling["options"]
                and "save_tracks" in self.config.sampling["options"]
                and self.config.sampling["options"]["save_tracks"] is True
                else None,
                nJets=self.config.sampling["options"]["njets_to_plot"],
            )
