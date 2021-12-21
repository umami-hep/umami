import json
import os

import h5py
import numpy as np
import pandas as pd
import yaml

from umami.configuration import logger
from umami.tools import yaml_loader


def Gen_default_dict(scale_dict):
    """
    Generates default value dictionary from scale/shift dictionary.

    Input:
    - scale_dict: Scale dict loaded from json.

    Output:
    - default_dict: Returns a dict with all given variables but with
                    default values. With these, NaNs can be filled.
    """

    default_dict = {}
    for elem in scale_dict:
        if "isDefaults" in elem["name"]:
            continue
        default_dict[elem["name"]] = elem["default"]
    return default_dict


def apply_scaling_trks(
    trks,
    variable_config: dict,
    scale_dict: dict,
):
    """
    Apply the scaling/shifting to the tracks

    Input:
    - trks: Loaded tracks as numpy array.
    - variable_config: Loaded variable config.
    - scale_dict: Loaded scale dict.

    Output:
    - scaled_trks: The tracks scaled and shifted.
    - trk_labels: The track labels, if defined in the variable config.
    """

    # Init a list for the variables
    var_arr_list = []

    # Check masking
    trk_mask = ~np.isnan(trks[trks.dtype.names[0]])

    # Get the track variables
    tracks_noNormVars = variable_config["track_train_variables"]["noNormVars"]
    tracks_logNormVars = variable_config["track_train_variables"]["logNormVars"]
    tracks_jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    tracks_variables = tracks_noNormVars + tracks_logNormVars + tracks_jointNormVars

    # Iterate over variables and scale/shift it
    for var in tracks_variables:
        x = trks[var]

        if var in tracks_logNormVars:
            x = np.log(x)
        if var in tracks_jointNormVars or var in tracks_logNormVars:
            shift = np.float32(scale_dict[var]["shift"])
            scale = np.float32(scale_dict[var]["scale"])
            x = np.where(
                trk_mask,
                x - shift,
                x,
            )
            x = np.where(
                trk_mask,
                x / scale,
                x,
            )
        var_arr_list.append(np.nan_to_num(x))

        # track vertex and origin labels
        if "track_labels" in variable_config:
            trkLabels = variable_config["track_labels"]
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

    def __init__(self, config, compression="gzip") -> None:
        """
        Init the needed configs and variables

        Input:
        - config: Loaded config file for the preprocessing.
        - compression: Type of compression which should be used.
                       Default: gzip
        """
        self.config = config
        self.scale_dict_path = config.dict_file
        self.bool_use_tracks = config.sampling["options"]["save_tracks"]
        self.compression = compression
        self.mask_value = 0

        with open(config.var_file, "r") as conf:
            self.variable_config = yaml.load(conf, Loader=yaml_loader)

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

        Input:
        - first_scale_dict: First scale dict with the variable and
                            their respective mean/std inside.
        - second_scale_dict: Second scale dict with the variable and
                            their respective mean/std inside.
        - variable: Variable which is to be combined.
        - first_N: Number of tracks/jets used to calculate mean/std for first dict.
        - second_N: Number of tracks/jets used to calculate mean/std for second dict.

        Output:
        combined_mean: Combined mean/shift.
        combined_std: Combined std/scale.
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
                    (((mean - combined_mean) ** 2 + std ** 2) * first_N)
                    + (((tmp_mean - combined_mean) ** 2 + tmp_std ** 2) * second_N)
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

        Input:
        - first_scale_dict: First scale dict to join.
        - second_scale_dict: Second scale dict to join.

        Output:
        - combined_scale_dict: The combined scale dict.
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

        return combined_scale_dict

    def join_scale_dicts_jets(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        first_nJets: int,
        second_nJets: int,
    ):
        """
        Combining the scale dicts of two track chunks.

        Input:
        - first_scale_dict: First scale dict two join.
        - second_scale_dict: Second scale dict two join.

        Output:
        - combined_scaled_dict: The combined scale dict list.
        """

        # Init a combined list for the dicts
        combined_dict_list = []

        # Loop over the list with the dicts from the variables
        for counter in range(len(first_scale_dict)):
            # Ensure the same variables are merged
            if first_scale_dict[counter]["name"] == second_scale_dict[counter]["name"]:
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
                        varname=first_scale_dict[counter]["name"],
                        average=combined_average,
                        std=combined_std,
                        default=first_scale_dict[counter]["default"],
                    )
                )

        return combined_dict_list

    def get_scaling_tracks(self, data, var_names, mask_value=0):
        """
        Calculate the scale dict for the tracks and return the dict.

        Input:
        - data: Loaded tracks with shape (nJets, nTrks, nTrkFeatures)
        - var_names: List of variables which are to be scaled
        - mask_value: Masking value to use. Default: 0

        Output:
        - scale_dict: Scale dict with scaling/shifting values for each variable
        - nTrks: Number of tracks used to calculate the scaling/shifting
        """
        # TODO add weight support for tracks

        # Track variables
        # data has shape nJets,nTrks,nFeatures,so to sort out the mask,
        # we need to find where the value is masked for a track over
        # all it's features
        # mask has shape nJets,nTrks
        mask = ~np.all(data == mask_value, axis=-1)

        scale_dict = {}
        for v, name in enumerate(var_names):
            f = data[:, :, v]
            slc = f[mask]
            nTrks = len(slc)
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {"shift": float(m), "scale": float(s)}

        return scale_dict, nTrks

    def get_scaling(self, vec, varname, custom_defaults_vars):
        """
        Calculates the weighted average and std for vector vec.

        Input:
        - vec: Array with variable values for the jets
        - varname: Name of the variable which is to be scaled
        - custom_defaults_var: Dict with custom default variable values

        Output:
        - varname: Name of the variable
        - average: Average of the variable
        - std: Std of the variable
        - default: Default value of the variable
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

    def dict_in(self, varname, average, std, default):
        """
        Creates dictionary entry containing scale and shift parameters.

        Input:
        - varname: Name of the variable
        - average: Average of the variable
        - std: Standard deviation of the variable
        - default: Default value of the variable

        Output:
        - Scale_dict: Dict with the name, shift, scale and default
        """

        return {
            "name": varname,
            "shift": average,
            "scale": std,
            "default": default,
        }

    def GetScaleDict(self, input_file: str = None, chunkSize: int = 1e5):
        """
        Calculates the scaling, shifting and default values and saves them to json.

        Input:
        - input_file: File which is used to calculate scaling/shifting

        Output:
        - Scale dict: Scale dict calculated using the given file
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
        n_chunks = int(np.ceil(file_length / chunkSize))

        # Get the jets scaling generator
        jets_scaling_generator = self.get_scaling_generator(
            input_file=input_file,
            nJets=file_length,
            chunkSize=chunkSize,
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
                scale_dict = self.join_scale_dicts_jets(
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

            # Load generator
            trks_scaling_generator = self.get_scaling_tracks_generator(
                input_file=input_file,
                nJets=file_length,
                chunkSize=chunkSize,
            )

            # Loop over chunks
            for chunk_counter in range(n_chunks):
                logger.info(
                    f"Calculating track scales for chunk {chunk_counter+1} of"
                    f" {n_chunks}"
                )
                # Check if this is the first time loading from the generator
                if chunk_counter == 0:
                    # Get the first chunk of scales from the generator
                    scale_dict_trk, nTrks_loaded = next(trks_scaling_generator)
                else:
                    # Get the next chunk of scales from the generator
                    tmp_dict_trk, tmp_nTrks_loaded = next(trks_scaling_generator)

                    # Combine the scale dicts coming from the generator
                    scale_dict_trk = self.join_scale_dicts_trks(
                        first_scale_dict=scale_dict_trk,
                        second_scale_dict=tmp_dict_trk,
                        first_nTrks=nTrks_loaded,
                        second_nTrks=tmp_nTrks_loaded,
                    )

        # save scale/shift dictionary to json file
        scale_dict = {"jets": scale_dict, "tracks": scale_dict_trk}
        os.makedirs(os.path.dirname(self.scale_dict_path), exist_ok=True)
        with open(self.scale_dict_path, "w") as outfile:
            json.dump(scale_dict, outfile, indent=4)
        logger.info(f"Saved scale dictionary as {self.scale_dict_path}")

    def get_scaling_generator(
        self,
        input_file: str,
        nJets: int,
        chunkSize: int = int(10000),
    ):
        """
        Set up a generator that loads the jets in chunks and calculates the mean/std.

        Input:
        - input_file: File which is to be scaled.
        - nJets: Number of jets which are to be scaled.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - scale_dict_trk: Dict with the scale/shift values for each variable.
        - nJets: Number of jets used for scaling/shifting.
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
                end_ind = int(start_ind + chunkSize)

                # Check if end index is bigger than Njets
                if end_ind > nJets:
                    end_ind = nJets

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
                    length = nJets if nJets < chunkSize else len(jets)
                    jets["weight"] = np.ones(int(length))

                # Iterate over the vars of the jets
                for var in jets.columns.values:
                    # Skip all label/weight variables
                    if var in [self.variable_config["label"], "weight"]:
                        continue

                    # Set Default values for isDefaults variables
                    elif "isDefaults" in var:
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
        chunkSize: int = int(10000),
    ):
        """
        Set up a generator that loads the tracks in chunks and calculates the mean/std.

        Input:
        - input_file: File which is to be scaled.
        - nJets: Number of jets which are to be scaled.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - scale_dict_trk: Dict with the scale/shift values for each variable.
        - nTrks: Number of tracks used for scaling/shifting.
        """

        # Load the variables which are scaled/shifted
        logNormVars = self.variable_config["track_train_variables"]["logNormVars"]
        jointNormVars = self.variable_config["track_train_variables"]["jointNormVars"]
        trkVars = logNormVars + jointNormVars

        # Open h5 file
        with h5py.File(input_file, "r") as infile_all:

            # Get the indices
            start_ind = 0
            tupled_indices = []

            # Loop over indicies
            while start_ind < nJets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunkSize)

                # Check if end index is bigger than Njets
                if end_ind > nJets:
                    end_ind = nJets

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            # Loop over the chunks
            for index_tuple in tupled_indices:

                # Load tracks
                trks = np.asarray(
                    infile_all["/tracks"][index_tuple[0] : index_tuple[1]]
                )

                # Stack the arrays by their variable
                X_trk_train = np.stack(
                    [np.nan_to_num(trks[v]) for v in trkVars], axis=-1
                )

                # Get the masking
                mask = ~np.all(X_trk_train == self.mask_value, axis=-1)

                # Add small value, so log(0) does not happen
                eps = 1e-8

                # Take the log of the desired variables
                for i, v in enumerate(logNormVars):
                    X_trk_train[:, :, i][mask] = np.log(
                        X_trk_train[:, :, i][mask] + eps
                    )

                # Scale the variables
                scale_dict_trk, nTrks = self.get_scaling_tracks(
                    data=X_trk_train[:, :, :],
                    var_names=logNormVars + jointNormVars,
                    mask_value=self.mask_value,
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
        chunkSize: int = int(10000),
    ):
        """
        Set up a generator who applies the scaling/shifting for the given jet variables.

        Input:
        - input_file: File which is to be scaled.
        - jets_variables: Variables of the jets which are to be scaled.
        - jets_scale_dict: Scale dict of the jet variables with the values inside.
        - jets_default_dict: Default scale dict of the jets.
        - nJets: Number of jets which are to be scaled.
        - tracks_scale_dict: Scale dict of the track variables.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Yield: The yielded scaled/shifted jets/tracks and the labels
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0
            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunkSize)
                if end_ind > nJets:
                    end_ind = nJets
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunkSize)

            for index_tuple in tupled_indices:

                # Load jets
                jets = pd.DataFrame(
                    f["/jets"][index_tuple[0] : index_tuple[1]]
                )
                labels = pd.DataFrame(
                    f["/labels"][index_tuple[0] : index_tuple[1]]
                )
                if "weight" not in jets:
                    length = nJets if nJets < chunkSize else len(jets)
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

                    else:
                        jets[elem["name"]] -= elem["shift"]
                        jets[elem["name"]] /= elem["scale"]

                if self.bool_use_tracks is False:
                    yield jets, labels, flavour

                elif self.bool_use_tracks is True:

                    # Load tracks
                    trks = np.asarray(
                        h5py.File(input_file, "r")["/tracks"][
                            index_tuple[0] : index_tuple[1]
                        ]
                    )

                    # Apply scaling to the tracks
                    trks, trk_labels = apply_scaling_trks(
                        trks=trks,
                        variable_config=self.variable_config,
                        scale_dict=tracks_scale_dict,
                    )

                    # Yield jets, labels and tracks
                    yield jets, trks, labels, trk_labels, flavour

            # TODO: Add plotting

    def ApplyScales(self, input_file: str = None, chunkSize: int = 1e6):
        """
        Apply the scaling and shifting.

        Input:
        - input_file: File which is to be scaled.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - scaled_file: Returns the scaled/shifted file
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

        n_chunks = int(np.ceil(file_length / chunkSize))

        # Get scale dict
        with open(self.scale_dict_path, "r") as infile:
            jets_scale_dict = json.load(infile)["jets"]

        # Define Scale dict with default values
        jets_default_dict = Gen_default_dict(jets_scale_dict)

        # Check if tracks are used
        if self.bool_use_tracks:
            # Get the scale dict for tracks
            with open(self.scale_dict_path, "r") as infile:
                tracks_scale_dict = json.load(infile)["tracks"]

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
            chunkSize=chunkSize,
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
                            compression=self.compression,
                            maxshape=(None,),
                        )
                        h5file.create_dataset(
                            "labels",
                            data=labels,
                            compression=self.compression,
                            maxshape=(None, labels.shape[1]),
                        )
                        h5file.create_dataset(
                            "flavour",
                            data=flavour,
                            compression=self.compression,
                            maxshape=(None,),
                        )

                        if self.bool_use_tracks is True:
                            h5file.create_dataset(
                                "tracks",
                                data=tracks,
                                compression=self.compression,
                                maxshape=(
                                    None,
                                    tracks.shape[1],
                                    tracks.shape[2],
                                ),
                            )
                            if track_labels is not None:
                                h5file.create_dataset(
                                    "track_labels",
                                    data=track_labels,
                                    compression=self.compression,
                                    maxshape=(
                                        None,
                                        track_labels.shape[1],
                                        track_labels.shape[2],
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
                            h5file["tracks"].resize(
                                (h5file["tracks"].shape[0] + tracks.shape[0]),
                                axis=0,
                            )
                            h5file["tracks"][-tracks.shape[0] :] = tracks

                            if track_labels is not None:
                                h5file["track_labels"].resize(
                                    (
                                        h5file["track_labels"].shape[0]
                                        + track_labels.shape[0]
                                    ),
                                    axis=0,
                                )
                                h5file["track_labels"][
                                    -track_labels.shape[0] :
                                ] = track_labels

                except StopIteration:
                    break

                chunk_counter += 1
