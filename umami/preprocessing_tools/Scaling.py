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

    def get_scaling_tracks(self, data, var_names, mask_value=0):
        """
        Calculate the scale dict for the tracks and return the dict.

        Input:
        - data: Loaded tracks with shape (nJets, nTrks, nTrkFeatures)
        - var_names: List of variables which are to be scaled
        - mask_value: Masking value to use. Default: 0

        Output:
        - scale_dict: Scale dict with scaling/shifting values for each variable
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
            logger.info(
                f"Scaling feature {v + 1} of {len(var_names)} ({name})."
            )
            f = data[:, :, v]
            slc = f[mask]
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {"shift": float(m), "scale": float(s)}

        return scale_dict

    def get_scaling(self, vec, w, varname, custom_defaults_vars):
        """
        Calculates the weighted average and std for vector vec and weight w.

        Input:
        - vec: Array with variable values for the jets
        - w: Jet weights
        - varname: Name of the variable which is to be scaled
        - custom_defaults_var: Dict with custom default variable values

        Output:
        - varname: Name of the variable
        - average: Average of the variable
        - std: Std of the variable
        - default: Default value of the variable
        """

        if np.sum(w) == 0:
            raise ValueError("Sum of weights has to be >0.")
        # find NaN values
        nans = np.isnan(vec)
        # check if variable has predefined default value
        if varname in custom_defaults_vars:
            default = custom_defaults_vars[varname]
        # NaN values are not considered in calculation for average
        else:
            w_without_nan = w[~nans]
            vec_without_nan = vec[~nans]
            default = np.ma.average(vec_without_nan, weights=w_without_nan)
        # replace NaN values with default values
        vec[nans] = default
        average = np.ma.average(vec, weights=w)
        std = np.sqrt(np.average((vec - average) ** 2, weights=w))
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

    def GetScaleDict(self, input_file: str = None):
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

        logger.info(f"Using {input_file} for calculation of scaling/shifting")

        # Load file
        infile_all = h5py.File(input_file, "r")

        # Extract the correct variables
        variables_header = self.variable_config["train_variables"]
        var_list = [i for j in variables_header for i in variables_header[j]]

        # Load jets from file
        jets = pd.DataFrame(infile_all["/jets"][:][var_list])

        # Replace inf values
        jets.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.info(
            "Retrieving scaling and shifting values for the jet variables"
        )
        scale_dict = []
        for var in jets.columns.values:
            if var in [self.variable_config["label"], "weight"]:
                continue
            elif "isDefaults" in var:
                logger.debug(
                    f"Default scaling/shifting values (0, 1) are used for {var}"
                )
                scale_dict.append(self.dict_in(var, 0.0, 1.0, None))
            else:
                dict_entry = self.get_scaling(
                    vec=jets[var].values,
                    w=jets["weight"].values
                    if "weight" in jets
                    else np.ones(len(jets)),
                    varname=var,
                    custom_defaults_vars=self.variable_config[
                        "custom_defaults_vars"
                    ],
                )
                scale_dict.append(self.dict_in(*dict_entry))

        scale_dict_trk = {}
        if self.bool_use_tracks is True:
            logger.info(
                "Retrieving scaling and shifting values for the track variables"
            )

            # Load the variables which are scaled/shifted
            logNormVars = self.variable_config["track_train_variables"][
                "logNormVars"
            ]
            jointNormVars = self.variable_config["track_train_variables"][
                "jointNormVars"
            ]
            trkVars = logNormVars + jointNormVars

            # Load the tracks as np arrays from file
            trks = np.asarray(infile_all["/tracks"][:])

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
            scale_dict_trk = self.get_scaling_tracks(
                data=X_trk_train[:, :, :],
                var_names=logNormVars + jointNormVars,
                mask_value=self.mask_value,
            )

        # save scale/shift dictionary to json file
        scale_dict = {"jets": scale_dict, "tracks": scale_dict_trk}
        os.makedirs(os.path.dirname(self.scale_dict_path), exist_ok=True)
        with open(self.scale_dict_path, "w") as outfile:
            json.dump(scale_dict, outfile, indent=4)
        logger.info(f"saved scale dictionary as {self.scale_dict_path}")

    def scale_generator(
        self,
        input_file: str,
        jets_variables: list,
        jets_scale_dict: dict,
        jets_default_dict: dict,
        nJets: int,
        tracks_noNormVars: list = None,
        tracks_logNormVars: list = None,
        tracks_jointNormVars: list = None,
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
        - tracks_noNormVars: Track variables which will not be scaled.
        - tracks_logNormVars: Track variables where the log is used and then scaled.
        - tracks_jointNormVars: Joint track variables.
        - tracks_scale_dict: Scale dict of the track variables.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Yield: The yielded scaled/shifted jets/tracks and the labels
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0
            end_ind = int(start_ind + chunkSize)

            tupled_indices = []
            while end_ind <= nJets or start_ind == 0:
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
                    length = nJets if nJets < chunkSize else chunkSize
                    jets["weight"] = np.ones(int(length))

                if "weight" not in jets_variables:
                    jets_variables += ["weight"]

                # Remove inf values
                jets = jets[jets_variables]
                jets = jets.replace([np.inf, -np.inf], np.nan)

                # Fill the nans with default values
                jets = jets.fillna(jets_default_dict)

                for elem in jets_scale_dict:
                    if (
                        "isDefaults" in elem["name"]
                        or "weight" in elem["name"]
                    ):
                        continue

                    else:
                        jets[elem["name"]] -= elem["shift"]
                        jets[elem["name"]] /= elem["scale"]

                if self.bool_use_tracks is False:
                    yield jets, labels

                elif self.bool_use_tracks is True:

                    # Load tracks
                    trks = np.asarray(
                        h5py.File(input_file, "r")["/tracks"][
                            index_tuple[0] : index_tuple[1]
                        ]
                    )

                    # Check masking
                    var_arr_list = []
                    trk_mask = ~np.isnan(trks["ptfrac"])
                    tracks_variables = (
                        tracks_noNormVars
                        + tracks_logNormVars
                        + tracks_jointNormVars
                    )

                    # Iterate over variables and scale/shift it
                    for var in tracks_variables:
                        if var in tracks_logNormVars:
                            x = np.log(trks[var])
                            x -= tracks_scale_dict[var]["shift"]
                            x /= tracks_scale_dict[var]["scale"]
                        elif var in tracks_jointNormVars:
                            x = np.where(
                                trk_mask,
                                x - tracks_scale_dict[var]["shift"],
                                x,
                            )
                            x = np.where(
                                trk_mask,
                                x / tracks_scale_dict[var]["scale"],
                                x,
                            )
                        else:
                            x = trks[var]
                        var_arr_list.append(np.nan_to_num(x))

                    # Stack the results for new dataset
                    d_arr = np.stack(var_arr_list, axis=-1)

                    # Yield jets, labels and tracks
                    yield jets, d_arr, labels

            # TODO: Add plotting

    def ApplyScales(self, input_file: str = None, chunkSize: int = 1e5):
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

        # Extract the correct variables
        variables_header_jets = self.variable_config["train_variables"]
        jets_variables = [
            i for j in variables_header_jets for i in variables_header_jets[j]
        ]

        file_length = len(
            h5py.File(input_file, "r")["/jets"][jets_variables[0]][:]
        )
        n_chunks = int(np.ceil(file_length / chunkSize))

        # Get scale dict
        with open(self.scale_dict_path, "r") as infile:
            jets_scale_dict = json.load(infile)["jets"]

        # Define Scale dict with default values
        jets_default_dict = Gen_default_dict(jets_scale_dict)

        # Check if tracks are used
        if self.bool_use_tracks:

            # Load variables for tracks
            noNormVars = self.variable_config["track_train_variables"][
                "noNormVars"
            ]
            logNormVars = self.variable_config["track_train_variables"][
                "logNormVars"
            ]
            jointNormVars = self.variable_config["track_train_variables"][
                "jointNormVars"
            ]

            # Get the scale dict for tracks
            with open(self.scale_dict_path, "r") as infile:
                tracks_scale_dict = json.load(infile)["tracks"]

        else:
            noNormVars = None
            jointNormVars = None
            logNormVars = None
            tracks_scale_dict = None

        # Load jets
        scale_generator = self.scale_generator(
            input_file=input_file,
            jets_variables=jets_variables,
            jets_scale_dict=jets_scale_dict,
            jets_default_dict=jets_default_dict,
            nJets=file_length,
            tracks_noNormVars=noNormVars,
            tracks_logNormVars=logNormVars,
            tracks_jointNormVars=jointNormVars,
            tracks_scale_dict=tracks_scale_dict,
            chunkSize=chunkSize,
        )

        logger.info("Applying scaling and shifting.")
        out_file = self.config.GetFileName(option="resampled_scaled")
        with h5py.File(out_file, "w") as h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            while chunk_counter <= n_chunks:
                try:
                    # Load jets from file
                    if self.bool_use_tracks is False:
                        jets, labels = next(scale_generator)

                    else:
                        jets, tracks, labels = next(scale_generator)

                    if chunk_counter == 0:
                        h5file.create_dataset(
                            "jets",
                            data=jets.to_records(index=False),
                            compression=self.compression,
                        )
                        h5file.create_dataset(
                            "labels",
                            data=labels,
                            compression=self.compression,
                        )
                        if self.bool_use_tracks is True:
                            h5file.create_dataset(
                                "tracks",
                                data=tracks,
                                compression=self.compression,
                            )

                    else:
                        # appending to existing dataset
                        h5file["jets"].resize(
                            (h5file["jets"].shape[0] + jets.shape[0]),
                            axis=0,
                        )
                        h5file["jets"][-jets.shape[0] :] = jets
                        h5file["labels"].resize(
                            (h5file["labels"].shape[0] + labels.shape[0]),
                            axis=0,
                        )
                        h5file["labels"][-labels.shape[0] :] = labels
                        if self.bool_use_tracks is True:
                            h5file["tracks"].resize(
                                (h5file["tracks"].shape[0] + tracks.shape[0]),
                                axis=0,
                            )
                            h5file["tracks"][-tracks.shape[0] :] = tracks

                except StopIteration:
                    break

                chunk_counter += 1
