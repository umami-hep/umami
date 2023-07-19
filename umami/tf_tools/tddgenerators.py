"""Data generator module to handle reading of TDD datasets
rescaling them and rearranging for training of models."""
import inspect
import json
import pickle

import h5py
import numpy as np
from numpy.lib.recfunctions import (
    append_fields,
    repack_fields,
    structured_to_unstructured,
)
from scipy.stats import binned_statistic_dd

from umami.configuration import logger
from umami.preprocessing_tools import (
    PreprocessConfiguration,
    apply_scaling_jets,
    apply_scaling_trks,
    binarise_jet_labels,
    get_track_mask,
    get_variable_dict,
)
from umami.preprocessing_tools.scaling import as_full
from umami.tf_tools.generators import (
    CadsGenerator,
    DipsGenerator,
    Dl1Generator,
    ModelGenerator,
    UmamiConditionGenerator,
    UmamiGenerator,
)


class TDDGenerator:
    """Base class for the generators of the datasets for the models.
    from the usual TDD format

    This class provides the base functionalites for the different
    models to load the dataset.
    """

    def __init__(
        self,
        train_file_path: str,
        n_jets: int,
        batch_size: int,
        sample_weights: bool,
        config_file: str,
        tracks_name: str = None,
        chunk_size: int = 1e5,
        excluded_var: list = None,
        n_conds: int = None,  # TODO implemnt
        print_logger: bool = False,
        use_validation_samples: bool = False,
        old_scaling_functions: bool = False,
    ) -> None:
        """
        Initialize the parameters needed for the generators.

        Parameters
        ----------
        train_file_path : str
            Path to the train file that is to be used.
        n_jets : int
            Number of jets that is to be used for training.
        batch_size : int
            Batch size for the training.
        sample_weights : bool
            Decide whether to use sample weights. If True, the sample weights
            need to be processed in the preprocessing. Otherwise, the values are ones.
        config_file : str
            Path to the preprocessing configuration file.
        tracks_name : str, optional
            Name of the tracks in the train tdd file. Leave empty if no tracks are used.
        chunk_size : int, optional
            Chunk size for loading the training jets.
        excluded_var : list, optional
            List with excluded variables. Only available for DL1 training.
        n_conds : int, optional
            Number of conditions used for training of CADS.
        print_logger : bool, optional
            Decide whether the logger outputs are printed or not.
        use_validation_samples: bool, optional
            Decide whether to use the validation samples for weights or not.
        old_scaling_functions : bool, optional
            Decide whether to use the old scaling functions or not.
        """

        self.old_scaling_functions = old_scaling_functions
        self.train_file_path = train_file_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.sample_weights = sample_weights
        self.tracks_name = tracks_name
        self.print_logger = print_logger
        self.excluded_var = excluded_var
        self.n_conds = n_conds

        # Load the configuration file and all the necessary information
        if isinstance(config_file, PreprocessConfiguration):
            config = config_file
        else:
            config = PreprocessConfiguration(config_file)

        self.sampling_options = config.sampling.options
        self.save_track_labels = config.sampling.options.save_track_labels
        self.class_labels = config.sampling.class_labels
        self.variable_config = get_variable_dict(config.general.var_file)
        self.jet_vars = sum(self.variable_config["train_variables"].values(), [])
        self.track_label_variables = self.variable_config.get("track_truth_variables")
        with open(config.general.dict_file, "r") as infile:
            self.scale_dict = json.load(infile)
        # done with the configuration file

        self.in_file = h5py.File(self.train_file_path, "r")

        if n_jets is not None:
            self.n_jets = int(n_jets)
        else:
            self.n_jets = self.in_file["/jets"].shape[0]
        self.length = int(self.n_jets / self.batch_size)
        self.step_size = self.batch_size * int(self.chunk_size / self.batch_size)
        self.x_in_mem = None
        self.weight_in_mem = None
        self.x_trk_in_mem = None
        self.y_in_mem = None
        self.use_validation_samples = use_validation_samples

        # Retrieving the dtypes of the variables to load
        self.datasets = {
            "jets": self.in_file["/jets"],
            "labels": self.in_file["/labels"],
        }  # handles to the datasets
        self.norm = {}
        self.dtype = {}
        self.variables = {"jets": self.jet_vars}
        self.dtype["jets"] = [
            (n, as_full(x)) for n, x in self.in_file["/jets"].dtype.descr
        ]
        self.norm["jets"] = self.get_normalisation_arrays(self.jet_vars, "jets")

        if tracks_name is not None:
            self.datasets[f"{self.tracks_name}"] = self.in_file[f"/{self.tracks_name}"]
            self.dtype[tracks_name] = [
                (n, as_full(x)) for n, x in self.in_file[f"/{tracks_name}"].dtype.descr
            ]
            self.trk_vars, self.trk_vars_lists_dict = self.get_track_vars()
            self.norm[self.tracks_name] = self.get_normalisation_arrays(
                self.trk_vars, self.tracks_name, self.trk_vars_lists_dict
            )
            self.variables[self.tracks_name] = self.trk_vars

    def get_normalisation_arrays(
        self, variables, input_type: str, trk_vars_lists_dict=None
    ):
        """
        Generate normalization arrays for the given variables.
        This function calculates the mean, standard deviation, and default values
        for the specified variables based on the provided input type
        and tracked variables dictionary (if available).
        It returns a dictionary containing arrays for mean,
        standard deviation, and default values.

        Parameters
        ----------
        variables : list
            A list of variables for which to generate normalization arrays.
        input_type : str
            The type of input data. Used to select the appropriate scale dictionary.
        trk_vars_lists_dict : dict, optional
            A dictionary containing lists of tracked variables.
            The keys represent categories, such as 'jointNormVars' or 'logNormVars',
            and the values are the corresponding lists of variables. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing the normalization arrays:
            - "mean" (numpy.ndarray):
                An array of mean values for the given variables.
            - "std" (numpy.ndarray):
                An array of standard deviation values for the given variables.
            - "default" (numpy.ndarray):
                An array of default values for the given variables.

        Notes
        -----
        If a variable does not have a corresponding entry in the scale dictionary
        or does not match any tracked variables, it will be assigned default
        normalization values of mean=0, std=1, and default=0.
        """

        scld = self.scale_dict[input_type]
        var0 = None
        for varr in variables:
            if varr in scld:
                var0 = varr
                mean_key = "mean" if "mean" in scld[var0] else "shift"
                std_key = "std" if "std" in scld[var0] else "scale"
                break
        if var0 is None:
            mean_key = "shift"
            std_key = "scale"
        default_key = "default"
        means = []
        stds = []
        defaults = []
        for varr in variables:
            if trk_vars_lists_dict is not None:
                condition = (
                    varr in trk_vars_lists_dict["jointNormVars"]
                    or varr in trk_vars_lists_dict["logNormVars"]
                )
            else:
                condition = True
            if (varr in scld) and condition:
                means.append(scld[varr][mean_key])
                stds.append(scld[varr][std_key])
                defaults.append(scld[varr].get(default_key) or 0)
            else:
                means.append(0)
                stds.append(1)
                defaults.append(0)
        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)
        defaults = np.array(defaults, dtype=np.float32)
        return {"mean": means, "std": stds, "default": defaults}

    def get_track_vars(self):
        """
        Retrieve track variables.

        This function retrieves the track variables based on the configuration settings.
        It returns a list of track variables
        and a dictionary containing categorized track variable lists.

        Returns
        -------
        tuple
            A tuple containing:
            - trk_vars (list): A list of track variables.
            - trk_vars_lists_dict (dict):
                A dictionary containing categorized track variable lists.

        Notes
        -----
        The track variables are obtained from the `variable_config`
        and categorized into three types:
        "noNormVars", "logNormVars", and "jointNormVars".
        If a specific variable type is not found in the configuration or is set to None,
        an empty list will be used as its value.
        """
        trk_vars = []
        trk_vars_lists_dict = {}
        # Get the variables for the tracks
        for var_type in ["noNormVars", "logNormVars", "jointNormVars"]:
            if (
                var_type
                in self.variable_config["track_train_variables"][self.tracks_name]
                and self.variable_config["track_train_variables"][self.tracks_name][
                    var_type
                ]
                is not None
            ):
                trk_vars_lists_dict[var_type] = self.variable_config[
                    "track_train_variables"
                ][self.tracks_name][var_type]

            else:
                logger.warning(
                    "No %s in variable dict for %s!", var_type, self.tracks_name
                )
                trk_vars_lists_dict[var_type] = []

            # Combine all variables into one list
        for _, item in trk_vars_lists_dict.items():
            trk_vars += item
        return trk_vars, trk_vars_lists_dict

    def scale_input(self, batch, input_type: str):
        """
        Normalize jet inputs.
        This function takes a batch of inputs
        and scales them based on the specified input type.
        It uses the mean and standard deviation values from the normalization dictionary
        to normalize the inputs. Any non-finite values in the inputs
        are replaced with the corresponding default values.

        Parameters
        ----------
        batch : numpy.ndarray
            A structured numpy array containing the batch of inputs.
        input_type : str
            The type of input data. Used to select
            the appropriate normalization parameters.

        Returns
        -------
        numpy.ndarray
            The normalized inputs as a NumPy array.
        """
        inputs = structured_to_unstructured(batch[self.variables[input_type]])
        inputs = inputs.astype(np.float32)
        inputs = np.where(
            ~np.isfinite(inputs),
            self.norm[input_type]["default"],
            inputs,
        )
        inputs = (inputs - self.norm[input_type]["mean"]) / self.norm[input_type]["std"]
        return inputs

    def scale_tracks(self, trks):
        """
        Scale track variables.
        This function scales the track variables in the given input
        based on the normalization parameters and track masks.
        It applies logarithmic scaling to the variables specified
        in the `logNormVars` list.
        Non-finite values in the input are replaced with appropriate values,
        and the scaling is performed using the mean and standard deviation values
        from the normalization dictionary.

        Parameters
        ----------
        trks : numpy.ndarray
            The input track variables to be scaled as a structured array.

        Returns
        -------
        numpy.ndarray
            The scaled track variables as a NumPy array.
        """
        track_mask = get_track_mask(trks)
        scaled_trks = structured_to_unstructured(trks[self.trk_vars])
        vars_to_log = np.array(
            [(var in self.trk_vars_lists_dict["logNormVars"]) for var in self.trk_vars]
        )
        track_mask_full = track_mask.reshape(track_mask.shape + (1,))
        scaled_trks[:, :, vars_to_log] = np.log(scaled_trks[:, :, vars_to_log])
        scaled_trks = np.where(
            ~track_mask_full, np.nan_to_num(scaled_trks), scaled_trks
        )
        scaled_trks = np.where(
            track_mask_full,
            (scaled_trks - self.norm[self.tracks_name]["mean"])
            / self.norm[self.tracks_name]["std"],
            scaled_trks,
        )
        return scaled_trks

    def calculate_weights(
        self,
        weights_dict: dict,
        jets: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Finds the according weight for the jet, with the weights calculated
        from the GetFlavorWeights method. Writes it onto the jets["weight"].

        Parameters
        ---------
        weights_dict : dict
            weights_dict per flavor and some additional info written into a
            pickle file at /hybrids/flavour_weights

            - 'bjets', etc.
            - 'bins_x' : pt bins
            - 'bins_y' : eta bins
            - 'bin_indices_flat' : flattened indices of the bins in the histogram
            - 'label_map' : {0: 'ujets', 1: 'cjets', 2: 'bjets'}

        jets : np.ndarray
            Containing values of jet variables
        labels : np.ndarray
            Binarized truth value of flavor for jet with shape
            (n_jets x (nFlavor x 1))
        """
        # Get the resampling variables and bins from the weight dict
        resampling_variables = weights_dict["resampling_variables"]
        resampling_bins = weights_dict["resampling_bins"]

        # scale to original values for binning
        with open(self.scale_dict, "r") as infile:
            jets_scale_dict = json.load(infile)
        for varname, scale_dict in jets_scale_dict["jets"].items():
            if varname in resampling_variables:
                jets[varname] *= scale_dict["scale"]
                jets[varname] += scale_dict["shift"]

        # Stack the jet variables
        sample_vector = np.column_stack(
            [np.asarray(jets[variable]) for variable in resampling_variables]
        )

        # Get binnumber of jet from 2D pt,eta grid
        _, _, binnumbers = binned_statistic_dd(
            sample=np.column_stack(
                [sample_vector[:, i] for i in range(sample_vector.shape[1])]
            ),
            values=None,
            statistic="count",
            bins=[value for _, value in resampling_bins.items()],
        )

        # transfrom labels into "bjets", "cjets"...
        label_keys = [weights_dict["label_map"][np.argmax(label)] for label in labels]
        for i, binnumber in enumerate(binnumbers):
            # look where in flattened 2D bin array this binnumber is
            index = np.where(weights_dict["bin_indices_flat"] == binnumber)
            # extract weight with flavour key and index
            weight = weights_dict[label_keys[i]][index]
            # if its out of the defined bin bounds, default to 1
            if not weight:
                weight = 1
            jets["weight"][i] = weight

    def load_in_memory(
        self,
        load_jets: bool,
        load_tracks: bool,
        part: int = 0,
    ):
        """
        This function loads jets, labels, tracks, weights,
        and performs scaling operations.
        The loaded data is stored in memory for faster
        access during training or evaluation.

        Parameters
        ----------
        load_jets : bool
            A flag indicating whether to load jets into memory.
        load_tracks : bool
            A flag indicating whether to load tracks into memory.
        part : int, optional
            The part number of the data to load. Defaults to 0.
        """
        if self.print_logger:
            logger.info(
                "\nloading in memory %i/%i", part + 1, 1 + self.n_jets // self.step_size
            )
        length = (
            min(self.step_size * (part + 1), len(self.datasets["labels"]))
            - self.step_size * part
        )

        ############################
        # Jets\
        if load_jets:
            # Load jets
            jets = np.array(0, dtype=self.dtype["jets"])
            shape = (length,) + self.datasets["jets"].shape[1:]
            jets.resize(shape, refcheck=False)
            indices = np.s_[self.step_size * part : self.step_size * part + length]
            self.datasets["jets"].read_direct(jets, indices)
        # Jets/
        ############################
        # Labels\
        # Load labels
        labels = self.datasets["labels"][
            self.step_size * part : self.step_size * (part + 1)
        ]
        label_classes = list(range(len(self.class_labels)))

        # Binarise labels
        labels_one_hot = binarise_jet_labels(
            labels=labels, internal_labels=label_classes
        )
        # Labels/
        ############################
        # Tracks\
        if load_tracks:
            # Get the tracks scale dict
            trk_scale_dict = self.scale_dict[self.tracks_name]

            # Load tracks
            tracks = np.array(0, dtype=self.dtype[self.tracks_name])
            shape = (length,) + self.datasets[self.tracks_name].shape[1:]
            tracks.resize(shape, refcheck=False)
            indices = np.s_[self.step_size * part : self.step_size * part + length]
            self.datasets[self.tracks_name].read_direct(tracks, indices)

            # Apply scaling to the tracks
            if self.old_scaling_functions:
                tracks, _, _ = apply_scaling_trks(
                    trks=tracks,
                    variable_config=self.variable_config,
                    scale_dict=trk_scale_dict,
                    tracks_name=self.tracks_name,
                    save_track_labels=self.save_track_labels,
                    track_label_variables=self.track_label_variables.get(
                        self.tracks_name
                    )
                    if self.save_track_labels
                    else None,
                )
            else:
                tracks = self.scale_tracks(tracks)

        else:
            tracks = None
        # Tracks/
        ############################
        # Weights\
        # If no weights are available, but asked, init ones as weights
        if self.sample_weights:
            if "weight" not in self.datasets["jets"].dtype.names:
                jets = append_fields(jets, "weight", np.ones(int(length)), dtypes="<i8")

            if self.sampling_options.bool_attach_sample_weights:
                weights_dict = None
                if self.use_validation_samples:
                    file_name = (
                        self.config.parameters["sample_path"]
                        + "/flavour_weights_training"
                    )
                else:
                    file_name = (
                        self.config.parameters["sample_path"]
                        + "/flavour_weights_validation"
                    )
                with open(file_name, "rb") as file:
                    weights_dict = pickle.load(file)
                self.calculate_weights(weights_dict, jets, labels)

            self.weight_in_mem = jets["weight"]
            jets = np.lib.recfunctions.drop_fields(jets, "weight")
        else:
            self.weight_in_mem = None
        # Weights/
        ############################
        # Jet scaling\
        # Scale jets only after weights are calculated
        if load_jets:
            if self.old_scaling_functions:
                # Loop over the columns and change all floats to full precision
                for iter_var in self.jet_vars:
                    if jets[iter_var].dtype.kind == "f":
                        jets[iter_var] = jets[iter_var].astype(np.float32)
                jets = apply_scaling_jets(
                    jets=jets,
                    variables_list=self.jet_vars,
                    scale_dict=self.scale_dict["jets"],
                )
                jets = repack_fields(jets[self.jet_vars])
                jets = structured_to_unstructured(jets).astype(np.float32)
            else:
                jets = repack_fields(jets[self.jet_vars])
                jets = self.scale_input(jets, "jets")
            jets = (
                np.delete(jets, self.excluded_var, 1)
                if self.excluded_var is not None
                else jets
            )
        else:
            jets = None
        # Jet scaling/

        self.x_in_mem = jets
        self.x_trk_in_mem = tracks
        self.y_in_mem = labels_one_hot

    def get_n_jets(self):
        """
        Get the number of jets.

        Returns
        -------
        int
            The number of jets.
        """
        return self.n_jets

    def get_n_dim(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of dimensions
        of the labels. It does not load jets or tracks.

        Returns
        -------
        int
            The number of dimensions of the labels.
        """
        self.load_in_memory(load_jets=False, load_tracks=False, part=0)
        return self.y_in_mem.shape[1]

    def get_n_jet_features(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of features
        for the jets. It does not load tracks.

        Returns
        -------
        int
            The number of jet features.
        """
        self.load_in_memory(load_jets=True, load_tracks=False, part=0)
        return self.x_in_mem.shape[1]

    def get_n_trk_features(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of features
        for the tracks. It does not load jets.

        Returns
        -------
        int
            The number of track features.
        """
        self.load_in_memory(load_jets=False, load_tracks=True, part=0)
        return self.x_trk_in_mem.shape[2]

    def get_n_trks(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of tracks.
        It does not load jets.

        Returns
        -------
        int
            The number of tracks.
        """
        self.load_in_memory(load_jets=False, load_tracks=True, part=0)
        return self.x_trk_in_mem.shape[1]


class TDDDipsGenerator(TDDGenerator, DipsGenerator):
    """Generator class for DIPS.

    This class provides the a generator that loads the TDD dataset
    for DIPS.
    """


class TDDDl1Generator(TDDGenerator, Dl1Generator):
    """Generator class for DL1*.

    This class provides the a generator that loads the TDD dataset
    for DL1*.
    """


class TDDUmamiGenerator(TDDGenerator, UmamiGenerator):
    """Generator class for UMAMI.

    This class provides the a generator that loads the TDD dataset
    for UMAMI.
    """


class TDDCadsGenerator(TDDGenerator, CadsGenerator):
    """Generator class for CADS.

    This class provides the a generator that loads the TDD dataset
    for CADS.
    """


class TDDUmamiConditionGenerator(TDDGenerator, UmamiConditionGenerator):
    """Generator class for UMAMI with conditional attention.

    This class provides the a generator that loads the TDD dataset
    for UMAMI with conditional attention.
    """


def filter_dictionary(dictionary, fields):
    """
    Filter a dictionary for a list of fields.

    This function takes a dictionary and a list of fields as input.
    It creates a new dictionary that contains only the key-value pairs
    where the key is present in the provided fields list.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be filtered.
    fields : list
        A list of fields to filter the dictionary.

    Returns
    -------
    dict
        A filtered dictionary that contains only the specified fields.

    """
    filtered_dict = {}
    for field in fields:
        if field in dictionary:
            filtered_dict[field] = dictionary[field]
    return filtered_dict


def get_generator(for_model, arg_dict, train_data_structure=None, small=False):
    """
    Get a generator for the specified model.
    This function returns a generator object based on the specified model.
    The generator is created using the provided arguments in the `arg_dict`.
    Optionally, the `train_data_structure`
    can be provided to customize the training data structure.
    The `small` parameter can be set to True to indicate a smaller dataset.

    Parameters
    ----------
    for_model : str
        The name or identifier of the model for which to get the generator.
    arg_dict : dict
        A dictionary containing the arguments for creating the generator.
    train_data_structure : Optional
        The data structure for training data. Defaults to None.
    small : bool, optional
        Flag indicating whether to use a small chunk size for loading
        that is equal to the size of 2 batches.
        Defaults to False.

    Returns
    -------
    generator
        A generator object for the specified model.

    Raises
    ------
    ValueError
        If the specified model is not supported.
    """
    if train_data_structure is None:
        with h5py.File(arg_dict["train_file_path"], "r") as in_file:
            if isinstance(in_file["/jets"], h5py.Group):
                train_data_structure = "Umami"
            else:
                train_data_structure = "TDD"

    if small:
        arg_dict["chunk_size"] = arg_dict["batch_size"] * 2

    generator = None
    if train_data_structure == "TDD":
        args_needed = inspect.getfullargspec(TDDGenerator.__init__).args
        arg_dict = filter_dictionary(arg_dict, args_needed)
        if for_model == "DIPS":
            generator = TDDDipsGenerator(**arg_dict)
        elif for_model == "Dl1":
            generator = TDDDl1Generator(**arg_dict)
        elif for_model == "Umami":
            generator = TDDUmamiGenerator(**arg_dict)
        elif for_model == "CADS":
            generator = TDDCadsGenerator(**arg_dict)
        elif for_model == "UmamiCondition":
            generator = TDDUmamiConditionGenerator(**arg_dict)
    elif train_data_structure == "Umami":
        args_needed = inspect.getfullargspec(ModelGenerator.__init__).args
        arg_dict = filter_dictionary(arg_dict, args_needed)
        if for_model == "DIPS":
            generator = DipsGenerator(**arg_dict)
        elif for_model == "Dl1":
            generator = Dl1Generator(**arg_dict)
        elif for_model == "Umami":
            generator = UmamiGenerator(**arg_dict)
        elif for_model == "CADS":
            generator = CadsGenerator(**arg_dict)
        elif for_model == "UmamiCondition":
            generator = UmamiConditionGenerator(**arg_dict)
    if generator is None:
        raise ValueError(f"Unknown model {for_model}")
    return generator
