"""Configuration module for preprocessing."""
import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PosixPath
from random import Random
from subprocess import check_output

import pydash

from umami.configuration import Configuration, logger
from umami.tools import check_option_definition, flatten_list


def check_key(location, old_key: str, new_key: str) -> None:
    """Helper function to check

    Parameters
    ----------
    location : object
        location in which to check the keys
    old_key : str
        name of old key/option
    new_key : str
        name of new key/option
    Raises
    ------
    KeyError
        If deprecated keys are being used
    """
    if old_key in location:
        raise KeyError(
            f"`{old_key}` was deprecated and is now called `{new_key}`. "
            "Please change that in your config"
        )


@dataclass
class Sample:
    """Class storing sample info.

    Parameters
    ----------
    name : str
        Name of sample
    type : str
        Sample type
    category : str
        Sample category, e.g. bjets
    n_jets : int
        Number of jets to load from sample
    cuts : dict
        Dictionary containing cuts which will be applied on sample
    output_name : str
        Name of output file
    """

    name: str = None
    type: str = None
    category: str = None
    n_jets: int = None
    cuts: dict = None
    output_name: str = None

    def __str__(self) -> str:
        return (
            f"{self.name=}, {self.type=}, {self.category=}, {self.n_jets=}, "
            f"{self.cuts=}, {self.output_name=}"
        )

    def __post_init__(self):
        """
        Process options and perform checks on them.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.name, "name", str, True),
            (self.type, "type", str, True),
            (self.category, "category", str, True),
            (self.n_jets, "n_jets", int, True),
            (self.cuts, "cuts", list, False),
            (self.output_name, "output_name", [str, PosixPath], True),
        ]

        # Check option definition and define return dict
        self.return_dict = {}

        for iter_var in needed_args:
            check_option_definition(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )
            self.return_dict[iter_var[1]] = iter_var[0]


@dataclass
class Sampling:
    """Class handling preprocessing options in `sampling` block."""

    class_labels: list = None
    method: str = None
    options: object = None
    use_validation_samples: bool = False

    def as_dict(self):
        """
        Return the class attributes as dict

        Returns
        -------
        dict
            Class attributes as dict
        """
        return self.return_dict

    def __post_init__(self):
        """
        Process options and perform checks on them.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.class_labels, "class_labels", list, True),
            (self.method, "method", str, True),
            (self.use_validation_samples, "use_validation_samples", bool, True),
        ]

        # Check option definition and define return dict
        self.return_dict = {}

        for iter_var in needed_args:
            check_option_definition(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )
            self.return_dict[iter_var[1]] = iter_var[0]


@dataclass
class SamplingOptions:
    """Class handling preprocessing options in `sampling` block."""

    sampling_variables: list = None
    samples_training: dict = None
    samples_validation: dict = None
    custom_n_jets_initial: dict = None
    fractions: dict = None
    max_upsampling_ratio: dict = None
    sampling_fraction: dict = None
    n_jets: int = None
    n_jets_validation: int = None
    n_jets_scaling: int = None
    save_tracks: bool = False
    tracks_names: list = None
    save_track_labels: bool = False
    intermediate_index_file: str = None
    intermediate_index_file_validation: str = None
    weighting_target_flavour: str = None
    bool_attach_sample_weights: bool = None
    n_jets_to_plot: int = None
    target_distribution: str = None

    def as_dict(self):
        """
        Return the class attributes as dict

        Returns
        -------
        dict
            Class attributes as dict
        """
        return self.return_dict

    def __post_init__(self):
        """
        Process options and perform checks on them.

        Raises
        ------
        ValueError
            If save_tracks is True but no tracks_names are given.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.sampling_variables, "sampling_variables", list, True),
            (self.samples_training, "samples", dict, True),
            (self.samples_validation, "samples", dict, True),
            (self.custom_n_jets_initial, "custom_n_jets_initial", dict, False),
            (self.fractions, "fractions", dict, False),
            (self.max_upsampling_ratio, "max_upsampling_ratio", dict, False),
            (self.sampling_fraction, "sampling_fraction", dict, False),
            (self.n_jets, "n_jets", int, True),
            (self.n_jets_validation, "n_jets_validation", int, False),
            (self.n_jets_scaling, "n_jets_scaling", int, False),
            (self.save_tracks, "save_tracks", bool, True),
            (self.tracks_names, "tracks_names", list, False),
            (self.save_track_labels, "save_track_labels", bool, True),
            (self.intermediate_index_file, "intermediate_index_file", str, True),
            (
                self.intermediate_index_file_validation,
                "intermediate_index_file_validation",
                str,
                False,
            ),
            (self.weighting_target_flavour, "weighting_target_flavour", str, False),
            (
                self.bool_attach_sample_weights,
                "bool_attach_sample_weights",
                bool,
                False,
            ),
            (self.n_jets_to_plot, "n_jets_to_plot", int, False),
            (self.target_distribution, "target_distribution", str, False),
        ]

        # Check option definition and define return dict
        self.return_dict = {}

        for iter_var in needed_args:
            check_option_definition(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )
            self.return_dict[iter_var[1]] = iter_var[0]

        # Check that tracks_name is correctly defined if save tracks is true
        if self.save_tracks and self.tracks_names is None:
            raise ValueError(
                "You defined save_tracks as True but gave no tracks_names! "
                "Please define them!"
            )


@dataclass
class GeneralSettings:
    """Class handling general preprocessing options."""

    outfile_name: str = None
    outfile_name_validation: str = None
    plot_name: str = None
    plot_type: str = "pdf"
    apply_atlas_style: bool = True
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    legend_sample_category: bool = True
    var_file: str = None
    dict_file: str = None
    compression: str = None
    precision: str = None
    concat_jet_tracks: bool = False
    convert_to_tfrecord: dict = None

    def as_dict(self):
        """
        Return the class attributes as dict

        Returns
        -------
        dict
            Class attributes as dict
        """
        return self.return_dict

    def plot_options_as_dict(self):
        """
        Return the plotting related class attributes as dict. These
        values are the ones which can be passed to PUMA.

        Returns
        -------
        dict
            Plotting related class attributes
        """
        return self.plot_options_dict

    def __post_init__(self):
        """
        Process options and perform checks on them.

        Raises
        ------
        ValueError
            If the one of the filenames does not have the right file extension.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.outfile_name, "outfile_name", str, True),
            (self.outfile_name_validation, "outfile_name_validation", str, False),
            (self.plot_name, "plot_name", str, True),
            (self.plot_type, "plot_type", str, True),
            (self.apply_atlas_style, "apply_atlas_style", bool, True),
            (self.use_atlas_tag, "use_atlas_tag", bool, True),
            (self.atlas_first_tag, "atlas_first_tag", str, False),
            (self.atlas_second_tag, "atlas_second_tag", str, False),
            (self.legend_sample_category, "legend_sample_category", bool, True),
            (self.var_file, "var_file", str, True),
            (self.dict_file, "dict_file", str, True),
            (self.compression, "compression", str, False),
            (self.precision, "precision", str, True),
            (self.concat_jet_tracks, "concat_jet_tracks", bool, True),
            (self.convert_to_tfrecord, "convert_to_tfrecord", dict, False),
        ]

        # Check option definition and define return dict
        self.return_dict = {}
        self.plot_options_dict = {}

        # Loop over list
        for iter_var in needed_args:
            check_option_definition(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )
            self.return_dict[iter_var[1]] = iter_var[0]

            if iter_var[1] not in [
                "outfile_name",
                "outfile_name_validation",
                "plot_name",
                "plot_type",
                "legend_sample_category",
                "var_file",
                "dict_file",
                "compression",
                "precision",
                "concat_jet_tracks",
                "convert_to_tfrecord",
            ]:
                self.plot_options_dict[iter_var[1]] = iter_var[0]

        # Check that .h5 files have the h5 in the name
        for varname, var, needed_extension in zip(
            ["outfile_name", "outfile_name_validation", "var_file", "dict_file"],
            [
                self.outfile_name,
                self.outfile_name_validation,
                self.var_file,
                self.dict_file,
            ],
            [".h5", ".h5", ".yaml", ".json"],
        ):
            # Skip variable if its None
            if not var:
                continue

            # Check that the given variable has the correct extension
            if not var.endswith(needed_extension):
                raise ValueError(
                    f"Your specified `{varname}` has to be a {needed_extension} file. "
                    f"You defined in the preprocessing config {var}"
                )


class Preparation:
    """Class handling preprocessing options in `preparation` block."""

    def __init__(self, settings: dict) -> None:
        """Initialise Preparation settings.

        Parameters
        ----------
        settings : dict
            Dictionary containing the preparation block.
        """
        # Number of jets loaded per batch from the files for preparation.
        self.settings = settings
        # TODO: rename also in config to batch_size
        if "batch_size" in settings:
            self.batch_size = int(settings.get("batch_size"))
        elif "batchsize" in settings:
            self.batch_size = int(settings.get("batchsize"))
        else:
            self.batch_size = 500_000
            logger.info(
                "Batch size not specified for prepare step. It will be set to 500k."
            )
        self.input_files = {}
        self.samples = {}
        self._init_samples(settings.get("samples"))
        # The sample categories are the keys of the input_h5 dict
        self.sample_categories = list(
            self.settings.get("input_h5", self.settings.get("ntuples")).keys()
        )

    def get_sample(self, sample_name: str):
        """Retrieve information about sample.

        Parameters
        ----------
        sample_name : str
            Name of sample

        Returns
        -------
        Sample
            sample class of specified sample

        Raises
        ------
        KeyError
            if specified sample not in config file
        """
        try:
            return self.samples[sample_name]
        except KeyError as error:
            raise KeyError(
                f"Requested sample `{sample_name}` is not defined in config file."
            ) from error

    def _init_samples(self, samples: dict) -> None:
        """Reading in samples from configuration.

        Parameters
        ----------
        samples : dict
            dictionary containing samples

        Raises
        ------
        KeyError
            if both `f_output` and `output_name` are specified for a sample
        """

        for sample_name, sample_settings in samples.items():
            # for now two config options are supported, either defining the full name
            # via `output_name` or in the old way giving a dict `f_output` with `path`
            # and `file` specified.
            if "f_output" in sample_settings and "output_name" in sample_settings:
                raise KeyError(
                    "You specified both `f_output` and `output_name` in your"
                    f"`{sample_name}`, you can only specify one of them."
                )

            # Get the f_output and the cuts
            f_output = sample_settings.get("f_output")
            cuts = sample_settings.get("cuts", [])
            sample = Sample(
                name=sample_name,
                type=sample_settings.get("type"),
                category=sample_settings.get("category"),
                n_jets=int(sample_settings.get("n_jets")),
                output_name=Path(f_output.get("path", ".")) / f_output.get("file")
                if f_output
                else Path(sample_settings.get("output_name")),
                cuts=flatten_list(cuts),
            )
            self.samples[sample_name] = sample
            logger.debug("Read in sample %s", sample)

    def _init_input_h5(self, input_h5: dict) -> None:
        """Reading in input_h5 from configuration.

        Parameters
        ----------
        input_h5 : dict
            dictionary containing input_h5

        Raises
        ------
        FileNotFoundError
            If there are no input h5 files found for a given sample.
        """
        for sample_type, value in input_h5.items():
            path = Path(value.get("path"))
            file_list = sorted(list(path.rglob(value.get("file_pattern"))))
            if len(file_list) == 0:
                raise FileNotFoundError(
                    f"Didn't find any input files for {sample_type}.\n"
                    f"\t- path: {value['path']}\n"
                    f"\t- pattern: {value['file_pattern']}"
                )
            if value.get("randomise"):
                Random(42).shuffle(file_list)
            self.input_files[sample_type] = file_list

    def get_input_files(self, sample_type: str):
        """Provides

        Parameters
        ----------
        sample_type : str
            Sample type, e.g. ttbar

        Returns
        -------
        list
            List of h5 input files
        """
        if not self.input_files:
            # avoid to call this in the init since it is not always needed,
            # make it available only when requested.
            # the keyword `ntuples` was renamed to `input_h5`, still supporting both
            self._init_input_h5(
                self.settings.get("input_h5", self.settings.get("ntuples"))
            )

        return self.input_files.get(sample_type)


class PreprocessConfiguration(Configuration):
    """Preprocessing Configuration class."""

    def __init__(self, yaml_config: str):
        """Init the Configuration class.

        Parameters
        ----------
        yaml_config : str
            Path to yaml config file.
        """
        super().__init__(yaml_config)
        self.yaml_default_config = (
            Path(os.path.dirname(__file__))
            / "configs/preprocessing_default_config.yaml"
        )
        self.load_config_file()
        self.get_configuration()
        self.check_resampling_options()

        # Get the git hash
        self.git_hash = (
            check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        )

    def get_configuration(self) -> None:
        """
        Assign configuration from file to class variables.
        """

        # Init the preparation block
        self.preparation = Preparation(self.config.get("preparation"))

        # Init the sampling block
        self.sampling = Sampling(
            class_labels=self.config["sampling"].get("class_labels"),
            method=self.config["sampling"].get("method"),
            options=SamplingOptions(**self.config["sampling"]["options"]),
        )

        # Get the parameters and cut parameters
        setattr(self, "parameters", self.config["parameters"])
        setattr(self, "cut_parameters", self.config["cut_parameters"])

        # Arguments in the config file which are not general
        non_general_args = ["preparation", "sampling", "parameters", "cut_parameters"]
        self.general = GeneralSettings(**pydash.omit(self.config, non_general_args))

    def check_resampling_options(self):
        """
        Checking that n_jets* are defined correctly for the given resampling method.

        Raises
        ------
        ValueError
            If the value is smaller than 1 for all methods beside pdf
        """

        # Loop over the n_jets which are to check
        for n_jets_iter, name_iter in zip(
            (self.sampling.options.n_jets, self.sampling.options.n_jets_validation),
            ("n_jets", "n_jets_validation"),
        ):
            if n_jets_iter is not None:
                # Check that n_jets
                if self.sampling.method != "pdf":
                    if n_jets_iter <= 0:
                        raise ValueError(
                            f"You defined resampling method {self.sampling.method} "
                            f"with {name_iter} <= 0! Only values above zero are "
                            "support for this method!"
                        )

                else:
                    if n_jets_iter < 1 and n_jets_iter != -1:
                        raise ValueError(
                            f"You defined resampling method {self.sampling.method} "
                            f"with {name_iter} <= 0! Only values above zero and -1 are"
                            " support for this method!"
                        )

        # Check that fractions is set except for the importance sampling
        if (
            self.sampling.options.fractions is None
            and self.sampling.method != "importance_no_replace"
        ):
            raise ValueError(
                "You havn't defined the target fractions for your resampling!"
            )

    def get_file_name(
        self,
        option: str = None,
        extension: str = ".h5",
        custom_path: str = None,
        use_val: bool = False,
    ) -> str:
        """
        Get the file name for different preprocessing steps.

        Parameters
        ----------
        option : str, optional
            Option name for file, by default None
        extension : str, optional
            File extension, by default ".h5"
        custom_path : str, optional
            Custom path to file, by default None
        use_val: bool, optinal
            Decide if the outfile name from the training or
            from the validation will be loaded. With True, the
            validation file name will be used. By default False.

        Returns
        -------
        str
            Path of the output file.
        """

        # Get the defined base output file name
        out_file = (
            self.general.outfile_name_validation
            if use_val
            else self.general.outfile_name
        )

        # Get the index of the file extension
        idx = out_file.index(".h5")

        # Insert the option into the output file name
        inserttxt = "" if option is None else f"-{option}"

        # Check for a custom path
        if custom_path is not None:
            name_base = out_file.rsplit("/", maxsplit=1)[-1]
            idx = name_base.index(".h5")
            return custom_path + name_base[:idx] + inserttxt + extension

        # Check if the pure base output file name should be returned
        if option is None:
            return out_file

        # Create the new outfile name
        out_file = out_file[:idx] + inserttxt + extension
        return out_file

    def copy_to_out_dir(self, suffix: str, out_dir: str = None) -> None:
        """
        Write the current config object to a new file, in the output dir
        of the current preprocessing job

        Parameters
        ----------
        suffix : str
            Append this string to the copied config file name
        out_dir : str
            Output directory to which the files are copied.
        """
        if out_dir is None:
            # don't run during tests
            if self.config["parameters"]["file_path"] == ".":
                return
            # get output directory of this preprocessing job and go up one level
            out_dir = Path(self.config["parameters"]["file_path"])
        else:
            out_dir = Path(out_dir)
        # make output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # deepcopy current config dict
        config = copy.deepcopy(self.config)

        # get path for copy of current conifg
        suffix = f"_{suffix}" if suffix not in self.yaml_config.stem else ""
        new_config_path = Path(out_dir) / (
            self.yaml_config.stem + suffix + self.yaml_config.suffix
        )

        # get new var dict path and update copied config
        new_var_dict_path = out_dir / Path(self.general.var_file).name
        config["var_file"] = str(new_var_dict_path.resolve())
        config["parameters"]["var_file"] = str(new_var_dict_path.resolve())

        # if scale dict file exists, copy it as well
        new_sd_path = out_dir / Path(self.general.dict_file).name
        if Path(self.general.dict_file).is_file() and not new_sd_path.is_file():
            logger.info("Scale dict exists and will be copied.")
            config["dict_file"] = str(new_sd_path.resolve())
            config["parameters"][".dict_file"] = str(new_sd_path.resolve())
            logger.info("Copying config file to %s", new_sd_path)
            if new_sd_path.is_file():
                logger.warning("Overwriting existing scale dict at %s", new_sd_path)
            shutil.copyfile(self.general.dict_file, new_sd_path)

        # copy config
        logger.info("Copying config file to %s", new_config_path)
        if new_config_path.is_file():
            logger.warning("Overwriting existing config at %s", new_config_path)
        self.yaml.dump(config, new_config_path)

        # copy var dict
        logger.info("Copying variable dict to %s", new_config_path)
        if new_var_dict_path.is_file():
            logger.warning(
                "Overwriting existing variable dict at %s", new_var_dict_path
            )
        if self.general.var_file != str(new_var_dict_path):
            shutil.copyfile(self.general.var_file, new_var_dict_path)
