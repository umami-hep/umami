"""Configuration module for preprocessing."""
import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random

from umami.configuration import Configuration, logger
from umami.tools import flatten_list


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
            sample = Sample()
            sample.name = sample_name
            f_output = sample_settings.get("f_output", None)
            if f_output is None:
                sample.output_name = Path(sample_settings.get("output_name"))
            else:
                sample.output_name = Path(f_output.get("path", ".")) / f_output.get(
                    "file"
                )
            sample.type = sample_settings.get("type")
            sample.category = sample_settings.get("category")
            if "n_jets" in sample_settings:
                sample.n_jets = int(sample_settings.get("n_jets"))
            else:
                sample.n_jets = int(4e6)
                logger.info(
                    "`n_jets` not specified for sample %s. It will be set to 10M.",
                    sample_name,
                )
            sample.cuts = flatten_list(sample_settings.get("cuts"))
            if sample.cuts is None:
                sample.cuts = []
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


@dataclass
class Sampling:
    """Class handling preprocessing options in `sampling` block."""

    class_labels: list = None
    method: str = None
    options: object = None


@dataclass
class SamplingOptions:
    """Class handling preprocessing options in `sampling` block."""

    sampling_variables: list = None
    samples: dict = None
    custom_n_jets_initial: dict = None
    fractions: dict = None
    max_upsampling_ratio: dict = None
    n_jets: int = None
    n_jets_scaling: int = None
    save_tracks: bool = None
    tracks_names: list = None
    save_track_labels: bool = None
    track_truth_variables: list = None
    intermediate_index_file: str = None
    weighting_target_flavour: str = None
    bool_attach_sample_weights: bool = None
    n_jets_to_plot: int = None


@dataclass
class GeneralSettings:
    """Class handling general preprocessing options."""

    outfile_name: str = None
    plot_name: str = None
    plot_sample_label: str = None
    var_file: str = None
    dict_file: str = None
    compression: str = None
    precision: str = None
    concat_jet_tracks: bool = None
    convert_to_tfrecord: dict = None


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
        self.check_tracks_names()
        self.check_deprecated_keys()

        # here the new syntax starts
        logger.info("Initialising preparation configuration.")
        self.preparation = Preparation(self.config.get("preparation"))

    def get_configuration(self) -> None:
        """Assign configuration from file to class variables.

        Raises
        ------
        KeyError
            if required config option is not present in passed config file
        """
        for elem in self.default_config:
            if elem == "preparation":
                continue
            if elem in self.config:
                if isinstance(self.config[elem], dict) and "f_" in elem:
                    if "file" not in self.config[elem]:
                        raise KeyError(
                            "You need to specify the 'file' for"
                            f"{elem} in your config file!"
                        )
                    if self.config[elem]["file"] is None:
                        raise KeyError(
                            "You need to specify the 'file' for"
                            f" {elem} in your config file!"
                        )
                    if "path" in self.config[elem]:
                        setattr(
                            self,
                            elem,
                            os.path.join(
                                self.config[elem]["path"],
                                self.config[elem]["file"],
                            ),
                        )
                    else:
                        setattr(self, elem, self.config[elem]["file"])

                else:
                    setattr(self, elem, self.config[elem])
            elif self.default_config[elem] is None:
                raise KeyError(f"You need to specify {elem} in yourconfig file!")
            else:
                logger.warning(
                    "Setting %s to default value %s", elem, self.default_config[elem]
                )
                setattr(self, elem, self.default_config[elem])

    def get_file_name(
        self,
        iteration: int = None,
        option: str = None,
        extension: str = ".h5",
        custom_path: str = None,
        use_val: bool = False,
    ) -> str:
        """
        Get the file name for different preprocessing steps.

        Parameters
        ----------
        iteration : int, optional
            Number of iterations, by default None
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

        Raises
        ------
        ValueError
            If the outfile is not a .h5 file.
        """
        if option is None and iteration is None:
            if use_val:
                return self.config["parameters"]["outfile_name_validation"]

            return self.outfile_name

        out_file = (
            self.outfile_name
            if not use_val
            else self.config["parameters"]["outfile_name_validation"]
        )
        try:
            idx = out_file.index(".h5")
        except ValueError as error:
            raise ValueError(
                "Your specified `outfile_name` has to be a .h5 file. "
                f"You defined in the preprocessing config {out_file}"
            ) from error

        if iteration is None:
            if option is None:
                inserttxt = ""
            else:
                inserttxt = f"-{option}"
        else:
            if option is None:
                inserttxt = (
                    f"-file-{iteration:.0f}"
                    f"_{self.sampling['options']['iterations']:.0f}"
                )
            else:
                inserttxt = (
                    f"-{option}-file-{iteration:.0f}"
                    f"_{self.sampling['options']['iterations']:.0f}"
                )
        if custom_path is not None:
            name_base = out_file.split("/")[-1]
            idx = name_base.index(".h5")
            return custom_path + name_base[:idx] + inserttxt + extension

        out_file = out_file[:idx] + inserttxt + extension
        return out_file

    def check_deprecated_keys(self) -> None:
        """Checks if deprecated keys are used in the config file and raise an error."""

        check_key(
            self.sampling["options"].keys(),
            "custom_njets_initial",
            "custom_n_jets_initial",
        )
        check_key(
            self.sampling["options"].keys(),
            "njets",
            "n_jets",
        )

    def check_tracks_names(self) -> None:
        """
        Checks if the option tracks_name is given.

        Raises
        ------
        ValueError
            When save_tracks is True but no tracks_names was given.
        """
        if self.sampling["options"].get("save_tracks", False) is True:
            if isinstance(self.sampling["options"].get("tracks_names"), str):
                self.sampling["options"]["tracks_names"] = [
                    self.sampling["options"]["tracks_names"]
                ]

            elif not isinstance(self.sampling["options"].get("tracks_names"), list):
                raise ValueError(
                    "You set save_tracks to True but gave not a string or a "
                    "list for tracks_names! You gave "
                    f'{isinstance(self.sampling["options"].get("tracks_names"))}'
                )

        else:
            self.sampling["options"]["save_tracks"] = False
            self.sampling["options"]["tracks_names"] = None

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
            out_dir = Path(self.config["parameters"]["file_path"]).parent
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
        new_var_dict_path = out_dir / Path(self.var_file).name
        config["var_file"] = str(new_var_dict_path.resolve())
        config["parameters"]["var_file"] = str(new_var_dict_path.resolve())

        # if scale dict file exists, copy it as well
        if Path(self.dict_file).is_file():
            logger.info("Scale dict exists and will be copied.")
            new_sd_path = out_dir / Path(self.dict_file).name
            config["dict_file"] = str(new_sd_path.resolve())
            config["parameters"][".dict_file"] = str(new_sd_path.resolve())
            logger.info("Copying config file to %s", new_sd_path)
            if new_sd_path.is_file():
                logger.warning("Overwriting existing scale dict at %s", new_sd_path)
            if self.dict_file != str(new_sd_path):
                shutil.copyfile(self.dict_file, new_sd_path)

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
        shutil.copyfile(self.var_file, new_var_dict_path)
