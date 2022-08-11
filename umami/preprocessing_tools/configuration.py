"""Configuration module for preprocessing."""
import copy
import os
import shutil
from pathlib import Path

from umami.configuration import Configuration, logger


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

    @property
    def parameter_config_path(self) -> str:
        """
        Return parameter config path, as found on some line in the config file.

        Returns
        -------
        str
            Preprocessing parameter filepath
        """

        with open(self.yaml_config, "r") as conf:
            line = conf.readline()
            while "!include" not in line:
                line = conf.readline()

        line = line.split("!include ")
        if line[0] != "parameters: ":
            logger.warning(
                "You did not specify in the first line of the preprocessing config  the"
                " 'parameters' with the !include option. Ignoring any parameter file."
            )
            return None

        preprocess_parameters_path = os.path.join(
            os.path.dirname(self.config_path),
            line[1].strip(),
        )
        return preprocess_parameters_path

    def get_configuration(self) -> None:
        """Assign configuration from file to class variables.

        Raises
        ------
        KeyError
            if required config option is not present in passed config file
        """
        for elem in self.default_config:
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
            return self.outfile_name
        out_file = self.outfile_name
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
        """Checks if the option tracks_name is given."""
        if (
            "tracks_names" not in self.sampling["options"]
            or self.sampling["options"]["tracks_names"] is None
        ):
            self.sampling["options"]["tracks_names"] = [
                "tracks",
            ]
            if self.sampling["options"]["save_tracks"]:
                logger.info(
                    "'tracks_names' option not given or None, using default value"
                    "'tracks'"
                )
        elif not isinstance(self.sampling["options"]["tracks_names"], list):
            self.sampling["options"]["tracks_names"] = [
                self.sampling["options"]["tracks_names"]
            ]

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
            new_scale_dict_path = out_dir / Path(self.dict_file).name
            config["dict_file"] = str(new_scale_dict_path.resolve())
            config["parameters"][".dict_file"] = str(new_scale_dict_path.resolve())
            logger.info("Copying config file to %s", new_scale_dict_path)
            if new_scale_dict_path.is_file():
                logger.warning(
                    "Overwriting existing scale dict at %s", new_scale_dict_path
                )
            shutil.copyfile(self.dict_file, new_scale_dict_path)

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
