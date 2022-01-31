"""Configuration module for preprocessing."""
import os
import warnings

import yaml

from umami.configuration import logger
from umami.tools import YAML, yaml_loader


class Configuration:
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super().__init__()
        self.yaml_config = yaml_config
        self.yaml_default_config = "configs/preprocessing_default_config.yaml"
        self.LoadConfigFiles()
        self.GetConfiguration()
        self.CheckTracksNames()

    @property
    def ConfigPath(self):
        """Return config path."""
        return self.yaml_config

    @property
    def ParameterConfigPath(self):
        """Return parameter config path."""
        with open(self.yaml_config, "r") as conf:
            first_line = conf.readline()
        first_line = first_line.split("!include ")
        if first_line[0] != "parameters: ":
            raise ValueError(
                "Please specify in the first line of the preprocessing config"
                " the 'parameters' with the !include option."
            )

        preprocess_parameters_path = os.path.join(
            os.path.dirname(self.ConfigPath),
            first_line[1].strip(),
        )
        return preprocess_parameters_path

    def LoadConfigFiles(self):
        """Load config file from disk."""
        self.yaml_default_config = os.path.join(
            os.path.dirname(__file__), self.yaml_default_config
        )
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = yaml.load(conf, Loader=yaml_loader)
        logger.info(f"Using config file {self.yaml_config}")
        umami_yaml = YAML(typ="safe", pure=True)
        with open(self.yaml_config, "r") as conf:
            self.config = umami_yaml.load(conf)

    def GetConfiguration(self):
        """Assigne configuration from file to class variables.

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
                warnings.warn(
                    f"setting {elem} to default value {self.default_config[elem]}"
                )
                setattr(self, elem, self.default_config[elem])

    def GetFileName(
        self, iteration=None, option=None, extension=".h5", custom_path=None
    ):
        """Get the file name for different preprocessing steps."""
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

    def CheckTracksNames(self):
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
