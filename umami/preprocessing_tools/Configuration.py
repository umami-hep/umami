"""Configuration module for preprocessing."""
import copy
import os
import shutil

import yaml

from umami.configuration import logger
from umami.tools import YAML


class Configuration:
    """docstring for Configuration."""

    def __init__(self, yaml_config: str):
        """Init the Configuration class.

        Parameters
        ----------
        yaml_config : str
            Path to yaml config file.
        """
        super().__init__()
        self.YAML = YAML(typ="safe", pure=True)
        self.yaml_config = yaml_config
        self.yaml_default_config = "configs/preprocessing_default_config.yaml"
        self.load_config_files()
        self.get_configuration()
        self.CheckTracksNames()

    @property
    def ConfigPath(self) -> str:
        """
        Return config path.

        Returns
        -------
        str
            Config path.
        """
        return self.yaml_config

    @property
    def ParameterConfigPath(self) -> str:
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
            os.path.dirname(self.ConfigPath),
            line[1].strip(),
        )
        return preprocess_parameters_path

    def load_config_files(self) -> None:
        """Load config file from disk."""
        self.yaml_default_config = os.path.join(
            os.path.dirname(__file__), self.yaml_default_config
        )
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = self.YAML.load(conf)
        logger.info("Using config file %s", self.yaml_config)

        with open(self.yaml_config, "r") as conf:
            self.config = self.YAML.load(conf)

    def get_configuration(self) -> None:
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
                logger.warning(
                    "Setting %s to default value %s", elem, self.default_config[elem]
                )
                setattr(self, elem, self.default_config[elem])

    def GetFileName(
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

    def CheckTracksNames(self) -> None:
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

    def copy_to_out_dir(self, suffix: str) -> None:
        """
        Write the current config object to a new file, in the output dir
        of the current preprocessing job

        Parameters
        ----------
        suffix : str
            Append this string to the copied config file name
        """

        # get output directory of this preprocessing job
        out_dir = os.path.dirname(self.config["parameters"]["file_path"])

        # don't run during tests
        if out_dir == ".":
            return

        # go up one level
        if os.path.basename(out_dir) == "preprocessed":
            out_dir = os.path.dirname(out_dir)

        # deepcopy current config dict
        config = copy.deepcopy(self.config)

        # get path for copy of current conifg
        root, ext = os.path.splitext(os.path.basename(self.yaml_config))
        if suffix not in root:
            new_config_fname = root + "_" + suffix + ext
        else:
            new_config_fname = root + ext
        new_config_path = os.path.join(out_dir, new_config_fname)

        # get new var dict path and update copied config
        new_var_dict_path = os.path.join(out_dir, os.path.basename(self.var_file))
        config["parameters"]["var_file"] = new_var_dict_path

        # make output dir
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)

        # copy config
        logger.info("Copying config file to %s", new_config_path)
        if os.path.exists(new_config_path):
            logger.warning("Overwriting existing config at %s", new_config_path)
        with open(new_config_path, "w") as f:
            yaml.dump(config, f)

        # copy var dict
        logger.info("Copying variable dict to %s", new_config_path)
        if os.path.exists(new_var_dict_path):
            logger.warning(
                "Overwriting existing variable dict at %s", new_var_dict_path
            )
        shutil.copyfile(self.var_file, new_var_dict_path)
