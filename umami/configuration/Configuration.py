import logging
import pathlib

import yaml

from umami.tools import yaml_loader


class Configuration(object):
    """This is a global configuration to allow certain settings which are hardcoded so far."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = f"{pathlib.Path(__file__).parent.absolute()}/../configs/global_config.yaml"
        self.LoadConfigFile()
        self.SetLoggingLevel()
        self.GetConfiguration()

    def LoadConfigFile(self):
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        config_items = ["pTvariable", "etavariable"]
        for item in config_items:
            if item in self.config:
                logging.debug(f"Setting {item} to {self.config[item]}.")
                setattr(self, item, self.config[item])
            else:
                raise KeyError(
                    f"You need to specify {item} in your" " config file!"
                )

    def SetLoggingLevel(self):
        # set DebugLevel for logging
        log_levels = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        if self.config["DebugLevel"] in log_levels.keys():
            logging.basicConfig(level=log_levels[self.config["DebugLevel"]])
        else:
            logging.error(
                f"The 'DebugLevel' option {self.config['DebugLevel']} set in the global config is not valid."
            )


global_config = Configuration()
logging.info(f"Loading global config {global_config.yaml_config}")
