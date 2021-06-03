import logging
import os
import pathlib

import yaml

from umami.tools import yaml_loader


class Configuration(object):
    """This is a global configuration to allow certain settings which are hardcoded so far."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = f"{pathlib.Path(__file__).parent.absolute()}/../configs/global_config.yaml"
        self.LoadConfigFile()
        self.logger = self.SetLoggingLevel()
        self.SetTFDebugLevel()
        self.GetConfiguration()

    def LoadConfigFile(self):
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        config_items = ["pTvariable", "etavariable"]
        for item in config_items:
            if item in self.config:
                self.logger.debug(f"Setting {item} to {self.config[item]}.")
                setattr(self, item, self.config[item])
            else:
                raise KeyError(
                    f"You need to specify {item} in your" " config file!"
                )

    def SetTFDebugLevel(self):
        """Setting the Debug level of tensorflow.
        For reference see https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error"""  # noqa
        self.logger.debug(
            f"Setting TFDebugLevel to {self.config['TFDebugLevel']}"
        )
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(self.config["TFDebugLevel"])

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
        logger = logging.getLogger("umami")
        if self.config["DebugLevel"] in log_levels.keys():
            logger.setLevel(log_levels[self.config["DebugLevel"]])
        else:
            logging.error(
                f"The 'DebugLevel' option {self.config['DebugLevel']} set in the global config is not valid."
            )
        ch = logging.StreamHandler()
        ch.setLevel(log_levels[self.config["DebugLevel"]])
        ch.setFormatter(CustomFormatter())

        logger.addHandler(ch)
        logger.propagate = False
        return logger


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    using implementation from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""  # noqa

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    debugformat = "%(asctime)s - %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"
    format = "%(levelname)s:%(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debugformat + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + debugformat + reset,
        logging.CRITICAL: bold_red + debugformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


global_config = Configuration()
logger = global_config.logger
logger.info(f"Loading global config {global_config.yaml_config}")
