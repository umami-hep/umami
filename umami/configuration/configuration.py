"""Configuration for logger of umami and tensorflow as well as reading global config."""
import logging
import os
import pathlib

import matplotlib
import yaml


class Configuration:
    """
    This is a global configuration to allow certain settings which are
    hardcoded so far.
    """

    def __init__(self):
        super().__init__()
        self.yaml_config = (
            f"{pathlib.Path(__file__).parent.absolute()}/../configs/global_config.yaml"
        )
        self.load_config_file()
        self.logger = self.set_logging_level()
        self.set_tf_debug_level()
        self.set_mpl_plotting_backend()
        self.get_configuration()

    def load_config_file(self):
        """Load config file from disk."""
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml.FullLoader)

    def get_configuration(self):
        """Assigne configuration from file to class variables.

        Raises
        ------
        KeyError
            if required config is not present in passed config file
        """
        config_items = [
            "pTvariable",
            "etavariable",
            "MPLPlottingBackend",
            "flavour_categories",
            "hist_err_style",
            "OriginType",
            "process_labels",
        ]
        for item in config_items:
            if item in self.config:
                self.logger.debug("Setting %s to %s.", item, self.config[item])
                setattr(self, item, self.config[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file!")

    def set_mpl_plotting_backend(self):
        """Setting the plotting backend of matplotlib."""
        self.logger.debug(
            "Setting Matplotlib's backend to %s", self.config["MPLPlottingBackend"]
        )

        matplotlib.use(self.config["MPLPlottingBackend"])

    def set_tf_debug_level(self):
        """Setting the Debug level of tensorflow.
        For reference see https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error"""  # noqa # pylint: disable=C0301
        self.logger.debug("Setting TFDebugLevel to %s", self.config["TFDebugLevel"])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(self.config["TFDebugLevel"])

    def set_logging_level(self) -> object:
        """Set DebugLevel for logging.

        Returns
        -------
        object
            Umami logger.
        """

        log_levels = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        umami_logger = logging.getLogger("umami")
        if self.config["DebugLevel"] in log_levels.keys():  # pylint: disable=C0201
            umami_logger.setLevel(log_levels[self.config["DebugLevel"]])
        else:
            logging.error(
                "The 'DebugLevel' option %s set in the global config is not valid.",
                self.config["DebugLevel"],
            )
        ch_handler = logging.StreamHandler()
        ch_handler.setLevel(log_levels[self.config["DebugLevel"]])
        ch_handler.setFormatter(CustomFormatter())

        umami_logger.addHandler(ch_handler)
        umami_logger.propagate = False
        return umami_logger


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    using implementation from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""  # noqa # pylint: disable=C0301

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    debugformat = (
        "%(asctime)s - %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"
    )
    date_format = "%(levelname)s:%(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debugformat + reset,
        logging.INFO: green + date_format + reset,
        logging.WARNING: yellow + date_format + reset,
        logging.ERROR: red + debugformat + reset,
        logging.CRITICAL: bold_red + debugformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_log_level(umami_logger, log_level: str):
    """Setting log level

    Parameters
    ----------
    umami_logger : logger
        logger object
    log_level : str
        logging level corresponding CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    umami_logger.setLevel(log_levels[log_level])
    for handler in umami_logger.handlers:
        handler.setLevel(log_levels[log_level])


global_config = Configuration()
logger = global_config.logger
logger.debug("Loading global config %s", global_config.yaml_config)
