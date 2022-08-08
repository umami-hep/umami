"""Base modul for process configuration (preprocessing, training, ...)."""
from pathlib import Path

from umami.configuration.configuration import logger
from umami.tools import YAML


class Configuration:
    """Configuration base class."""

    def __init__(self, yaml_config: str):
        """Init the Configuration class.

        Parameters
        ----------
        yaml_config : str
            Path to yaml config file.

        Raises
        ------
        ValueError
            if `yaml_config` does not exist as file
        """
        super().__init__()
        self.YAML = YAML(typ="safe", pure=True)
        self.yaml_config = Path(yaml_config)
        if not self.yaml_config.exists():
            raise ValueError(
                f"Your specified config file {yaml_config} does not exist."
            )
        self.yaml_default_config = None
        self.default_config = None
        self.config = None

    @property
    def config_path(self) -> str:
        """
        Return config path.

        Returns
        -------
        str
            Config path.
        """
        return self.yaml_config

    def load_config_file(self) -> None:
        """Load config file from disk."""
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = self.YAML.load(conf)
        logger.info("Using config file %s", self.yaml_config)

        with open(self.yaml_config, "r") as conf:
            self.config = self.YAML.load(conf)
