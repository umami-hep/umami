import yaml

from umami.tools import yaml_loader


class Configuration(object):
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = yaml_config
        self.LoadConfigFile()
        self.GetConfiguration()

    def LoadConfigFile(self):
        print("Using Input plots configuration file ", self.yaml_config)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        config_items = [
            "test_file",
            "test_file_Zext",
            "comparison_file",
            "nJets",
            "plot_settings_jets",
            "plot_settings_tracks",
            "binning_jets",
            "binning_tracks",
            "flavors",
        ]
        for item in config_items:
            if item in self.config:
                setattr(self, item, self.config[item])
            else:
                raise KeyError(
                    f"You need to specify {item} in your" " config file!"
                )