import yaml

from umami.tools import yaml_loader


class Configuration(object):
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = yaml_config
        self.yaml_default_config = "configs/preprocessing_default_config.yaml"
        self.LoadConfigFile()
        self.GetConfiguration()

    def LoadConfigFile(self):
        print("Using train config file", self.yaml_config)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        config_items = [
            "model_name",
            "preprocess_config",
            "train_file",
            "validation_file",
            "var_dict",
            "NN_structure",
            "add_validation_file",
            "test_file",
            "add_test_file",
            "Eval_parameters_validation",
            "bool_use_taus",
            "comparison_file",
            "add_comparison_file",
        ]
        for item in config_items:
            if item in self.config:
                setattr(self, item, self.config[item])
            else:
                raise KeyError(
                    f"You need to specify {item} in your" " config file!"
                )
