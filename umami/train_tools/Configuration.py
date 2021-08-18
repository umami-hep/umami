import yaml

from umami.configuration import logger
from umami.tools import yaml_loader
from umami.train_tools.NN_tools import get_class_label_ids


class Configuration(object):
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = yaml_config
        self.LoadConfigFile()
        self.GetConfiguration()

    def LoadConfigFile(self):
        logger.info(f"Using train config file {self.yaml_config}")
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        config_train_items = [
            "model_name",
            "preprocess_config",
            "model_file",
            "train_file",
            "validation_file",
            "add_validation_file",
            "var_dict",
            "NN_structure",
            "Eval_parameters_validation",
            "bool_use_taus",
            "ttbar_test_files",
            "zpext_test_files",
        ]
        config_evaluation_items = [
            "model_name",
            "ttbar_test_files",
            "zpext_test_files",
            "evaluate_trained_model",
            "Eval_parameters_validation",
            "bool_use_taus",
        ]
        if "evaluate_trained_model" in self.config.keys():
            if self.config["evaluate_trained_model"] is True:
                iterate_list = config_train_items

            elif self.config["evaluate_trained_model"] is False:
                iterate_list = config_evaluation_items

        else:
            iterate_list = config_train_items

        for item in iterate_list:
            if item in self.config:
                setattr(self, item, self.config[item])
            else:
                raise KeyError(
                    f"You need to specify {item} in your" " config file!"
                )

        # Define a security to check if label_value is used twice
        class_ids = get_class_label_ids(self.NN_structure["class_labels"])
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("label_value is used twice in the used classes!")
