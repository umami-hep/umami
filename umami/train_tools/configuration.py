"""Configuration module for NN trainings."""
import os

import pydash
import yaml

from umami.configuration import logger
from umami.helper_tools import get_class_label_ids, get_class_label_variables
from umami.plotting_tools.utils import translate_kwargs
from umami.tools import yaml_loader


class Configuration:
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super().__init__()
        self.yaml_config = yaml_config
        self.config = {}
        self.yaml_default_config = "configs/default_train_config.yaml"
        self.load_config_file()
        self.get_configuration()

    @property
    def plot_args(self):
        """Plotting arguments for the training plots
        Returns
        -------
        dict
            Arguments for plotting API
        """
        # List of arguments are non-plotting arguments
        omit_args = [
            "n_jets",
            "val_batch_size",
            "eval_batch_size",
            "taggers_from_file",
            "trained_taggers",
            "tagger_label",
            "WP",
        ]
        if self.config is not None and "Validation_metrics_settings" in self.config:
            plot_arguments = pydash.omit(
                self.config["Validation_metrics_settings"], omit_args
            )
            return translate_kwargs(plot_arguments)
        return {}

    def load_config_file(self):
        """Load config file from disk."""
        self.yaml_default_config = os.path.join(
            os.path.dirname(__file__), self.yaml_default_config
        )
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = yaml.load(conf, Loader=yaml_loader)

        logger.info(f"Using train config file {self.yaml_config}")
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

        # Check if values in default config are defined in loaded config
        # If not, set default values
        for elem in self.default_config:
            if elem not in self.config or self.config[elem] is None:
                self.config[elem] = self.default_config[elem]

            if isinstance(self.default_config[elem], dict):
                for item in self.default_config[elem]:
                    if item not in self.config[elem]:
                        self.config[elem][item] = self.default_config[elem][item]

    def get_configuration(self):
        """Assigne configuration from file to class variables.

        Raises
        ------
        KeyError
            if required config is not present in passed config file
        KeyError
            deprecation error for Plotting_settings
        ValueError
            if label_value is used twice in the used classes
        ValueError
            if label definitions are mixed
        """
        config_train_items = [
            "model_name",
            "preprocess_config",
            "model_file",
            "continue_training",
            "train_file",
            "validation_files",
            "test_files",
            "var_dict",
            "NN_structure",
            "Validation_metrics_settings",
            "Eval_parameters_validation",
            "tracks_name",
        ]
        config_evaluation_items = [
            "model_name",
            "test_files",
            "evaluate_trained_model",
            "NN_structure",
            "Validation_metrics_settings",
            "Eval_parameters_validation",
            "tracks_name",
        ]

        if "validation_file" in self.config or "add_validation_file" in self.config:
            raise KeyError(
                "You have specified the entries 'validation_file' and/or "
                "'add_validation_file' in your training config. This structure is no "
                "longer supported. Please move both filepaths in the common "
                "'validation_files' section of the train config."
            )

        if "evaluate_trained_model" in self.config:
            if self.config["evaluate_trained_model"] is True:
                iterate_list = config_train_items
                bool_evaluate_trained_model = True

            elif self.config["evaluate_trained_model"] is False:
                iterate_list = config_evaluation_items
                bool_evaluate_trained_model = False

        else:
            iterate_list = config_train_items
            bool_evaluate_trained_model = True

        if "Plotting_settings" in self.config:
            raise KeyError(
                """
                You defined Plotting_settings. This option is deprecated. Please use
                Validation_metrics_settings instead.
                """
            )

        for item in iterate_list:
            if item in self.config:
                if item == "tracks_name":
                    setattr(self, "tracks_key", f"X_{self.config[item]}_train")

                elif item == "Validation_metrics_settings":
                    try:
                        if (
                            self.config["Validation_metrics_settings"]["val_batch_size"]
                            is None
                            and self.config["Eval_parameters_validation"][
                                "eval_batch_size"
                            ]
                            is None
                        ):
                            logger.warning(
                                "Neither eval_batch_size nor "
                                "val_batch_size was defined. Using "
                                "training batch_size for "
                                "validation/evaluation!"
                            )

                            self.config["Validation_metrics_settings"][
                                "val_batch_size"
                            ] = int(self.config["NN_structure"]["batch_size"])
                            self.config["Eval_parameters_validation"][
                                "eval_batch_size"
                            ] = int(self.config["NN_structure"]["batch_size"])

                        elif (
                            self.config["Validation_metrics_settings"]["val_batch_size"]
                            is None
                        ):
                            logger.warning(
                                "No val_batch_size defined. Using training batch size"
                                " for validation"
                            )

                            self.config["Validation_metrics_settings"][
                                "val_batch_size"
                            ] = int(self.config["NN_structure"]["batch_size"])

                        elif (
                            self.config["Eval_parameters_validation"]["eval_batch_size"]
                            is None
                        ):
                            logger.warning(
                                "No eval_batch_size defined. Using validation batch"
                                " size for evaluation."
                            )

                            self.config["Eval_parameters_validation"][
                                "eval_batch_size"
                            ] = int(
                                self.config["Validation_metrics_settings"][
                                    "val_batch_size"
                                ]
                            )

                    except KeyError as Error:
                        if bool_evaluate_trained_model:
                            raise ValueError("No batch size given!") from Error

                setattr(self, item, self.config[item])

            elif item == "tracks_name":
                if "dl1" not in self.config["NN_structure"]["tagger"]:
                    setattr(self, item, "tracks")
                    setattr(self, "tracks_key", "X_trk_train")
                    logger.warning(
                        'Using old version of tracks keys nomenclautre ("X_trk_train")'
                    )

            elif item not in self.config and item == "continue_training":
                setattr(self, item, False)

            else:
                raise KeyError(f"You need to specify {item} in your config file!")

        # Define a security to check if label_value is used twice
        class_ids = get_class_label_ids(self.config["NN_structure"]["class_labels"])
        class_label_vars, _ = get_class_label_variables(
            self.config["NN_structure"]["class_labels"]
        )
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("label_value is used twice in the used classes!")

        # Define a security check for using the jets twice
        # Check if the extended b labeling is used
        if 55 in class_ids or 54 in class_ids:

            # check if the b label is in class_ids and if its extended or not
            if 5 in class_ids:
                b_index = class_ids.index(5)

                if class_label_vars[b_index] == "HadronConeExclTruthLabelID":
                    raise ValueError(
                        "You defined default bjets and extended bjets"
                        " simultaneously using the simple flavour labelling"
                        " scheme! Please modify class_labels."
                    )

        # Check if the extended c labeling is used
        if 44 in class_ids:
            if 4 in class_ids:
                c_index = class_ids.index(4)

                if class_label_vars[c_index] == "HadronConeExclTruthLabelID":
                    raise ValueError(
                        "You defined default cjets and extended cjets"
                        " simultaneously using the simple flavour labelling"
                        " scheme! Please modify class_labels."
                    )
