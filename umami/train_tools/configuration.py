"""Configuration module for NN trainings."""
from dataclasses import dataclass, field

import numpy as np
import pydash
import yaml

from umami.configuration import logger
from umami.data_tools import get_cut_list
from umami.plotting_tools.utils import translate_kwargs
from umami.preprocessing_tools import PreprocessConfiguration
from umami.tools import yaml_loader


def check_option_defintion(
    variable_to_check,
    variable_name: str,
    needed_type: list,
    check_for_nan: bool,
) -> None:
    """
    Check if the given variable is correctly defined.

    Parameters
    ----------
    variable_to_check
        Variable which is to be checked.
    variable_name : str
        Name of the variable (for logger).
    needed_type : list
        List of allowed types for the variable.
    check_for_nan : bool
        Bool, if the variable needs to be set (True) or if a NaN
        value is also allowed (False).

    Raises
    ------
    ValueError
        If you havn't/wronly defined a variable which is needed.
    """
    # Check that needed_types is a list to loop over
    if not isinstance(needed_type, list):
        needed_type = [needed_type]

    # If given values is a int, continue
    if type(variable_to_check) in needed_type:
        return

    # Check case where the given type is string but you need a list
    if (
        isinstance(variable_to_check, str)
        and list in needed_type
        and str not in needed_type
    ):
        variable_to_check = [variable_to_check]

    # If a flaot value was found but it should be int
    elif (
        isinstance(variable_to_check, float)
        and int in needed_type
        and float not in needed_type
    ):
        logger.warning(
            "You defined a float for %s! Translating to int value %s",
            variable_name,
            int(variable_to_check),
        )
        variable_to_check = int(variable_to_check)

    else:
        # Check if the value is allowed to be None
        if check_for_nan is False and variable_to_check is None:
            return

        # Raise error for all other cases
        raise ValueError(
            f"You havn't/wrongly defined {variable_name}! "
            f"You gave a {type(variable_to_check)} but it should be one of these types "
            f"{needed_type}. Please define this correctly!"
        )


@dataclass
class TrainConfigurationObject:
    """Dataclass for the global train config options."""

    # Global needed options
    model_name: str = None
    preprocess_config: str = None
    evaluate_trained_model: bool = True
    tracks_name: str = None

    # File options
    train_file: str = None
    validation_files: dict = None
    test_files: dict = None

    # Training specific options
    model_file: str = None
    continue_training: bool = False
    exclude: list = None

    def __post_init__(self):
        """
        Process options and perform checks on them.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.model_name, "model_name", str, True),
            (self.evaluate_trained_model, "evaluate_trained_model", bool, True),
            (self.tracks_name, "tracks_name", str, False),
            (self.validation_files, "validation_files", dict, False),
            (self.test_files, "test_files", dict, False),
            (self.model_file, "model_file", str, False),
            (self.continue_training, "continue_training", bool, True),
            (self.exclude, "exclude", list, False),
        ]

        if self.evaluate_trained_model:
            needed_args += [
                (self.preprocess_config, "preprocess_config", str, True),
                (self.train_file, "train_file", str, True),
            ]

        # Check option definition
        for iter_var in needed_args:
            check_option_defintion(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )

        if self.evaluate_trained_model:
            # Load the preprocessing config and var dict
            self.preprocess_config = PreprocessConfiguration(self.preprocess_config)
            self.var_dict = self.preprocess_config.var_file

        # Setting the tracks key
        self.tracks_key = f"{self.tracks_name}/inputs" if self.tracks_name else None


@dataclass
class NNStructureConfig:
    """Dataclass for the nn_structure config options."""

    # General options
    evaluate_trained_model: bool = True
    class_labels: list = None
    tagger: str = None
    batch_size: int = None
    learning_rate: float = None
    epochs: int = None
    main_class: str = None
    n_jets_train: int = None
    load_optimiser: bool = False
    use_sample_weights: bool = False
    nfiles_tfrecord: int = 5

    # Used for all models
    batch_normalisation: bool = False

    # Learning rate reducer options
    lrr: bool = False
    lrr_monitor: str = "loss"
    lrr_factor: float = 0.8
    lrr_patience: int = 3
    lrr_verbose: int = 1
    lrr_mode: str = "auto"
    lrr_cooldown: int = 5
    lrr_min_learning_rate: float = 0.000001

    # Both DIPS and DL1*
    dense_sizes: list = None
    dropout_rate: list = None

    # DIPS specific
    ppm_sizes: list = None

    # DL1 specific
    activations: list = None
    repeat_end: list = None
    dl1_units: list = None

    # Umami specific
    dropout_rate_phi: list = None
    dropout_rate_f: list = None
    dips_ppm_units: list = None
    dips_dense_units: list = None
    dips_loss_weight: float = None
    intermediate_units: list = None

    # Umami and Umami Conditional Attention specific
    dips_ppm_condition: bool = None

    # CADS specific
    ppm_condition: bool = None

    # Conditional attention specific (Umami Cond Att, CADS)
    pooling: str = "attention"
    attention_sizes: list = None
    attention_condition: bool = None
    n_conditions: int = 0
    dense_condition: bool = None

    def __post_init__(self):
        """Process options and perform checks on them."""
        # Get all the variables that need to be defined
        self.check_options()
        self.check_class_labels()

    def check_options(self):
        """
        Check the given options for wrongly/mis-defined values/types. Also
        check that all needed variables are set.

        Raises
        ------
        ValueError
            If the given tagger is not supported.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.class_labels, "class_labels", list, True),
            (self.tagger, "tagger", str, True),
            (self.main_class, "main_class", [str, list], True),
        ]

        if self.evaluate_trained_model:
            needed_args += [
                (self.learning_rate, "learning_rate", float, True),
                (self.epochs, "epochs", int, True),
                (self.batch_size, "batch_size", int, True),
                (self.n_jets_train, "n_jets_train", int, False),
                (self.batch_normalisation, "batch_normalisation", bool, True),
                (self.dropout_rate, "dropout_rate", list, False),
                (self.load_optimiser, "load_optimiser", bool, True),
                (self.use_sample_weights, "use_sample_weights", bool, True),
                (self.nfiles_tfrecord, "nfiles_tfrecord", int, True),
                (self.lrr, "lrr", bool, True),
                (self.lrr_monitor, "lrr_monitor", str, True),
                (self.lrr_factor, "lrr_factor", float, True),
                (self.lrr_patience, "lrr_patience", int, True),
                (self.lrr_verbose, "lrr_verbose", int, True),
                (self.lrr_mode, "lrr_mode", str, True),
                (self.lrr_cooldown, "lrr_cooldown", int, True),
                (self.lrr_min_learning_rate, "lrr_min_learning_rate", float, True),
            ]

            if self.tagger.casefold() == "dips":
                needed_args += [
                    (self.ppm_sizes, "ppm_sizes", list, True),
                    (self.dropout_rate_phi, "dropout_rate_phi", list, False),
                    (self.dense_sizes, "dense_sizes", list, True),
                ]

            elif self.tagger.casefold() == "dips_attention":
                needed_args += [
                    (self.pooling, "pooling", str, True),
                    (self.attention_sizes, "attention_sizes", list, True),
                    (self.dense_sizes, "dense_sizes", list, True),
                ]

            elif self.tagger.casefold() == "cads":
                needed_args += [
                    (self.ppm_condition, "ppm_condition", bool, True),
                    (self.dense_condition, "dense_condition", bool, True),
                    (self.n_conditions, "n_conditions", int, True),
                    (self.pooling, "pooling", str, True),
                    (self.attention_sizes, "attention_sizes", list, True),
                    (self.attention_condition, "attention_condition", bool, True),
                    (self.dense_sizes, "dense_sizes", list, True),
                ]

            elif self.tagger.casefold() == "dl1":
                needed_args += [
                    (self.activations, "activations", list, True),
                    (self.repeat_end, "repeat_end", list, False),
                    (self.dense_sizes, "dense_sizes", list, True),
                ]

            elif self.tagger.casefold() in ["umami", "umami_cond_att"]:
                needed_args += [
                    (self.dips_ppm_units, "dips_ppm_units", list, True),
                    (self.dips_dense_units, "dips_dense_units", list, True),
                    (self.dl1_units, "dl1_units", list, True),
                    (self.dips_loss_weight, "dips_loss_weight", [float, int], True),
                    (self.intermediate_units, "intermediate_units", list, True),
                    (self.dropout_rate_phi, "dropout_rate_phi", list, False),
                    (self.dropout_rate_f, "dropout_rate_f", list, False),
                ]

                if self.tagger.casefold() == "umami_cond_att":
                    needed_args += [
                        (self.dips_ppm_condition, "dips_ppm_condition", bool, True),
                        (self.dense_condition, "dense_condition", bool, True),
                        (self.n_conditions, "n_conditions", int, True),
                        (self.pooling, "pooling", str, True),
                        (self.attention_sizes, "attention_sizes", list, True),
                        (self.attention_condition, "attention_condition", bool, True),
                    ]

            else:
                raise ValueError(f"Tagger {self.tagger} is not supported!")

        # Check option definition
        for iter_var in needed_args:
            check_option_defintion(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )

    def check_n_conditions(self):
        """
        Check that the n_conditions are correctly defined together with
        the tagger.

        Raises
        ------
        ValueError
            If n_conditions is == 0 and the tagger is cads.
        ValueError
            If n_conditions is != 0 and the tagger is dips_attention.
        """
        # Check cads and dips_attention n_conditions
        if self.tagger.casefold() == "cads" and self.n_conditions == 0:
            raise ValueError(
                "You defined tagger as CADS but with n_conditions = 0! "
                "Please change the n_conditions or switch the tagger to dips_attention"
            )

        if self.tagger.casefold() == "dips_attention" and self.n_conditions != 0:
            raise ValueError(
                "You defined tagger as dips_attention but with n_conditions != 0! "
                "Please change the n_conditions or switch the tagger to CADS"
            )

    def check_class_labels(self):
        """
        Check the class_labels that no jet is used twice or a wrongly
        defined class_labels list is used.

        Raises
        ------
        ValueError
            If the label value was used twice.
        ValueError
            When the default and extended bjets are defined.
        ValueError
            When the default and extended cjets are defined.
        """
        # Define a security to check if label_value is used twice
        class_cuts = get_cut_list(self.class_labels)
        class_cuts_flatten = np.hstack((np.array(cuts) for cuts in class_cuts.values()))

        if "HadronConeExclTruthLabelID == 5" in class_cuts_flatten and (
            "HadronConeExclExtendedTruthLabelID == 55" in class_cuts_flatten
            or "HadronConeExclExtendedTruthLabelID == 54" in class_cuts_flatten
            or ("HadronConeExclExtendedTruthLabelID == [5, 54]" in class_cuts_flatten)
        ):
            raise ValueError(
                "You defined default bjets and extended bjets"
                " simultaneously using the simple flavour labelling"
                " scheme! Please modify class_labels."
            )
        if "HadronConeExclTruthLabelID == 4" in class_cuts_flatten and (
            "HadronConeExclExtendedTruthLabelID == 44" in class_cuts_flatten
            or "HadronConeExclExtendedTruthLabelID == [4, 44]" in class_cuts_flatten
        ):
            raise ValueError(
                "You defined default cjets and extended cjets"
                " simultaneously using the simple flavour labelling"
                " scheme! Please modify class_labels."
            )


@dataclass
class ValidationSettingsConfig:
    """Dataclass for the validation_settings config options."""

    n_jets: int = None
    working_point: float = None
    plot_datatype: str = "pdf"
    taggers_from_file: dict = None
    tagger_label: str = None
    trained_taggers: dict = None
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = "$\\sqrt{s}=13$ TeV, PFlow jets"
    val_batch_size: int = None

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
            "taggers_from_file",
            "trained_taggers",
            "tagger_label",
            "working_point",
        ]
        plot_arguments = pydash.omit(self, omit_args)
        return translate_kwargs(plot_arguments)

    def __post_init__(self):
        """
        Process options and perform checks on them.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.n_jets, "n_jets", int, True),
            (self.working_point, "working_point", float, True),
            (self.plot_datatype, "plot_datatype", str, True),
            (self.taggers_from_file, "taggers_from_file", dict, False),
            (self.tagger_label, "tagger_label", str, True),
            (self.trained_taggers, "trained_taggers", dict, False),
            (self.use_atlas_tag, "use_atlas_tag", bool, True),
            (self.atlas_first_tag, "atlas_first_tag", str, True),
            (self.atlas_second_tag, "atlas_second_tag", str, True),
            (self.val_batch_size, "val_batch_size", int, False),
        ]

        # Check option definition
        for iter_var in needed_args:
            check_option_defintion(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )


@dataclass
class EvaluationSettingsConfig:
    """Dataclass for the evaluation_settings config options."""

    evaluate_traind_model: bool = True
    results_filename_extension: str = ""
    n_jets: int = None
    tagger: list = field(default_factory=list)
    frac_values_comp: dict = field(default_factory=dict)
    frac_values: dict = field(default_factory=dict)
    working_point: float = None
    eff_min: float = 0.49
    eff_max: float = 1.0
    frac_step: float = 0.01
    frac_min: float = 0.01
    frac_max: float = 1.0
    add_eval_variables: list = field(default_factory=list)
    eval_batch_size: int = None
    extra_classes_to_evaluate: list = field(default_factory=list)
    shapley: dict = field(default_factory=dict)
    calculate_saliency: bool = False
    saliency_ntrks: int = 8
    saliency_effs: list = None
    x_axis_granularity: int = 100

    def __post_init__(self):
        """
        Process options and perform checks on them.
        """

        # List of tuples for check. Each tuple contains:
        # The variable, the name of the variable, which type the variable should be
        # and if the variable needs to be set (True if None is not ok, False if
        # None is ok)
        needed_args = [
            (self.n_jets, "n_jets", int, True),
            (self.working_point, "working_point", float, True),
            (self.eval_batch_size, "eval_batch_size", int, False),
            (self.results_filename_extension, "results_filename_extension", str, False),
            (self.eff_min, "eff_min", float, True),
            (self.eff_max, "eff_max", float, True),
            (self.frac_step, "frac_step", float, True),
            (self.frac_min, "frac_min", float, True),
            (self.frac_max, "frac_max", float, True),
            (self.add_eval_variables, "add_eval_variables", list, False),
            (self.extra_classes_to_evaluate, "extra_classes_to_evaluate", list, False),
            (self.shapley, "shapley", dict, False),
            (self.calculate_saliency, "calculate_saliency", bool, False),
            (self.saliency_ntrks, "saliency_ntrks", int, False),
            (self.saliency_effs, "saliency_effs", list, False),
            (self.x_axis_granularity, "x_axis_granularity", int, True),
        ]

        if self.evaluate_traind_model:
            needed_args += [
                (self.frac_values, "frac_values", dict, True),
                (self.tagger, "tagger", list, False),
                (self.frac_values_comp, "frac_values_comp", dict, False),
            ]

        else:
            needed_args += [
                (self.tagger, "tagger", list, True),
                (self.frac_values_comp, "frac_values_comp", dict, True),
            ]

        # Check option definition
        for iter_var in needed_args:
            check_option_defintion(
                variable_to_check=iter_var[0],
                variable_name=iter_var[1],
                needed_type=iter_var[2],
                check_for_nan=iter_var[3],
            )


class TrainConfiguration:
    """docstring for Configuration."""

    def __init__(self, yaml_config: str = None, **kwargs):
        super().__init__(**kwargs)
        self.yaml_config = yaml_config
        self.config = {}
        self.yaml_default_config = "configs/default_train_config.yaml"
        self.load_config_file()
        self.get_configuration()

    def load_config_file(self):
        """Load config file from disk."""
        logger.info("Using train config file %s", self.yaml_config)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def get_configuration(self):
        """Initalise the dataclasses and check the values."""
        # Get all the config keys
        config_keys = list(self.config.keys())

        # Remove all anchors from the config (not needed once loaded)
        for key in config_keys:
            if key.startswith("."):
                self.config.pop(key)

        # Arguments in the config file which are not general
        non_general_args = [
            "nn_structure",
            "validation_settings",
            "evaluation_settings",
        ]

        # Define the general settings
        self.general = TrainConfigurationObject(
            **pydash.omit(self.config, non_general_args)
        )

        # Define the nn_structure settings
        self.nn_structure = NNStructureConfig(
            self.general.evaluate_trained_model,
            **self.config["nn_structure"],
        )

        # Check for validation_settings
        if self.config["validation_settings"]:

            # Define the validation_settings
            self.validation_settings = ValidationSettingsConfig(
                **self.config["validation_settings"]
            )

            # Check if validation batch size is set
            if self.validation_settings.val_batch_size is None:
                logger.warning(
                    "No val_batch_size defined. Using training batch size for"
                    " validation"
                )
                self.validation_settings.val_batch_size = self.nn_structure.batch_size

        # Check for evaluation_settings
        if self.config["evaluation_settings"]:

            # Define the evaluation_settings
            self.evaluation_settings = EvaluationSettingsConfig(
                self.general.evaluate_trained_model,
                **self.config["evaluation_settings"],
            )

            # Check if evaluation batch size is set
            if self.evaluation_settings.eval_batch_size is None:
                logger.warning(
                    "No eval_batch_size defined. Using training batch size for"
                    " evaluation"
                )
                self.evaluation_settings.eval_batch_size = self.nn_structure.batch_size
