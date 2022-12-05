"""Script to validate an lwtnn model between output of keras and from the TDD."""
import argparse

import h5py
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.models import load_model  # pylint: disable=E0401
from tensorflow.keras.utils import CustomObjectScope  # pylint: disable=E0401

import umami.train_tools as utt
from umami.configuration import global_config, logger
from umami.data_tools import retrieve_cut_string
from umami.tf_tools import Sum
from umami.tools import yaml_loader


def get_parser():
    """
    Argparse option for create_vardict script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="""Options lwtnn check""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="writes the jet scores into a file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config yaml file.",
    )

    return parser.parse_args()


def prepare_config(yaml_config: str) -> dict:
    """
    Load the config for checking the model
    and return the values in a dict.

    Parameters
    ----------
    yaml_config : str
        Path to the yaml config file

    Returns
    -------
    dict
        All parameters needed for checking the lwtnn model.

    Raises
    ------
    ValueError
        If one of the needed options is not given or is None.
    """

    logger.info("Using config file %s", yaml_config)
    with open(yaml_config, "r") as conf:
        config = yaml.load(conf, Loader=yaml_loader)

    # Init a list of the needed inputs
    needed_options = [
        "input_file",
        "scale_dict",
        "var_dict",
        "tagger",
        "model_file",
        "class_labels",
    ]

    # If tracks are used, assert that all needed track settings are present and set
    if "dips" in config["tagger"].casefold() or "umami" in config["tagger"].casefold():
        needed_options.append("ntracks_max")
        needed_options.append("tracks_name")

    # If no tracks are used, set placeholder values.
    else:
        config["ntracks_max"] = np.inf
        config["tracks_name"] = None

    # Check for batch size
    if "batch_size" not in config or config["batch_size"] is None:
        config["batch_size"] = 5000
        logger.warning("No batch size for evaluation was given. Using default 5000.")

    elif isinstance(config["batch_size"], float):
        config["batch_size"] = int(config["batch_size"])
        logger.warning("Batch size was given as float. Converted it to int.")

    # Assert that the needed variables are present and set.
    for var in needed_options:
        if not (var in config and config[var] is not None):
            raise ValueError(f"Needed option {var} is not given or is None!")

    return config


def load_model_umami(
    model_file: str,
    x_test_trk: np.ndarray,
    x_test_jet: np.ndarray,
    batch_size: int = 5000,
):
    """Load umami model

    Parameters
    ----------
    model_file : str
        file name of the model to load
    x_test_trk : np.ndarray
        test array for tracks
    x_test_jet : np.ndarray
        test array for jets
    batch_size : int, optional
        Number of jets used per batch for
        evaluation. By default 5000.

    Returns
    -------
    np.ndarray
        dips predictions
    np.ndarray
        umami predictions
    """
    with CustomObjectScope({"Sum": Sum}):
        model = load_model(model_file)
    pred_dips, pred_umami = model.predict(
        [x_test_trk, x_test_jet],
        batch_size=batch_size,
        verbose=0,
    )

    return pred_dips, pred_umami


# workaround to not use the full preprocessing config
class MinimalPreprocessingConfig:
    """
    Minimal implementation of preprocessing config. Sets a few
    values which are needed here.
    """

    def __init__(
        self,
        scale_dict: str,
        class_labels: list,
        tracks_name: str = None,
    ):
        """
        Initalise the minimal preprocessing config so the loading
        of the files is done correctly

        Parameters
        ----------
        scale_dict : str
            Path to the used scale dict.
        class_labels : list
            Class labels that where used in training the tagger.
        tracks_name : str, optional
            Track name inside the h5 files.
        """
        self.dict_file = scale_dict
        self.sampling = {"class_labels": class_labels}
        self.tracks_name = tracks_name


def main():
    """
    Main function is called when executing the script.
    """

    # Get the arguments
    args = get_parser()

    # Load the config file
    eval_config = prepare_config(args.config)
    scale_dict = eval_config["scale_dict"]
    class_labels = eval_config["class_labels"]
    tracks_name = eval_config["tracks_name"]
    tagger = eval_config["tagger"]
    model_file = eval_config["model_file"]
    input_file = eval_config["input_file"]
    var_dict = eval_config["var_dict"]
    ntracks_max = eval_config["ntracks_max"]
    batch_size = eval_config["batch_size"]

    # Get the class ids for removing
    cut_strings = retrieve_cut_string(class_labels)

    logger.info("Evaluating %s", model_file)

    # Load the input file
    with h5py.File(input_file, "r") as file:
        df_in = pd.DataFrame(file["jets"][:])

    # Remove all jets which are not trained on
    df_in.query(
        "|".join([cut_strings[class_label] for class_label in class_labels]),
        inplace=True,
    )

    # Init a pred_model
    pred_model = None

    # Get prediction for umami
    if "umami" in tagger.casefold():
        x_test_jet, x_test_trk, y_test = utt.get_test_file(
            input_file=input_file,
            var_dict=var_dict,
            scale_dict=scale_dict,
            class_labels=class_labels,
            tracks_name=tracks_name,
            n_jets=int(10e6),
            exclude=None,
        )
        logger.info("Evaluated jets: %i", len(y_test))

        # Get the umami and dips predictions
        pred_dips, pred_umami = load_model_umami(
            model_file,
            x_test_trk,
            x_test_jet,
            batch_size=batch_size,
        )
        pred_model = pred_dips if "dips" in tagger.casefold() else pred_umami

    # Get prediction for dips
    elif "dips" in tagger.casefold():
        x_test_trk, y_test = utt.get_test_sample_trks(
            input_file=input_file,
            var_dict=var_dict,
            scale_dict=scale_dict,
            class_labels=class_labels,
            tracks_name=tracks_name,
            n_jets=int(10e6),
        )
        logger.info("Evaluated jets: %i", len(y_test))

        # Load the model
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(model_file)

        # Predict the test sample with the loaded model
        pred_model = model.predict(
            x_test_trk,
            batch_size=batch_size,
            verbose=0,
        )

    # Get prediction for dl1
    elif "dl1" in tagger.casefold():
        x_test_jet, y_test = utt.get_test_sample(
            input_file=input_file,
            var_dict=var_dict,
            scale_dict=scale_dict,
            class_labels=class_labels,
            n_jets=int(10e6),
            exclude=None,
        )
        logger.info("Evaluated jets: %i", len(y_test))

        # Load the model
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(model_file)

        # Predict the test sample with the loaded model
        pred_model = model.predict(
            x_test_jet,
            batch_size=batch_size,
            verbose=0,
        )

    if "dips" in tagger.casefold() or "umami" in tagger.casefold():
        trk_mask = np.sum(x_test_trk, axis=-1) != 0
        ntrks = trk_mask.sum(axis=1)
        df_in["ntrks"] = ntrks

    else:
        df_in["ntrks"] = -1 * np.ones(len(y_test))

    for counter, key in enumerate(class_labels):
        # Get the probability short form of the class
        prob_key = global_config.flavour_categories[key]["prob_var_name"]

        # Add the evaluation probabilites to the dict
        df_in[f"eval_prob_{prob_key}"] = pred_model[:, counter]

    # Add an index to the dataframe
    df_in["index"] = range(len(df_in))

    # Add the truth to the dataframe
    df_in["y"] = np.argmax(y_test, axis=1)
    logger.info("Jets: %i", len(df_in))

    # Get the first class defined in class labels to calculate the difference
    prob_key = global_config.flavour_categories[class_labels[0]]["prob_var_name"]
    evaluated = f"eval_prob_{prob_key}"

    # Calculate the difference between the lwtnn output and the keras model output
    df_in["diff"] = abs(df_in[evaluated] - df_in[f"{tagger}_{prob_key}"])

    # Define the difference regions
    sample_diffs = np.array(
        [
            np.linspace(1e-6, 5e-6, 5),
            np.linspace(1e-5, 5e-5, 5),
            np.linspace(1e-4, 5e-4, 5),
            np.linspace(1e-3, 5e-3, 5),
            np.linspace(1e-2, 5e-2, 5),
            np.linspace(1e-1, 5e-1, 5),
        ]
    ).flatten()

    # Iterate over the different difference regions
    for sample_diff in sample_diffs:
        df_select = df_in.query(f"diff>{sample_diff} and ntrks<{ntracks_max}")
        diff = round(len(df_select) / len(df_in[df_in["ntrks"] < ntracks_max]) * 100, 2)
        print(f"Differences off {sample_diff:.1e} {diff}%")
        if diff == 0:
            break

    if args.output is not None:
        df_select = df_in[
            [
                "diff",
                "eval_pu",
                f"{args.tagger}_pu",
                "eval_pc",
                f"{args.tagger}_pc",
                "eval_pb",
                f"{args.tagger}_pb",
                "index",
                "ntrks",
            ]
        ]
        df_select = df_select.copy()
        df_select.sort_values("diff", ascending=False, inplace=True)
        out_file = f"{args.output}.csv"
        logger.info("Writing output file %s", out_file)
        df_select.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
