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
from umami.helper_tools import get_class_label_ids
from umami.tf_tools import Sum
from umami.tools import yaml_loader


def GetParser():
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


def prepareConfig(yaml_config: str) -> dict:
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

    logger.info(f"Using config file {yaml_config}")
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

    # Assert that the needed variables are present and set.
    for var in needed_options:
        if not (var in config and config[var] is not None):
            raise ValueError(f"Needed option {var} is not given or is None!")

    return config


def load_model_umami(
    model_file: str,
    X_test_trk: np.ndarray,
    X_test_jet: np.ndarray,
):
    """Load umami model

    Parameters
    ----------
    model_file : str
        file name of the model to load
    X_test_trk : np.ndarray
        test array for tracks
    X_test_jet : np.ndarray
        test array for jets

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
        [X_test_trk, X_test_jet], batch_size=5000, verbose=0
    )

    return pred_dips, pred_umami


# workaround to not use the full preprocessing config
class minimal_preprocessing_config:
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
    args = GetParser()

    # Load the config file
    eval_config = prepareConfig(args.config)
    scale_dict = eval_config["scale_dict"]
    class_labels = eval_config["class_labels"]
    tracks_name = eval_config["tracks_name"]
    tagger = eval_config["tagger"]
    model_file = eval_config["model_file"]
    input_file = eval_config["input_file"]
    var_dict = eval_config["var_dict"]
    ntracks_max = eval_config["ntracks_max"]

    # Init the minimal preprocessing config for loading of the jets
    preprocess_config = minimal_preprocessing_config(
        scale_dict=scale_dict,
        class_labels=class_labels,
        tracks_name=tracks_name,
    )

    # Get the class ids for removing
    class_ids = get_class_label_ids(class_labels)

    logger.info(f"Evaluating {model_file}")

    # Load the input file
    with h5py.File(input_file, "r") as file:
        df = pd.DataFrame(file["jets"][:])

    # Remove all jets which are not trained on
    df.query(f"HadronConeExclTruthLabelID in {class_ids}", inplace=True)

    # Init a pred_model
    pred_model = None

    # Get prediction for umami
    if "umami" in tagger.casefold():
        X_test_jet, X_test_trk, Y_test = utt.GetTestFile(
            input_file,
            var_dict,
            preprocess_config,
            class_labels,
            tracks_name=tracks_name,
            nJets=int(10e6),
            exclude=None,
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")

        # Get the umami and dips predictions
        pred_dips, pred_umami = load_model_umami(model_file, X_test_trk, X_test_jet)
        pred_model = pred_dips if "dips" in tagger.casefold() else pred_umami

    # Get prediction for dips
    elif "dips" in tagger.casefold():
        X_test_trk, Y_test = utt.GetTestSampleTrks(
            input_file,
            var_dict,
            preprocess_config,
            class_labels,
            tracks_name=tracks_name,
            nJets=int(10e6),
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")

        # Load the model
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(model_file)

        # Predict the test sample with the loaded model
        pred_model = model.predict(X_test_trk, batch_size=5000, verbose=0)

    # Get prediction for dl1
    elif "dl1" in tagger.casefold():
        X_test_jet, Y_test = utt.GetTestSample(
            input_file,
            var_dict,
            preprocess_config,
            class_labels,
            nJets=int(10e6),
            exclude=None,
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")

        # Load the model
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(model_file)

        # Predict the test sample with the loaded model
        pred_model = model.predict(X_test_jet, batch_size=5000, verbose=0)

    if "dips" in tagger.casefold() or "umami" in tagger.casefold():
        trk_mask = np.sum(X_test_trk, axis=-1) != 0
        ntrks = trk_mask.sum(axis=1)
        df["ntrks"] = ntrks

    else:
        df["ntrks"] = -1 * np.ones(len(Y_test))

    for counter, key in enumerate(class_labels):
        # Get the probability short form of the class
        prob_key = global_config.flavour_categories[key]["prob_var_name"]

        # Add the evaluation probabilites to the dict
        df[f"eval_prob_{prob_key}"] = pred_model[:, counter]

    # Add an index to the dataframe
    df["index"] = range(len(df))

    # Add the truth to the dataframe
    df["y"] = np.argmax(Y_test, axis=1)
    logger.info(f"Jets: {len(df)}")

    # Get the first class defined in class labels to calculate the difference
    prob_key = global_config.flavour_categories[class_labels[0]]["prob_var_name"]
    evaluated = f"eval_prob_{prob_key}"

    # Calculate the difference between the lwtnn output and the keras model output
    df["diff"] = abs(df[evaluated] - df[f"{tagger}_{prob_key}"])

    # Define the difference regions
    sampleDiffs = np.array(
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
    for sampleDiff in sampleDiffs:
        df_select = df.query(f"diff>{sampleDiff} and ntrks<{ntracks_max}")
        diff = round(len(df_select) / len(df[df["ntrks"] < ntracks_max]) * 100, 2)
        print(f"Differences off {sampleDiff:.1e} {diff}%")
        if diff == 0:
            break

    if args.output is not None:
        df_select = df[
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
        logger.info(f"Writing output file {out_file}")
        df_select.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
