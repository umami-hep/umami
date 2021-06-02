from umami.configuration import logger  # isort:skip
import argparse
import os
import re

import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.preprocessing_tools as upt
import umami.train_tools as utt
import umami.validation_dumper_tools as uvdt


def GetParser():
    """Argument parser for evaluate dumper script."""
    parser = argparse.ArgumentParser(
        description="Evaluate dumper command line options."
    )

    parser.add_argument(
        "-c",
        "--dumper_comparison_config",
        type=str,
        required=True,
        help="""Name/Path to dumper_comparison_config.yaml""",
    )

    parser.add_argument(
        "-d",
        "--dips",
        action="store_true",
        help="Evaluating Dips tagger.",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="""Saving the probabilities from DL2-DIPS and
        normal DIPS in .csv""",
    )

    args = parser.parse_args()
    return args


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def EvaluateDumperDips(dc_config):
    """
    Evaluates the dips model applied inside
    of the dumper vs applied after dumping with
    trained dips model.
    """

    # Laod needed configs and lists
    train_config = utt.Configuration(dc_config.train_config)
    preprocess_config = upt.Configuration(dc_config.preprocess_config)
    var_dict = dc_config.var_dict
    test_file = dc_config.test_file
    model_file = dc_config.model_file

    if os.path.isdir(test_file) is True:
        test_file = list(absoluteFilePaths(test_file))

    # Load test file
    if isinstance(test_file, list):
        df_list = []
        for file in sorted(test_file, key=natural_keys):
            with h5py.File(file, "r") as file:
                df_list.append(pd.DataFrame(file["jets"][:]))

        df = pd.concat(df_list)
        df_list = 0

        X_test_trk_list = []
        Y_test_list = []

        for file in sorted(test_file, key=natural_keys):
            logger.info(f"Using {file}")
            # Get X and Y from test file
            X_test_trk_tmp, Y_test_tmp = utt.GetTestSampleTrks(
                file,
                var_dict,
                preprocess_config,
                nJets=int(10e6),
            )

            X_test_trk_list.append(X_test_trk_tmp)
            Y_test_list.append(Y_test_tmp)

        X_test_trk = np.vstack(X_test_trk_list)
        Y_test = np.vstack(Y_test_list)

        X_test_trk_list = 0
        Y_test_list = 0

    else:
        with h5py.File(test_file, "r") as file:
            df = pd.DataFrame(file["jets"][:])

        logger.info(f"Using {test_file}")
        # Get X and Y from test file
        X_test_trk, Y_test = utt.GetTestSampleTrks(
            test_file,
            var_dict,
            preprocess_config,
            nJets=int(10e6),
        )

    # Select truth labels
    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

    # Load pretrained dips model
    logger.info(f"Model used for evaluating {model_file}")
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    # Predict test jets
    pred_dips = model.predict(
        x=X_test_trk,
        batch_size=int(train_config.NN_structure["batch_size"]),
        verbose=0,
    )

    # Save predictions
    df["dips_eval_pu"] = pred_dips[:, :1]
    df["dips_eval_pc"] = pred_dips[:, 1:2]
    df["dips_eval_pb"] = pred_dips[:, 2:]

    # Setting y_true
    Y_true = np.argmax(Y_test, axis=1)

    # Save y_true
    df["Y_true"] = Y_true

    # Add index
    df["index"] = range(len(df))

    # Define boolean mask to filter placeholder tracks
    boolMask = (np.sum(X_test_trk, axis=-1) != 0).astype(bool)

    # Define the number of true tracks per jet as a mask
    nTrks = np.sum(boolMask, axis=-1)

    # Add nTrks to dataframe
    df["nTrks"] = nTrks

    logger.info(f"Number of Jets: {len(df)}")

    # Print and add to dataframe percent difference
    df_select = df.query("abs(dips_eval_pu-dips_pu)>1e-6")
    print(
        "Differences off 1e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query("abs(dips_eval_pu-dips_pu)>2e-6")
    print(
        "Differences off 2e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query("abs(dips_eval_pu-dips_pu)>3e-6")
    print(
        "Differences off 3e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query("abs(dips_eval_pu-dips_pu)>4e-6")
    print(
        "Differences off 4e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query("abs(dips_eval_pu-dips_pu)>5e-6")
    print(
        "Differences off 5e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query("abs(dips_eval_pu-dips_pu)>1e-5")
    print(
        "Differences off 1e-5", round(len(df_select) / len(df) * 100, 2), "%"
    )

    # Calcualte difference for all jets
    Diff = abs(df["dips_eval_pu"] - df["dips_pu"])

    # Set diff to df
    df["diff"] = Diff

    # Only save wanted variables
    df = df[
        [
            "diff",
            "dips_eval_pu",
            "dips_pu",
            "dips_eval_pc",
            "dips_pc",
            "dips_eval_pb",
            "dips_pb",
            "index",
            "nTrks",
            "Y_true",
        ]
    ]

    if args.save:
        # Create results folder to save csv
        models_dirname = os.path.dirname(model_file)
        ResultsDir = models_dirname + "/results"
        if not os.path.isdir(ResultsDir):
            os.makedirs(ResultsDir)

        # Define final filepath
        FilePath = ResultsDir + "/dumper_val.csv"

        # Save selected df to csv
        df.to_csv(FilePath)
        logger.info(f"File saved to {FilePath}")


if __name__ == "__main__":
    args = GetParser()
    dc_config = uvdt.Configuration(args.dumper_comparison_config)

    if args.dips:
        EvaluateDumperDips(dc_config=dc_config)
