import argparse

import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.preprocessing_tools as upt
import umami.train_tools as utt
from umami.configuration import logger
from umami.tf_tools import Sum


def GetParser():
    """Argparse option for create_vardict script.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Options lwtnn check""")

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="hdf5 input with taggers included for comparison.",
    )
    parser.add_argument(
        "-s",
        "--scale_dict",
        type=str,
        default=None,
        help="""scale_dict file containing scaling and shifting
                        values.""",
    )
    parser.add_argument(
        "-v",
        "--var_dict",
        required=True,
        type=str,
        help="""Dictionary (json) with training variables.""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="writes the jet scores into a file",
    )
    parser.add_argument(
        "-t",
        "--tagger",
        type=str,
        required=True,
        help="tagger shortcut, corresponding to the name in the ntuples",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Keras model for comparison.",
    )

    parser.add_argument(
        "-n",
        "--ntracks_max",
        type=int,
        default=np.inf,
        help="Number of tracks per jet to ignore.",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Training Config yaml file.",
    )

    return parser.parse_args()


def load_model_umami(model_file, X_test_trk, X_test_jet):
    with CustomObjectScope({"Sum": Sum}):
        model = load_model(model_file)
    pred_dips, pred_umami = model.predict(
        [X_test_trk, X_test_jet], batch_size=5000, verbose=0
    )

    return pred_dips, pred_umami


# workaround to not use the full preprocessing config
class config:
    def __init__(self, preprocess_config):
        self.dict_file = preprocess_config
        self.preparation = {"class_labels": ["ujets", "cjets", "bjets"]}


def __run():
    args = GetParser()
    logger.info(f"Opening input file {args.input}")
    with h5py.File(args.input, "r") as file:
        df = pd.DataFrame(file["jets"][:])

    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

    if args.config is not None:
        if args.scale_dict is not None:
            raise ValueError(
                "Both --confing and --scale_dict options were given, "
                "only one of them needs to be used"
            )
        training_config = utt.Configuration(args.config)
        preprocess_config = upt.Configuration(
            training_config.preprocess_config
        )
        class_labels = training_config.NN_structure["class_labels"]
    elif args.scale_dict is not None:
        preprocess_config = config(args.scale_dict)
        class_labels = preprocess_config.preparation["class_labels"]
    else:
        raise ValueError(
            "Missing option, either --config or --scale_dict "
            "needs to be specified (only one of them)"
        )

    logger.info(f"Evaluating {args.model}")

    pred_model = None
    if "umami" in args.tagger.lower():
        X_test_jet, X_test_trk, Y_test = utt.GetTestFile(
            args.input,
            args.var_dict,
            preprocess_config,
            class_labels,
            nJets=int(10e6),
            exclude=None,
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")
        pred_dips, pred_umami = load_model_umami(
            args.model, X_test_trk, X_test_jet
        )
        pred_model = pred_dips if "dips" in args.tagger.lower() else pred_umami

    elif "dips" in args.tagger.lower():
        X_test_trk, Y_test = utt.GetTestSampleTrks(
            args.input,
            args.var_dict,
            preprocess_config,
            class_labels,
            nJets=int(10e6),
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(args.model)
        pred_model = model.predict(X_test_trk, batch_size=5000, verbose=0)

    elif "dl1" in args.tagger.lower():
        X_test_jet, Y_test = utt.GetTestSample(
            args.input,
            args.var_dict,
            preprocess_config,
            class_labels,
            nJets=int(10e6),
            exclude=None,
        )
        logger.info(f"Evaluated jets: {len(Y_test)}")

        with CustomObjectScope({"Sum": Sum}):
            model = load_model(args.model)
        pred_model = model.predict(X_test_jet, batch_size=5000, verbose=0)

    if "dips" in args.tagger.lower() or "umami" in args.tagger.lower():
        trk_mask = np.sum(X_test_trk, axis=-1) != 0
        ntrks = trk_mask.sum(axis=1)
        df["ntrks"] = ntrks
    else:
        df["ntrks"] = -1 * np.ones(len(Y_test))

    df["eval_pu"] = pred_model[:, :1]
    df["eval_pc"] = pred_model[:, 1:2]
    df["eval_pb"] = pred_model[:, 2:3]

    df["index"] = range(len(df))

    df["y"] = np.argmax(Y_test, axis=1)
    logger.info(f"Jets: {len(df)}")

    evaluated = "eval_pu"
    df["diff"] = abs(df[evaluated] - df[f"{args.tagger}_pu"])
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
    for sampleDiff in sampleDiffs:
        df_select = df.query(f"diff>{sampleDiff} and ntrks<{args.ntracks_max}")
        diff = round(
            len(df_select) / len(df[df["ntrks"] < args.ntracks_max]) * 100, 2
        )
        print(f"Differences off {'{:.1e}'.format(sampleDiff)} {diff}%")
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
    __run()
