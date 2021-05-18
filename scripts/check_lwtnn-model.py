import argparse

import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.train_tools as utt
from umami.train_tools import Sum


def GetParser():
    """Argparse option for create_vardict script."""
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
        required=True,
        type=str,
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


def __run():
    args = GetParser()
    with h5py.File(args.input, "r") as file:
        df = pd.DataFrame(file["jets"][:])

    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
    preprocess_config = config(args.scale_dict)

    print("Evaluating", args.model)

    pred_model = None
    if "umami" in args.tagger.lower():
        X_test_jet, X_test_trk, Y_test = utt.GetTestFile(
            args.input,
            args.var_dict,
            preprocess_config,
            nJets=int(10e6),
            exclude=None,
        )
        print("Evaluated jets:", len(Y_test))
        pred_dips, pred_umami = load_model_umami(
            args.model, X_test_trk, X_test_jet
        )
        pred_model = pred_dips if "dips" in args.tagger.lower() else pred_umami

    elif "dips" in args.tagger.lower():
        X_test_trk, Y_test = utt.GetTestSampleTrks(
            args.input,
            args.var_dict,
            preprocess_config,
            nJets=int(10e6),
        )
        print("Evaluated jets:", len(Y_test))
        with CustomObjectScope({"Sum": Sum}):
            model = load_model(args.model)
        pred_model = model.predict(X_test_trk, batch_size=5000, verbose=0)

    elif "dl1" in args.tagger.lower():
        X_test_jet, Y_test = utt.GetTestSample(
            args.input,
            args.var_dict,
            preprocess_config,
            nJets=int(10e6),
            exclude=None,
        )
        print("Evaluated jets:", len(Y_test))

        with CustomObjectScope({"Sum": Sum}):
            model = load_model(args.model)
        pred_model = model.predict(X_test_jet, batch_size=5000, verbose=0)

    if "dips" in args.tagger.lower() or "umami" in args.tagger.lower():
        trk_mask = np.sum(X_test_trk, axis=-1) != 0
        ntrks = trk_mask.sum(axis=1)
        df["ntrks"] = ntrks
    else:
        df["ntrks"] = np.zeros(len(Y_test))

    df["eval_pu"] = pred_model[:, :1]
    df["eval_pc"] = pred_model[:, 1:2]
    df["eval_pb"] = pred_model[:, 2:]

    df["index"] = range(len(df))

    df["y"] = np.argmax(Y_test, axis=1)
    print("Jets:", len(df))

    evaluated = "eval_pu"
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>1e-6")
    print(
        "Differences off 1e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>2e-6")
    print(
        "Differences off 2e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>3e-6")
    print(
        "Differences off 3e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>4e-6")
    print(
        "Differences off 4e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>5e-6")
    print(
        "Differences off 5e-6", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>1e-5")
    print(
        "Differences off 1e-5", round(len(df_select) / len(df) * 100, 2), "%"
    )
    df_select = df.query(f"abs({evaluated}-{args.tagger}_pu)>1e-6")
    df_select["diff"] = abs(
        df_select[evaluated] - df_select[f"{args.tagger}_pu"]
    )
    df_select.sort_values("diff", ascending=False, inplace=True)

    if args.output is not None:
        df_select = df_select[
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
        # df_select.query("diff>1e-5", inplace=True)
        df_select.to_csv(f"{args.output}.csv")


if __name__ == "__main__":
    __run()
