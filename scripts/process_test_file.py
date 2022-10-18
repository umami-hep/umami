"""
This script processes an inclusive test file to scale,
 shift it and provide a ready
output which can be used for evaluating a tagger.
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from umami.configuration import logger
from umami.train_tools import get_test_file


def get_parser():
    """Argparse option for compute_workingpoint script.

    Returns
    -------
    args: parse_args
    """
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("-v", "--var_dict")
    parser.add_argument("-s", "--scale_dict")
    parser.add_argument("-t", "--tracks_name")
    parser.add_argument("-n", "--n_jets", default=1_000_000, type=int)
    parser.add_argument(
        "-c", "--class_labels", nargs="+", default=["ujets", "cjets", "bjets"]
    )
    parser.add_argument("-o", "--out_file")
    return parser.parse_args()


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
    """Process test file and write it to disk."""
    args = get_parser()
    logger.info("Start processing")

    X_test_jet, X_test_trk, Y_test = get_test_file(
        input_file=args.input_file,
        var_dict=args.var_dict,
        scale_dict=args.scale_dict,
        class_labels=args.class_labels,
        tracks_name=args.tracks_name,
        n_jets=int(args.n_jets),
    )
    logger.info("Shape of X_test_jet: %s", np.shape(X_test_jet))
    logger.info("Shape of X_test_trk: %s", np.shape(X_test_trk))
    logger.info("Shape of Y_test: %s", np.shape(Y_test))
    # TODO: also add track labels

    logger.info("Writing file %s ...", args.out_file)
    with h5py.File(args.out_file, "w") as f_h5:
        f_h5.create_dataset("jets/inputs", data=X_test_jet)
        f_h5.create_dataset(f"{args.tracks_name}/inputs", data=X_test_trk)
        f_h5.create_dataset("jets/labels_one_hot", data=Y_test)
    logger.info("Saved file %s ", args.out_file)


if __name__ == "__main__":
    main()
