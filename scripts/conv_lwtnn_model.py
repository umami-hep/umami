"""Script to convert hdf5 keras models to separate weights hdf5 and
architecture json files."""
import argparse

from tensorflow.keras.initializers import glorot_uniform  # pylint: disable=E0401
from tensorflow.keras.models import load_model  # pylint: disable=E0401
from tensorflow.keras.utils import CustomObjectScope  # pylint: disable=E0401

from umami.tf_tools import Sum


def get_parser():
    """
    Argparse option for conv_model script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="""Options for DL1 conv_model""",
    )

    parser.add_argument(
        "-m",
        "--model_file",
        required=True,
        type=str,
        help="""HDF5 keras model which should be converted.""",
    )
    parser.add_argument(
        "-o",
        "--output_base",
        default="lwtnn_model",
        type=str,
    )

    return parser.parse_args()


def __run():
    args = get_parser()
    with CustomObjectScope({"GlorotUniform": glorot_uniform(), "Sum": Sum}):
        model = load_model(args.model_file)
    # get the architecture as a json string
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(f"architecture-{args.output_base}.json", "w") as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(f"weights-{args.output_base}.h5")


if __name__ == "__main__":
    __run()
