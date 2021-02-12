"""Script to convert hdf5 keras models to separate weights hdf5 and
architecture json files."""
import argparse

from keras.initializers import glorot_uniform
from keras.models import load_model
from keras.utils import CustomObjectScope

from umami.train_tools import Sum


def GetParser():
    """Argparse option for conv_model script."""
    parser = argparse.ArgumentParser(
        description="""Options for DL1
                                     conv_model"""
    )

    parser.add_argument(
        "-m",
        "--model_file",
        required=True,
        type=str,
        help="""HDF5 keras model which should be converted.""",
    )
    parser.add_argument("-o", "--output_base", type=str, required=True)

    return parser.parse_args()


def __run():
    args = GetParser()
    with CustomObjectScope({"GlorotUniform": glorot_uniform(), "Sum": Sum}):
        model = load_model(args.model_file)
    # get the architecture as a json string
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open("architecture-%s.json" % args.output_base, "w") as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights("weights-%s.h5" % args.output_base)


if __name__ == "__main__":
    __run()
