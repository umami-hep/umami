import umami.preprocessing_tools as upt
import h5py
import numpy as np
import argparse
import yaml
from umami.tools import yaml_loader
import json


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Enter the name of the config file to create the"
                        "hybrid sample.")
    parser.add_argument('-t', '--tracks', action='store_true',
                        help="Stores also track information.")
    # parser.add_argument('-v', '--var_dict', required=True, default=None,
    #                     help="Dictionary with input variables of tagger.",
    #                     type=str)
    # possible job options for the different preprocessing steps
    # action = parser.add_mutually_exclusive_group(required=True)
    # action.add_argument('-u', '--undersampling', action='store_true',
    #                     help="Runs undersampling.")
    # # action.add_argument('--weighting', action='store_true',
    # #                     help="Runs weighting.")
    # action.add_argument('-s', '--scaling', action='store_true',
    #                     help="Retrieves scaling and shifting factors.")
    # action.add_argument('-a', '--apply_scales', action='store_true',
    #                     help="Apllies scaling and shifting factors.")
    # action.add_argument('-w', '--write', action='store_true',
    #                     help="Shuffles sample and writes training sample and"
    #                          "training labels to disk")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = GetParser()
    config = upt.Configuration(args.config_file)
