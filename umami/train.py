import argparse

import tensorflow as tf

import umami.models as utm
import umami.preprocessing_tools as upt
import umami.train_tools as utt


def GetParser():
    """Argument parser for the train executable."""
    parser = argparse.ArgumentParser(description="Train command line options.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to the training config file",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, help="Number of training epochs."
    )

    parser.add_argument(
        "-o",
        "--overwrite_config",
        action="store_true",
        help="Overwrite the configs files saved in metadata folder",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get the args from parser
    args = GetParser()

    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get the train and preprocess config
    train_config = utt.Configuration(args.config_file)
    preprocess_config = upt.Configuration(train_config.preprocess_config)

    # Get the tagger which is to be trained from the train config
    tagger_name = train_config.NN_structure["tagger"]

    # Create the metadatafolder
    utt.create_metadata_folder(
        train_config_path=args.config_file,
        var_dict_path=train_config.var_dict,
        model_name=train_config.model_name,
        preprocess_config_path=train_config.preprocess_config,
        overwrite_config=True if args.overwrite_config else False,
    )

    # Check for DIPS
    # TODO: Switch to case syntax with python 3.10
    if tagger_name == "dips":
        utm.Dips(
            args=args,
            train_config=train_config,
            preprocess_config=preprocess_config,
        )

    elif tagger_name == "dl1":
        utm.TrainLargeFile(
            args=args,
            train_config=train_config,
            preprocess_config=preprocess_config,
        )

    elif tagger_name == "umami":
        utm.Umami(
            args=args,
            train_config=train_config,
            preprocess_config=preprocess_config,
        )

    elif tagger_name == "dips_cond_att":
        utm.DipsCondAtt(
            args=args,
            train_config=train_config,
            preprocess_config=preprocess_config,
        )

    else:
        raise ValueError(
            f"""
            Tagger {tagger_name} is not supported! Possible taggers are
            dips, dl1, umami and dips_cond_att!
            """
        )
