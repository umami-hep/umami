import h5py
import numpy as np
import yaml
from numpy.lib.recfunctions import repack_fields, structured_to_unstructured

from umami.configuration import logger
from umami.tools import yaml_loader


class TrainSampleWriter:
    def __init__(self, config, compression="gzip") -> None:
        """
        Init the needed configs and variables

        Input:
        - config: Loaded config file for the preprocessing.
        - compression: Type of compression which should be used.
                       Default: gzip
        """
        self.config = config
        self.bool_use_tracks = config.sampling["options"]["save_tracks"]
        self.compression = compression

        with open(config.var_file, "r") as conf:
            self.variable_config = yaml.load(conf, Loader=yaml_loader)

    def load_generator(
        self, input_file: str, nJets: int, chunkSize: int = 10000
    ):
        """
        Set up a generator who loads the scaled file and save it in the format for training.

        Input:
        - input_file: File which is to be scaled.
        - nJets: Number of jets used.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Yield: The yielded jets/tracks and labels loaded from file
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0

            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunkSize)
                if end_ind > nJets:
                    end_ind = nJets

                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunkSize)

            for index_tuple in tupled_indices:

                # Load jets
                jets = f["/jets"][index_tuple[0] : index_tuple[1]]
                labels = f["/labels"][index_tuple[0] : index_tuple[1]]

                if self.bool_use_tracks is False:
                    yield jets, labels

                elif self.bool_use_tracks is True:
                    # Load tracks
                    trks = np.asarray(
                        h5py.File(input_file, "r")["/tracks"][
                            index_tuple[0] : index_tuple[1]
                        ]
                    )

                    yield jets, trks, labels

    def WriteTrainSample(
        self,
        input_file: str = None,
        output_file: str = None,
        chunkSize: int = int(1e5),
    ):
        """
        Input:
        - input_file: File with scaled/shifted jets. Default is name from config + resampled_scaled
        - output_file: Name of the output file. Default is name from config + resampled_scaled_shuffled.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Train File: File which ready for training with the NN's.
        """

        # Get the input files for writing/merging
        if input_file is None:
            input_file = self.config.GetFileName(option="resampled_scaled")

        # Define outfile name
        if output_file is None:
            out_file = self.config.GetFileName(
                option="resampled_scaled_shuffled"
            )

        # Extract the correct variables
        variables_header_jets = self.variable_config["train_variables"]
        jets_variables = [
            i for j in variables_header_jets for i in variables_header_jets[j]
        ]

        # Get the max length of the input file
        file_length = len(h5py.File(input_file, "r")["/jets"])

        # Get the number of chunks that need to be processed
        n_chunks = int(np.ceil(file_length / chunkSize))

        load_generator = self.load_generator(
            input_file=input_file,
            nJets=file_length,
            chunkSize=chunkSize,
        )

        logger.info(f"Saving sample to {out_file}")
        with h5py.File(out_file, "w") as h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            while chunk_counter <= n_chunks:
                try:
                    # Load jets from file
                    if self.bool_use_tracks is False:
                        jets, labels = next(load_generator)

                    else:
                        jets, tracks, labels = next(load_generator)

                    # Get weights from jets
                    weights = jets["weight"]

                    # Reform jets to unstructured arrays
                    jets = repack_fields(jets[jets_variables])
                    jets = structured_to_unstructured(jets)

                    if chunk_counter == 0:
                        h5file.create_dataset(
                            "X_train",
                            data=jets,
                            compression=self.compression,
                            maxshape=(None, jets.shape[1]),
                        )

                        h5file.create_dataset(
                            "Y_train",
                            data=labels,
                            compression=self.compression,
                            maxshape=(None, labels.shape[1]),
                        )

                        h5file.create_dataset(
                            "weight",
                            data=weights,
                            compression=self.compression,
                            maxshape=(None,),
                        )

                        if self.bool_use_tracks is True:
                            h5file.create_dataset(
                                "X_trk_train",
                                data=tracks,
                                compression=self.compression,
                                maxshape=(
                                    None,
                                    tracks.shape[1],
                                    tracks.shape[2],
                                ),
                            )

                    else:
                        # appending to existing dataset
                        h5file["X_train"].resize(
                            (h5file["X_train"].shape[0] + jets.shape[0]),
                            axis=0,
                        )
                        h5file["X_train"][-jets.shape[0] :] = jets

                        # Appending Truth labels
                        h5file["Y_train"].resize(
                            (h5file["Y_train"].shape[0] + labels.shape[0]),
                            axis=0,
                        )
                        h5file["Y_train"][-labels.shape[0] :] = labels

                        # Appending weights
                        h5file["weight"].resize(
                            (h5file["weight"].shape[0] + weights.shape[0]),
                            axis=0,
                        )
                        h5file["weight"][-weights.shape[0] :] = weights

                        # Appending tracks if used
                        if self.bool_use_tracks is True:
                            h5file["X_trk_train"].resize(
                                (
                                    h5file["X_trk_train"].shape[0]
                                    + tracks.shape[0]
                                ),
                                axis=0,
                            )
                            h5file["X_trk_train"][-tracks.shape[0] :] = tracks

                except StopIteration:
                    break

                chunk_counter += 1
