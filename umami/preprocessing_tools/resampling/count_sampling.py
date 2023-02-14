"""Count sampling module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import os

import h5py
import numpy as np

from umami.configuration import logger
from umami.plotting_tools import plot_resampling_variables, preprocessing_plots
from umami.preprocessing_tools.resampling.resampling_base import (
    ResamplingTools,
    correct_fractions,
)
from umami.preprocessing_tools.utils import get_variable_dict


class UnderSampling(ResamplingTools):
    """Undersampling class."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.indices_to_keep = None
        self.x_y_after_sampling = None

    def get_indices(self):
        """
        Applies the UnderSampling to the given arrays.

        Returns
        -------
        Returns the indices for the jets to be used separately for each
        category and sample.

        """
        # concatenate samples with the same category
        # no fraction between different samples is ensured at this stage,
        # this will be done at the end
        self.concatenate_samples()

        # calculate the 2D bin statistics for each sample
        self.get_pt_eta_bin_statistics()

        min_count_per_bin = np.amin(
            [self.concat_samples[sample]["stat"] for sample in self.concat_samples],
            axis=0,
        )

        rng = np.random.default_rng(seed=self.rnd_seed)
        indices_to_keep = {elem: [] for elem in self.class_categories}
        # retrieve the indices of the jets to keep
        for bin_i, count in zip(self.bin_indices_flat, min_count_per_bin):
            for class_category in self.class_categories:
                indices_to_keep[class_category].append(
                    rng.choice(
                        np.nonzero(
                            self.concat_samples[class_category]["binnumbers"] == bin_i
                        )[0],
                        int(count),
                        replace=False,
                    )
                )
        # join the indices into one array per class category
        for class_category in self.class_categories:
            indices_to_keep[class_category] = np.concatenate(
                indices_to_keep[class_category]
            )
        # make sure the sample fractions are correct
        # since undersampling is applied at this stage, it is enough to
        # calculate the fractions for one flavour class
        # there might be some fluctuations but that should be negligible
        # TODO: could be done for each of them
        reference_class_category = self.class_categories[0]
        target_fractions = []
        n_jets = []
        sample_categories = self.concat_samples[reference_class_category]["jets"][
            indices_to_keep[reference_class_category], 4
        ]
        for sample_category in self.sample_categories:
            target_fractions.append(self.options.fractions[sample_category])
            n_jets.append(
                len(
                    np.nonzero(
                        sample_categories == self.sample_categories[sample_category]
                    )[0]
                )
            )
        target_n_jets = correct_fractions(
            n_jets=n_jets,
            target_fractions=target_fractions,
            class_names=list(self.sample_categories.keys()),
        )
        target_n_jets = {
            list(self.sample_categories.keys())[i]: target_n_jets[i]
            for i in range(len(target_n_jets))
        }
        for class_category in self.class_categories:
            sample_categories = self.concat_samples[class_category]["jets"][
                indices_to_keep[class_category], 4
            ]
            indices_to_keep_tmp = np.asarray([])
            for sample_category in self.sample_categories:
                location = np.nonzero(
                    sample_categories == self.sample_categories[sample_category]
                )[0]
                indices_to_keep_tmp = np.append(
                    indices_to_keep_tmp,
                    rng.choice(
                        indices_to_keep[class_category][location],
                        size=target_n_jets[sample_category]
                        if target_n_jets[sample_category]
                        <= len(indices_to_keep[class_category][location])
                        else len(indices_to_keep[class_category][location]),
                        replace=False,
                    ),
                )
            indices_to_keep[class_category] = indices_to_keep_tmp.astype(int)

        # Check which n_jets to use (training or validation)
        n_jets_requested = (
            self.options.n_jets
            if not self.use_validation_samples
            else self.options.n_jets_validation
        )

        # check if more jets are available as requested in the config file
        # we assume that all classes have the same number of jets now
        if len(indices_to_keep[reference_class_category]) > n_jets_requested // len(
            self.class_categories
        ):
            size_per_class = n_jets_requested // len(self.class_categories)
            for class_category in self.class_categories:
                indices_to_keep[class_category] = rng.choice(
                    indices_to_keep[class_category],
                    size=int(size_per_class)
                    if size_per_class <= len(indices_to_keep[class_category])
                    else len(indices_to_keep[class_category]),
                    replace=False,
                )
        else:
            size_per_class = len(indices_to_keep[reference_class_category])
            size_total = len(indices_to_keep[reference_class_category]) * len(
                self.class_categories
            )
            logger.warning(
                "You asked for %i jets, however, only %i are available.",
                n_jets_requested,
                size_total,
            )

        # get indices per single sample
        self.indices_to_keep = {}
        self.x_y_after_sampling = {}
        size_total = 0

        # decide which index file to use
        index_file = (
            self.options.intermediate_index_file_validation
            if self.use_validation_samples
            else self.options.intermediate_index_file
        )

        with h5py.File(index_file, "w") as f_index:
            for class_category in self.class_categories:
                self.x_y_after_sampling[class_category] = self.concat_samples[
                    class_category
                ]["jets"][indices_to_keep[class_category], :2]
                sample_categories = self.concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 4
                ]
                sample_indices = self.concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 2
                ]
                for sample_category in self.sample_categories:
                    sample_name = self.sample_map[sample_category][class_category]
                    self.indices_to_keep[sample_name] = np.sort(
                        sample_indices[
                            sample_categories == self.sample_categories[sample_category]
                        ]
                    ).astype(int)
                    f_index.create_dataset(
                        sample_name,
                        data=self.indices_to_keep[sample_name],
                        compression="gzip",
                    )
                    sample_size = len(self.indices_to_keep[sample_name])
                    size_total += sample_size
                    logger.info("Using %i jets from %s.", sample_size, sample_name)
        logger.info("Using in total %i jets.", size_total)
        return self.indices_to_keep

    def Run(self):
        """Run function executing full chain."""
        logger.info("Starting undersampling.")
        self.initialise_samples()
        self.get_indices()

        # Make the resampling plots for the resampling variables before resampling
        plot_resampling_variables(
            concat_samples=self.concat_samples,
            var_positions=[0, 1],
            variable_names=[self.var_x, self.var_y],
            sample_categories=self.config.preparation.sample_categories,
            output_dir=os.path.join(
                self.resampled_path,
                "plots/resampling/",
            ),
            bins_dict={
                self.var_x: 200,
                self.var_y: 20,
            },
            atlas_second_tag=self.config.general.plot_sample_label,
            logy=True,
            ylabel="Normalised number of jets",
        )

        # Resample the files and write them to disk
        self.write_file(self.indices_to_keep)

        # Plot the variables from the output file of the resampling process
        if self.options.n_jets_to_plot:
            logger.info("Plotting resampled distributions...")
            preprocessing_plots(
                sample=self.config.get_file_name(option="resampled"),
                var_dict=get_variable_dict(self.config.var_file),
                class_labels=self.config.sampling["class_labels"],
                plots_dir=os.path.join(
                    self.resampled_path,
                    "plots/resampling/",
                ),
                track_collection_list=self.options.tracks_names
                if self.options.save_tracks is True
                else None,
                n_jets=self.options.n_jets_to_plot,
                atlas_second_tag=self.config.general.plot_sample_label,
                logy=True,
                ylabel="Normalised number of jets",
            )
