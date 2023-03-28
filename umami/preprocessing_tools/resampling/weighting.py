"""Weighting module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import os
import pickle

import h5py
import numpy as np

from umami.configuration import logger
from umami.plotting_tools import plot_resampling_variables, preprocessing_plots
from umami.preprocessing_tools.resampling.resampling_base import ResamplingTools
from umami.preprocessing_tools.utils import get_variable_dict


class Weighting(ResamplingTools):
    """Weighting class."""

    def get_flavour_weights(self):
        """
        Calculate ratios (weights) from bins in 2d (pt,eta) histogram between
        different flavours.
        """

        # calculate the 2D bin statistics for each sample and add it to
        # concat_samples dict with keys 'binnumbers','bin_indices_flat', 'stat'
        self.get_pt_eta_bin_statistics()

        # target distribution
        target_jets_stats = self.concat_samples[self.options.weighting_target_flavour][
            "stat"
        ]

        # Write out weights_dict for later use
        weights_dict = {}

        # calculate weights between flavours of pt, eta distribution
        for flavour in self.class_categories:
            # jets by flavours to be ratioed
            flavour_jets_stats = self.concat_samples[flavour]["stat"]
            # make sure there is no zero divsion
            weights_dict[flavour] = np.divide(
                target_jets_stats,
                flavour_jets_stats,
                out=np.zeros_like(target_jets_stats),
                where=flavour_jets_stats != 0,
            )
        # Add the binning
        for variable_name, bin_values in self.resampling_bins.items():
            weights_dict[variable_name] = bin_values

        # Add the flat bin indices
        weights_dict["bin_indices_flat"] = self.bin_indices_flat

        # Add the resampling variables/bins to the weights dict
        weights_dict["resampling_variables"] = self.resampling_variables
        weights_dict["resampling_bins"] = self.resampling_bins

        # map inverse -> {0: 'ujets', 1: 'cjets', 2: 'bjets'}
        weights_dict["label_map"] = {v: k for k, v in self.class_labels_map.items()}
        save_name = os.path.join(
            self.outfile_path,
            "flavour_weights_training"
            if not self.use_validation_samples
            else "flavour_weights_validation",
        )

        with open(save_name, "wb") as file:
            pickle.dump(weights_dict, file)
        logger.info("Saved flavour weights to: %s", save_name)

    def get_indices(self):
        """
        Applies the UnderSampling to the given arrays.

        Returns
        -------
        Returns the indices for the jets to be used separately for each
        category and sample.

        """
        # To include it into the preprocessing chain write out indices.h5 and
        # fill indices_to_keep with all jets for weighting method
        indices_to_keep = {elem: [] for elem in self.class_categories}
        for class_category in self.class_categories:
            size_class_category = len(self.concat_samples[class_category]["jets"])
            indices_to_keep[class_category] = np.arange(size_class_category)

        # Write out indices.h5 with keys as samplenames, e.g.
        # "training_ttbar_bjets"
        size_total = 0
        self.indices_to_keep = {}  # pylint: disable=attribute-defined-outside-init

        # decide which index file to use
        index_file = (
            self.options.intermediate_index_file_validation
            if self.use_validation_samples
            else self.options.intermediate_index_file
        )

        with h5py.File(index_file, "w") as f_index:
            for class_category in self.class_categories:
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
        """Run function for Weighting class."""
        logger.info("Starting weights calculation")
        # loading pt and eta from files and put them into dict
        self.initialise_samples()
        # put ttbar and zprime together
        self.concatenate_samples()
        # calculate ratios between 2d (pt,eta) bin distributions of different
        # flavours
        self.get_flavour_weights()
        # write out indices.h5 to use preprocessing chain
        self.get_indices()

        # Make the resampling plots for the resampling variables before resampling
        plot_resampling_variables(
            concat_samples=self.concat_samples,
            var_positions=[0, 1],
            variable_names=self.resampling_variables,
            sample_categories=self.config.preparation.sample_categories,
            output_dir=os.path.join(
                self.resampled_path,
                "plots/resampling/",
            ),
            bins_dict={
                self.resampling_variables[0]: 200,
                self.resampling_variables[1]: 20,
            },
            atlas_second_tag=self.config.general.plot_sample_label,
            logy=True,
            ylabel="Normalised number of jets",
        )

        self.write_file(self.indices_to_keep)

        # Plot the variables from the output file of the resampling process
        if self.options.n_jets_to_plot:
            logger.info("Plotting resampled distributions...")
            preprocessing_plots(
                sample=self.config.get_file_name(
                    option="resampled",
                    use_val=self.use_validation_samples,
                ),
                var_dict=get_variable_dict(self.config.general.var_file),
                class_labels=self.config.sampling.class_labels,
                plots_dir=os.path.join(
                    self.resampled_path,
                    "plots/resampling/",
                    "validation/" if self.use_validation_samples else "",
                ),
                track_collection_list=self.options.tracks_names
                if self.options.save_tracks is True
                else None,
                n_jets=self.options.n_jets_to_plot,
                atlas_second_tag=self.config.general.plot_sample_label,
                logy=True,
                ylabel="Normalised number of jets",
            )
