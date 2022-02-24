"""Weighting module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import os
import pickle

import h5py
import numpy as np

from umami.configuration import logger
from umami.preprocessing_tools.resampling.resampling_base import ResamplingTools
from umami.preprocessing_tools.utils import ResamplingPlots, generate_process_tag


class Weighting(ResamplingTools):
    """Weighting class."""

    def GetFlavourWeights(self):
        """
        Calculate ratios (weights) from bins in 2d (pt,eta) histogram between
        different flavours.
        """

        # calculate the 2D bin statistics for each sample and add it to
        # concat_samples dict with keys 'binnumbers','bin_indices_flat', 'stat'
        self.GetPtEtaBinStatistics()

        # target distribution
        target_jets_stats = self.concat_samples[
            self.options["weighting_target_flavour"]
        ]["stat"]

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
        # Some additional infos
        weights_dict["bins_x"] = self.bins_x
        weights_dict["bins_y"] = self.bins_y
        weights_dict["bin_indices_flat"] = self.bin_indices_flat
        # map inverse -> {0: 'ujets', 1: 'cjets', 2: 'bjets'}
        weights_dict["label_map"] = {v: k for k, v in self.class_labels_map.items()}
        save_name = os.path.join(
            self.outfile_path,
            "flavour_weights",
        )

        with open(save_name, "wb") as file:
            pickle.dump(weights_dict, file)
        logger.info(f"Saved flavour weights to: {save_name}")

    def Plotting(self):
        """Plot weighting results."""

        # Check if the directory for the plots exists
        plot_dir_path = os.path.join(
            self.resampled_path,
            "plots/",
        )
        os.makedirs(plot_dir_path, exist_ok=True)

        plot_name_raw = self.config.GetFileName(
            extension="",
            option="pt_eta_raw_",
            custom_path=plot_dir_path,
        )
        ResamplingPlots(
            concat_samples=self.concat_samples,
            positions_x_y=[0, 1],
            variable_names=["pT", "abseta"],
            plot_base_name=plot_name_raw,
            binning={"pT": 200, "abseta": 20},
            Log=True,
            use_weights=False,
            second_tag=generate_process_tag(self.config.preparation["ntuples"].keys()),
        )

    def GetIndices(self):
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
        with h5py.File(self.options["intermediate_index_file"], "w") as f:
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
                    f.create_dataset(
                        sample_name,
                        data=self.indices_to_keep[sample_name],
                        compression="gzip",
                    )
                    sample_size = len(self.indices_to_keep[sample_name])
                    size_total += sample_size
                    logger.info(f"Using {sample_size} jets from {sample_name}.")
        logger.info(f"Using in total {size_total} jets.")
        return self.indices_to_keep

    def Run(self):
        """Run function for Weighting class."""
        logger.info("Starting weights calculation")
        # loading pt and eta from files and put them into dict
        self.InitialiseSamples()
        # put ttbar and zprime together
        self.ConcatenateSamples()
        # calculate ratios between 2d (pt,eta) bin distributions of different
        # flavours
        self.GetFlavourWeights()
        logger.info("Making Plots")
        # Plots raw pt and eta
        self.Plotting()
        # write out indices.h5 to use preprocessing chain
        self.GetIndices()
        self.WriteFile(self.indices_to_keep)
