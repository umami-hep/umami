# Preprocessing
Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the tables in the FTAG documentation section [here](https://ftag.docs.cern.ch/software/samples/). However, the training ntuples are not yet optimal for training the different _b_-taggers and require preprocessing. The configuration of the preprocessing is done with a [`.yaml`](https://en.wikipedia.org/wiki/YAML) file which steers the whole preprocessing. An example can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml). The different parts of the config file will be explained in the different steps of the preprocessing.

## Motivation
The motivation for preprocessing the training samples results from the fact that the input datasets are highly imbalanced in their flavour composition. While there are large quantities of light jets, the fraction of _b_-jets is small and the fraction of other flavours is even smaller.
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).
In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.
Another approach can be to tell the network how important samples from each class are. For e.g. a majority class you can reduce the impact of samples from this class to the training. You can do this by assigning a weight to each sample and use it to weight the loss function used in the training.

## Hybrid Samples
Umami/DIPS and DL1r are trained on so-called hybrid samples which are created using both $t\bar{t}$ and $Z'$ input jets.
The hybrid samples for PFlow jets are created by combining events from $t\bar{t}$ and $Z'$ samples based on a pt threshold, which is defined by the `pt_btagJes` variable for all jet-flavours.
Below a certain pt threshold (which needs to be defined for the preprocessing), $t\bar{t}$ events are used in the hybrid sample. Above this pt threshold, the jets are taken from $Z'$ events.
The advantage of these hybrid samples is the availability of sufficient jets with high pt, as the $t\bar{t}$ samples typically have lower-pt jets than those jets from the $Z'$ sample.

![Pt distribution of hybrid samples being composed from ttbar and Zjets samples](../assets/pt_btagJes-cut_spectrum.png)

The production of the hybrid samples in the preprocessing stage requires preparation of input files which are created from the training ntuples.

Additional preprocessing steps for PFlow jets are required to ensure similar kinematic distributions for the jets of different flavours in the training samples in order to avoid kinematic biases. One of these techniques is downsampling which is used in the `Undersampling` approach.

![pT distribution of downsampled hybrid samples](../assets/pt_btagJes-downsampled.png)

Although we are using here for example reasons $t\bar{t}$ and $Z'$ to create a hybrid sample, you can use any kind of samples. Also, you don't need to create a hybrid sample. You can still use only one sample and for the preprocessing.
All these steps are implemented in the `preprocessing.py` script, whose usage is described in the follwing documentation.

## Preprocessing Steps
For the preprocessing, four steps need to be done:

1. [Preparation step](ntuple_preparation.md): Extract the different flavours from the `.h5` files from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) and separate them into flavour-specific files. Also the the split in training/validation/evaluation is done at this step.
2. [Resampling step](resampling.md): Combine and resample the different processes/flavours to achieve similar $p_T$ and $\eta$ distributions for all used flavours.
3. [Scaling/Shifting step](scaling_shifting.md): Calculate scaling/shifting values for all variables that are about to be used in the training. Then apply the scaling/shifting with the just calculated values.
4. [Writing step](write_train_sample.md): Write the final training sample to disk. In this step, the collections of jets and track variables are encoded and flattened so that we can load/use them for training.


## Apply Preprocessing with stand-alone script

In some cases you might want to apply the scaling and shifting to a data set using a stand-alone script.
For instance if you train with pytorch and just need a validation/test sample.

This can be done with the script `scripts/process_test_file.py`.