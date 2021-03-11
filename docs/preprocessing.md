## Introduction
For the training of umami, the ntuples can be used which are specified in the section [MC Samples](mc-samples.md).

Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the table in the file [MC-Samples.md](mc-samples.md). However, the training ntuples are not yet optimal for training the different b-taggers and require preprocessing.

### Preprocessing
The motivation for preprocessing the training samples results from the fact that the input datasets are highly imbalanced in their flavour composition. While there are large quantities of light jets, the fraction of b-jets is small.
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).
In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.

### Hybrid samples
Umami is trained on so-called hybrid samples which are created using both ttbar and Z' input jets.
The hybrid samples for PFlow jets are created by combining events from ttbar and Z' samples based on a pt threshold, which is defined by the `GhostBHadronsFinalPt` variable for b-jets and by the `pt_btagJes` variable for other jet-flavours.
Below a certain pt threshold (which needs to be defined for the preprocessing), ttbar events are used in the hybrid sample. Above this pt threshold, the jets are taken from Z' events.
The advantage of these hybrid samples is the avaliability of sufficient jets with high pt, as the ttbar samples typically have lower-pt jets than those jets from the Z' sample.

![Pt distribution of hybrid samples being composed from ttbar and Zjets samples](assets/pt_btagJes-cut_spectrum.png)

The production of the hybrid samples in the preprocessing stage requires preparation of input files which are created from the training ntuples.

Additional preprocessing steps for PFlow jets include downsampling to ensure similar kinematic distributions for the jets of different flavours in the training samples in order to avoid kinematic biases.

![Pt distribution of downsampled hybrid samples](assets/pt_btagJes-downsampled.png)

Finally, the input features are scaled and shifted to normalise the range of the independent variables.
[Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling) gives a motivation for the scaling + shifting step:

> Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, many classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance. Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

All these steps are implemented in the `preprocessing.py` script, whose usage is described below in the documentation.

### Jet truth labelling
The standard labelling is provided via the `HadronConeExclTruthLabelID` variable while an extended jet labelling is available via the `HadronConeExclExtendedTruthLabelID` variable.
For more information, consider the [FTAG TWiki about flavour labelling](https://twiki.cern.ch/twiki/bin/view/AtlasProtected/FlavourTaggingLabeling).

| HadronConeExclTruthLabelID | Category         |
| -------------------------- | ---------------- |
| 0                          | light jets       |
| 4                          | c-jets           |
| 5                          | single b-jets    |
| 15                         | tau-jets         |

| HadronConeExclExtendedTruthLabelID | Category         |
| ---------------------------------- | ---------------- |
| 0                                  | light jets       |
| 4                                  | c-jets           |
| 5, 54                              | single b-jets    |
| 15                                 | tau-jets         |
| 44                                 | double c-jets    |
| 55                                 | double b-jets    |

For the `HadronConeExclTruthLabelID` labelling, the categories `4` and `44` as well as `5`, `54` and `55` are combined.


## Ntuple preparation for b-,c- & light-jets
The jets used for the training and validation of the taggers are taken from ttbar and Z' events.

After the ntuple production the samples have to be further processed using the Umami [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script. The preprocessing script is configured using a dedicated configuration file.
See [`examples/PFlow-Preprocessing.yaml`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml) for an example of a preprocessing config file.


### Configuration files
The ntuple preparation and is configured by the parameters in the `preparation` block.
Note that this file is formatted according to [`yaml`](https://en.wikipedia.org/wiki/YAML) specifications, so keeping an eye on the indentation is very important.

```
parameters:
# you can define paths which are used in several positions of the config file in advance to avoid duplication
  # ntuple path
  ntuple_path: &ntuple_path <path to your ntuple directory>
  # prepared sample path
  sample_path: &sample_path <path to the output of the "preprocessing: preparation" step>
  # merged sample output file path
  file_path: &file_path <path to the output of the "preprocessing: merging" step>

preparation:
# this block configures the sample preparation and merging stages
  ntuples:
  # path and pattern used to match the input ntuple files for the sample preparation
  # both ttbar and zprime ntuples are required (see documentation on MC samples)
    ttbar:
      path: *ntuple_path
      file_pattern: user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5/*.h5
    zprime:
      path: *ntuple_path
      file_pattern: user.mguth.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-15-T225316-R8334_output.h5/*.h5

  samples:
  # here you can define the output of the prepare sample stage
  # choose a name for the sample in order to process it using "preprocessing.py --prepare --sample <sample name>"
  # below are two examples for samples with the names "training_ttbar_bjets"  and "testing_zprime"

    training_ttbar_bjets:
      type: ttbar               # use ttbar ntuples

      category: bjets           # filter b-jets (only relevant for ttbar, not for zprime samples)

      n_jets: 10000000          # number of jets in output sample

      parity: even              # only store events with even event number (training sample)

      n_split: 10               # split output in 10 files
                                # (which need to be merged in second step)
                                # to reduce memory consumption

      pt_cut: true              # use pt cut in hybrid sample creation
                                # the pt cut is defined below using the
                                # "bhad_pTcut" keyword
                                # - for ttbar samples the cut is an upper cut
                                # - for Z' samples the cut is a lower cut

      f_output:                 # output file of prepared samples (need to be merged)
        path: *sample_path
        file: MC16d_hybrid-bjets_even_1_PFlow.h5

      merge_output: f_tt_bjets  # output file of merging the prepared samples
                                # can be either an explicit path or
                                # referring to a property defined below in the
                                # config file

    testing_zprime:
      type: zprime
      n_jets: 4000000
      parity: odd
      n_split: 2
      f_output:
        path: *sample_path
        file: MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts.h5


# pT cut for b-jets (used in hybrid sample creation)
bhad_pTcut: 2.5e5

# output path for prepared ttbar b-jet training sample
f_tt_bjets:
  path: *file_path
  file: MC16d_hybrid-bjets_even_1_PFlow-merged.h5
```

### Running the sample preparation

The preparation of the samples consists of two steps:

1. Sample preparation
2. Merging the files

In case of the default umami (3 categories:b, c, light) the label `HadronConeExclTruthLabelID` is used.
In the example above, only the b-jet category is shown.

To run the sample preparation for the ttbar b-jet sample `training_ttbar_bjets`, which has been defined in the config file in the `preparation: samples:` block, execute:

```
preprocessing.py --config <path to config file> --sample training_ttbar_bjets --prepare
```

As a result, 10 output files (because you specified `n_split: 10`) will be written to the output path your specified via `sample_path`. These files will follow the pattern `MC16d_hybrid-bjets_even_1_PFlow-file_*.h5`.
The rationale behind writing out several output files and then merging them and not writing one large file is to reduce the memory consumption of the preparation step.

Since there are several output files, they have to be merged. This can be achieved by executing:

```
preprocessing.py --config <path to config file> --sample training_ttbar_bjets --merge
```

### Running the preprocessing

After the preparation of the samples, the next step is the processing for the training itself which is also done with the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/preprocessing.py) script. Again, tthe configurations for the preprocessing are defined in the config file [PFlow-Preprocessing.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml) which you need to adapt to your needs.

1. Running the undersampling:

```bash
preprocessing.py -c examples/PFlow-Preprocessing.yaml --undersampling --tracks
```

2. Retrieving scaling and shifting factors:

```bash
preprocessing.py -c examples/PFlow-Preprocessing.yaml --scaling --tracks
```

3. Applying shifting and scaling factors

```bash
preprocessing.py -c examples/PFlow-Preprocessing.yaml --apply_scales --tracks
```

4. Shuffling the samples and writing the samples to disk

```bash
preprocessing.py -c examples/PFlow-Preprocessing.yaml --write --tracks
```

The training Variables for DL1r are defined in [DL1r_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/DL1r_Variables.yaml).


### Storing track content
The default setting of the preprocessing script is to write out only the jet content and not the associated tracks. If you also want to store the tracks, which is **required for training Dips and the Umami tagger**, you need to add the argument `'--tracks` to the preprocessing script.


Only jets written to output files:

```bash
preprocessing.py --config <path to config> --sample <sample> --prepare
preprocessing.py --config <path to config> --sample <sample> --merge
```


Jets and associated tracks written to output files:

```bash
preprocessing.py --config <path to config> --sample <sample> --prepare --tracks
preprocessing.py --config <path to config> --sample <sample> --merge --tracks
```


## Full example

There are several training and validation/test samples to produce. See below a list of all the necessary ones in a complete example configuration file:

```
parameters:
  # ntuple path
  ntuple_path: &ntuple_path <path to your data>/ntuples/
  # prepared sample path
  sample_path: &sample_path <path to your data>/processed/prepared_samples/
  # merged sample path
  file_path: &file_path <path to your data>/processed/merged_samples/

preparation:
  ntuples:
    ttbar:
      path: *ntuple_path
      file_pattern: user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5/*.h5
    zprime:
      path: *ntuple_path
      file_pattern: user.mguth.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-15-T225316-R8334_output.h5/*.h5

  samples:
    training_ttbar_bjets:
      type: ttbar
      category: bjets
      n_jets: 10000000
      parity: even
      n_split: 10
      pt_cut: true
      f_output:
        path: *sample_path
        file: MC16d_hybrid-bjets_even_1_PFlow-merged.h5
      merge_output: f_tt_bjets

    training_ttbar_cjets:
      type: ttbar
      category: cjets
      n_jets: 12745953
      parity: even
      n_split: 13
      pt_cut: true
      f_output:
        path: *sample_path
        file: MC16d_hybrid-cjets_even_1_PFlow-merged.h5
      merge_output: f_tt_cjets

    training_ttbar_ujets:
      type: ttbar
      category: ujets
      n_jets: 20000000
      parity: even
      n_split: 20
      pt_cut: true
      f_output:
        path: *sample_path
        file: MC16d_hybrid-ujets_even_1_PFlow-merged.h5
      merge_output: f_tt_ujets

    training_zprime:
      type: zprime
      n_jets: 9593092
      parity: even
      split: 2
      pt_cut: true
      f_output:
        path: *sample_path
        file: MC16d_hybrid-ext_even_0_PFlow-merged.h5
      merge_output: f_z

    testing_ttbar:
      type: ttbar
      n_jets: 4000000
      parity: odd
      n_split: 2
      f_output:
        path: *sample_path
        file: MC16d_hybrid_odd_100_PFlow-no_pTcuts.h5

    testing_zprime:
      type: zprime
      n_jets: 4000000
      parity: odd
      n_split: 2
      f_output:
        path: *sample_path
        file: MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts.h5

# amount of b-jets which are used
# njets: 1e5
njets: 5.5e6
# fraction of ttbar jets wrt. Z'
ttbar_frac: 0.65
# outputfiles are split into 5
# iterations: 1
iterations: 5
# pT cut for hybrid creation (for light and c-jets)
pTcut: 2.5e5
# pT cut for b-jets
bhad_pTcut: 2.5e5
# upper pT limit for all jets
pT_max: False
f_z:
  path: *file_path
  file: MC16d_hybrid-ext_even_0_PFlow-merged.h5
f_tt_bjets:
  path: *file_path
  file: MC16d_hybrid-bjets_even_1_PFlow-merged.h5
f_tt_cjets:
  path: *file_path
  file: MC16d_hybrid-cjets_even_1_PFlow-merged.h5
f_tt_ujets:
  path: *file_path
  file: MC16d_hybrid-ujets_even_1_PFlow-merged.h5
outfile_name: /nfs/dust/atlas/user/pgadow/ftag/data/processed/20201216-defaulttracks/output/PFlow-hybrid_70-test.h5
plot_name: PFlow_ext-hybrid

# Dictfile for the scaling and shifting (json)
dict_file: "examples/PFlow-scale_dict-22M.json"

# cut definitions to be applied to remove outliers
# possible operators: <, ==, >, >=, <=
Cuts:
  JetFitterSecondaryVertex_mass:
    operator: <
    condition: 25000
    NaNcheck: True
  JetFitterSecondaryVertex_energy:
    operator: <
    condition: 1e8
    NaNcheck: True
  JetFitter_deltaR:
    operator: <
    condition: 0.6
    NaNcheck: True
  softMuon_pt:
    operator: <
    condition: 0.5e9
    NaNcheck: True
  softMuon_momentumBalanceSignificance:
    operator: <
    condition: 50
    NaNcheck: True
  softMuon_scatteringNeighbourSignificance:
    operator: <
    condition: 600
    NaNcheck: True
  softMuon_momentumBalanceSignificance:
    operator: <
    condition: 4000
    NaNcheck: True

```

### Sample preparation

#### Training Samples (even EventNumber)

* ttbar (pT < 250 GeV)
    * b-jets
        ```bash
        preprocessing.py --config <path to config file> --sample training_ttbar_bjets --tracks --prepare
        preprocessing.py --config <path to config file> --sample training_ttbar_bjets --tracks --merge
        ```
    * c-jets
        ```bash
        preprocessing.py --config <path to config file> --sample training_ttbar_cjets --tracks --prepare
        preprocessing.py --config <path to config file> --sample training_ttbar_cjets --tracks --merge
        ```
    * light-jets
        ```bash
        preprocessing.py --config <path to config file> --sample training_ttbar_ujets --tracks --prepare
        preprocessing.py --config <path to config file> --sample training_ttbar_ujets --tracks --merge
        ```
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined
        ```bash
        preprocessing.py --config <path to config file> --sample training_zprime --tracks --prepare
        preprocessing.py --config <path to config file> --sample training_zprime --tracks --merge
        ```


#### Validation and Test Samples (odd EventNumber)

* ttbar
    ```bash
    preprocessing.py --config <path to config file> --sample testing_ttbar --tracks --prepare
    ```
* Z' (extended and standard)
    ```bash
    preprocessing.py --config <path to config file> --sample testing_zprime --tracks --prepare
    ```

### Preprocessing for hybrid sample creation
After you have created all files by running the preparation and merging commands, you can create the hybrid sample used for the tagger training by executing:

```bash
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict <path to dictionary with input variables of tagger> --undersampling --tracks
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict <path to dictionary with input variables of tagger> --scaling --tracks
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict <path to dictionary with input variables of tagger> --apply_scales --tracks
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict <path to dictionary with input variables of tagger> --write --tracks
```


## Ntuple Preparation for bb-jets

The double b-jets will be taken from Znunu and Zmumu samples. The framework still requires some updates in order to process those during the hybrid sample creation stage.

Since the double b-jets represent only a fraction of the jets, they can be filtered out using the [`merge_ntuples.py`](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_ntuples.py) script from the [hdf5-manipulator](https://gitlab.cern.ch/mguth/hdf5_manipulator).
