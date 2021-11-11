## Introduction
For the training of umami, the ntuples can be used which are specified in the section [MC Samples](mc-samples.md).

Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the table in the file [MC-Samples.md](mc-samples.md). However, the training ntuples are not yet optimal for training the different b-taggers and require preprocessing.

### Preprocessing
The motivation for preprocessing the training samples results from the fact that the input datasets are highly imbalanced in their flavour composition. While there are large quantities of light jets, the fraction of b-jets is small and the fraction of other flavours is even smaller.
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).
In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.

### Hybrid samples
Umami/DIPS and DL1r are trained on so-called hybrid samples which are created using both ttbar and Z' input jets.
The hybrid samples for PFlow jets are created by combining events from ttbar and Z' samples based on a pt threshold, which is defined by the `pt_btagJes` variable for all jet-flavours.
Below a certain pt threshold (which needs to be defined for the preprocessing), ttbar events are used in the hybrid sample. Above this pt threshold, the jets are taken from Z' events.
The advantage of these hybrid samples is the avaliability of sufficient jets with high pt, as the ttbar samples typically have lower-pt jets than those jets from the Z' sample.

![Pt distribution of hybrid samples being composed from ttbar and Zjets samples](assets/pt_btagJes-cut_spectrum.png)

The production of the hybrid samples in the preprocessing stage requires preparation of input files which are created from the training ntuples.

Additional preprocessing steps for PFlow jets are required to ensure similar kinematic distributions for the jets of different flavours in the training samples in order to avoid kinematic biases. One of these techniques is downsampling which is used in the `Undersampling` approach.

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
| 5                          | b-jets    |
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


## Ntuple preparation
The jets used for the training and validation of the taggers are taken from ttbar and Z' events. Different flavours can be used and combined to prepare different datasets for training/evaluation. The standard classes used are `bjets`, `cjets` and `ujets` (light jets).   
After the ntuple production (training-dataset-dumper) the samples have to be further processed using the Umami [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script. The preprocessing script is configured using a dedicated configuration file.
See [`examples/PFlow-Preprocessing.yaml`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml) for an example of a preprocessing config file.


### Configuration files
Note that this file is formatted according to [`yaml`](https://en.wikipedia.org/wiki/YAML) specifications, so keeping an eye on the indentation is very important. An explanation of the features of this file is given in the example below:

#### Parameters
```yaml
parameters: !include Preprocessing-settings-Freiburg.yaml
```

This line specifies where the ntuples (which are used) are stored and where to save the output of the preprocessing. You can find the file [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Preprocessing-settings-Freiburg.yaml).

#### Cut Templates
```yaml
# Defining yaml anchors to be used later, avoiding duplication
.cuts_template_ttbar_train: &cuts_template_ttbar_train
  cuts:
    - eventNumber:
        operator: mod_6_<=
        condition: 3
    - pt_btagJes:
        operator: <=
        condition: 2.5e5

.cuts_template_zprime_train: &cuts_template_zprime_train
  cuts:
    - eventNumber:
        operator: mod_6_<=
        condition: 3
    - pt_btagJes:
        operator: >=
        condition: 2.5e5

.cuts_template_validation: &cuts_template_validation
  cuts:
    - eventNumber:
        operator: mod_6_==
        condition: 4

.cuts_template_test: &cuts_template_test
  cuts:
    - eventNumber:
        operator: mod_6_==
        condition: 5
```

The cuts defined in this section are templates for the cuts of the different flavour for ttbar/zprime. We also split the ttbar/zprime in train/validation/test to ensure no jet is used twice. `ttbar_train` and `zprime_train` are the jets which are used for training while validation/test are the templates for validation and test.
The cuts which are to be applied can be defined in these templates. For example, we can define a cut on the `eventNumber` with a modulo operator. This modulo operator defines that all jets are used, where the `eventNumber` is equal to something. The something can be defined by the `condition`.
Another cut which can be applied is the `pt_btagJes`, which is a cut on the jet pT. Works the same as the modulo operator.

#### File- and Flavour Preparation
```yaml
preparation:
  batchsize: 50000

  ntuples:
    ttbar:
      path: *ntuple_path
      file_pattern: user.alfroch.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2021-07-28-T130145-R11969_output.h5/*.h5

    zprime:
      path: *ntuple_path
      file_pattern: user.alfroch.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2021-07-28-T130145-R11969_output.h5/*.h5

  class_labels: [ujets, cjets, bjets]

  samples:
    training_ttbar_bjets:
      type: ttbar
      category: bjets
      n_jets: 10e6
      <<: *cuts_template_ttbar_train
      f_output:
        path: *sample_path
        file: MC16d-bjets_training_ttbar_PFlow.h5

    training_ttbar_cjets:
      type: ttbar
      category: cjets
      # Number of c jets available in MC16d
      n_jets: 12745953
      <<: *cuts_template_ttbar_train
      f_output:
        path: *sample_path
        file: MC16d-cjets_training_ttbar_PFlow.h5

    training_ttbar_ujets:
      type: ttbar
      category: ujets
      n_jets: 20e6
      <<: *cuts_template_ttbar_train
      f_output:
        path: *sample_path
        file: MC16d-ujets_training_ttbar_PFlow.h5

    training_ttbar_taujets:
      type: ttbar
      category: taujets
      n_jets: 13e6
      <<: *cuts_template_ttbar_train
      f_output:
        path: *sample_path
        file: MC16d-taujets_training_ttbar_PFlow.h5

    training_zprime_bjets:
      type: zprime
      category: bjets
      n_jets: 10e6
      <<: *cuts_template_zprime_train
      f_output:
        path: *sample_path
        file: MC16d-bjets_training_zprime_PFlow.h5

    training_zprime_cjets:
      type: zprime
      category: cjets
      # Number of c jets available in MC16d
      n_jets: 10e6
      <<: *cuts_template_zprime_train
      f_output:
        path: *sample_path
        file: MC16d-cjets_training_zprime_PFlow.h5

    training_zprime_ujets:
      type: zprime
      category: ujets
      n_jets: 10e6
      <<: *cuts_template_zprime_train
      f_output:
        path: *sample_path
        file: MC16d-ujets_training_zprime_PFlow.h5

    training_zprime_taujets:
      type: zprime
      category: taujets
      n_jets: 10e6
      <<: *cuts_template_zprime_train
      f_output:
        path: *sample_path
        file: MC16d-taujets_training_zprime_PFlow.h5

    validation_ttbar:
      type: ttbar
      category: inclusive
      n_jets: 4e6
      <<: *cuts_template_validation
      f_output:
        path: *sample_path
        file: MC16d-inclusive_validation_ttbar_PFlow.h5

    testing_ttbar:
      type: ttbar
      category: inclusive
      n_jets: 4e6
      <<: *cuts_template_test
      f_output:
        path: *sample_path
        file: MC16d-inclusive_testing_ttbar_PFlow.h5

    validation_zprime:
      type: zprime
      category: inclusive
      n_jets: 4e6
      <<: *cuts_template_validation
      f_output:
        path: *sample_path
        file: MC16d-inclusive_validation_zprime_PFlow.h5

    testing_zprime:
      type: zprime
      category: inclusive
      n_jets: 4e6
      <<: *cuts_template_test
      f_output:
        path: *sample_path
        file: MC16d-inclusive_testing_zprime_PFlow.h5
```
In the `Preparation`, the size of the batches which are be loaded from the ntuples is defined in `batchsize`. The exact path of the ntuples are defined in `ntuples`. You define there where the ttbar and zprime ntuples are saved and which files to use (You can use wildcards here!). The `file_pattern` defines the files while `path` defines the absolut path to the folder where they are saved. `*ntuple_path` is the path to the ntuples defined in the `parameters` file.   

Another important part are the `class_labels` which are defined here. You can define here which flavours are used in the preprocessing. The name of the available flavours can be find [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/alfroch-scaling-followup/umami/configs/global_config.yaml). Add the names of those to the list here to add them to the preprocessing. PLEASE KEEP THE ORDERING CONSTANT! THIS IS VERY IMPORTANT. This list must be the same as the one in the train config!

The last part is the exact splitting of the flavours. In `samples`, you define for each of ttbar/zprime and training/validation/testing the flavours you want to use. You need to give a type (ttbar/zprime), a category (flavour or `inclusive`) and the number of jets you want for this specific flavour. Also you need to apply the template cuts we defined already. The `f_output` defines where the output files is saved. `path` defines the folder, `file` defines the name.

 Setting | Explanation      |
| ------ | ---------------- |
| `batchsize` | size of loaded batches |
#### Sampling

```yaml
sampling:
  method: count
  # The options depend on the sampling method
  options:
    sampling_variables:
      - pt_btagJes:
          # bins take either a list containing the np.linspace arguments
          # or a list of them
          bins: [[0, 600000, 351], [650000, 6000000, 84]]
      - absEta_btagJes:
          bins: [0, 2.5, 10]
    samples:
      ttbar:
        - training_ttbar_bjets
        - training_ttbar_cjets
        - training_ttbar_ujets
      zprime:
        - training_zprime_bjets
        - training_zprime_cjets
        - training_zprime_ujets
    # this optional option allows to specify the jets which should be used per sample
    custom_njets_initial:
      # these are empiric values ensuring a smooth hybrid sample
      training_ttbar_bjets: 5.5e6
      training_ttbar_cjets: 11.5e6
      training_ttbar_ujets: 13.5e6
    fractions:
      ttbar: 0.7
      zprime: 0.3
    # number of training jets
    njets: 25e6
    save_tracks: True
    # this stores the indices per sample into an intermediate file
    intermediate_index_file: indices.h5
```

In `sampling`, we can define the method which is used in the preprocessing for resampling. `method` defines the method which is used. Currently available are:

| Method | Explanation      |
| ------ | ---------------- |
| `count`                   | Standard undersampling approach. Undersamples all flavours to the statistically lowest flavour used |
| `probability_ratio`       | NOTE: If your sample's statistics are small and/or your lowest distribution is other than the b-jet distribution, you can force the b-jet distribution shape on the other jet flavor distributions. This will ensure ensure all the distributions have the b-shape and the same fractions. Additionally, when building the target distribution for "probability_ratio", `pT_max` (set in the config file [PFlow-Preprocessing.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml)) will be used to compute the probability ratios or PDFs. Not setting `pT_max` will allow you to keep more jets (bigger fractions) but with more noise (uncertainty) loosing the guarantee that all the distributions will have the same b-jet distribution shape. WARNING: The `probability_ratio`does not work well with taus as of now.|

The `options` are some options for the different resampling methods. You need to define the sampling variables which are used for resampling. For example, if you want to resample in `pt_btagJes` and `absEta_btagJes` bins, you just define them with their respective bins. 
Another thing you need to define are the `samples` which are to be resampled. You need to define them for `ttbar` and `zprime`. The samples defined in here are the ones we prepared in the step above. To ensure a smooth hybrid sample of ttbar and zprime, we need to define some empirically derived values for the ttbar samples in `custom_njets_initial`.
`fractions` gives us the fractions of ttbar and zprime in the final training sample. These values need to add up to 1! The rest of the variables are pretty self-explanatory.


### General settings 

| Setting | Explanation      |
| ------ | ---------------- |
| `outfile_name` | name of the output file of the preprocessing |
| `plot_name` | defines the names of the control plots which are produced in the preprocessing |
| `var_file` | path to the variable dict |
| `dict_file` | path to the scale dict |

```yaml
# Name of the output file from the preprocessing
outfile_name: *outfile_name
plot_name: PFlow_ext-hybrid

# Variable dict which is used for scaling and shifting
var_file: *var_file

# Dictfile for the scaling and shifting (json)
dict_file: *dict_file

# TODO: move these cuts to the preparation step
# cut definitions to be applied to remove outliers
# possible operators: <, ==, >, >=, <=
cuts:
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
```
In the last part, the path to the variable dict `var_file` and the scale dict `dict_file` is defined. Those values are set in the `parameters` file. For example, the training variables for DL1r are defined in [DL1r_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/DL1r_Variables.yaml).

Also the `outfile_name` is defined (which is also included in `parameters`). The `plot_name` here defines the names of the control plots which are produced in the preprocessing.

### Running the sample preparation

To run the sample preparation for the ttbar b-jet sample `training_ttbar_bjets`, which has been defined in the config file in the `preparation: samples:` block, execute:

```bash
preprocessing.py --config <path to config file> --sample training_ttbar_bjets --prepare
```

As a result, an output file will be written to the output path you specified via `sample_path`. The file will have the name defined in the `preparation` block.

If you want to prepare all the samples defined in the `preparation: samples:` block, just leave out the `--sample` option. Also, if you want to use tracks, you need to add the flag `--tracks`. An example command would look like this:

```bash
preprocessing.py --config <path to config file> --prepare --tracks
```

### Running the preprocessing

After the preparation of the samples, the next step is the processing for the training itself which is also done with the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/preprocessing.py) script. Again, the configurations for the preprocessing are defined in the config file [PFlow-Preprocessing.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml) which you need to adapt to your needs.

The steps defined in the following segment are only performed on the training samples! You do not need to resample/scale/write the validation/test samples!

1. Running the resampling:

```bash
preprocessing.py --config <path to config file> --resampling
```

If you want to also use the tracks of the jets, you need to give an extra flag `--tracks`. Track information are not needed for the DL1r but for DIPS and Umami. If you want to train one of those, you need to process the track information too with setting the `--tracks` flag:

```bash
preprocessing.py --config <path to config file> --resampling --tracks
```

2. Retrieving scaling and shifting factors:

```bash
preprocessing.py --config <path to config file> --scaling --tracks
```

3. Applying shifting and scaling factors

```bash
preprocessing.py --config <path to config file> --apply_scales --tracks
```

4. Writing the samples to disk in the correct format for training.

```bash
preprocessing.py --config <path to config file> --write --tracks
```

## Full example

There are several training and validation/test samples to produce. See the following link for a list of all the necessary ones in a complete configuration file: [`examples/PFlow-Preprocessing.yaml`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml)

## Ntuple Preparation for bb-jets

TODO: Rewrite this!
The double b-jets will be taken from Znunu and Zmumu samples. The framework still requires some updates in order to process those during the hybrid sample creation stage. 
