# Umami framework tutorial

## Introduction

In this tutorial, you will learn to setup and use the [Umami framework](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami). Umami is a high-level framework which combines the necessary preprocessing
for training taggers, the actual training and validation of the training of the taggers living inside of Umami and the evaluation of the training results. In addition,
easy plotting scripts are given to plot the results of the evaluation using the [`puma`](https://github.com/umami-hep/puma) package.

In this tutorial, we cover the following functionalities of Umami:

1. Preprocessing the `.h5` files coming from the [training-dataset-dumper](https://training-dataset-dumper.docs.cern.ch)
2. Train one of the taggers (here DL1d) available in Umami
3. Validate and check the training of the taggers
4. Evaluate the taggers on an un-preprocessed sample
5. Plot the results of the evaluation
6. [extra task] plotting of input variables using Umami

The tutorial is meant to be followed in a self-guided manner. You will be prompted to do certain tasks by telling you what the desired outcome will be, without telling you how to do it. Using the [documentation of Umami](../index.md), you can find out how to achieve your goal. In case you are stuck, you can click on the "hint" toggle box to get a hint. If you tried for more than 10 min at a problem, feel free to toggle also the solution with a working example.

In case you encounter some errors or you are completely stuck, you can reach out to the dedicated [Umami mattermost channel](https://mattermost.web.cern.ch/aft-algs/channels/umami) (click [here]() to sign up)

**You can find the introduction talk by Alexander Froch on the [FTAG workshop indico page](https://indico.cern.ch/event/1193206/timetable/?view=standard#b-477082-day-3-afternoon-tutor).**

## Prerequisites

For this tutorial, you need access to a shell on either CERN's `lxplus` or your local cluster with `/cvmfs` access to retrieve the `singularity` image needed. To set this up, please follow the instructions [here](../setup/installation.md#singularity-images-on-cvmfs).

You can also run this tutorial on a cluster without `/cvmfs` access, but with `singularity` installed. To do so, please follow the instructions given [here](../setup/installation.md#launching-containers-using-singularity-lxplusinstitute-cluster). The image needed for this tutorial is the `umamibase-plus:0-15`. If you are doing this tutorial to prepare yourself for a training (and you have a GPU available for training), you need to get the GPU image of Umami to be able to utilize the GPU. You can get this image by adding the `-gpu` to the image name. The final image name is than `umamibase-plus:0-15-gpu`.

After running the `singularity shell` or the `singularity exec` command, you can re-source your .bashrc to get the "normal" look of your terminal back by running 

```bash
source ~/.bashrc
```

??? warning "Solution"

    The FTAG group provides ready singularity images via `/cvmfs/unpacked.cern.ch` on lxplus (or any cluster which has `/cvmfs` mounted). You can use these cvmfs images. There are two ways how you can run umami, either directly from the image (not recommended for code development) or install it on top of a base image which provides the requirements. Below, commands for both options are provided.
    
    **image with already installed umami**
    ```
    singularity shell -B /eos,/tmp,/cvmfs /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:0-15
    ```

    **image with only requirements** (recommended for this tutorial)
    ```
    singularity shell -B /eos,/tmp,/cvmfs /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:0-15
    ```
    if you use this image, you need to set up umami, this can be done via the `run_setup.sh` script which you will obtain in the next step by cloning the git repository

    ```
    source run_setup.sh
    ```


    **In case you cannot go to your `/eos` directory, simply type `bash` into your terminal and it should work.**


## Tutorial tasks

### 1. Fork, clone and install Umami

Before you can start with the other tasks, you need to retrieve a version of Umami (mainly the config files). 
To do so, you need to do the following steps:

1. Create a personal fork of Umami in Gitlab.
2. Clone the forked repository to your machine using `git`.
3. (Optional) Run the setup to switch to development mode.

Go to the GitLab project page of Umami to begin with the task: <https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami>

It is highly recommended to **NOT** perform this tutorial on your `lxplus` home, because we will need a bit more than 15 GB of free disk space! Try to do the tutorial in your personal EOS space `/eos/user/${USER:0:1}/$USER`.

??? info "Hint: Create a personal fork of Umami in Gitlab"

    In case you are stuck how to create your personal fork of the project, you can find some general information on git and the forking concept [here in the GitLab documentation](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).

??? info "Hint: Clone the forked repository to your machine using `git`"

    The command `git clone` is the one you need. You can look up the usage [here](../setup/installation.md#cloning-the-repository)

??? warning "Solution"

    Open the website <https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami> in a browser. You may need to authenticate with your CERN login credentials. In the top right corner of the Umami project you see three buttons which show a bell (notifications), a star (to favourite the project) next to a number, and a forking graph (to fork the project) with the text "Fork" next to a number. Click on the word "Fork" to open a new website, allowing you to specify the namespace of your fork. Click on "Select a namespace", choose your CERN username, and create the fork by clicking on "Fork project".

    Next, you need to clone the project using `git`. Open a fresh terminal on the cluster your are working on, create a new folder and proceed with the cloning. To do so, open your forked project in a browser. The address typically is `https://gitlab.cern.ch/<your CERN username>/umami`. When clicking on the blue "Clone" button at the right hand-side of the page, a drop-down mini-page appears with the ssh path to the forked git project. Let's check out your personal fork. It's explained [here](../setup/installation.md#cloning-the-repository)

    You now forked and cloned Umami and should be ready to go!


### 2. Download the test files for the tutorial

**If you work on lxplus, there is actually no need to copy the files, you can just directly read them from the provided directory.**

For this tutorial, we provide you with some `.h5` files coming from the dumper which already passed the `preparation` step of Umami (due to time constraints of the tutorial session, we will skip that part but you can have a look at that afterwards). Also, if you are unable to perform one of the following steps, we provide checkpoint files, with which you can continue. The name of the files for the checkpoints are given in the end of the respective section.

To get access to the files, you can either copy them directly (on lxplus) or download them using `wget`. To access them directly, the path to all the files in `eos` is `/eos/user/u/umamibot/www/ci/tutorial/umami/`. If you want to download the files via `wget`, the link is `https://umami-ci-provider.web.cern.ch/tutorial/umami/` where you just need to add the filename in the end to download it.

The command you need to run on lxplus is:

```bash
mkdir prepared_samples
cp /eos/user/u/umamibot/www/ci/tutorial/umami/*.h5 prepared_samples/.
```

The commands you need to run with `wget` are:

```bash
mkdir prepared_samples && cd prepared_samples
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/bjets_training_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/cjets_training_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/ujets_training_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/bjets_training_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/cjets_training_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/ujets_training_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/bjets_validation_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/cjets_validation_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/ujets_validation_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/bjets_validation_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/cjets_validation_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/ujets_validation_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/inclusive_validation_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/inclusive_validation_zprime_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/inclusive_testing_ttbar_PFlow.h5
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/inclusive_testing_zprime_PFlow.h5
```

### 3. Preprocessing of the .h5 files

The preprocessing of Umami consists of multiple small steps. These are

1. Preparation
2. Resampling
3. Scaling/Shifting
4. Writing

which are explained in more detail in the [Umami preprocessing documentation](../preprocessing/Overview.md).

For this tutorial, the `Preparation` step was already performed due to the large amount of time this step can consume. A very detailed explanation how to run this step is given in the documentation [here](../preprocessing/ntuple_preparation.md)

#### 3.0. (Optional) Preparation


???+ info "Optional part of the tutorial"

    This part of the tutorial is optional. You can skip it and proceed directly with part 3.1 (recommended) by using the `.h5` files which are provided and which you downloaded in part 2 of the tutorial.


???+ warning "Large input files"
    
    In order to perform this step, you need to download the output files from the h5 dumper. You can find an overview with the latest training samples in the [algorithm docs](https://ftag.docs.cern.ch/software/samples/) (*it will take quite a while to download these samples*).

The first step of the whole Umami chain is the preparation of the `.h5` samples coming from the dumper. The preprocessing of Umami (and nearly all other features) are steered by `yaml` config files. Examples for most of these config files can be found in the `examples/` folder of the Umami repository. For the first part, the preprocessing, you will need the examples files in `examples/preprocessing/`.

In the preparation part, we split the `.h5` samples in training, validation and testing. This split is needed to ensure an unbiased training and evaluation of the taggers. The jets are split into the three categories: training, validation and testing.
For training the jets will also be separated in their respective flavours (b-jets, c-jets, light-flavour jets). This is needed for the resampling algorithm which is covered in the next part of the tutorial.
For the validation of the algorithms, both an inclusive sample (not separated by flavour) and samples separated by flavour are created. The inclusive sample can be used to validate the training in a scenario resembling a real physics (composition-wise) use-case, e.g. for checking the performance in top quark pair events.
The samples separated by flavour are used for the creation of a hybrid validation sample, which is used for checking against overfitting and other training problems. The creation of the hybrid validation sample is covered in the next part.
For testing of the algorithms, test samples are created. Only inclusive (not separated by flavour) samples are created because the performance of the tagger is always evaluated on real physics samples.

The first task is to have a look at the `PFlow-Preprocessing.yaml` config file which is the main config file here. In there you will find different sections with options for different parts of the preprocessing. The important part for this step is the `preparation` part. Here you can add the path to the output files of the training-dataset-dumper. In the example, the samples `ttbar` and `zprime` are defined. The `*ntuple_path` is a [yaml anchor](https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/) and is defined in the `Preprocessing-parameters.yaml` file. Change the `file_pattern` of the two entries so they match your samples. An explanation of the options of `preparation` is given [here](../preprocessing/ntuple_preparation.md#file-and-flavour-preparation).

After the you added the correct `file_pattern`, we also need to change the path to these folders. All global paths (input and output) are defined as yaml anchors in the `Preprocessing-parameters.yaml` file. In there you will find all the paths were we will store or load files from. The first step is to adapt all the paths for you.

??? info "Hint: Adapting the `Preprocessing-parameters.yaml` configs"

    The explanation of the options of the `PFlow-Preprocessing.yaml` are given [here](../preprocessing/ntuple_preparation.md#config-file).

??? warning "Solution: Adapting the `Preprocessing-parameters.yaml` configs"

    Replace `<path_place_holder>` with your path to the test files we provided. Also you need to replace the `<base_path_place_holder>` with the path to where the preprocessed samples should be stored. For the `var_file`, you need to give the path to your variable config file.

    ```yaml
    # Path where the ntuples are saved
    ntuple_path: &ntuple_path <path_place_holder>/ntuples/

    # Base path where to store preprocessing results
    .base_path: &base_path <base_path_place_holder>

    # Path where the hybrid samples will be saved
    sample_path: &sample_path !join [*base_path, /prepared_samples]

    # Path where the merged and ready-to-train samples are saved
    file_path: &file_path !join [*base_path, /preprocessed]

    # Name of the output file from the preprocessing used for training (has to be a .h5 file, no folder)
    .outfile_name: &outfile_name !join [*base_path, /PFlow-hybrid.h5]

    # List of variables for training (yaml)
    .var_file: &var_file <path_place_holder>/umami/umami/configs/DL1r_Variables_R22.yaml

    # Dictfile for the scaling and shifting (json)
    .dict_file: &dict_file !join [*base_path, /scale_dicts/PFlow-scale_dict.json]

    # Intermediate file for the training sample indicies used in h5 format
    .intermediate_index_file: &intermediate_index_file !join [*base_path, /preprocessed/indicies.h5]

    # Name of the output file from the preprocessing used for hybrid validation (has to be a .h5 file, no folder)
    # Will be ignored if hybrid validation is not used
    outfile_name_validation: !join [*base_path, /PFlow-hybrid-validation.h5]

    # Intermediate file for the hybrid validation sample indicies used in h5 format
    # Will be ignored if hybrid validation is not used
    intermediate_index_file_validation: !join [*base_path, /preprocessed/indicies-hybrid-validation.h5]
    ```

After all the paths are prepared, you can now have a look at the `Preprocessing-samples.yaml` config file, in which all the inclusive samples and samples separated by flavour are listed and defined. Here, you need to remove the $\tau$ jets and change the number of jets for the training samples to `1e6`, validation and test samples to `3e5`. These numbers are **only** for the tutorial. If you want to make use of your whole statistics available, choose the number of jets for the training jets as high as possible! This is only the number how much jets are extracted from the `.h5` files and **not** the numbers you will use for training. With a large number here, the resampling algorithms just have more jets to choose from. The validation and testing sample number shouldn't be too large (not larger than `4e6`) otherwise the loading of these files will take a huge amount of time.

??? info "Hint: Prepare the Preprocessing-samples.yaml"

    A detailed description about that part is given [here](../preprocessing/ntuple_preparation.md#file-and-flavour-preparation)

??? warning "Solution: Prepare the Preprocessing-samples.yaml"

    ```yaml
    training_ttbar_bjets:
        type: ttbar
        category: bjets
        n_jets: 1e6
        <<: *cuts_template_training_ttbar
        output_name: !join [*sample_path, /bjets_training_ttbar_PFlow.h5]

    training_ttbar_cjets:
        type: ttbar
        category: cjets
        n_jets: 1e6
        <<: *cuts_template_training_ttbar
        output_name: !join [*sample_path, /cjets_training_ttbar_PFlow.h5]

    training_ttbar_ujets:
        type: ttbar
        category: ujets
        n_jets: 1e6
        <<: *cuts_template_training_ttbar
        output_name: !join [*sample_path, /ujets_training_ttbar_PFlow.h5]

    training_zprime_bjets:
        type: zprime
        category: bjets
        n_jets: 1e6
        <<: *cuts_template_training_zprime
        output_name: !join [*sample_path, /bjets_training_zprime_PFlow.h5]

    training_zprime_cjets:
        type: zprime
        category: cjets
        n_jets: 1e6
        <<: *cuts_template_training_zprime
        output_name: !join [*sample_path, /cjets_training_zprime_PFlow.h5]

    training_zprime_ujets:
        type: zprime
        category: ujets
        n_jets: 1e6
        <<: *cuts_template_training_zprime
        output_name: !join [*sample_path, /ujets_training_zprime_PFlow.h5]

    validation_ttbar:
        type: ttbar
        category: inclusive
        n_jets: 3e5
        <<: *cuts_template_validation
        output_name: !join [*sample_path, /inclusive_validation_ttbar_PFlow.h5]

    validation_ttbar_bjets:
        type: ttbar
        category: bjets
        n_jets: 3e5
        <<: *cuts_template_validation_ttbar_hybrid
        output_name: !join [*sample_path, /bjets_validation_ttbar_PFlow.h5]

    validation_ttbar_cjets:
        type: ttbar
        category: cjets
        n_jets: 3e5
        <<: *cuts_template_validation_ttbar_hybrid
        output_name: !join [*sample_path, /cjets_validation_ttbar_PFlow.h5]

    validation_ttbar_ujets:
        type: ttbar
        category: ujets
        n_jets: 3e5
        <<: *cuts_template_validation_ttbar_hybrid
        output_name: !join [*sample_path, /ujets_validation_ttbar_PFlow.h5]

    validation_zprime:
        type: zprime
        category: inclusive
        n_jets: 3e5
        <<: *cuts_template_validation
        output_name: !join [*sample_path, /inclusive_validation_zprime_PFlow.h5]

    validation_zprime_bjets:
        type: zprime
        category: bjets
        n_jets: 3e5
        <<: *cuts_template_validation_zprime_hybrid
        output_name: !join [*sample_path, /bjets_validation_zprime_PFlow.h5]

    validation_zprime_cjets:
        type: zprime
        category: cjets
        n_jets: 3e5
        <<: *cuts_template_validation_zprime_hybrid
        output_name: !join [*sample_path, /cjets_validation_zprime_PFlow.h5]

    validation_zprime_ujets:
        type: zprime
        category: ujets
        n_jets: 3e5
        <<: *cuts_template_validation_zprime_hybrid
        output_name: !join [*sample_path, /ujets_validation_zprime_PFlow.h5]

    testing_ttbar:
        type: ttbar
        category: inclusive
        n_jets: 3e5
        <<: *cuts_template_testing
        output_name: !join [*sample_path, /inclusive_testing_ttbar_PFlow.h5]

    testing_zprime:
        type: zprime
        category: inclusive
        n_jets: 3e5
        <<: *cuts_template_testing
        output_name: !join [*sample_path, /inclusive_testing_zprime_PFlow.h5]
    ```

The `cuts` which are applied here are defined in the `Preprocessing-cut_parameters.yaml` file. The cuts in there are outlier cuts and $p_T$ cuts. Also, a cut on the `eventNumber` is applied to split the samples in training/validation/testing and ensure their orthogonality. If you want to apply other cuts on the samples, you can change them as you like, but for the tutorial, we will go with the default settings.

Now that all our config files are prepared, we can start the preparation step. Try to run the preparation!

??? info "Hint: Run the preparation step"

    An explanation how to run the preparation step can be found [here](../preprocessing/ntuple_preparation.md#run-the-preparation)

??? warning "Solution: Run the preparation step"

    To run the preparation, switch to the `umami/umami` folder of your forked repo and run the following command:

    ```bash
    preprocessing.py --config <path to config file> --prepare
    ```

    where `<path to config file>` is the path to the `PFlow-Preprocessing.yaml` file. There is also an option to parallelize this due to the large number of samples that need to be prepared. An explanation is given [here](../preprocessing/ntuple_preparation.md#run-the-preparation).

#### 3.1. Resampling

In this step, we are going to combine the different flavours, which were split in the `Preparation` step, such that the combined sample provides a desired composition after resampling. To retrieve the resampling factors and which jets are resampled and which are not, Umami provides different resampling strategies. For this tutorial, you can either use the `count` or the `pdf` method, although we are encouraging you to use the `count` method, due to the huge size the `pdf` training dataset will have.

**Note: If you did the `preparation` part, you don't need to adapt the `Preprocessing-parameters.yaml` file if you set all options already.**
The first task for you is to adapt the example `Preprocessing-parameters.yaml` and add your paths. A very detailed explanation of the all the options and paths in these files are given in the [Umami preprocessing documentation](../preprocessing/Overview.md). Important here is to use the correct variable dict file. The one we want to use is `umami/umami/configs/DL1r_Variables_R22.yaml`

??? info "Hint: Adapting the `Preprocessing-parameters.yaml` configs"

    The explanation of the options of the `PFlow-Preprocessing.yaml` are given [here](../preprocessing/ntuple_preparation.md#config-file).

??? warning "Solution: Adapting the `Preprocessing-parameters.yaml` configs"

    Replace `<path_place_holder>` with your path to the test files we provided and you retrieved in part 2 of this tutorial. Also you need to replace the `<base_path_place_holder>` with the path to where the preprocessed samples should be stored. For the `var_file`, you need to give the path to your variable config file. 

    ```yaml
    # Path where the ntuples are saved
    ntuple_path: &ntuple_path <path_place_holder>/ntuples/

    # Base path where to store preprocessing results
    .base_path: &base_path <base_path_place_holder>

    # Path where the hybrid samples will be saved
    sample_path: &sample_path !join [*base_path, /prepared_samples]

    # Path where the merged and ready-to-train samples are saved
    file_path: &file_path !join [*base_path, /preprocessed]

    # Name of the output file from the preprocessing used for training (has to be a .h5 file, no folder)
    .outfile_name: &outfile_name !join [*base_path, /PFlow-hybrid.h5]

    # List of variables for training (yaml)
    .var_file: &var_file <path_place_holder>/umami/umami/configs/DL1r_Variables_R22.yaml

    # Dictfile for the scaling and shifting (json)
    .dict_file: &dict_file !join [*base_path, /scale_dicts/PFlow-scale_dict.json]

    # Intermediate file for the training sample indicies used in h5 format
    .intermediate_index_file: &intermediate_index_file !join [*base_path, /preprocessed/indicies.h5]

    # Name of the output file from the preprocessing used for hybrid validation (has to be a .h5 file, no folder)
    # Will be ignored if hybrid validation is not used
    outfile_name_validation: !join [*base_path, /PFlow-hybrid-validation.h5]

    # Intermediate file for the hybrid validation sample indicies used in h5 format
    # Will be ignored if hybrid validation is not used
    intermediate_index_file_validation: !join [*base_path, /preprocessed/indicies-hybrid-validation.h5]
    ```

An important next step is to check the variable dict file we want to use. In there are all variables defined we will use for the training of DL1d. Have a look at the `DL1r_Variables_R22.yaml` and check which variables are present in there. You will see that still RNNIP values are used for the training. For DL1d, we need to switch that to DIPS. Replace the RNNIP values with their corresponding DIPS values. The DIPS model name here is `dipsLoose20220314v2` (more information on that algorithm [here](https://ftag-docs.docs.cern.ch/algorithms/available_taggers/#trained-on-r22-p4931-with-23m-jets)).

??? info "Hint: Add DIPS to Variable Dict"

    The variable names consists of the model name (e.g `rnnip`) and the output probability (e.g `pb`).

??? info "Solution: Add DIPS to Variable Dict"

    The correct variable dict will look like this:

    ```yaml
    label: HadronConeExclTruthLabelID
    train_variables:
    JetKinematics:
        - absEta_btagJes
        - pt_btagJes
    JetFitter:
        - JetFitter_isDefaults
        - JetFitter_mass
        - JetFitter_energyFraction
        - JetFitter_significance3d
        - JetFitter_nVTX
        - JetFitter_nSingleTracks
        - JetFitter_nTracksAtVtx
        - JetFitter_N2Tpair
        - JetFitter_deltaR
    JetFitterSecondaryVertex:
        - JetFitterSecondaryVertex_isDefaults
        - JetFitterSecondaryVertex_nTracks
        - JetFitterSecondaryVertex_mass
        - JetFitterSecondaryVertex_energy
        - JetFitterSecondaryVertex_energyFraction
        - JetFitterSecondaryVertex_displacement3d
        - JetFitterSecondaryVertex_displacement2d
        - JetFitterSecondaryVertex_maximumTrackRelativeEta
        - JetFitterSecondaryVertex_minimumTrackRelativeEta
        - JetFitterSecondaryVertex_averageTrackRelativeEta
        - JetFitterSecondaryVertex_maximumAllJetTrackRelativeEta  # Modified name in R22. Was: maximumTrackRelativeEta
        - JetFitterSecondaryVertex_minimumAllJetTrackRelativeEta  # Modified name in R22. Was: minimumTrackRelativeEta
        - JetFitterSecondaryVertex_averageAllJetTrackRelativeEta  # Modified name in R22. Was: averageTrackRelativeEta
    SV1:
        - SV1_isDefaults
        - SV1_NGTinSvx
        - SV1_masssvx
        - SV1_N2Tpair
        - SV1_efracsvx
        - SV1_deltaR
        - SV1_Lxy
        - SV1_L3d
        - SV1_correctSignificance3d # previously SV1_significance3d
    DIPS:
        - dipsLoose20220314v2_pb
        - dipsLoose20220314v2_pc
        - dipsLoose20220314v2_pu

    custom_defaults_vars:
        JetFitter_energyFraction: 0
        JetFitter_significance3d: 0
        JetFitter_nVTX: -1
        JetFitter_nSingleTracks: -1
        JetFitter_nTracksAtVtx: -1
        JetFitter_N2Tpair: -1
        SV1_N2Tpair: -1
        SV1_NGTinSvx: -1
        SV1_efracsvx: 0
        JetFitterSecondaryVertex_nTracks: 0
        JetFitterSecondaryVertex_energyFraction: 0
    ```

After the adaptation of the `Preprocessing-parameters.yaml` and the variable dict is done, you also need to adapt the `PFlow-Preprocessing.yaml` config file, which is the main config file for the preprocessing. While the first part of the file is for the `preparation` step, we will focus now on the `sampling` part.
In this particular section, the options for the resampling are defined. Your next task is to adapt this accordingly to the resampling method you want to use. 

For the `count` method, you need to set the number of jets in the final training file to `1.5e6` and deactivate the tracks.

For the `pdf` method, you need to set the maximum oversampling ratio for the `cjets` to 5 and the number of jets per class in the final training file to `2e6` and deactivate the tracks.

??? info "Hint: Adapt the `PFlow-Preprocessing.yaml` config file"

    You can find a detailed description about the options of the `sampling` part [here](../preprocessing/resampling.md#general-config-file-options)

??? warning "Solution: Adapt the `PFlow-Preprocessing.yaml` config file"

    For the `count` approach, the part should look like this:

    ```yaml
    sampling:
        # Classes which are used in the resampling. Order is important.
        # The order needs to be the same as in the training config!
        class_labels: [ujets, cjets, bjets]

        # Decide, which resampling method is used.
        method: count

        # The options depend on the sampling method
        options:
            sampling_variables:
            - pt_btagJes:
                # bins take either a list containing the np.linspace arguments
                # or a list of them
                # For PDF sampling: must be the np.linspace arguments.
                #   - list of list, one list for each category (in samples)
                #   - define the region of each category.
                bins: [[0, 600000, 351], [650000, 6000000, 84]]

            - absEta_btagJes:
                # For PDF sampling: same structure as in pt_btagJes.
                bins: [0, 2.5, 10]

            # Decide, which of the in preparation defined samples are used in the resampling.
            samples_training:
                ttbar:
                    - training_ttbar_bjets
                    - training_ttbar_cjets
                    - training_ttbar_ujets

                zprime:
                    - training_zprime_bjets
                    - training_zprime_cjets
                    - training_zprime_ujets

            samples_validation:
                ttbar:
                    - validation_ttbar_bjets
                    - validation_ttbar_cjets
                    - validation_ttbar_ujets

                zprime:
                    - validation_zprime_bjets
                    - validation_zprime_cjets
                    - validation_zprime_ujets

            custom_n_jets_initial:
                # these are empiric values ensuring a smooth hybrid sample.
                # These values are retrieved for a hybrid ttbar + zprime sample for the count method!
                training_ttbar_bjets: 5.5e6
                training_ttbar_cjets: 11.5e6
                training_ttbar_ujets: 13.5e6

            # Fractions of ttbar/zprime jets in final training set. This needs to add up to one.
            fractions:
                ttbar: 0.7
                zprime: 0.3

            # number of training jets
            # For PDF sampling: the number of target jets per class!
            #                   So if you set n_jets=1_000_000 and you have 3 output classes
            #                   you will end up with 3_000_000 jets
            # For other sampling methods: total number of jets after resampling
            # If set to -1: max out to target numbers (limited by fractions ratio)
            n_jets: 1.5e6

            # number of validation jets in the hybrid validation sample
            # Same rules as above for n_jets when it comes to PDF sampling
            n_jets_validation: 3e5

            # Bool, if track information (for DIPS etc.) are saved.
            save_tracks: False

            # Name(s) of the track collection(s) to use.
            tracks_names: null

            # Bool, if track labels are processed
            save_track_labels: False

            # String with the name of the track truth variable
            track_truth_variables: null

            # this stores the indices per sample into an intermediate file
            intermediate_index_file: *intermediate_index_file

            # for method: weighting
            # relative to which distribution the weights should be calculated
            weighting_target_flavour: 'bjets'

            # If you want to attach weights to the final files
            bool_attach_sample_weights: False

            # How many jets you want to use for the plotting of the results
            # Give null (the yaml None) if you don't want to plot them
            n_jets_to_plot: 3e4
    ```

    For the `pdf` method, this should look like this:

    ```yaml
    sampling:
        # Classes which are used in the resampling. Order is important.
        # The order needs to be the same as in the training config!
        class_labels: [ujets, cjets, bjets]

        # Decide, which resampling method is used.
        method: pdf

        # The options depend on the sampling method
        options:
            sampling_variables:
            - pt_btagJes:
                # bins take either a list containing the np.linspace arguments
                # or a list of them
                # For PDF sampling: must be the np.linspace arguments.
                #   - list of list, one list for each category (in samples)
                #   - define the region of each category.
                bins: [[0, 25e4, 100], [25e4, 6e6, 100]]

            - absEta_btagJes:
                # For PDF sampling: same structure as in pt_btagJes.
                bins: [[0, 2.5, 10], [0, 2.5, 10]]

            # Decide, which of the in preparation defined samples are used in the resampling.
            samples_training:
                ttbar:
                    - training_ttbar_bjets
                    - training_ttbar_cjets
                    - training_ttbar_ujets

                zprime:
                    - training_zprime_bjets
                    - training_zprime_cjets
                    - training_zprime_ujets

            samples_validation:
                ttbar:
                    - validation_ttbar_bjets
                    - validation_ttbar_cjets
                    - validation_ttbar_ujets

                zprime:
                    - validation_zprime_bjets
                    - validation_zprime_cjets
                    - validation_zprime_ujets

            # This is empty for pdf!
            custom_n_jets_initial:

            # Fractions of ttbar/zprime jets in final training set. This needs to add up to one.
            fractions:
                ttbar: 0.7
                zprime: 0.3

            # For PDF sampling, this is the maximum upsampling rate (important to limit tau upsampling)
            # File are referred by their key (as in custom_njets_initial)
            max_upsampling_ratio:
                training_ttbar_cjets: 5
                training_zprime_cjets: 5

            # number of training jets
            # For PDF sampling: the number of target jets per class!
            #                   So if you set n_jets=1_000_000 and you have 3 output classes
            #                   you will end up with 3_000_000 jets
            # For other sampling methods: total number of jets after resampling
            # If set to -1: max out to target numbers (limited by fractions ratio)
            n_jets: 5e5

            # number of validation jets in the hybrid validation sample
            # Same rules as above for n_jets when it comes to PDF sampling
            n_jets_validation: 1e5

            # Bool, if track information (for DIPS etc.) are saved.
            save_tracks: False

            # Name(s) of the track collection(s) to use.
            tracks_names: null

            # Bool, if track labels are processed
            save_track_labels: False

            # String with the name of the track truth variable
            track_truth_variables: null

            # this stores the indices per sample into an intermediate file
            intermediate_index_file: *intermediate_index_file

            # for method: weighting
            # relative to which distribution the weights should be calculated
            weighting_target_flavour: 'bjets'

            # If you want to attach weights to the final files
            bool_attach_sample_weights: False

            # How many jets you want to use for the plotting of the results
            # Give null (the yaml None) if you don't want to plot them
            n_jets_to_plot: 3e4
    ```

After the `sampling` options are set, you can also have a look at the more general options at the bottom of the file. For this tutorial, the default values provided in the file are fine.

Now you need to start the first (or second when you have done the prepare step on your own) part of the preprocessing. The different steps of the preprocessing can be run sequentially, one by one. Start by running the resampling for the main training sample. You also need to run the resampling for the hybrid validation sample. The hybrid validation sample is used while training to validate the performance of the model in metrics of loss, accuracy and rejection per epoch. Due to the resampling, using the un-resampled validation files for checking against overtraining is no recommended due to the different composition of flavours in these samples.

??? info "Hint: Run the resampling step of the preprocessing"

    An explanation how to run the different steps of the preprocessing can be found in their respective sections in the [Umami documentation](../preprocessing/Overview.md)

??? warning "Solution: Run the resampling step of the preprocessing"

    ```bash
    preprocessing.py --config <path to config file> --resampling
    ```

    where `<path to config file>` is the path to your `PFlow-Preprocessing.yaml`.

    To produce the hybrid validation sample, you just need to run the resampling with the extra `--hybrid_validation` flag. The command looks like this

    ```bash
    preprocessing.py --config <path to config file> --resampling --hybrid_validation
    ```

While the resampling is running, plots from the variables in the variable config file before and after the resampling are created. You can check if the resampling was done correctly by checking this plot. The plots are stored in the `file_path` path in a new folder called `plots/`. 

#### 3.2. Scaling/Shifting

After the resampling is finished, the next task is to calculate the scaling and shifting values for the training set. For that, you don't need to adapt any config file. The files should already be prepared for this step. The output of this will be the scale dict which will be saved in a `.json` file.

??? info "Hint: Run the Scaling/Shifting calculation"

    An explanation how to run the different steps of the preprocessing can be found in their respective sections in the [Umami documentation](../preprocessing/Overview.md)

??? warning "Solution: Run the Scaling/Shifting calculation"

    You need to run the following command:

    ```bash
    preprocessing.py --config <path to config file> --scaling
    ```

    where `<path to config file>` is the path to your `PFlow-Preprocessing.yaml`.

#### 3.3. Writing

In the final step of the preprocessing, the final training file with only the scaled/shifted train variables is written to disk. Like the step before, the config files are already prepared and you only need to run the writing command.

??? info "Hint: Run the writing"

    An explanation how to run the different steps of the preprocessing can be found in their respective sections in the [Umami documentation](../preprocessing/Overview.md)

??? warning "Solution: Run the writing"

    You need to run the following command:

    ```bash
    preprocessing.py --config <path to config file> --write
    ```

    where `<path to config file>` is the path to your `PFlow-Preprocessing.yaml`.

After the writing step is done, you can check the content of the files by running the following command:

```bash
h5ls -vr <file>
```

This will show you the structure of the final training file which should contain a group called `jets` with some entries. These entries are `inputs`, `labels`, `labels_one_hot` and `weight`. The first entry, the `inputs` are the inputs for the network. The group `jets` tells you that these are the jet inputs for DL1d. The `labels` and `labels_one_hot` are the truth/target labels of the jets. The `ont_hot` are the same labels, but one hot encoded (which is used in Umami, the other ones are used for the GNN). The `weight` is only used if the weighting resampling was chosen, otherwise these are ones for all jets (but are not used).
A nice feature of the command is, that you can see which variables are stored for the entries (with the correct ordering). This is a good way to check which flavours were used for creating this samples and which variables were used. You can also so how many jets are in the training file.

**NOTE** If you run this command on a large file with a lot of jets (like above 10M or so), this command is very slow and could die! Be careful on which sample you cast this.

#### 3.4 Checkpoint Files Preprocessing

If for some reason, the preprocessing didn't work for you and there is no time to retry, you can download and continue with the following files. Please keep in mind that you need to safe the files to the correct places! To get the files, run the following command:

```bash
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/checkpoint_preprocessing/my_preprocessing_checkpoint.zip
```

After downloading, you can unpack the `.zip` file by running `unzip my_preprocessing_checkpoint.zip`. Please keep in mind that you need to adapt the paths in the `Preprocessing-parameters.yaml` file to your paths! Otherwise the training will not work!

On `lxplus` you can instead run the `cp` command to copy from EOS:

```bash
cp /eos/user/u/umamibot/www/ci/tutorial/umami/checkpoint_preprocessing/my_preprocessing_checkpoint.zip ./
```

### 4. Train DL1d for the preprocessed samples

After the preprocessing is finished, the next step is the actual training of the tagger. For this, we need first another config file. As a base config file, we will now use the `DL1r-PFlow-Training-config.yaml` from the `examples` folder from Umami. The config file consists of 4 big parts:

1. General Options (Everything before `nn_structure`)
2. The settings for the neural network (`nn_structure`)
3. The settings for the validation of the training (`validation_settings`)
4. The settings for the evaluation of the performance of one model (`evaluation_settings`)

The latter one will be covert in the next section of this tutorial. First, we will now focus on the general options we need to change.

#### 4.1. Adapt the general options

For the first task in this section, you should adapt the first part of the config file (everything until `nn_structure`) with your paths and settings. Also, you should add your hybrid validation file to the `validation_files` part.

??? info "Hint: Adapt the general options"

    You can find the detailed description of these options [here](../trainings/train.md#global-settings).
    You don't need to adapt the `variable_cuts_*` here. Those are already correctly defined for the cuts we apply in the preprocessing.
    Also, your hybrid validation file does not need any further cuts! Those were already applied when the file was created.

??? warning "Solution: Adapt the general options"

    The general options are rather easy to set. This could look like this:

    ```yaml
    # Set modelname and path to Pflow preprocessing config file
    model_name: My_DL1d_Tutorial_Model
    preprocess_config: <path_to_your_preprocessing_config>

    # Add here a pretrained model to start with.
    # Leave empty for a fresh start
    model_file:

    # Add training file
    train_file: <path_place_holder>/PFlow-hybrid-resampled_scaled_shuffled.h5

    # Defining templates for the variable cuts
    .variable_cuts_ttbar: &variable_cuts_ttbar
        variable_cuts:
            - pt_btagJes:
                operator: "<="
                condition: 2.5e5

    .variable_cuts_zpext: &variable_cuts_zpext
        variable_cuts:
            - pt_btagJes:
                operator: ">"
                condition: 2.5e5

    # Add validation files
    validation_files:
        r22_hybrid_val:
            path: <path_place_holder>/PFlow-hybrid-validation-resampled.h5
            label: "Hybrid Validation"

        ttbar_r22_val:
            path: <path_place_holder>/inclusive_validation_ttbar_PFlow.h5
            label: "$t\\bar{t}$ Validation"
            <<: *variable_cuts_ttbar

        zprime_r22_val:
            path: <path_place_holder>/inclusive_validation_zprime_PFlow.h5
            label: "$Z'$ Validation"
            <<: *variable_cuts_zpext

    test_files:
        ttbar_r22:
            path: /work/ws/nemo/fr_af1100-Training-Simulations-0/tutorial_hybrids/inclusive_testing_ttbar_PFlow.h5
            <<: *variable_cuts_ttbar

        zpext_r22:
            path: /work/ws/nemo/fr_af1100-Training-Simulations-0/tutorial_hybrids/inclusive_testing_zprime_PFlow.h5
            <<: *variable_cuts_zpext


    exclude: null
    ```

    where `<path_place_holder>` is the path to either the `preprocessed` folder where the train file is stored or the path to the files we provided for you. The `preprocess_config` option also needs to be set accordingly to the path where the adapted preprocessing config is stored. The `model_name` is the foldername which is created while running the training where everything will be stored.
    For the validation- and test files, you need to set the variable cuts correctly for the physics (non-resampled) files. For the hybrid validation sample, these cuts were already applied when creating the sample. Therefore we need no further cuts here.


#### 4.2. Adapt the network settings

The second part of the config, the `nn_structure` defines the architecture and the main tagging options of the network we are going to train. The default network size is quite large. Try to reduce the networks size by removing the first two layers. Also, change the number of epochs to 25. 
You can also try to change some of the other settings, i.e deactivate the learning rate reducer (`lrr`) or add more dropout to the layers. That's up to you. But for the tutorial, it is suggested to leave them as they are.

??? info "Hint: Remove the first two layers"

    The layers are defined via `dense_sizes` and `activations`. You can have a look [here](../trainings/train.md#network-settings) for a more detailed explanation.

??? warning "Solution: Remove the first two layers"

    The layers are defined in chronological order in these list. The `nn_structure` part should look like this:

    ```yaml
    nn_structure:
        # Decide, which tagger is used
        tagger: "dl1"

        # NN Training parameters
        lr: 0.001
        batch_size: 15000
        epochs: 25

        # Number of jets used for training
        # To use all: Fill nothing
        n_jets_train:

        # Dropout rates for the dense layers
        # --> has to be a list of same length as the `dense_sizes` list
        # The example here would use a dropout rate of 0.2 for the two middle layers but
        # no dropout for the other layers
        dropout_rate: [0, 0.2, 0.2, 0, 0, 0]

        # Define which classes are used for training
        # These are defined in the global_config
        class_labels: ["ujets", "cjets", "bjets"]

        # Main class which is to be tagged
        main_class: "bjets"

        # Decide if Batch Normalisation is used
        batch_normalisation: False

        # Nodes per dense layer. Starting with first dense layer.
        dense_sizes: [60, 48, 36, 24, 12, 6]

        # Activations of the layers. Starting with first dense layer.
        activations: ["relu", "relu", "relu", "relu", "relu", "relu"]

        # Variables to repeat in the last layer (example)
        repeat_end: ["pt_btagJes", "absEta_btagJes"]

        # Options for the Learning Rate reducer
        lrr: True

        # Option if you want to use sample weights for training
        use_sample_weights: False
    ```

#### 4.3. Adapt the validation settings

Before we can start the actual training, the validation settings need to be set because the validation metrics are calculated either on-the-fly after each epoch or after the training itself (the latter one will be covered after the training in this section).
For now, try to deactivate the on-the-fly calculation of the validation metrics.

??? info "Hint: Activate/Deactivate the on-the-fly calculation of validation metrics"

    Have a look in the [Umami documentation](../trainings/train.md#running-the-training) and look for how to run the training.

??? warning "Solution: Activate/Deactivate the on-the-fly calculation of validation metrics"

    To activate/deactivate the on-the-fly calculation of the validation metrics (validation loss, validation rejection per epoch etc.), you need to give a value for the `n_jets` option in the `validation_settings` part of the train config file. A value of `None` or `0` deactivates it, while a value greater than `0` activates it. Also, this value is the number of jets which are to be used for the calculation of the metrics.

Another thing you can already change is the label of the tagger that you are going to train. Also, you can check which taggers you want to plot as comparison to the rejection per epoch plots.

??? info "Hint: Change tagger name and comparison tagger"

    Have a look in the [Umami documentation](../trainings/train.md#running-the-training) and look for the `taggers_from_file` and `tagger_label` option.

??? warning "Solution: Change tagger name and comparison tagger"

    The `tagger_label` will be the name of the tagger displayed in the validation plots in the legend. The `taggers_from_file` are taggers that are present in the `.h5` validation files. If this option is active, horizontal lines are plotted in the rejection vs epoch validation plots which provide a comparison of the freshly trained tagger to the reference taggers.

#### 4.4. Run the training

After network and validation settings are prepared, we can prepare the real training. In the preparation, your config files, scale- and variable dict are copied to the model folder (which will be created). Also, the paths inside of the configs are changed to the new model folder. The configs/dicts will be now in a folder in `umami/umami` `<your_model_name/metadata>`. First step here is to run the preparation step.

??? info "Hint: Run the Preparation"

    Have a closer look at the Umami documentation [here](../trainings/train.md#running-the-training)

??? warning "Solution: Run the Preparation"

    To run the training, you need first to switch to the `umami/umami` directory in your forked repo. Here you can simply run the following command:

    ```bash
    train.py -c <path to train config file> --prepare
    ```

    where `<path to train config file>` is the path to your train config file. This will not start the training, but the preparation of the model folder.

After the preparation we can now start the training of the tagger! From now on, the path to all our config files is always `<your_model_name>/metadata/<config_file>`! We are now using the ones stored in the `metadata` folder and we will also only adapt them! Try to run the training now with this!

??? info "Hint: Running the training"

    Have a closer look at the Umami documentation [here](../trainings/train.md#running-the-training)

??? warning "Solution: Running the training"

    To run the training, you need first to switch to the `umami/umami` directory in your forked repo. Here you can simply run the following command:

    ```bash
    train.py -c <path to train config file>
    ```

    where `<path to train config file>` is the path to your train config file.

#### 4.5. Validate the Training

After the training successfully finished, the next step is to figure out which epoch to use for the evaluation. To do so, we can use the different validation samples produced during the preprocessing. In the `Preparation` step, the different validation and test samples are produced except the hybrid validation sample. This is produced during the `Resampling` step. Due to the fact that we already added all the validation- and test samples in the first step of this section, we just need to reactivate the validation again by setting the `n_jets` in `validation_settings` to a higher value than `0`. After that, you can run
the validation.

??? info "Hint: Running the validation"

    To run the validation, you need to switch again to the `umami/umami` directory in your forked repo. For the correct command, have a closer look [here](../trainings/validate.md#running-the-validation)

??? warning "Solution: Running the validation"

    To run the validation, you need to execute the following command in the `umami/umami` folder of your repo:

    ```bash
    plotting_epoch_performance.py -c <path to train config file> --recalculate
    ```

    The `--recalculate` option tells the script to load the validation samples and (re)calculate the validation metrics, like validation loss, validation accuracy and the rejection per epoch. The results will be saved in a `.json` file. Also, the script will automatically plot the metrics after the calculation is done. If you just want to re-plot the plots, run the command without the `--recalculate` option. Further explanation is given [here](../trainings/validate.md#running-the-validation)

Now have a look at the plots. You will notice that all plots are rather small and the legend is colliding with a lot of other stuff. This is due to the default figure size of `Puma` which is used to create these plots. But, Umami can handle that! Just set the `Puma` argument `figsize` in the `validation_settings` block of your train config and re-run the validation plotting. Note: You don't need the `--recalculate` option to change something which is purely plot-related!

??? warning "Solution: Re-run the validation"

    To re-run the validation, you need to execute the following command in the `umami/umami` folder of your repo:

    ```bash
    plotting_epoch_performance.py -c <path to train config file>
    ```

    This will re-run only the plotting of the results and will not recalculate all the metrics for the validation samples.

After all our plots are nice and presentable (you can of course adapt them further with more `Puma` arguments. These are listed [here](../plotting/plotting_inputs.md#list-of-puma-parameters)), you need to find an epoch you want to use for further evaluation based on the loss, accuracy and the rejections. Keep that number and go to the next part!

#### 4.6 Checkpoint Files Training

If for some reason, the training didn't work for you and there is no time to retry, you can download and continue with the following files. Please keep in mind that you need to safe the files to the correct places! To get the files, run the following command:

```bash
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/checkpoint_training/My_DL1d_Tutorial_Model.zip
```

After downloading, you can unpack the `.zip` file by running `unzip My_DL1d_Tutorial_Model.zip`. Copy the unzipped folder in your `umami/umami/` folder of your forked repo and adapt the paths inside the config files in `metadata/`!

On `lxplus` you can instead run the `cp` command to copy from EOS:

```bash
cp /eos/user/u/umamibot/www/ci/tutorial/umami/checkpoint_preprocessing/My_DL1d_Tutorial_Model.zip ./
```

### 5. Evaluate the freshly trained DL1d tagger

After the training and validation of the tagger is done and you have chosen an epoch for further evaluation, we can start the evaluation process. The evaluation (in comparison to the validation) will use the model from your chosen epoch and more detailed performance measures will be calculated using the testing samples. But before we can run this, we need to adapt the config again.

#### 5.1 Adapt the evaluation settings

The `evaluation_settings` are located in your train config file (at the bottom). It is the last big part of the train config file. The first task here is to add `dipsLoose20220314v2` to the comparison tagger list with the fraction values of $f_c = 0.005$ and $f_u = 0.995$.

??? info "Hint: Add DIPS to the comparison taggers"

    The comparison tagger list is called `tagger`. The fraction values are stored in `frac_values_comp` for comparison taggers. For a further explanation of the options, look [here](../trainings/evaluate.md#config)

??? warning "Solution: Add DIPS to the comparison taggers"

    To add DIPS to the tagger comparison list, you need to add the full name of the tagger to `tagger`. Also you need to make an entry in `frac_values_comp` with the respective fraction values for DIPS. In file, this will look like this.

    ```yaml
    # Eval parameters for validation evaluation while training
    evaluation_settings:
        # Number of jets used for evaluation
        n_jets: 3e5

        # Define taggers that are used for comparison in evaluate_model
        # This can be a list or a string for only one tagger
        tagger: ["rnnip", "DL1r", "dipsLoose20220314v2"]

        # Define fc values for the taggers
        frac_values_comp: {
            "rnnip": {
                "cjets": 0.07,
                "ujets": 0.93,
            },
            "DL1r": {
                "cjets": 0.018,
                "ujets": 0.982,
            },
            "dipsLoose20220314v2": {
                "cjets": 0.005,
                "ujets": 0.995,
            }
        }

        # Charm fraction value used for evaluation of the trained model
        frac_values: {
            "cjets": 0.018,
            "ujets": 0.982,
        }

        # A list to add available variables to the evaluation files
        add_eval_variables: ["actualInteractionsPerCrossing"]

        # Working point used in the evaluation
        working_point: 0.77
    ```

    The other options are already set, although you can try and play around with the `working_point` or the `frac_values` if you want.

#### 5.2 Run the evaluation

After all settings are done, try to run the evaluation! This will produce several output files in the `results/` folder of your model folder. In these files, all the information needed for plotting are stored. We don't need the raw testing samples anymore. With these files, we can continue to make plots!

??? info "Hint: Add DIPS to the comparison taggers"

    Have a look [here](../trainings/evaluate.md#running-the-evaluation)

??? warning "Solution: Add DIPS to the comparison taggers"

    To run the evaluation, you need to switch to the `umami/umami` folder of your repo (if you are not already there) and execute the following command

    ```bash
    evaluate_model.py -c <path to train config file> -e <epoch to evaluate>
    ```

    where the `-e` option defines the epoch you chose for evaluation in the last step (the validation).


#### 5.4 Checkpoint Files Evaluation

If for some reason, the evaluation didn't work for you and there is no time to retry, you can download and continue with the following files. Please keep in mind that you need to safe the files to the correct places! To get the files, run the following command:

```bash
wget https://umami-ci-provider.web.cern.ch/tutorial/umami/checkpoint_evaluation/My_DL1d_Tutorial_Model.zip
```

After downloading, you can unpack the `.zip` file by running `unzip My_DL1d_Tutorial_Model.zip`. Copy the unzipped folder in your `umami/umami/` folder of your forked repo and adapt the paths inside the config files in `metadata/`! The difference between this version and the checkpoint files after the training is the configured `evaluation_settings` part and the now existing `results/` folder with the result files from the evaluation inside. Also, the SHAPley plots are present in the `plots/` folder.

On `lxplus` you can instead run the `cp` command to copy from EOS:

```bash
cp /eos/user/u/umamibot/www/ci/tutorial/umami/checkpoint_preprocessing/My_DL1d_Tutorial_Model.zip ./
```

### 6. Make performance plots of the evaluation results

In addition to the whole preprocessing, training and evaluation which is done, Umami also has some high level plotting functions based on [`Puma`](https://github.com/umami-hep/puma). The plotting is (again) completely configurable via `yaml` files. Examples for them can be found in the `examples/` folder of Umami.
Although we are working with DL1r(d) here, you can also check the DIPS version of that file. Most of the plots we will cover in the following sub-sections are given there.
The structure of the config file is rather simple. Each block is one plot. The name of the block is the filename of the output plot. In general, all plots have some common options, which are `type`, `models_to_plot` and `plot_settings`. The plot settings are mainly arguments for `puma`, which are explained a bit more [here](../plotting/plotting_umami.md#yaml-config-file). The `type` tells the plotting script, which type of plot will be plotted and the `models_to_plot` are the different inputs for the plot. We will cover that a bit more in detail in the sub-sections.

#### 6.1 Adapt the General Options

The first step is to create a new folder in your model directory called `eval_plots` for example. Create a `.yaml` file in there and try to add the first block of the plotting config called `Eval_paramters` with your settings.

??? info "Hint: Adapt the General Options"

    An example is given in the `examples/` folder in Umami named `plotting_umami_config_DL1r.yaml`. An explanation of the options is given [here](../plotting/plotting_umami.md#yaml-config-file)

??? warning "Solution: Adapt the General Options"

    The `Eval_parameters` are the general options we need to change. It should look like this in your file:

    ```yaml
    # Evaluation parameters
    Eval_parameters:
      Path_to_models_dir: <Path where your model directory is stored>
      model_name: <Name of your model (the model directory)>
      epoch: <The epoch you chose for evaluation>
      epoch_to_name: <True or False. Decide if the epoch number is added to your plot names>
    ```

#### 6.2 Probability Output Plot and Running the Script

After adapting the general options, we start with the first plot(s). The probability output plots which is simply plotting the 3 outputs of the network for our testing samples. Try to make plots for the `pb` output class with only your new DL1d version inside. The `tagger_name` of your freshly trained DL1d model is `dl1`. The name for the freshly trained model is derived from which tagger you trained (`dl1` for DL1* models, `dips` for DIPS models).

??? info "Hint: Probability Output"

    An example of this type of plot can be found in the `plotting_umami_config_dips.yaml`. Also, an explanation about this particular plot type can be found [here](../plotting/plotting_umami.md#probability)
    
??? warning "Solution: Probability Output"

    The plot config is the following:

    ```yaml
    DL1d_prob_pb:
      type: "probability"
      prob_class: "bjets"
      models_to_plot:
        My_DL1d_Model:
          data_set_name: "ttbar_r22"
          label: "My DL1d tagger"
          tagger_name: "dl1"
          class_labels: ["ujets", "cjets", "bjets"]
      plot_settings:
        logy: True
        bins: 50
        y_scale: 1.5
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets"
    ```

Now you can try to run the script. This can be done by switching again to the `umami/umami` folder and running the following command:

```bash
plotting_umami.py -c <path_to_your_plotting_config> -o <Name_of_the_output_folder> -f <plot_type>
```

The `-c` option defines again the path to your plotting config file, the `-o` defines the name of the output folder where all your plots will be stored which are produced by this script (If it is `test` for examples, a folder named `test` will be created in your model directory in which everything is stored) and the `-f` gives the output plot type, like `pdf` or `png`. For this, use the `-o` option with your `eval_plots` folder.
More details how to run the script can be found [here](../plotting/plotting_umami.md#executing-the-script)

To go a step further, you can now also add another entry to the `models_to_plot`, like the for Run 2 used DL1r. The `tagger_name` for this version is `DL1r`. Also, add this one above your DL1d model. The base (to which all ratios are calculated) is the first model in `models_to_plot`. 

??? warning "Solution: Probability Output - Multiple Models"

    The plot config is the following:

    ```yaml
    DL1d_prob_pb:
      type: "probability"
      prob_class: "bjets"
      models_to_plot:
        Recommended_DL1r:
          data_set_name: "ttbar_r22"
          label: "Recomm. DL1r"
          tagger_name: "DL1r"
          class_labels: ["ujets", "cjets", "bjets"]
        My_DL1d_Model:
          data_set_name: "ttbar_r22"
          label: "My DL1d tagger"
          tagger_name: "dl1"
          class_labels: ["ujets", "cjets", "bjets"]
      plot_settings:
        logy: True
        bins: 50
        y_scale: 1.5
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets"
    ```

#### 6.3 Discriminant Scores

The next type of plot is the combination of the probability outputs: The $b$-tagging discriminant score. The task is to now plot again the scores of your model. Afterwards, you can add DL1r again. Also, add the different working point vertical lines. When activating the working points lines, the working points are calculated using the sample you are currently plotting! These are not the official working points for the taggers!

??? info "Hint: Discriminant Scores"

    An example of this type of plot can be found in the `plotting_umami_config_DL1r.yaml`. Also, an explanation about this particular plot type can be found [here](../plotting/plotting_umami.md#scores)
    
??? warning "Solution: Discriminant Scores"

    The plot config is the following:

    ```yaml
    scores_DL1r:
      type: "scores"
      main_class: "bjets"
      models_to_plot:
        Recommended_DL1r:
          data_set_name: "ttbar_r22"
          label: "Recomm. DL1r"
          tagger_name: "DL1r"
          class_labels: ["ujets", "cjets", "bjets"]
        My_DL1d_Model:
          data_set_name: "ttbar_r22"
          label: "My DL1d tagger"
          tagger_name: "dl1"
          class_labels: ["ujets", "cjets", "bjets"]
      plot_settings:
        working_points: [0.60, 0.70, 0.77, 0.85] # Set Working Point Lines in plot
        bins: 50
        y_scale: 1.4
        figsize: [8, 6]
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets"
        Ratio_Cut: [0.5, 1.5]
    ```

#### 6.4 ROC Curves

Up next are the (in-)famous ROC plots. The ROC plots we are using in flavour tagging are a bit different to the ones used in pure ML talks/topics. We are plotting our signal efficiency (for $b$-tagging obviously the $b$-efficiency) on the x-axis vs the background rejections (here $c$- and light-flavour rejection) on the y-axis. Rejection means here the $\frac{1}{\text{efficiency}}$. Similar to the already discussed plot types, the ROC plots also use the `models_to_plot` option but has a twist on it. You need to set one entry per rejection you want to plot. `Puma` (and therefore Umami) are currently supporting up to two rejection types per ROC plot. The next task is to create the ROC plot for your new model with the recommended DL1r as the baseline with both $c$- and light-flavour rejection in the plot.

??? info "Hint: ROC Curves"

    An example of this type of plot can be found in the `plotting_umami_config_DL1r.yaml`. Also, an explanation about this particular plot type can be found [here](../plotting/plotting_umami.md#roc-curves)

??? warning "Solution: ROC Curves"

    ```yaml
    DL1d_Comparison_ROC_ttbar:
      type: "ROC"
      models_to_plot:
        DL1r_urej:
          data_set_name: "ttbar_r22"
          label: "recomm. DL1r"
          tagger_name: "DL1r"
          rejection_class: "ujets"
        DL1r_crej:
          data_set_name: "ttbar_r22"
          label: "recomm. DL1r"
          tagger_name: "DL1r"
          rejection_class: "cjets"
        My_DL1d_Model_urej:
          data_set_name: "ttbar_r22"
          label: "My DL1d Model"
          tagger_name: "dl1"
          rejection_class: "ujets"
        My_DL1d_Model_crej:
          data_set_name: "ttbar_r22"
          label: "My DL1d Model"
          tagger_name: "dl1"
          rejection_class: "cjets"
      plot_settings:
        draw_errors: True
        xmin: 0.5
        ymax: 1000000
        figsize: [9, 9]
        working_points: [0.60, 0.70, 0.77, 0.85]
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets,\n$t\\bar{t}$ validation sample, fc=0.018"
    ```

    One important thing to mention here: The `label` option for the two rejections of one tagger should be exactly the same. When they are exactly the same, it will only be shown once in the legend. The difference between $c$- and light-flavour rejection will be automatically added to the legend.

#### 6.5 Variable vs Efficiency/Rejection

Next up are the (nearly every time by Valerio or Alex :D requested) variable vs efficiency plots. These plots are (based on which flavour is chosen for the y-axis) either variable vs efficiency or variable vs rejection. These are binned plots and therefore we also need to provide a binning here. For now, we want to plot the $p_T$ vs $b$ efficiency for again both DL1r and your trained DL1d model.

??? info "Hint: pT vs c-rejection"

    An example of this type of plot can be found in the `plotting_umami_config_dips.yaml`. Also, an explanation about this particular plot type can be found [here](../plotting/plotting_umami.md#variable-vs-efficiency)

??? warning "Solution: pT vs c-rejection"

    ```yaml
    DL1d_pT_vs_crej:
      type: "pT_vs_eff"
      models_to_plot:
        Recommended_DL1r:
          data_set_name: "ttbar_r22"
          label: "Recomm. DL1r"
          tagger_name: "DL1r"
        My_DL1d_Model:
          data_set_name: "ttbar_r22"
          label: "My DL1d Model"
          tagger_name: "dl1"
      plot_settings:
        bin_edges: [20, 30, 40, 60, 85, 110, 140, 175, 250] # This is the recommended ttbar binning for pT
        flavour: "cjets" # This the flavour for the y-axis, in this case this corresponds to c-rejection
        variable: "pt"
        class_labels: ["ujets", "cjets", "bjets"] # The used classes
        main_class: "bjets" # The main class to define the b-tagging discriminant to calculate the working points
        working_point: 0.77
        fixed_eff_bin: False # Choose between an inclusive working point (False) or a per bin working point (True)
        figsize: [7, 5]
        logy: False
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets,\n$t\\bar{t}$ test sample"
        y_scale: 1.3
    ```

#### 6.6 Fraction Contour

The final plot we are going to cover in this tutorial is the fraction contour plot. The fraction contour plot is a plot to distinguish the best fraction value combinations for a given model (or compare different models). While the `evaluate_model.py` script is running, different combinations are of fraction values are calculated using the working point defined in the `evaluation_settings` in the train config. The default scanned values are combinations from $0.01$ to $1$ with $0.01$ steps. All combinations which adds up to $1$ are chosen and tested. If you want a finer scan, you need to add the `frac_step`, `frac_min` and `frac_max` options to the `evaluation_settings` in the train config and re-run `evaluate_model.py` but only the `rej_per_frac` part, which can be achieved with the following command:

```bash
evaluate_model.py -c <path to train config file> -e <epoch to evaluate> -s rej_per_frac
```

But now to the task. We want a fraction contour plot with for your DL1d model and the recommended DL1r model with markers at $f_c = 0.02$ and $f_u = 0.98$.

??? info "Hint: Fraction Contour"

    An example of this type of plot can be found in the `plotting_umami_config_DL1r.yaml`. Also, an explanation about this particular plot type can be found [here](../plotting/plotting_umami.md#fraction-contour-plot)

??? warning "Solution: Fraction Contour"

    ```yaml
    contour_fraction_ttbar:
      type: "fraction_contour"
      rejections: ["ujets", "cjets"]
      models_to_plot:
        dl1r:
          tagger_name: "DL1r"
          colour: "b"
          linestyle: "--"
          label: "Recomm. DL1r"
          data_set_name: "ttbar_r22"
          marker:
            cjets: 0.02
            ujets: 0.98
        My_DL1d:
          tagger_name: "dl1"
          colour: "r"
          linestyle: "-"
          label: "My DL1d"
          data_set_name: "ttbar_r22"
          marker:
            cjets: 0.02
            ujets: 0.98
      plot_settings:
        y_scale: 1.3
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets,\n$t\\bar{t}$ test sample, WP = 77 %"
    ```

### 7. (Optional) Plot the input variables for the given .h5 files

The last feature of Umami (which we will not cover in the tutorial if there is no time left) is the plotting of input variables from `.h5` files coming directly from the dumper. Using `Puma`, Umami is able to plot all the jet/track variables in the `.h5` files using a `yaml` config file. An example config file (`plotting_input_vars.yaml`) can be found in the `examples/` folder of the Umami repository. A detailed description how to use the config and run the input variable plotting is given [here](../plotting/plotting_inputs.md)

#### 7.1 Jet Input Variables

The first task here is to plot some jet-level input variables. Using the example config from the `examples/` folder of Umami (`plotting_input_vars.yaml`), the first step would be to remove the $\tau$ jet category from the `jet_input_vars` and also adapt the `Datasets_to_plot`. The files want to plot is the `inclusive_validation_ttbar_PFlow.h5` and the `inclusive_validation_zprime_PFlow.h5` which you can get when you follow the steps in chapter 2. Also, you need to change the `rnnip_p*` values to the `dipsLoose20220314v2_p*` values.

Also, you will need to change the default plot settings and adapt the `atlas_second_tag` to remove the $t\bar{t}$ from there.

??? info "Hint: Jet Input Variables"

    A detailed explanation of all availabe options is given [here](../plotting/plotting_inputs.md#input-variables-jets)

??? warning "Solution: Jet Input Variables"

    Just putting the correct `jets_input_vars` part of the config here.

    ```yaml
    jets_input_vars:
        variables: "jets"
        folder_to_save: jets_input_vars
        Datasets_to_plot:
            ttbar:
                files: <path_palce_holder>/inclusive_validation_ttbar_PFlow.h5
                label: "$t\\bar{t}$"
            zprime:
                files: <path_palce_holder>/inclusive_validation_zprime_PFlow.h5
                label: "$Z'$"
        plot_settings:
            <<: *default_plot_settings
        class_labels: ["bjets", "cjets", "ujets"]
        special_param_jets:
            SV1_NGTinSvx:
                lim_left: 0
                lim_right: 19
            JetFitterSecondaryVertex_nTracks:
                lim_left: 0
                lim_right: 17
            JetFitter_nTracksAtVtx:
                lim_left: 0
                lim_right: 19
            JetFitter_nSingleTracks:
                lim_left: 0
                lim_right: 18
            JetFitter_nVTX:
                lim_left: 0
                lim_right: 6
            JetFitter_N2Tpair:
                lim_left: 0
                lim_right: 200
        xlabels:
            # here you can define xlabels, if a variable is not in this dict, the variable name
            # will be used (i.e. for pT this would be 'pt_btagJes')
            pt_btagJes: "$p_T$ [MeV]"
        binning:
            JetFitter_mass: 100
            JetFitter_energyFraction: 100
            JetFitter_significance3d: 100
            JetFitter_deltaR: 100
            JetFitter_nVTX: 7
            JetFitter_nSingleTracks: 19
            JetFitter_nTracksAtVtx: 20
            JetFitter_N2Tpair: 201
            JetFitter_isDefaults: 2
            JetFitterSecondaryVertex_minimumTrackRelativeEta: 11
            JetFitterSecondaryVertex_averageTrackRelativeEta: 11
            JetFitterSecondaryVertex_maximumTrackRelativeEta: 11
            JetFitterSecondaryVertex_maximumAllJetTrackRelativeEta: 11
            JetFitterSecondaryVertex_minimumAllJetTrackRelativeEta: 11
            JetFitterSecondaryVertex_averageAllJetTrackRelativeEta: 11
            JetFitterSecondaryVertex_displacement2d: 100
            JetFitterSecondaryVertex_displacement3d: 100
            JetFitterSecondaryVertex_mass: 100
            JetFitterSecondaryVertex_energy: 100
            JetFitterSecondaryVertex_energyFraction: 100
            JetFitterSecondaryVertex_isDefaults: 2
            JetFitterSecondaryVertex_nTracks: 18
            pt_btagJes: 100
            absEta_btagJes: 100
            SV1_Lxy: 100
            SV1_N2Tpair: 8
            SV1_NGTinSvx: 20
            SV1_masssvx: 100
            SV1_efracsvx: 100
            SV1_significance3d: 100
            SV1_deltaR: 10
            SV1_L3d: 100
            SV1_isDefaults: 2
            dipsLoose20220314v2_pb: 50
            dipsLoose20220314v2_pc: 50
            dipsLoose20220314v2_pu: 50
        flavours:
            b: 5
            c: 4
            u: 0
    ```

#### 7.2 Track Input Variables

Similar to the jet-level input variable plotting, you can also plot the track-level input variables. Using the example config file again, you now need to change the `tracks_input_vars`. Of course we need to change the `Datasets_to_plot` again but add the correct track collection which is `tracks_loose`. Also, you can remove all entries from the `n_leading` list except for the `None`. For the binning, please remove the `btagIp_` from the `d0` ad the `z0SinTheta`. You can also look up all available track variables used `h5ls -v`.

??? info "Hint: Track Input Variables"

    A detailed explanation of all availabe options is given [here](../plotting/plotting_inputs.md#input-variables-tracks)

??? warning "Solution: Track Input Variables"

    Just putting the correct `tracks_input_vars` part of the config here.

    ```yaml
    tracks_input_vars:
        variables: "tracks"
        folder_to_save: tracks_input_vars
        Datasets_to_plot:
            ttbar:
                files: <path_palce_holder>/inclusive_validation_ttbar_PFlow.h5
                label: "$t\\bar{t}$"
                tracks_name: "tracks_loose"
            zprime:
                files: <path_palce_holder>/inclusive_validation_zprime_PFlow.h5
                label: "$Z'$"
            tracks_name: "tracks_loose"
        plot_settings:
            <<: *default_plot_settings
            sorting_variable: "ptfrac"
            n_leading: [None]
            ymin_ratio_1: 0.5
            ymax_ratio_1: 1.5
        binning:
            IP3D_signed_d0_significance: 100
            IP3D_signed_z0_significance: 100
            numberOfInnermostPixelLayerHits: [0, 4, 1]
            numberOfNextToInnermostPixelLayerHits: [0, 4, 1]
            numberOfInnermostPixelLayerSharedHits: [0, 4, 1]
            numberOfInnermostPixelLayerSplitHits: [0, 4, 1]
            numberOfPixelSharedHits: [0, 4, 1]
            numberOfPixelSplitHits: [0, 9, 1]
            numberOfSCTSharedHits: [0, 4, 1]
            ptfrac: [0, 5, 0.05]
            dr: 100
            numberOfPixelHits: [0, 11, 1]
            numberOfSCTHits: [0, 19, 1]
            d0: 100
            z0SinTheta: 100
        class_labels: ["bjets", "cjets", "ujets"]
    ```
#### 7.3 Number of Tracks per Jet

One final variable (which needs its own entry here) is the number of tracks per jet. The strcuture here is similar to the track input variables. Using the example config again, we only need to change the `Datasets_to_plot`. Change them like the change you made to the track variables.

??? info "Hint: Number of Tracks per Jet"

    A detailed explanation of all availabe options is given [here](../plotting/plotting_inputs.md#number-of-tracks-per-jet)

??? warning "Solution: Number of Tracks per Jet"

    Just putting the correct `nTracks` part of the config here.

    ```yaml
    nTracks:
        variables: "tracks"
        folder_to_save: nTracks
        nTracks: True
        Datasets_to_plot:
            ttbar:
                files: <path_palce_holder>/inclusive_validation_ttbar_PFlow.h5
                label: "$t\\bar{t}$"
                tracks_name: "tracks_loose"
            zprime:
                files: <path_palce_holder>/inclusive_validation_zprime_PFlow.h5
                label: "$Z'$"
            tracks_name: "tracks_loose"
        plot_settings:
            <<: *default_plot_settings
            ymin_ratio_1: 0.5
            ymax_ratio_1: 2
        class_labels: ["bjets", "cjets", "ujets"]
    ```

#### 7.4 Run the Input Variable Plotting

The final step is to run the jet- and track-level variable plotting. Try to run this using the `plot_input_variables.py` script.

??? info "Hint: Run the Input Variable Plotting"

    A detailed explanation how to run the different input variable plotting parts is given [here](../plotting/plotting_inputs.md)

??? warning "Solution: Run the Input Variable Plotting"

    To run the plotting, you need to switch to the `umami/umami` folder and run the following command:

    ```bash
    plot_input_variables.py -c <path/to/config> --jets
    ```

    The `--jets` flag here tells the script to run the jet-level input variable plotting. For the tracks, you need to run the following command:

    ```bash
    plot_input_variables.py -c <path/to/config> --tracks
    ```

    The config file mentioned here is your adapted `plotting_input_vars.yaml` config file.
