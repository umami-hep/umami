## Ntuple preparation

After the ntuple production (training-dataset-dumper), the first step of the preprocessing is the preparation of the different flavour files. In this step, the different flavours that are to be used for the training are extracted from the `.h5` files and written into extra files. While extracting the jets, different cuts are applied and the splitting into training/validation/test is done.

### Config file
As already mentioned in the [overview](preprocessing/Overview.md), the preprocessing is configured using [`.yaml`](https://en.wikipedia.org/wiki/YAML) config files. We start with some general options that are needed by multiple preprocessing steps and should be set at the very beginning of the preprocessing:

```yaml
§§§examples/preprocessing/PFlow-Preprocessing.yaml:121:148§§§
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `outfile_name` | `str` | Name of the output file of the preprocessing. The different steps will add append some info to this name to produce their output files (Like `_resampled`). |
| `outfile_name_validation` | `str` | Name of the validation output file of the preprocessing. The different steps will add append some info to this name to produce their output files (Like `_resampled`). |
| `plot_name` | `str` | Defines the names of the control plots which are produced in the preprocessing. |
| `plot_type` | `str` | Defines the filetype in which the preprocessing plots are saved. Default is "pdf" |
| `use_atlas_tag` | `bool` | Define if you want to have the ATLAS Logo at the top left of the plot. |
| `atlas_first_tag` | `str` | Define the text after the ATLAS Logo. By default "Simulation Internal". |
| `atlas_second_tag` | `str` | Defines the label in the control plots which are made in the preprocessing. This is the text which is written under the "ATLAS". |
| `legend_sample_category` | `bool` | Whether to include sample categories in the legends of plots before resampling. Set to False for Boosted tagging. |
| `var_file` | `str` | Path to the variable dict which is used. Default configs can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/umami/configs) |
| `dict_file` | `str` | Full path (with filename) to the scale dict. This must be a `.json` file! |

For plot related variables, Umami supports here also all available options of [`PUMA`](https://github.com/umami-hep/puma/).
The `var_dict` and `dict_file` options are normally set in the [`Preprocessing-parameters.yaml`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/preprocessing/Preprocessing-parameters.yaml) file. A snapshot of these two variables is shown here:

```yaml
§§§examples/preprocessing/Preprocessing-parameters.yaml:16:20§§§
```

For the preparation step, we also need the some more parts of the preprocessing config, which are described in the following sections.

#### Preprocessing Parameters
```yaml
parameters: !include Preprocessing-parameters.yaml
```

This line specifies where the ntuples (which are used) are stored and where to save the output of the preprocessing. You can find an example file [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/preprocessing/Preprocessing-parameters.yaml). In the following the options from the `Preprocessing-parameters.yaml`, which are needed for the preparation step, will be explained:

```yaml
§§§examples/preprocessing/Preprocessing-parameters.yaml:1:11§§§
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `ntuple_path` | `str` | The path where the ntuples are stored. This is the folder where the different process folders with the `.h5` files are stored (the folder with the `user.*` folders from `rucio`) |
| `sample_path` | `str` | The path were the prepared samples will be stored (the output files of this preprocessing step). |

#### Cut Templates
```yaml
cut_parameters: !include Preprocessing-cut_parameters.yaml
```
This line includes the cut parameter file `Preprocessing-cut_parameters.yaml`

```yaml
§§§examples/preprocessing/Preprocessing-cut_parameters.yaml§§§
```

The cuts defined in this section are templates for the cuts of the different flavour for $t\bar{t}$/$Z'$. `ttbar_train` and `zprime_train` are the jets which are used for training while validation/test are the templates for validation and test.
The cuts which are to be applied can be defined in these templates. For example, we can define a cut on the `eventNumber` with a modulo operator. This modulo operator defines that all jets are used, where the `eventNumber` is equal to something. The something can be defined by the `condition`. With this specific cut on the `eventNumber`, we are splitting the $t\bar{t}$/$Z'$ in train/validation/test to ensure no jet is used twice. In the default case, $\frac{2}{3}$ of the jets are used for training, $\frac{1}{6}$ for validation and $\frac{1}{6}$ for evaluation.
Another cut which can be applied is the `pt_btagJes`, which is a cut on the jet $p_T$. Works the same as the modulo operator. In the default case, we want $t\bar{t}$ for the jet $p_T$ region from $20\,\text{GeV}$ to $250\,\text{GeV}$ and $Z'$ for the region above $250\,\text{GeV}$.

??? info "Nested cuts on same variable"
    It is possible to also apply nested cuts on the same variable e.g. like this
    ```yaml
    .cuts_template_zprime_train: &cuts_template_zprime_train
      cuts:
        - eventNumber:
            operator: mod_6_<=
            condition: 3
        - pt_btagJes:
            operator: ">="
            condition: 2.5e5
        - pt_btagJes:
            operator: "<="
            condition: 3e6
    ```

#### File- and Flavour Preparation
```yaml
§§§examples/preprocessing/PFlow-Preprocessing.yaml:5:14§§§
```
In the `Preparation` section, different options need to be set and files/flavours defined. The options that need to be set are given in the following table:

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `batchsize` | `int` | Number of jets that are loaded per iteration step from the `.h5` files. This is to not load the whole `.h5` file at once, which could lead to exhaustion of the available RAM. This number can adjusted to the amount of RAM that is available. |
| `input_h5` | `dict` | The dict with the file types which are used in the preprocessing. Here `ttbar` and `zprime` are the internal names of these files. Both are also dicts. |
| `path` | `str` | Dict entry of `ttbar` and `zprime`. This gives the path to the folder where the process folders are stored (this is the same as `ntuple_path` in `Preprocessing-Parameters.yaml`). |
| `file_pattern` | `str` | Dict entry of `ttbar` and `zprime`. This is the specific path to the `.h5` files of the process. The `path` and `file` are in the script merged to form the global path to the `.h5` files. Wildcards are supported! |
| `randomise` | `bool` | Optional setting to randomise the samples which are read in. can be useful if you have several data taking campaigns and you want a representative sample, especially important for validation and testing. (a random seed is set to maintain reproducibility.) by default `False` |
| `jets_name` | `str` | Optional setting the name of the jet collection. The default is "jets". After the preparation step, the jet collection will always be called "jets". |
| `collection_name` | `str` | Optional setting to define a top level in which the jet collection and track collections are stored. By default, this is not set. If you have, for example, "collection_1/jets", this option would be "collection_1". After the preparation step, the jet- and track collections will always be at the top level. |

In the example above, we specify the paths for `ttbar` and `zprime` ntuples. Since we define them there, we can then use these ntuples in the `samples` section. So if you want to use e.g. Z+jets ntuples for $bb$-jets, define the corresponding `zjets` entry in the ntuples section before using it in the `samples` section.

```yaml
§§§examples/preprocessing/Preprocessing-samples.yaml§§§
```

The last part is the exact splitting of the flavours. In `samples`, you define for each of $t\bar{t}$/$Z'$ and training/validation/testing the flavours you want to use. In the example case, these samples are stored in another yaml file called `Preprocessing-samples.yaml` to keep the config file a bit smaller. But you can also simply add them directly to the config file.

The sample are defined as dicts with the following options:

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `type` | `str` | Type of process that this file will be. |
| `category` | `str` | This defines that flavour that will be extracted in this file. You can either use a flavour like `bjets` or `inclusive`, which will use all tracks regardless of their flavour. |
| `n_jets` | `int` | Number of jets you want for this specific flavour. If not specified, it is set to 4M.|
| `cuts` | `list` | A list of cuts that are applied. In the default case, this is added via templates which are added with `<<:`. |
| `output_name` | `str` | Name of the output file where the prepared file will be stored. |


**Note**: The `n_jets` should be as high as possible for the train files! This is just the number of jets for this flavour which are extracted from the `.h5` files coming from the dumper. The resampling algorithm uses these samples to get the jets for building the final training sample, but it only uses as much as needed! Only for the validation and testing files we suggest to use something around `4e6` (otherwise the loading later on takes quite some time).

??? info "Create samples list automatically"
    If you don't want to define all the different samples one by one, you can also use the [`create_preprocessing_samples.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/scripts/create_preprocessing_samples.py) script. To use the script, you just need to adapt it to your needs:

    ```python
    categories = ["ujets", "cjets", "bjets"]
    sample_types = ["ttbar", "zprime"]
    n_jets = {
        "training": int(10e6),
        "validation": int(4e6),
        "testing": int(4e6),
    }
    ```
    | Setting | Type | Explanation |
    | ------- | ---- | ----------- |
    | `categories` | `list` | List with the flavours to extract. |
    | `sample_types` | `list` | List with the sample types you want to use. |
    | `n_jets` | `dict` | Dict with the number of jets which are to be extracted from the `.h5` files for the different usages of the samples (this must be training/validation/testing! You can't rename them!). |

    This will create the content of the `samples` dict of the preprocessing config file. The different training samples, i.e. `training_ttbar_bjets` etc. and also the validation (`validation_ttbar` and `validation_zprime`) and testing samples (`testing_ttbar` and `testing_zprime`) will be created. In addition, the by flavour separated validation files (i.e `validation_ttbar_bjets`) needed for the hybrid validation creation are also prepared. Which cut template is used is also based on the name of the cut template. This must be `.cuts_template_training_ttbar` for the `training` case of `ttbar`.

    To add your file to the preprocessing config, you can simply `!include` it like the preprocessing parameters. Just exchange the `samples` with the different samples defined in it with:

    ```yaml
    samples: !include <Path to your samples yaml file>
    ```


### Run the Preparation
To run the preparation step, switch to the `umami/umami/` folder and run the following command:

```bash
preprocessing.py --config <path to config file> --prepare
```

The preprocessing will start in order of the files defined in `samples:` to preprare the different selected samples. This step is one of the longest steps of the preprocessing if not parallelised. You can run the preparation in for the different defines samples one per job, by defined which sample is to be prepared.

For example, to run the sample preparation for the prepared training _b_-jet sample `training_ttbar_bjets`, which has been defined in the config file in the `preparation: samples:` block, execute:

```bash
preprocessing.py --config <path to config file> --prepare --sample training_ttbar_bjets
```

The result of the commands are the prepared samples which are ready for resampling. Also, please keep in mind that in this step also the validation and testing files are prepared. You can also run them in separate with the same as the `training_ttbar_bjets` with, for example, the command:

```bash
preprocessing.py --config <path to config file> --prepare --sample testing_ttbar
```

## Ntuple Preparation for VR track jets

The preparation of VR track jet input files for training, validation, and testing datasets is very similar to the workflow described above for PFlow jets.
The main difference is that a special set of config files is used, which accounts for the different hybrid sample composition when using VR track jets. The main differences to PFlow jets are:

- implementation of the VR track jet overlap removal
- only the four leading jets in pt are used for the ttbar sample
- only the two leading jets in pt are used for the Z' sample
- the transition from ttbar to Z' occurs in a region in pt and not by a fixed pt cut

## Ntuple Preparation for bb-jets

TODO: Rewrite this!
The double b-jets will be taken from Znunu and Zmumu samples. The framework still requires some updates in order to process those during the hybrid sample creation stage.
