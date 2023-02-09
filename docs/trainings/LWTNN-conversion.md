# LWTNN Conversion

To run the models also within [ATHENA](https://gitlab.cern.ch/atlas/athena?nav_source=navbar), it is necessary to convert the keras models to `json` files compatible with [lwtnn](https://github.com/lwtnn/lwtnn).



To convert the architecture to the `json` format used by lwtnn, the script [kerasfunc2json.py](https://github.com/lwtnn/lwtnn/blob/master/converters/kerasfunc2json.py) will be used. The usage of this script is described as
```
usage: kerasfunc2json.py [-h] arch_file hdf5_file [variables_file]

Converter from Keras saved NN to JSON

positional arguments:
  arch_file       architecture json file
  hdf5_file       Keras weights file
  variables_file  variable spec as json

optional arguments:
  -h, --help      show this help message and exit

____________________________________________________________________
Variable specification file

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file.

Here `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

If no file is provided, a template will be generated.
```

Thus the `arch_file`, `hdf5_file` and `variables_file` need to be produced.

### Variables File

To produce the json `variables_file` a script is provided which can be executed like

```bash
python scripts/create_lwtnn_vardict.py -s <scale_dict.json> -v <Variables.yaml> -t <TAGGER>
```

Currently, 3 taggers are supported `DL1`, `DIPS` and `UMAMI`. The output will be a file called `lwtnn_vars.json`. If you
want to rename this file, you can provide a different name with the `-o` or `--output` command line option.

If you are converting a track-based tagger, like DIPS or UMAMI, you also need to set the name of the
track collection in athena and the name of the track collection inside umami.

```bash
python scripts/create_lwtnn_vardict.py -s <scale_dict.json> -v <Variables.yaml> -t <TAGGER> -n <athena-track-collection> --tracks_name <umami-track-collection>
```

the umami track collection is the name of the tracks which were used for training. By default this is `tracks`. If you
are not sure which tracks you used, have a look in your train config. The option `tracks_name` defines this. Use the value of that.
The `--sequence_name` or `-n` option defines the required track selection in athena described in the follwing.

#### Track Selection

There are different track selections [defined in Athena](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx#L398-531) which can be used and are specified in the lwtnn json file with their definitions [here](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx#L739-759) together with the sort order of the tracks. The following options are usually used for DIPS:

* Standard (R22): `tracks_r22default_sd0sort`
* Loose (R22): `tracks_r22loose_sd0sort`

There are also other track selections, which have been used in the past or are used for specialised R&D studies:

* Standard: `tracks_ip3d_sd0sort`
* Loose: `tracks_dipsLoose202102_sd0sort`
* Tight (Run-4 Upgrade): `tracks_dipsTightUpgrade_sd0sort`
* Loose (Run-4 Upgrade): `tracks_dipsLooseUpgrade_sd0sort`
* Loose (no IP cuts): `tracks_loose202102NoIpCuts_sd0sort`
* All: `tracks_all_sd0sort`

### Architecture and Weight File
Splitting the model into architecture `arch_file` and weight file `hdf5_file` can be done via

```bash
 python scripts/conv_lwtnn_model.py -m <model.h5>
```
This script will return two files which are in this case `architecture-lwtnn_model.json` and `weights-lwtnn_model.h5`. If you want to change the basename of
those files, which per default `lwtnn_model`, you can give the command line option `-o` or `--output_base` and your desired basename.

### Final JSON File
Finally, the three produced files can be merged via [kerasfunc2json.py](https://github.com/lwtnn/lwtnn/blob/master/converters/kerasfunc2json.py).
Before you can use this script, you have to clone [the lwtnn repo](https://github.com/lwtnn/lwtnn) to the directory where your architecture, weight and variables files are located.
Afterwards, you can run the script with

```bash
git clone git@github.com:lwtnn/lwtnn.git
python lwtnn/converters/kerasfunc2json.py architecture-lwtnn_model.json weights-lwtnn_model.h5 lwtnn_vars.json > FINAL-model.json
```

To test if the created model is properly working you can use the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) and add the created model to a config (e.g. [EMPFlow.json](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/EMPFlow.json)). This is explained a bit more detailed [here](https://training-dataset-dumper.docs.cern.ch/configuration/#dl2-config). To add the values also to the output, you need to add your probability variables also to the [single-btag-variables-all.json](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/fragments/pflow-variables-all.json) or whatever variables file you are using. Just add them to the other tagger probability values.

The training-dataset-dumper provides documentation for [running locally](https://training-dataset-dumper.docs.cern.ch/basic_usage/) and for [running on the grid](https://training-dataset-dumper.docs.cern.ch/grid/).

In both cases **it is critically important** that you run with full precision, as the flavour tagging algorithms in Athena are also evaluated with full precision (32bit). For local dumps this can be achieved with the command line argument `--force-full-precision`. 

After having the hdf5 ntuples produced, the script [`check_lwtnn_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/scripts/check_lwtnn_model.py) can be used to compare the athena evaluation with the keras evaluation. The script requiries a short config files which can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/scripts/check_lwtnn-model_config.yaml). This looks like this:

```
§§§scripts/check_lwtnn-model_config.yaml§§§
```

You can simply run the script via

```bash
python scripts/check_lwtnn_model.py -c scripts/check_lwtnn-model_config.yaml
```

the `-c` option is the path to the config file.

The output should look similiar to the following:
```
Differences off 1e-6 1.34 %
Differences off 2e-6 0.12 %
Differences off 3e-6 0.03 %
Differences off 4e-6 0.02 %
Differences off 5e-6 0.01 %
Differences off 1e-5 0.0 %
```
This means that the networks scores are matching within a precision of 1e-5 for all the jets in the produced ntuple, 0.01% of the tested jets have a difference in the predicted probabilities between 5e-6 and 1e-6, and so on... Typically, we are happy when the scores match within 1e-5.