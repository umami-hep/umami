# LWTNN Conversion

To run the models also within [ATHENA](https://gitlab.cern.ch/atlas/athena?nav_source=navbar), it is necessary to convert the keras models to `json` files compatible with [lwtnn](https://github.com/lwtnn/lwtnn).


## DIPS

For now there is only a workaround available, and needs to be generalised.

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
```
python scripts/create_lwtnn_vardict.py -s <scale_dict-22M.json> -v umami/configs/Dips_Variables.yaml -o lwtnn_vars-dips.json -t DIPS -n tracks_ip3d_sd0sort
```
the `--sequence_name` or `-n` option defines the required track selection described in the follwing.
#### Track Selection

There are different track selections [defined in Athena](https://gitlab.cern.ch/atlas/athena/blob/21.2/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DL2.cxx#L302-331) which can be used and are specified in the lwtnn json file with their definitions [here](https://gitlab.cern.ch/atlas/athena/-/blob/21.2/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DL2HighLevel.cxx#L148-156) together with the sort order of the tracks. The following options are usually used for DIPS:
    * Standard: `tracks_ip3d_sd0sort`
    * Loose: `tracks_dipsLoose202102_sd0sort`


### Architecture and Weight File
Splitting the model into architecture `arch_file` and weight file `hdf5_file` can be done via

```
 python scripts/conv_lwtnn_model.py -m <model.h5> -o lwtnn_dips
```
This script will return two files which are in this case `architecture-lwtnn_dips.json` and `weights-lwtnn_dips.h5`

### Final JSON File
Finally, the three produced files can be merged via [kerasfunc2json.py](https://github.com/lwtnn/lwtnn/blob/master/converters/kerasfunc2json.py) 

```
python kerasfunc2json.py architecture-lwtnn_dips.json weights-lwtnn_dips.h5 lwtnn_vars-dips.json > DIPS-model.json
```

To test if the created model is properly working you can use the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) and add the created model to a config (e.g. [EMPFlow.json](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/configs/single-b-tag/EMPFlow.json)). This can exemplarily look like
```json
{
    "jet_collection": "AntiKt4EMPFlowJets_BTagging201903",
    "jet_calibration_collection": "AntiKt4EMPFlow",
    "jet_calib_file": "JES_data2017_2016_2015_Consolidated_PFlow_2018_Rel21.config",
    "cal_seq": "JetArea_Residual_EtaJES_GSC_Smear",
    "cal_area": "00-04-82",
    "do_calibration": "true",
    "run_augmenters": "false",
    "vr_cuts": "false",
    "jvt_cut": 0.5,
    "pt_cut": 20000,
    "n_tracks_to_save": 40,
    "track_sort_order": "d0_significance",
    "track_pt_minimum": 1000,
    "track_d0_maximum": 1,
    "track_z0_maximum": 1.5,
    "dl2_configs": [
        {
            "nn_file_path": "DIPS-model.json",
            "output_remapping": {
                "DIPS_pu": "dips_pu",
                "DIPS_pc": "dips_pc",
                "DIPS_pb": "dips_pb"
            }
        }
    ],
    "variables": {
        "btag": {
            "file": "single-btag-variables.json",
            "doubles": [
                "dips_pu",
                "dips_pc",
                "dips_pb"
                ]
        },
        "track": {
            "file": "single-btag-track-variables.json"
        }
    }
}
```
